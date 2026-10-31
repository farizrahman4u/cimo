"""Linking: cimo's runtime library, the Python hooks, and host libraries on request.

Link model (requirements.md, "Language" and "Execution model"):

- The runtime (`cimo/runtime/**/*.c`: libc subset from musl, cimo's heap, stdout) is an
  archive with one member per source file. A member is compiled by cimo itself the first
  time a module needs it, and cached as assembly text in `<cache>/runtime-<key>/` (in memory
  if there is no writable cache). `runtime/symbols.txt` (generated) says which member
  defines which symbol, so nothing else is compiled; what a member uses is read from its
  object. The key hashes the runtime sources and headers, the bundled headers, and the
  compiler's own sources, so any change recompiles.
- Each module is linked **statically**: like `ld` with an archive, only the runtime objects
  that resolve its undefined symbols (transitively) are pulled in and merged with the module
  into one object, which `cimo.loader` maps. So every module has its own copy of the runtime
  code and data it uses: its own heap and stdout buffer, nothing shared between modules, no
  stubs between kernel and runtime, and later the middle end can inline runtime functions.
- What remains undefined must be a hook into Python (`__cimo_write`, the heap's `grow`, ...),
  bound to a ctypes callback, or a symbol of a library named in `link=` (resolved with dlsym,
  reached through the loader's stubs and GOT). Anything else is a CompileError naming the
  symbol. The default build resolves nothing from the host process.
"""

import codecs
import ctypes
import ctypes.util
import hashlib
import json
import os
import sys
import threading
import time

from . import assembler as _asm
from . import resources as _res
from .execution import CTX_OUT

__all__ = ['LinkError', 'runtime', 'link', 'open_library', 'Stdout', 'RUNTIME_DIR']

RUNTIME_DIR = _res.path('runtime')
RUNTIME_INCLUDE = os.path.join(RUNTIME_DIR, 'internal')
_APPLE = sys.platform == 'darwin'
_WINDOWS = sys.platform == 'win32'

# Compiler sources whose changes can change the runtime's code (part of the cache key):
# conservative, so editing the driver rebuilds the runtime too (~9 s).
_COMPILER = ('preprocessor.py', 'frontend', 'middle', 'backend', 'include', 'wasm', 'driver.py',
             'linker.py', 'execution.py')

SYMBOLS = os.path.join(RUNTIME_DIR, 'symbols.txt')

# The last link that compiled runtime members: {'seconds': ..., 'files': ...}, for tests.
last_build = None

# progress(event, **info) while a link compiles runtime members, instead of a line on stderr:
# 'file' (name=member) before each, 'done' (files=n, seconds=s) after. The CLI sets it.
progress = None


class LinkError(Exception):
    pass


def runtime_sources():
    """Every .c file under cimo/runtime, as paths relative to it."""
    return [os.path.relpath(p, RUNTIME_DIR) for p in _res.files(RUNTIME_DIR, ('.c',))]


def _hash(h, paths):
    for p in paths:
        h.update(os.path.relpath(p, _res.ROOT).replace(os.sep, '/').encode() + b'\0')
        h.update(_res.read(p))
        h.update(b'\0')


def _compiler_paths():
    paths = []
    for c in _COMPILER:
        p = _res.path(c)
        paths += [p] if c.endswith('.py') else _res.files(p, ('.py', '.h'))
    return paths


_compiler_key = None
_TARGETS = ('arm64', 'arm64_apple', 'arm64_win')


def _bundle_keys():
    """The keys a bundle was built with (bundle-keys.json, see cimo.bundle): its embedded
    sources are stripped of comments, so they hash differently from the sources the keys
    describe. None outside a bundle."""
    p = _res.path('bundle-keys.json')
    return json.loads(_res.read_text(p)) if _res.isfile(p) else None


def compiler_key():
    """Hash of the compiler's sources, computed once per process: part of every cache key, so
    a changed compiler never reuses assembly the old one generated."""
    global _compiler_key
    if _compiler_key is None:
        keys = _bundle_keys()
        if keys:
            _compiler_key = keys['compiler']
        else:
            h = hashlib.sha256(b'cimo-compiler-1\0')
            _hash(h, _compiler_paths())
            _compiler_key = h.hexdigest()[:20]
    return _compiler_key


def cache_keys():
    """{'compiler': key, 'runtime-<target>': key}: what a bundle records in bundle-keys.json."""
    out = {'compiler': compiler_key()}
    for t in _TARGETS:
        out['runtime-' + t] = _key(t)
    return out


def _key(target):
    keys = _bundle_keys()
    if keys:
        return keys['runtime-' + target]
    h = hashlib.sha256(b'cimo-runtime-1\0' + target.encode() + b'\0')
    _hash(h, _res.files(RUNTIME_DIR, ('.c', '.h')) + _compiler_paths())
    return h.hexdigest()[:20]


def _globals(obj):
    """(defined global symbols, undefined symbols) of an object."""
    defs, undefs = set(), set()
    for y in obj.symbols.values():
        if y.defined:
            if y.globl:
                defs.add(y.name)
        elif y.common:
            defs.add(y.name)
        else:
            undefs.add(y.name)
    return defs, undefs


def compile_runtime_source(path, target):
    """One runtime source file to assembly text, with the options the runtime is built with:
    the driver's optimizations and math builtins, but no FMA contraction (musl's algorithms
    are written for -ffp-contract=off, which is how musl itself is built). __FP_FAST_FMA(F) as gcc defines them
    on arm64: musl's pow, log, exp2f, ... then use fma() (one instruction) for exact products."""
    from . import driver
    from . import execution as _exe
    from .preprocessor import preprocess
    src = _res.read_text(path)
    pp = preprocess(src, path, include_dirs=(RUNTIME_INCLUDE,),
                    defines={'__FP_FAST_FMA': '1', '__FP_FAST_FMAF': '1'}, target=target,
                    warn=lambda *a, **k: None)
    ssa, params = driver._frontend(pp, path, target, fma=False, builtins=True, prune=True)
    return driver._qbe(ssa, params, stkchk=_exe.stack_checked(ssa), target=target)


def read_symbols(path=SYMBOLS):
    """runtime/symbols.txt -> {member: [C names it defines]}."""
    out = {}
    for line in _res.read_text(path).splitlines():
        if line and not line.startswith('#'):
            member, _, names = line.partition(':')
            out[member] = names.split()
    return out


def symbol_table(target=None):
    """runtime/symbols.txt's text, by compiling every runtime source (`cimo
    __update-runtime-symbols` writes it)."""
    from .driver import TARGET
    target = target or TARGET
    syntax = 'macho' if target == 'arm64_apple' else 'elf'
    pre = '_' if syntax == 'macho' else ''
    lines = ['# Generated by `cimo __update-runtime-symbols`: the global symbols each runtime',
             '# source defines, so a link compiles only the members it needs. Checked by',
             '# tests/test_runtime.py.']
    for rel in runtime_sources():
        obj = _asm.assemble(compile_runtime_source(os.path.join(RUNTIME_DIR, rel), target),
                            syntax)
        defs = sorted(d[len(pre):] if d.startswith(pre) else d for d in _globals(obj)[0])
        lines.append('%s: %s' % (rel.replace(os.sep, '/'), ' '.join(defs)))
    return '\n'.join(lines) + '\n'


class Runtime:
    """The runtime library as an archive. Members (one per source file) are compiled when a
    link first needs them, then cached: as assembly in `dir` (None: in memory) and assembled
    objects for the life of the process."""

    def __init__(self, key, target, syntax, dir):
        self.key, self.target, self.syntax, self.dir = key, target, syntax, dir
        pre = '_' if syntax == 'macho' else ''
        self.members = read_symbols()  # member -> C names it defines
        self.provider = {}  # symbol (with the target's prefix) -> member
        for m, names in self.members.items():
            for n in names:
                self.provider.setdefault(pre + n, m)
        self._texts = {}
        self._objs = {}
        self._lock = threading.RLock()
        self.compiled = []  # members compiled by this process, in order

    def text(self, member):
        with self._lock:
            t = self._texts.get(member)
            if t is not None:
                return t
            p = self.dir and os.path.join(self.dir, member_file(member))
            if p and os.path.exists(p):
                with open(p) as f:
                    t = f.read()
            else:
                t = self._compile(member)
                if p:
                    try:
                        os.makedirs(self.dir, exist_ok=True)
                        tmp = p + '.tmp%d' % os.getpid()
                        with open(tmp, 'w') as f:
                            f.write(t)
                        os.replace(tmp, p)
                    except OSError:
                        pass
            self._texts[member] = t
            return t

    def _compile(self, member):
        from .driver import CompileError
        if progress:
            progress('file', name=member)
        try:
            t = compile_runtime_source(os.path.join(RUNTIME_DIR, member), self.target)
        except Exception as e:
            raise CompileError('building the runtime: %s: %s' % (member, e)) from None
        self.compiled.append(member)
        return t

    def object(self, member):
        """(object, defs, undefs) of a member."""
        with self._lock:
            o = self._objs.get(member)
            if o is None:
                obj = _asm.assemble(self.text(member), self.syntax)
                defs, undefs = _globals(obj)
                pre = '_' if self.syntax == 'macho' else ''
                if sorted(d[len(pre):] if d.startswith(pre) else d for d in defs) != \
                        sorted(self.members[member]):
                    raise LinkError('runtime/symbols.txt is out of date for %s: run `cimo '
                                    '__update-runtime-symbols`' % member)
                o = self._objs[member] = (obj, defs, undefs - defs)
            return o


_runtime = None


def member_file(member):
    """The file name of a member's assembly in a runtime directory."""
    return member.replace('/', '__') + '.s'


def user_cache(var):
    """A cache directory: $var if set, else $XDG_CACHE_HOME/cimo, %LOCALAPPDATA%\\cimo on
    Windows, or ~/.cache/cimo, created if needed. None if it cannot be created or written (a
    read-only home, an embedded interpreter): callers then cache in memory."""
    env = os.environ
    d = env.get(var) or os.path.join(
        env.get('XDG_CACHE_HOME') or (_WINDOWS and env.get('LOCALAPPDATA'))
        or os.path.expanduser('~/.cache'), 'cimo')
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return d if os.access(d, os.W_OK | os.X_OK) else None


def _cache_root():
    """Where built runtimes live: CIMO_RUNTIME_CACHE, else the user cache (not
    CIMO_CACHE_DIR: entries are keyed by content, so a fresh module cache, as tests and
    benchmarks use, need not rebuild the runtime)."""
    return user_cache('CIMO_RUNTIME_CACHE')


def runtime():
    """The runtime archive for this machine. Members are compiled on demand and cached on disk
    (in memory if there is no writable cache directory)."""
    global _runtime
    if _runtime is None:
        from .driver import TARGET as target
        key = _key(target)
        root = _cache_root()
        _runtime = Runtime(key, target, 'macho' if _APPLE else 'elf',
                           root and os.path.join(root, 'runtime-' + key))
    return _runtime


# ---------------------------------------------------------------------------------------------
# host libraries

_ALIASES = {
    # the C library proper (not the whole process); on Windows the Universal CRT
    'libc': (lambda: 'ucrtbase.dll' if _WINDOWS else ctypes.util.find_library('c') or (
        'libSystem.B.dylib' if _APPLE else 'libc.so.6')),
    'libm': (lambda: 'ucrtbase.dll' if _WINDOWS else ctypes.util.find_library('m') or (
        'libSystem.B.dylib' if _APPLE else 'libm.so.6')),
    # the Python C API: the running interpreter
    'python': None,
}
_libcache = {}


def open_library(name):
    """A ctypes handle for a `link=` entry: 'libc', 'libm', 'python', a soname
    ('libz.so.1'), a short name ('z', as for -lz) or a path."""
    if name in _libcache:
        return _libcache[name]
    if name == 'python':
        lib = ctypes.pythonapi
    else:
        real = _ALIASES[name]() if name in _ALIASES else name
        try:
            lib = ctypes.CDLL(real)
        except OSError as e:
            found = ctypes.util.find_library(name) if os.sep not in name else None
            if not found:
                raise LinkError('link=%r: cannot load library: %s' % (name, e)) from None
            lib = ctypes.CDLL(found)
    _libcache[name] = lib
    return lib


def _dlsym(lib, name):
    try:
        return ctypes.cast(lib[name], ctypes.c_void_p).value
    except AttributeError:
        return None


# ---------------------------------------------------------------------------------------------
# merging objects

def merge(objs):
    """Merge assembler Objects into one: sections with the same name are concatenated (each
    object's part aligned), local symbols are renamed apart, globals must be defined once
    (common symbols merge, a definition wins over a common)."""
    out = _asm.Object(objs[0].syntax)
    secs = {}
    for n, o in enumerate(objs):
        base = {}
        for s in o.sections:
            m = secs.get(s.name)
            if m is None:
                m = secs[s.name] = _asm.Section(s.name, s.exec, s.write, s.alloc, s.nobits)
                out.sections.append(m)
            elif (m.exec, m.write, m.nobits) != (s.exec, s.write, s.nobits):
                raise LinkError('section %s has different flags in different objects' % s.name)
            off = (m.size + s.align - 1) & -s.align
            if not m.nobits:
                m.data += bytes(off - m.size)
                m.data += s.data
            m.size = off + s.size
            m.align = max(m.align, s.align)
            base[s] = (m, off)
        rename = {}
        for y in o.symbols.values():
            if y.defined and not y.globl:
                rename[y.name] = '%s@%d' % (y.name, n) if n else y.name
        for y in o.symbols.values():
            name = rename.get(y.name, y.name)
            z = out.symbols.get(name)
            if y.defined:
                m, off = base[y.section]
                if z is not None and z.defined:
                    raise LinkError('multiple definition of %s' % (
                        name[len(out.prefix):] if name.startswith(out.prefix) else name))
                if z is None:
                    z = out.symbols[name] = _asm.Symbol(name)
                z.section, z.value, z.globl, z.common = m, off + y.value, y.globl, None
            elif y.common:
                if z is None:
                    z = out.symbols[name] = _asm.Symbol(name)
                    z.common = y.common
                elif not z.defined:
                    a, b = z.common or (0, 1), y.common
                    z.common = (max(a[0], b[0]), max(a[1], b[1]))
            elif z is None:
                out.symbols[name] = _asm.Symbol(name)
        for s in o.sections:
            m, off = base[s]
            for r in s.relocs:
                m.relocs.append(_asm.Reloc(off + r.offset, r.type, rename.get(r.sym, r.sym),
                                           r.addend))
    return out


# ---------------------------------------------------------------------------------------------
# linking a module

def _report(rt, ncompiled, t0):
    """After a link that compiled runtime members: tell progress, or one line on stderr."""
    global last_build
    n = len(rt.compiled) - ncompiled
    if not n:
        return
    last_build = {'seconds': time.time() - t0, 'files': n}
    if progress:
        progress('done', **last_build)
    else:
        sys.stderr.write('cimo: compiled %d runtime library file%s in %.1f s (%s)\n' % (
            n, '' if n == 1 else 's', last_build['seconds'],
            'cached for next time' if rt.dir else 'no writable cache: kept for this process'))


def link(obj, hooks=None, libs=()):
    """Link a module's object with the runtime.

    hooks: {C name: address} of the Python callbacks the runtime may import.
    libs: `link=` entries; their symbols resolve what neither the module nor the runtime
    defines.
    Returns (merged object, resolve) where resolve(C name) -> address serves cimo.loader for
    the symbols that stay external; it never falls back to the process."""
    rt = runtime()
    ncompiled, t0 = len(rt.compiled), time.time()
    hooks = hooks or {}
    pre = obj.prefix
    defs, undefs = _globals(obj)
    objs = [obj]
    where = {s: None for s in defs}  # symbol -> runtime member (None: the module)
    pulled = set()
    work = sorted(undefs - defs)
    while work:
        s = work.pop()
        if s in where:
            continue
        m = rt.provider.get(s)
        if m is None or m in pulled:
            continue
        pulled.add(m)
        obj, mdefs, mundefs = rt.object(m)
        objs.append(obj)
        for d in sorted(mdefs):
            if d in where:
                c = d[len(pre):] if d.startswith(pre) else d
                if where[d] is None:
                    raise LinkError("'%s' is defined by your code and by cimo's runtime (%s, "
                                    "pulled in for '%s'); rename your function"
                                    % (c, m, s[len(pre):]))
                raise LinkError("'%s' is defined twice in cimo's runtime (%s and %s)"
                                % (c, where[d], m))
            where[d] = m
        work.extend(u for u in sorted(mundefs) if u not in where)
    _report(rt, ncompiled, t0)
    external = set(undefs)
    for m in pulled:
        external.update(rt.object(m)[2])
    external -= set(where)
    handles = [(n, open_library(n)) for n in libs]
    resolved = {}
    missing = []
    for s in sorted(external):
        c = s[len(pre):] if s.startswith(pre) else s
        if c in hooks:
            resolved[c] = hooks[c]
            continue
        for _, h in handles:
            a = _dlsym(h, c)
            if a is not None:
                resolved[c] = a
                break
        else:
            missing.append(c)
    if missing:
        if libs:
            hint = 'it is not in cimo\'s runtime nor in %s' % ', '.join(libs)
        else:
            hint = ("cimo's runtime does not provide it; to call a host library, pass e.g. "
                    "link=['libc'] (or the library's soname or path)")
        raise LinkError('undefined symbol%s %s: %s' % (
            's' if len(missing) > 1 else '', ', '.join("'%s'" % c for c in missing), hint))
    merged = merge(objs) if len(objs) > 1 else obj

    def resolve(name):
        try:
            return resolved[name]
        except KeyError:
            raise LinkError('undefined symbol: %s' % name) from None

    return merged, resolve


# ---------------------------------------------------------------------------------------------
# stdout: the Python side of runtime/stdio/stdout.c

_tls = threading.local()


def _write(buf, n):
    """__cimo_hook_write: C output (UTF-8) to whatever sys.stdout is at the time, so
    contextlib.redirect_stdout and Jupyter's capture see it. Decoded per thread: a character
    split between two writes of one thread is not mixed with another thread's output."""
    try:
        try:
            dec = _tls.decoder
        except AttributeError:
            dec = _tls.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        text = dec.decode(ctypes.string_at(buf, n))
        if text:
            sys.stdout.write(text)
    except Exception:  # never let an exception cross into C
        pass


_WRITE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t)(_write)


class Stdout:
    """A module's stdout: binds its write hook and flushes the calling thread's buffer (in its
    execution context) after each call."""

    __slots__ = ('_flush',)

    @classmethod
    def attach(cls, image):
        """For an image linked with runtime/stdio/stdout.c, else None."""
        sym, pre = image.symbols, image.prefix
        if pre + '__cimo_flush' not in sym:
            return None
        ctypes.c_void_p.from_address(sym[pre + '__cimo_hook_write']).value = \
            ctypes.cast(_WRITE, ctypes.c_void_p).value
        self = cls()
        self._flush = ctypes.CFUNCTYPE(None)(sym[pre + '__cimo_flush'])
        return self

    def flush(self, ctx):
        """ctx: the calling thread's context (cimo.execution)."""
        if ctypes.c_size_t.from_address(ctypes.addressof(ctx) + CTX_OUT).value:
            self._flush()

