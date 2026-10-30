"""Linking: cimo's runtime library, the Python hooks, and host libraries on request.

Link model (requirements.md, "Language" and "Execution model"):

- The runtime (`cimo/runtime/**/*.c`: libc subset from musl, cimo's heap, stdout) is compiled
  by cimo itself on first use, one object per source file, and cached as assembly text in
  `<cache>/runtime-<key>/` with an index of the symbols each file defines and uses (in memory
  if there is no writable cache; `cimo bundle --prebuilt` embeds it in the bundle). The key
  hashes the runtime sources and headers, the bundled headers, and the compiler's own
  sources, so any change rebuilds it.
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

# Compiler sources whose changes can change the runtime's code (part of the cache key):
# conservative, so editing the driver rebuilds the runtime too (~9 s).
_COMPILER = ('preprocessor.py', 'frontend', 'middle', 'backend', 'include', 'driver.py',
             'linker.py', 'execution.py')

# Last first-use build: {'seconds': ..., 'files': ...}, for tests and reports.
last_build = None

# progress(event, **info) during the runtime's first-use build, instead of messages on stderr:
# 'start' (files=n), 'file' (name=member), 'done' (seconds=s). The command line sets it.
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
_TARGETS = ('arm64', 'arm64_apple')


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
    ssa, params = driver._frontend(pp, path, fma=False, builtins=True, prune=True)
    return driver._qbe(ssa, params, stkchk=_exe.stack_checked(ssa))


class Runtime:
    """The runtime library as an archive: members (one per source file) are assembled on
    first use and kept for the life of the process. text(member) is a member's assembly."""

    def __init__(self, key, index, syntax, text):
        self.key = key  # the cache key (_key)
        self.index = index  # member -> {'defs': [...], 'undefs': [...]}
        self.syntax = syntax
        self.text = text
        self.provider = {}
        for m, e in index.items():
            for s in e['defs']:
                self.provider.setdefault(s, m)
        self._objs = {}

    def object(self, member):
        o = self._objs.get(member)
        if o is None:
            o = self._objs[member] = _asm.assemble(self.text(member), self.syntax)
        return o


_runtime = None


def member_file(member):
    """The file name of a member's assembly in a runtime directory."""
    return member.replace('/', '__') + '.s'


def user_cache(var):
    """A cache directory: $var if set, else $XDG_CACHE_HOME/cimo or ~/.cache/cimo, created if
    needed. None if it cannot be created or written (a read-only home, an embedded
    interpreter): callers then cache in memory."""
    d = os.environ.get(var) or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'cimo')
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


def _build_runtime(target, syntax):
    """Compile every runtime source: (index, {member: assembly text})."""
    from .driver import CompileError
    t0 = time.time()
    srcs = runtime_sources()
    if progress:
        progress('start', files=len(srcs))
    else:
        sys.stderr.write('cimo: building the runtime library (%d files, first use only)...\n'
                         % len(srcs))
    index, texts = {}, {}
    for rel in srcs:
        member = rel.replace(os.sep, '/')
        if progress:
            progress('file', name=member)
        try:
            text = compile_runtime_source(os.path.join(RUNTIME_DIR, rel), target)
            obj = _asm.assemble(text, syntax)
        except Exception as e:
            raise CompileError('building the runtime: %s: %s' % (rel, e)) from None
        defs, undefs = _globals(obj)
        index[member] = {'defs': sorted(defs), 'undefs': sorted(undefs - defs)}
        texts[member] = text
    global last_build
    last_build = {'seconds': time.time() - t0, 'files': len(srcs)}
    if progress:
        progress('done', seconds=last_build['seconds'])
    else:
        sys.stderr.write('cimo: runtime built in %.1f s\n' % last_build['seconds'])
    return index, texts


def _save_runtime(d, index, texts):
    """Write a built runtime to cache directory d (atomically: index.json marks it complete)."""
    tmp = d + '.tmp%d' % os.getpid()
    os.makedirs(tmp, exist_ok=True)
    for member, text in texts.items():
        with open(os.path.join(tmp, member_file(member)), 'w') as f:
            f.write(text)
    with open(os.path.join(tmp, 'index.json'), 'w') as f:
        json.dump(index, f, indent=0, sort_keys=True)
    try:
        os.rename(tmp, d)
    except OSError:  # another process finished first
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def runtime():
    """The runtime archive for this machine: prebuilt in the bundle (`cimo bundle
    --prebuilt`), else built on first use and cached on disk, or in memory if there is no
    writable cache directory."""
    global _runtime
    if _runtime is not None:
        return _runtime
    target = 'arm64_apple' if _APPLE else 'arm64'
    syntax = 'macho' if _APPLE else 'elf'
    key = _key(target)
    d = _res.path('prebuilt/runtime-' + key)
    if not _res.isfile(os.path.join(d, 'index.json')):
        root = _cache_root()
        d = root and os.path.join(root, 'runtime-' + key)
        if d is None or not os.path.exists(os.path.join(d, 'index.json')):
            index, texts = _build_runtime(target, syntax)
            if d is not None:
                try:
                    _save_runtime(d, index, texts)
                except OSError:
                    d = None
            if d is None:  # no writable cache: keep it in memory
                _runtime = Runtime(key, index, syntax, texts.__getitem__)
                return _runtime
    index = json.loads(_res.read_text(os.path.join(d, 'index.json')))
    _runtime = Runtime(key, index, syntax,
                       lambda m: _res.read_text(os.path.join(d, member_file(m))))
    return _runtime


# ---------------------------------------------------------------------------------------------
# host libraries

_ALIASES = {
    # the C library proper (not the whole process)
    'libc': (lambda: ctypes.util.find_library('c') or ('libSystem.B.dylib' if _APPLE
                                                       else 'libc.so.6')),
    'libm': (lambda: ctypes.util.find_library('m') or ('libSystem.B.dylib' if _APPLE
                                                       else 'libm.so.6')),
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

def link(obj, hooks=None, libs=()):
    """Link a module's object with the runtime.

    hooks: {C name: address} of the Python callbacks the runtime may import.
    libs: `link=` entries; their symbols resolve what neither the module nor the runtime
    defines.
    Returns (merged object, resolve) where resolve(C name) -> address serves cimo.loader for
    the symbols that stay external; it never falls back to the process."""
    rt = runtime()
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
        objs.append(rt.object(m))
        e = rt.index[m]
        for d in e['defs']:
            if d in where:
                c = d[len(pre):] if d.startswith(pre) else d
                if where[d] is None:
                    raise LinkError("'%s' is defined by your code and by cimo's runtime (%s, "
                                    "pulled in for '%s'); rename your function"
                                    % (c, m, s[len(pre):]))
                raise LinkError("'%s' is defined twice in cimo's runtime (%s and %s)"
                                % (c, where[d], m))
            where[d] = m
        work.extend(u for u in e['undefs'] if u not in where)
    external = set(undefs)
    for m in pulled:
        external.update(rt.index[m]['undefs'])
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
