"""Linking: cimo's runtime library, the Python hooks, and host libraries on request.

Link model (requirements.md, "Language" and "Execution model"):

- The runtime (`cimo/runtime/**/*.c`: libc subset from musl, cimo's heap, stdout) is compiled
  by cimo itself on first use, one object per source file, and cached as assembly text in
  `<cache>/runtime-<key>/` with an index of the symbols each file defines and uses. The key
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
import time

from . import assembler as _asm

__all__ = ['LinkError', 'runtime', 'link', 'open_library', 'Stdout', 'RUNTIME_DIR']

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(_HERE, 'runtime')
RUNTIME_INCLUDE = os.path.join(RUNTIME_DIR, 'internal')
_APPLE = sys.platform == 'darwin'

# Compiler sources whose changes can change the runtime's code (part of the cache key):
# conservative, so editing the driver rebuilds the runtime too (~9 s).
_COMPILER = ('preprocessor.py', 'frontend', 'middle', 'backend', 'include', 'driver.py',
             'linker.py', 'execution.py')

# Last first-use build: {'seconds': ..., 'files': ...}, for tests and reports.
last_build = None


class LinkError(Exception):
    pass


def _files(top, exts):
    out = []
    for d, dirs, files in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x != '__pycache__')
        for f in sorted(files):
            if f.endswith(exts):
                out.append(os.path.join(d, f))
    return out


def runtime_sources():
    """Every .c file under cimo/runtime, as paths relative to it."""
    return [os.path.relpath(p, RUNTIME_DIR) for p in _files(RUNTIME_DIR, ('.c',))]


def _key(target):
    h = hashlib.sha256(b'cimo-runtime-1\0' + target.encode() + b'\0')
    paths = _files(RUNTIME_DIR, ('.c', '.h'))
    for c in _COMPILER:
        p = os.path.join(_HERE, c)
        paths += _files(p, ('.py', '.h')) if os.path.isdir(p) else [p]
    for p in paths:
        h.update(os.path.relpath(p, _HERE).encode() + b'\0')
        with open(p, 'rb') as f:
            h.update(f.read())
        h.update(b'\0')
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
    are written for -ffp-contract=off, which is how musl itself is built) and float
    arithmetic declared as evaluated in float (__FLT_EVAL_METHOD__ 0, which is what cimo
    does), so float_t is float as in a gcc build of musl. __FP_FAST_FMA(F) as gcc defines them
    on arm64: musl's pow, log, exp2f, ... then use fma() (one instruction) for exact products."""
    from . import driver
    from . import execution as _exe
    from .preprocessor import preprocess
    with open(path) as f:
        src = f.read()
    pp = preprocess(src, path, include_dirs=(RUNTIME_INCLUDE,),
                    defines={'__FLT_EVAL_METHOD__': '0', '__FP_FAST_FMA': '1',
                             '__FP_FAST_FMAF': '1'}, target=target,
                    warn=lambda *a, **k: None)
    ssa, params = driver._frontend(pp, path, fma=False, builtins=True)
    return driver._qbe(ssa, params, stkchk=_exe.stack_checked(ssa))


class Runtime:
    """The runtime library as an archive: members (one per source file) are assembled on
    first use and kept for the life of the process."""

    def __init__(self, directory, index, syntax):
        self.dir = directory
        self.index = index  # member -> {'defs': [...], 'undefs': [...]}
        self.syntax = syntax
        self.provider = {}
        for m, e in index.items():
            for s in e['defs']:
                self.provider.setdefault(s, m)
        self._objs = {}

    def object(self, member):
        o = self._objs.get(member)
        if o is None:
            with open(os.path.join(self.dir, member.replace('/', '__') + '.s')) as f:
                o = self._objs[member] = _asm.assemble(f.read(), self.syntax)
        return o


_runtime = None


def _cache_root():
    """Where built runtimes live: CIMO_RUNTIME_CACHE, else the user cache (not
    CIMO_CACHE_DIR: entries are keyed by content, so a fresh module cache, as tests and
    benchmarks use, need not rebuild the runtime)."""
    d = os.environ.get('CIMO_RUNTIME_CACHE') or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'cimo')
    os.makedirs(d, exist_ok=True)
    return d


def runtime():
    """The runtime archive for this machine, built and cached on first use."""
    global _runtime, last_build
    if _runtime is not None:
        return _runtime
    target = 'arm64_apple' if _APPLE else 'arm64'
    syntax = 'macho' if _APPLE else 'elf'
    d = os.path.join(_cache_root(), 'runtime-' + _key(target))
    idx = os.path.join(d, 'index.json')
    if not os.path.exists(idx):
        from .driver import CompileError
        t0 = time.time()
        srcs = runtime_sources()
        sys.stderr.write('cimo: building the runtime library (%d files, first use only)...\n'
                         % len(srcs))
        tmp = d + '.tmp%d' % os.getpid()
        os.makedirs(tmp, exist_ok=True)
        index = {}
        for rel in srcs:
            member = rel.replace(os.sep, '/')
            try:
                text = compile_runtime_source(os.path.join(RUNTIME_DIR, rel), target)
                obj = _asm.assemble(text, syntax)
            except Exception as e:
                raise CompileError('building the runtime: %s: %s' % (rel, e)) from None
            defs, undefs = _globals(obj)
            index[member] = {'defs': sorted(defs), 'undefs': sorted(undefs - defs)}
            with open(os.path.join(tmp, member.replace('/', '__') + '.s'), 'w') as f:
                f.write(text)
        with open(os.path.join(tmp, 'index.json'), 'w') as f:
            json.dump(index, f, indent=0, sort_keys=True)
        last_build = {'seconds': time.time() - t0, 'files': len(srcs)}
        try:
            os.rename(tmp, d)
        except OSError:  # another process finished first
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        sys.stderr.write('cimo: runtime built in %.1f s\n' % last_build['seconds'])
    with open(idx) as f:
        index = json.load(f)
    _runtime = Runtime(d, index, syntax)
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

_decoder = codecs.getincrementaldecoder('utf-8')('replace')


def _write(buf, n):
    """__cimo_hook_write: C output (UTF-8) to whatever sys.stdout is at the time, so
    contextlib.redirect_stdout and Jupyter's capture see it."""
    try:
        text = _decoder.decode(ctypes.string_at(buf, n))
        if text:
            sys.stdout.write(text)
    except Exception:  # never let an exception cross into C
        pass


_WRITE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t)(_write)


class Stdout:
    """A module's stdout buffer: binds its write hook and flushes it after each call."""

    __slots__ = ('_len', '_flush')

    @classmethod
    def attach(cls, image):
        """For an image linked with runtime/stdio/stdout.c, else None."""
        sym, pre = image.symbols, image.prefix
        if pre + '__cimo_stdout' not in sym:
            return None
        ctypes.c_void_p.from_address(sym[pre + '__cimo_hook_write']).value = \
            ctypes.cast(_WRITE, ctypes.c_void_p).value
        self = cls()
        self._len = ctypes.c_size_t.from_address(sym[pre + '__cimo_stdout'])
        self._flush = ctypes.CFUNCTYPE(None)(sym[pre + '__cimo_flush'])
        return self

    def flush(self):
        if self._len.value:
            self._flush()
