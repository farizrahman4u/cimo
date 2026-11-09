"""The wasm target: C to an Emscripten side module that Pyodide loads into its own memory.

    build(src, ...) -> (module bytes, signatures)

compiles with the wasm backend (cimo.backend.wasm), links the runtime members the module needs
and cimo/wasm/glue.c (cimo.wasm.link), and caches the result like the native build. Loading
and calling kernels is cimo.wasm.pyodide (only inside Pyodide).
"""

import hashlib
import os
import threading

from .. import resources as _res
from . import link as _link

GLUE = _res.path('wasm/glue.c')
WASM_RUNTIME = _res.path('wasm/runtime')
# runtime members replaced for wasm: symbol -> file in WASM_RUNTIME
OVERRIDES = {'fma': 'fma.c', 'fmaf': 'fma.c'}
# runtime/omp.c: a team of one (Pyodide has no threads)
OVERRIDES.update((n, 'omp.c') for n in '''__cimo_par __cimo_par_next __cimo_lock __cimo_unlock __cimo_flush_mem __cimo_crit
    __cimo_atomic __cimo_reduce omp_get_thread_num omp_get_num_threads omp_in_parallel omp_get_max_threads
    omp_set_num_threads omp_get_num_procs omp_get_thread_limit omp_get_level omp_get_active_level omp_get_dynamic
    omp_set_dynamic omp_get_nested omp_set_nested omp_get_max_active_levels omp_set_max_active_levels omp_get_wtime
    omp_get_wtick omp_init_lock omp_destroy_lock omp_set_lock omp_unset_lock omp_test_lock'''.split())
_VERSION = b'cimo-wasm-1'


def _driver():
    from .. import driver
    return driver


def compile_object(src, filename='<string>', include_dirs=(), defines=(), runtime=False):
    """C source to a wasm object (cimo.backend.wasm.obj.Object). runtime: the runtime's own
    include directory and options (no FMA contraction, which wasm lacks anyway)."""
    d = _driver()
    from ..linker import RUNTIME_INCLUDE
    if runtime:
        include_dirs = tuple(include_dirs) + (RUNTIME_INCLUDE,)
    pp = d._preprocess(src, filename, include_dirs, False, defines, 'arm64')
    from ..frontend import FrontendError
    try:
        ssa, params = d._frontend(pp, filename, 'arm64', fma=False, fmaok=False, builtins=True,
                                  diagnostics=True, prune=True, vastruct='slots', bitops=False, wasmptr=True)
    except FrontendError as e:
        raise d.CompileError(str(e)) from None
    return d._qbe_wasm(ssa, params), ssa, params


class Runtime:
    """The runtime archive for wasm: members compiled on first use, cached as JSON objects
    (in `<cache>/runtime-wasm-<key>/`, or in memory)."""

    def __init__(self):
        from .. import linker as L
        self.L = L
        self.members = L.read_symbols()
        self.provider = {}
        for m, names in self.members.items():
            for n in names:
                self.provider.setdefault(n, m)
        for n in OVERRIDES:  # wasm's own versions of runtime members
            self.provider[n] = 'wasm:' + OVERRIDES[n]
        d = (L.user_cache if L.cache_writes.get() else L.cache_path)('CIMO_CACHE_DIR')
        self.dir = d and os.path.join(d, 'runtime-wasm-' + _key())
        self._objs = {}
        self._lock = threading.RLock()

    def object(self, member):
        from ..backend.wasm.obj import Object
        with self._lock:
            o = self._objs.get(member)
            if o is not None:
                return o
            p = self.dir and os.path.join(self.dir, member.replace('/', '__').replace(':', '_')
                                          + '.json')
            if p and os.path.exists(p):
                with open(p) as f:
                    o = Object.loads(f.read())
            else:
                path = (os.path.join(WASM_RUNTIME, member[5:]) if member.startswith('wasm:')
                        else os.path.join(self.L.RUNTIME_DIR, member))
                try:
                    o = compile_object(_res.read_text(path), path, runtime=True)[0]
                except Exception as e:
                    raise _driver().CompileError('building the wasm runtime: %s: %s'
                                                 % (member, e)) from None
                if p and self.L.cache_writes.get():
                    try:
                        os.makedirs(self.dir, exist_ok=True)
                        with open(p + '.tmp', 'w') as f:
                            f.write(o.dumps())
                        os.replace(p + '.tmp', p)
                    except OSError:
                        pass
            self._objs[member] = o
            return o

    def provide(self, name):
        m = self.provider.get(name)
        return self.object(m) if m else None


_runtime = None
_glue = None


def _key():
    from .. import linker as L
    h = hashlib.sha256(_VERSION + L.compiler_key().encode())
    h.update(_res.read(GLUE))
    for p in _res.files(WASM_RUNTIME, ('.c', '.h')):
        h.update(_res.read(p))
    for p in _res.files(L.RUNTIME_DIR, ('.c', '.h')):
        h.update(os.path.relpath(p, _res.ROOT).replace(os.sep, '/').encode() + b'\0')
        h.update(_res.read(p))
    return h.hexdigest()[:20]


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


def glue():
    global _glue
    if _glue is None:
        _glue = compile_object(_res.read_text(GLUE), GLUE, runtime=True)[0]
    return _glue


_memory_cache = {}


def build(src, filename='<string>', include_dirs=(), defines=()):
    """C source -> (module bytes, signatures text as cimo.driver._sigs gives)."""
    return _build(src, filename, include_dirs, defines)[:2]


def _build(src, filename='<string>', include_dirs=(), defines=()):
    """build(), and whether it came from the cache."""
    d = _driver()
    pp = d._preprocess(src, filename, include_dirs, False, defines, 'arm64')
    from .. import linker as L
    key = hashlib.sha256(_VERSION + L.compiler_key().encode() + (b'm' if d.MIDDLE else b'')
                         + pp.encode('utf-8', 'surrogateescape')).hexdigest()[:24]
    hit = _cache_get(key)
    if hit is not None:
        return hit[0], hit[1], True, key
    obj, ssa, params = compile_object(src, filename, include_dirs, defines)
    sigs = d._sigs(ssa, params)
    names = [line.split(' ')[0] for line in sigs.splitlines()]
    try:
        objs, undefined = _link.resolve([obj, glue()], runtime().provide)
        mod = _link.link(objs, names)
    except _link.LinkError as e:
        raise d.CompileError(str(e)) from None
    _cache_put(key, mod, sigs)
    return mod, sigs, False, key


def _cache_get(key):
    from .. import linker as L
    if key in _memory_cache:
        return _memory_cache[key]
    c = L.user_cache('CIMO_CACHE_DIR')
    if c:
        p = os.path.join(c, key + '.wasm')
        if os.path.exists(p) and os.path.exists(p + '.sig'):
            with open(p, 'rb') as f, open(p + '.sig') as g:
                return f.read(), g.read()
    return None


def _cache_put(key, mod, sigs):
    from .. import linker as L
    c = L.user_cache('CIMO_CACHE_DIR')
    if c and L.cache_writes.get():
        p = os.path.join(c, key + '.wasm')
        try:
            with open(p + '.sig.tmp', 'w') as f:
                f.write(sigs)
            os.replace(p + '.sig.tmp', p + '.sig')
            with open(p + '.tmp', 'wb') as f:
                f.write(mod)
            os.replace(p + '.tmp', p)
            return
        except OSError:
            pass
    _memory_cache[key] = (mod, sigs)
