"""cimo.gpu: explicit GPU kernels in C, compiled to PTX and loaded through the NVIDIA driver alone.

    gpu = cimo.gpu.device()                   # the first CUDA device; no toolkit, just the driver (libcuda)
    k = cimo.compile(src, device=gpu)         # src includes <cimo/gpu.h> and defines __gpu_kernel functions
    a = gpu.array(host_buffer)                # device memory, kept across calls
    k.f(a, n, grid=(blocks,), block=(256,))   # a launch; it doesn't wait
    gpu.sync()
    a.to_host(host_buffer)

    cimo.gpu.ptx(src, sm=80)                  # the PTX text alone, on any machine

The PTX backend is cimo.gpu.backend.ptx; the host side is gpu/host/cuda.c, compiled by cimo and linked to libcuda by
dlsym. Device math (expf, ...) is C in gpu/lib, compiled into the same module when a kernel calls it.
"""

import hashlib
import os
import re
import threading

from .. import resources as _res

INCLUDE = _res.path('gpu/include')
LIB = _res.path('gpu/lib')
# device library functions: name -> file in gpu/lib
LIBFUNCS = {'expf': 'expf.c', 'memcpy': 'string.c', 'memmove': 'string.c', 'memset': 'string.c'}

# the lowest PTX ISA version that has each target (PTX ISA, "Release notes"); a module states the lowest that works
SM_PTX = {50: (4, 0), 52: (4, 1), 53: (4, 2), 60: (5, 0), 61: (5, 0), 62: (5, 0), 70: (6, 0), 72: (6, 1),
          75: (6, 3), 80: (7, 0), 86: (7, 1), 87: (7, 4), 89: (7, 8), 90: (7, 8), 100: (8, 6), 101: (8, 6),
          103: (8, 8), 110: (9, 0), 120: (8, 7), 121: (8, 8)}
# instructions of an architecture-specific target (sm_90a, sm_100a): their presence in asm text adds the "a"
_ARCHSPECIFIC = re.compile(r'\b(wgmma|setmaxnreg|tcgen05|multimem)\b')
_lock = threading.RLock()


def _version_for(sm):
    known = [s for s in SM_PTX if s <= sm]
    if not known:
        raise ValueError('cimo.gpu: compute capability %d.%d is older than any PTX target cimo knows (5.0 and newer)'
                         % (sm // 10, sm % 10))
    return SM_PTX[sm] if sm in SM_PTX else SM_PTX[max(known)]


def _driver():
    from .. import driver
    return driver


def _ir(src, filename, include_dirs, defines, extra='', nowrap=True):
    """C source (plus device library sources) to (IR text, frontend info)."""
    d = _driver()
    pp = d._preprocess(src + extra, filename, (INCLUDE,) + tuple(include_dirs), False, defines, 'arm64')
    from ..frontend import FrontendError
    try:
        return d._frontend(pp, filename, 'arm64', fma=d._contract_gpu(), builtins=True, bitops=True,
                           diagnostics=True, prune=True, nowrap=nowrap and not d._wraps(), vecops=('half',), gpu=True,
                           bigcopy=None)
    except FrontendError as e:
        raise d.CompileError(str(e)) from None


_CALLS = re.compile(r'call (?:extern )?\$"?([A-Za-z_][\w.]*)')
_DEFS = re.compile(r'^(?:export\s+)?function\s+(?:\S+\s+)?\$"?([A-Za-z_][\w.]*)', re.M)


def frontend(src, filename='<string>', include_dirs=(), defines=(), nowrap=True):
    """C source to (IR text, frontend info), with the device library functions it calls."""
    extra, have = '', set()
    named = sorted({f for n, f in LIBFUNCS.items() if re.search(r'\b%s\b' % n, src)})  # most often: one compile
    for f in named:
        have.add(f)
        extra += '\n#line 1 "<cimo/gpu/lib/%s>"\n' % f + _res.read_text(os.path.join(LIB, f))
    while True:
        ssa, info = _ir(src, filename, include_dirs, defines, extra, nowrap)
        defined = set(_DEFS.findall(ssa))
        need = sorted({LIBFUNCS[n] for n in _CALLS.findall(ssa) if n in LIBFUNCS and n not in defined} - have)
        if not need:
            return ssa, info
        for f in need:
            have.add(f)
            extra += '\n#line 1 "<cimo/gpu/lib/%s>"\n' % f + _res.read_text(os.path.join(LIB, f))


INLINE = int(os.environ.get('CIMO_GPU_INLINE', '100000'))  # inline everything (no recursion in GPU code anyway)
MIDDLE = os.environ.get('CIMO_GPU_MIDDLE', '1') != '0'  # full unrolling of small constant-trip loops


def codegen(ssa, info, passes=None):
    """IR text to a cimo.gpu.backend.ptx.emit.Module."""
    d = _driver()
    from ..backend import defs as D
    from ..backend.main import _Driver
    from ..backend.parse import parse
    from ..backend import ext as _ext
    from .. import middle as _middle
    from .backend.ptx.targ import T_ptx
    from .backend.ptx.emit import Module
    mod = Module(info)
    opts = dict(math=True, fma=True, volatile=True, vla=True, cse=True, inline=INLINE, nowrap=True)
    passes = dict(passes or {})
    middle = passes.pop('middle', MIDDLE)  # passes: cimo.backend.ext options (cse, inline, nowrap), and middle
    opts.update(passes)
    try:
        with d._compile_lock, _middle.options(enabled=middle, gpu=True, vectorizing=False, params=info or {}), _ext.options(**opts), \
                d._Recursion():
            D.T = T_ptx
            drv = _Driver(mod, 0)
            parse(ssa.encode('latin-1'), '<cimo>', drv.dbgfile, drv.data, drv.func)
            drv.finish()
    except D.BackendError as e:
        raise d.CompileError(str(e)) from None
    return mod


def target(sm, mod=None):
    """(.target, .version) for compute capability sm (e.g. 89) and the module's asm text."""
    t = 'sm_%d' % sm
    v = _version_for(sm)
    if mod is not None and sm >= 90 and any(_ARCHSPECIFIC.search(a) for a in mod.features()):
        t += 'a'
    return t, '%d.%d' % v


def ptx(src, sm=80, include_dirs=(), defines=(), filename='<string>', version=None, passes=None):
    """GPU kernel C to PTX text for compute capability sm (80 is 8.0), on any machine.
    version: a PTX ISA version to state instead of the lowest the target has, as (major, minor)."""
    return _ptx(src, sm, include_dirs, defines, filename, version, passes)[0]


_gpukey = None


def _sources_key():
    """Hash of the compiler's sources and of gpu/ (the backend, headers, device library): part of a GPU kernel's
    cache key only, so editing gpu/ leaves CPU kernels' keys alone."""
    global _gpukey
    if _gpukey is None:
        from .. import linker
        h = hashlib.sha256(b'cimo-gpu-1\0' + linker.compiler_key().encode())
        for p in sorted(_res.files(_res.path('gpu'), ('.py', '.h', '.c'))):
            h.update(p.replace(os.sep, '/').rsplit('/gpu/', 1)[-1].encode() + b'\0')
            h.update(_res.read_text(p).encode('utf-8', 'surrogateescape'))
        _gpukey = h.hexdigest()[:20]
    return _gpukey


def _ptx(src, sm, include_dirs, defines, filename, version, passes):
    """(PTX text, kernel signatures), from the compile cache (`gpu-<key>.json`) when it has them."""
    import json
    d = _driver()
    pp = d._preprocess(src, filename, (INCLUDE,) + tuple(include_dirs), False, defines, 'arm64')
    key = hashlib.sha256(repr((_sources_key(), d._code(pp), sm, version, sorted((passes or {}).items()),
                               d._contract_gpu(), d._wraps(), MIDDLE, INLINE)).encode('utf-8', 'surrogateescape')
                         ).hexdigest()[:24]
    hit = _memo.get(key)
    cdir = d._cache_dir()
    path = cdir and os.path.join(cdir, 'gpu-%s.json' % key)
    if hit is None and path and os.path.exists(path):
        try:
            with open(path) as f:
                j = json.load(f)
            hit = j['ptx'], {k: [(c, tuple(v) if isinstance(v, list) else v) for c, v in ps]
                             for k, ps in j['kernels'].items()}
        except (OSError, ValueError, KeyError):
            hit = None
    if hit is not None:
        _memo[key] = hit
        return hit
    hit = _ptx1(src, sm, include_dirs, defines, filename, version, passes)
    _memo[key] = hit
    from .. import linker
    if path and linker.cache_writes.get():
        try:
            tmp = path + '.tmp%d' % os.getpid()
            with open(tmp, 'w') as f:
                json.dump({'ptx': hit[0], 'kernels': hit[1]}, f)
            os.replace(tmp, path)
        except OSError:
            pass
    return hit


_memo = {}


def _ptx1(src, sm, include_dirs, defines, filename, version, passes):
    with _lock:
        ssa, info = frontend(src, filename, include_dirs, defines, (passes or {}).get('nowrap', True))
        mod = codegen(ssa, info, passes)
    t, v = target(sm, mod)
    if version is not None:
        v = '%d.%d' % tuple(version)
    try:
        text = mod.text(t, v)
    except _driver_backend_error() as e:
        raise _driver().CompileError(str(e)) from None
    return text, _kernel_sigs(info, mod)


def _driver_backend_error():
    from ..backend import defs as D
    return D.BackendError


def _kernel_sigs(info, mod):
    """name -> [(C type, class)] of each kernel's parameters (class: 'w', 'l', 's', 'd' or ('struct', size, align))."""
    sigs = {n.replace('.', '$'): s for n, s in info.get('#signatures', {}).items()}
    out = {}
    for f in mod.funcs:
        if f.entry:
            ctypes_ = sigs.get(f.name, ('', ()))[1]
            if len(ctypes_) != len(f.params):
                ctypes_ = ('',) * len(f.params)
            out[f.name] = list(zip(ctypes_, f.params))
    return out


def _cache_key(*parts):
    return hashlib.sha256(repr(parts).encode('utf-8', 'surrogateescape')).hexdigest()


def __getattr__(name):
    if name in ('device', 'Device', 'DeviceArray', 'GpuModule', 'GpuKernel', 'devices', 'GpuError', 'driver_version',
                'max_ptx'):
        from . import cuda
        return getattr(cuda, name)
    raise AttributeError(name)
