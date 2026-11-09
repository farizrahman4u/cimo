"""NVIDIA devices through the CUDA driver API alone: gpu/host/cuda.c, compiled by cimo and linked to libcuda.

    gpu = cimo.gpu.device()                  # device 0's primary context
    k = gpu.compile(src)                      # or cimo.compile(src, device=gpu)
    a = gpu.array(buf)                        # a copy on the device
    k.f(a, n, grid=(n // 256,), block=(256,)) # doesn't wait
    gpu.sync()
    a.to_host(buf)
"""

import array
import ctypes
import hashlib
import struct
import sys
import threading

from .. import resources as _res

_WINDOWS = sys.platform == 'win32'
_lock = threading.RLock()
_host = None
_version = None
_count = None
_devices = {}


_closing = []  # set at exit: the host module's code may be gone, so nothing calls the driver from __del__ then


def _atexit():
    _closing.append(True)


import atexit as _atexit_mod  # noqa: E402
_atexit_mod.register(_atexit)


class GpuError(RuntimeError):
    """A CUDA driver call failed: its error name and text."""


def _libname():
    return 'nvcuda.dll' if _WINDOWS else 'libcuda.so.1'


def _cuda():
    """The host module (cuda.c compiled and linked to the driver), the driver's version and the device count."""
    global _host, _version, _count
    with _lock:
        if _host is None:
            from ..driver import compile as _compile, CompileError
            from ..linker import LinkError
            try:
                lib = _res.read_text(_res.path('gpu/host/cuda.c'))
                h = _compile(lib, link=[_libname()])
            except (LinkError, CompileError, OSError) as e:
                raise GpuError('no NVIDIA driver (%s): %s' % (_libname(), e)) from None
            v, c = array.array('i', [0]), array.array('i', [0])
            r = h.cimo_cuda_init(v, c)
            _host = h
            _check(r, 'cuInit')
            _version, _count = v[0], c[0]
        return _host


def _check(r, what):
    if r:
        n, t = array.array('Q', [0]), array.array('Q', [0])
        _host.cimo_cuda_error(r, n, t)
        name = ctypes.string_at(n[0]).decode() if n[0] else 'CUDA error %d' % r
        text = ctypes.string_at(t[0]).decode() if t[0] else ''
        raise GpuError('%s: %s (%s)' % (what, text or name, name))


def driver_version():
    """The driver's CUDA version as (major, minor), e.g. (12, 4)."""
    _cuda()
    return _version // 1000, _version % 1000 // 10


def max_ptx(version=None):
    """The newest PTX ISA version a driver of this CUDA version reads (PTX ISA release notes)."""
    major, minor = version or driver_version()
    if major >= 13:
        return (9, minor)
    if major == 12:
        return (8, min(minor, 8) if minor < 9 else 8)
    if major == 11:
        return (7, minor)
    return (6, 4)


def devices():
    """How many CUDA devices there are."""
    _cuda()
    return _count


def device(ordinal=0):
    """CUDA device `ordinal` (0: the first), through its primary context."""
    _cuda()
    with _lock:
        d = _devices.get(ordinal)
        if d is None:
            if not 0 <= ordinal < _count:
                raise GpuError('no CUDA device %d (%d found)' % (ordinal, _count))
            d = _devices[ordinal] = Device(ordinal)
        return d


class Device:
    """One CUDA device. compile() makes modules of kernels for it; array() keeps data on it."""

    def __init__(self, ordinal):
        h = _cuda()
        out = array.array('q', [0] * 8)
        name = bytearray(256)
        _check(h.cimo_cuda_device(ordinal, out, name, len(name)), 'cuDeviceGet')
        self.ordinal = ordinal
        self._dev, self._ctx = out[0], out[1]
        self.sm = out[2]  # compute capability as one number: 89 is 8.9
        self.multiprocessors = out[3]
        self.shared_per_block = out[4]
        self.memory = out[5]
        self.name = bytes(name).split(b'\0', 1)[0].decode()
        self._modules = {}

    def __repr__(self):
        return '<cimo.gpu.Device %d: %s, sm_%d>' % (self.ordinal, self.name, self.sm)

    @property
    def compute_capability(self):
        return self.sm // 10, self.sm % 10

    # ---- compiling

    def ptx(self, src, include_dirs=(), defines=(), filename='<string>', passes=None):
        """The PTX text cimo makes of src for this device."""
        from . import _ptx
        return _ptx(src, self.sm, include_dirs, defines, filename, None, passes)[0]

    def compile(self, src, include_dirs=(), defines=(), filename='<string>', opt=4, passes=None, cache=True,
                version=None):
        """GPU kernel C to a module loaded on this device: its __gpu_kernel functions are attributes. The driver
        rejecting the PTX raises cimo.CompileError with its log. opt: the driver's optimization level (0 to 4).
        version: the PTX ISA version to state, (major, minor), when asm needs a newer one than the target's lowest."""
        from . import _ptx, _cache_key
        from ..driver import CompileError, _fma, _wrapv
        key = _cache_key(src, self.sm, include_dirs, defines, filename, opt, passes, _fma.get(), _wrapv.get(), version)
        m = self._modules.get(key) if cache else None
        if m is not None:
            return m
        import time
        t0 = time.perf_counter()
        text, kernels = _ptx(src, self.sm, include_dirs, defines, filename, version, passes)
        t1 = time.perf_counter()
        v = max_ptx()
        need = tuple(int(x) for x in text.split('.version ', 1)[1].split('\n', 1)[0].split('.'))
        if need > v:
            raise CompileError('the NVIDIA driver (CUDA %d.%d) reads PTX up to %d.%d, and sm_%d needs %d.%d: update '
                               'the driver' % (driver_version() + v + (self.sm,) + need))
        m = GpuModule(self, text, kernels, opt)
        m.compile_time = t1 - t0
        if cache:
            self._modules[key] = m
        return m

    def load(self, image, kernels, opt=4):
        """A module made elsewhere: PTX text (str) or a cubin or fatbin (bytes), with the parameters of its kernels
        as GpuModule.signatures gives them ({name: [(C type, class)]}, class 'w', 'l', 's', 'd')."""
        return GpuModule(self, image, kernels, opt)

    # ---- memory

    def array(self, a=None, nbytes=None, shape=None, typestr=None):
        """Device memory: a copy of the host buffer a (anything with the buffer protocol, numpy arrays too), or
        nbytes of it uninitialized. shape and typestr (numpy's '<f4', ...) describe it for
        __cuda_array_interface__; from a buffer they are its own."""
        h = _cuda()
        mv = None
        if a is not None:
            mv = memoryview(a)
            if not mv.contiguous:
                raise ValueError('cimo.gpu: a host buffer must be contiguous')
            mv = mv.cast('B') if mv.ndim != 1 or mv.format != 'B' else mv
            if nbytes is None:
                nbytes = mv.nbytes
            if shape is None:
                ai = getattr(a, '__array_interface__', None)
                if ai is not None:
                    shape, typestr = tuple(ai['shape']), ai['typestr']
                else:
                    m0 = memoryview(a)
                    shape, typestr = tuple(m0.shape), _typestr(m0.format, m0.itemsize)
        if nbytes is None:
            raise ValueError('cimo.gpu: array() needs a host buffer or nbytes')
        p = array.array('Q', [0])
        _check(h.cimo_cuda_alloc(self._ctx, nbytes, p), 'cuMemAlloc')
        d = DeviceArray(self, p[0], nbytes, shape or (nbytes,), typestr or '|u1', owner=True)
        if mv is not None and nbytes:
            d.from_host(mv)
        return d

    def empty(self, nbytes):
        return self.array(nbytes=nbytes)

    def zeros(self, nbytes):
        d = self.array(nbytes=nbytes)
        _check(_host.cimo_cuda_memset(self._ctx, d.ptr, 0, nbytes), 'cuMemsetD8')
        return d

    def sync(self):
        """Wait for every launch and copy on this device so far."""
        _check(_host.cimo_cuda_sync(self._ctx), 'cuCtxSynchronize')

    def time(self, fn, repeat=1):
        """fn() `repeat` times between two events on the device: ms per call, as the GPU measures it."""
        h = _host
        a, b = array.array('Q', [0]), array.array('Q', [0])
        _check(h.cimo_cuda_event(self._ctx, a), 'cuEventCreate')
        _check(h.cimo_cuda_event(self._ctx, b), 'cuEventCreate')
        try:
            _check(h.cimo_cuda_record(self._ctx, a[0]), 'cuEventRecord')
            for _ in range(repeat):
                fn()
            _check(h.cimo_cuda_record(self._ctx, b[0]), 'cuEventRecord')
            ms = array.array('f', [0])
            _check(h.cimo_cuda_elapsed(self._ctx, a[0], b[0], ms), 'cuEventElapsedTime')
            return ms[0] / repeat
        finally:
            h.cimo_cuda_event_free(self._ctx, a[0])
            h.cimo_cuda_event_free(self._ctx, b[0])


_FMT = {'b': '|i1', 'B': '|u1', 'h': '<i2', 'H': '<u2', 'i': '<i4', 'I': '<u4', 'l': '<i8', 'L': '<u8', 'q': '<i8',
        'Q': '<u8', 'e': '<f2', 'f': '<f4', 'd': '<f8', 'c': '|u1', '?': '|b1'}


def _typestr(fmt, itemsize):
    return _FMT.get(fmt.lstrip('@=<'), '|V%d' % itemsize)


class DeviceArray:
    """Memory on a device. ptr is its device address; __cuda_array_interface__ lets other libraries (CuPy, PyTorch,
    Numba) use it without a copy."""

    def __init__(self, device, ptr, nbytes, shape, typestr, owner=False):
        self.device, self.ptr, self.nbytes, self.shape, self.typestr = device, ptr, nbytes, tuple(shape), typestr
        self._owner = owner

    @property
    def __cuda_array_interface__(self):
        return {'version': 3, 'data': (self.ptr, False), 'shape': self.shape, 'typestr': self.typestr,
                'strides': None, 'stream': None}

    def from_host(self, buf):
        """Copy a host buffer of at most nbytes in (waits for the device's work before it)."""
        mv = memoryview(buf).cast('B')
        if mv.nbytes > self.nbytes:
            raise ValueError('cimo.gpu: %d bytes into a %d-byte device array' % (mv.nbytes, self.nbytes))
        _check(_host.cimo_cuda_htod(self.device._ctx, self.ptr, mv if not mv.readonly else bytes(mv), mv.nbytes),
               'cuMemcpyHtoD')

    def to_host(self, out=None):
        """Copy to a writable host buffer (or a new bytearray), after the device's work before it; returns it."""
        if out is None:
            out = bytearray(self.nbytes)
        mv = memoryview(out).cast('B')
        n = min(mv.nbytes, self.nbytes)
        _check(_host.cimo_cuda_dtoh(self.device._ctx, out, self.ptr, n), 'cuMemcpyDtoH')
        return out

    def free(self):
        if self._owner and self.ptr and _host is not None and not _closing:
            _host.cimo_cuda_free(self.device._ctx, self.ptr)
        self.ptr = 0
        self._owner = False

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self):
        return '<cimo.gpu.DeviceArray %d bytes at 0x%x>' % (self.nbytes, self.ptr)


class GpuModule:
    """A PTX module loaded on a device: one GpuKernel per __gpu_kernel function. ptx is its text; load_time the
    driver's compile time (ms), compile_time cimo's (s); info the driver's log."""

    def __init__(self, device, text, kernels, opt):
        from ..driver import CompileError
        self.device, self.ptx = device, text
        h = _host
        err, info = bytearray(16384), bytearray(16384)
        wall, mod = array.array('f', [0]), array.array('Q', [0])
        image = text.encode() if isinstance(text, str) else bytes(text)
        r = h.cimo_cuda_load(device._ctx, image, opt, err, len(err), info, len(info), wall, mod)
        self.info = bytes(info).split(b'\0', 1)[0].decode('utf-8', 'replace')
        if r:
            log = bytes(err).split(b'\0', 1)[0].decode('utf-8', 'replace').strip()
            n, t = array.array('Q', [0]), array.array('Q', [0])
            h.cimo_cuda_error(r, n, t)
            name = ctypes.string_at(n[0]).decode() if n[0] else str(r)
            raise CompileError('the NVIDIA driver rejected the PTX (%s)%s' % (name, ':\n' + log if log else ''))
        self._mod = mod[0]
        self.load_time = wall[0]
        self.signatures = dict(kernels)
        self.kernels = {}
        for name, params in kernels.items():
            f, regs = array.array('Q', [0]), array.array('i', [0])
            _check(h.cimo_cuda_function(device._ctx, self._mod, name.encode(), f, regs), 'cuModuleGetFunction')
            self.kernels[name] = GpuKernel(self, name, f[0], params, regs[0])

    def __getattr__(self, name):
        k = self.__dict__.get('kernels', {}).get(name)
        if k is None:
            raise AttributeError('no __gpu_kernel %r in this module' % name)
        return k

    def __dir__(self):
        return list(self.kernels) + ['ptx', 'device', 'load_time', 'compile_time', 'info', 'signatures']

    def __del__(self):
        try:
            if self._mod and _host is not None and not _closing:
                _host.cimo_cuda_unload(self.device._ctx, self._mod)
        except Exception:
            pass


class _Host(Exception):
    """A host buffer where a pointer goes: the launch takes the slow path, with copies."""


def _dptr(a):
    if type(a) is DeviceArray:
        return a.ptr
    if isinstance(a, int):
        return a
    cai = getattr(a, '__cuda_array_interface__', None)
    if cai is not None:
        return cai['data'][0]
    raise _Host


def _half(a):
    if isinstance(a, float):
        return struct.unpack('<H', struct.pack('<e', a))[0]
    return int(a) & 0xffffffff


def _bf16(a):
    if isinstance(a, float):
        b = struct.unpack('<I', struct.pack('<f', a))[0]
        return (b + 0x7fff + ((b >> 16) & 1)) >> 16
    return int(a) & 0xffffffff


def _bytes(n, name):
    def conv(a):
        b = bytes(memoryview(a).cast('B'))
        if len(b) != n:
            raise TypeError('%s(): a %d-byte struct argument, not %d bytes' % (name, n, len(b)))
        return b
    return conv


_dims = {}


def _dim(v, what):
    if isinstance(v, list):
        v = tuple(v)
    d = _dims.get(v)
    if d is not None:
        return d
    t = (v,) if isinstance(v, int) else v
    t = tuple(int(x) for x in t)
    if not 1 <= len(t) <= 3 or any(x < 1 for x in t):
        raise ValueError('cimo.gpu: %s must be 1 to 3 positive sizes, not %r' % (what, v))
    t = t + (1,) * (3 - len(t))
    if len(_dims) < 4096:
        _dims[v] = t
    return t


class GpuKernel:
    """A kernel: k(*args, grid=, block=, shared=0). Pointer parameters take a DeviceArray, an object with
    __cuda_array_interface__, a device address (int), or a host buffer, which is copied in and, unless the
    parameter points to const, back after the kernel, and then the call waits. Otherwise a launch doesn't wait
    (Device.sync())."""

    def __init__(self, module, name, handle, params, regs):
        self.module, self.name, self._f, self.registers = module, name, handle, regs
        self.params = params  # [(C type, class)]
        fmt, conv, off = ['<'], [], 0
        for ctype, k in params:
            size, align = (4, 4) if k in ('w', 's') else (8, 8) if k in ('l', 'd') else (k[1], min(k[2], 16))
            pad = ((off + align - 1) & -align) - off
            if pad:
                fmt.append('%dx' % pad)
            off += pad + size
            if ctype.endswith('*'):
                fmt.append('Q')
                conv.append(_dptr)
            elif k == 's':
                fmt.append('f')
                conv.append(float)
            elif k == 'd':
                fmt.append('d')
                conv.append(float)
            elif k == 'w':
                fmt.append('I')
                conv.append(_half if ctype in ('_Float16', '__fp16') else _bf16 if ctype == '__bf16'
                            else lambda a: int(a) & 0xffffffff)
            elif k == 'l':
                fmt.append('Q')
                conv.append(lambda a: int(a) & 0xffffffffffffffff)
            else:
                fmt.append('%ds' % k[1])
                conv.append(_bytes(k[1], name))
        self._struct = struct.Struct(''.join(fmt) if off else '<x')
        self._conv = conv
        self._size = self._struct.size
        self._shared = 48 * 1024
        self._launch = _host.cimo_cuda_launch
        self._tls = threading.local()
        self._ctx = module.device._ctx

    def __call__(self, *args, grid, block, shared=0):
        conv = self._conv
        if len(args) != len(conv):
            raise TypeError('%s() takes %d arguments (%d given)' % (self.name, len(conv), len(args)))
        try:  # the driver copies the parameters at the launch, so each thread's buffer is reused
            buf, addr = self._tls.b
        except AttributeError:
            buf = bytearray(self._size)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            self._tls.b = (buf, addr)
        try:
            self._struct.pack_into(buf, 0, *[c(a) for c, a in zip(conv, args)])
        except _Host:
            return self._withcopies(args, grid, block, shared)
        g = _dim(grid, 'grid')
        b = _dim(block, 'block')
        if shared > self._shared:
            self._allow(shared)
        r = self._launch(self._ctx, self._f, g[0], g[1], g[2], b[0], b[1], b[2], shared, 0, addr, self._size)
        if r:
            _check(r, 'cuLaunchKernel')

    def _allow(self, shared):
        _check(_host.cimo_cuda_shared(self._ctx, self._f, shared), 'cuFuncSetAttribute')
        self._shared = shared

    def _withcopies(self, args, grid, block, shared):
        dev = self.module.device
        copies, vals = [], []
        for (ctype, k), a in zip(self.params, args):
            if ctype.endswith('*'):
                try:
                    a = _dptr(a)
                except _Host:  # a host buffer: asked for a copy
                    d = dev.array(a)
                    copies.append((d, a, not ctype.lstrip().startswith('const')))
                    a = d.ptr
            vals.append(a)
        try:
            self(*vals, grid=grid, block=block, shared=shared)
            for d, a, back in copies:
                if back:
                    d.to_host(a)
            dev.sync()
        finally:
            for d, a, back in copies:
                d.free()

    def __repr__(self):
        return '<cimo.gpu kernel %s>' % self.name
