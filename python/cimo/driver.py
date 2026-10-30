"""cimo.compile(): C source in, Python callables out.

Pipeline, all pure Python, in this process:

    C --[cimo.preprocessor]--> preprocessed C --[cimo.frontend]--> QBE IR --[cimo.backend]--> arm64 asm
      --[cimo.assembler]--> object --[cimo.loader]--> executable memory

The module is then linked statically with the parts of cimo's runtime library it uses
(cimo.linker): nothing resolves from the host process unless libraries are named in `link=`.
The assembly is cached by a hash of the preprocessed source; assembling, linking and loading
take milliseconds.
"""

import ctypes
import hashlib
import io
import os
import platform
import re
import sys

from . import assembler as _asm
from . import execution as _exe
from . import linker as _link
from . import loader as _jit
from .execution import KernelError

__all__ = ['compile', 'load', 'Module', 'CompileError', 'KernelError']

_PRELUDE = '_Noreturn void __builtin_trap(void);\n'
_CTYPE = {'w': ctypes.c_int32, 'l': ctypes.c_int64, 's': ctypes.c_float, 'd': ctypes.c_double}
_SIG = re.compile(r'^export\s*\nfunction\s+(?:([wlsd])\s+)?\$(\w+)\(([^)]*)\)', re.M)


class CompileError(Exception):
    pass


def _cache_dir():
    d = os.environ.get('CIMO_CACHE_DIR') or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'cimo')
    os.makedirs(d, exist_ok=True)
    return d


_APPLE = sys.platform == 'darwin'
_VERSION = b'cimo-m5-exec1-rt1'  # part of the cache key: bump when the generated assembly changes


# the middle end's loop optimizations (cimo.middle); CIMO_MIDDLE=0 turns them off
MIDDLE = os.environ.get('CIMO_MIDDLE', '1') != '0'
# fused multiply-add where C allows contraction (within one expression, like clang's
# -ffp-contract=on); CIMO_FMA=0 turns it off, for results identical to -ffp-contract=off
FMA = os.environ.get('CIMO_FMA', '1') != '0'


def _qbe(ssa, params=None, middle=None, stkchk=None):
    """QBE IR text to assembly text, with the Python QBE port, in this process.
    params: the frontend's parameter info (restrict flags) for the middle end.
    stkchk: names of the functions that get the stack guard (cimo.execution)."""
    from .backend import defs as D
    from .backend.main import _Driver, _targets
    from .backend.parse import parse
    from . import middle as _middle
    from .backend import ext as _ext
    D.T = _targets()[1 if _APPLE else 0]  # arm64_apple: Apple's ABI variant
    out = io.StringIO()
    try:
        with _middle.options(enabled=MIDDLE if middle is None else middle,
                             params=params or {}), \
                _ext.options(fma=True, ldp=True, layout=True, math=True, trap=True,
                             stkchk=stkchk):
            parse(ssa.encode('latin-1'), '<cimo>', _Driver(out, 0).dbgfile,
                  _Driver(out, 0).data, _Driver(out, 0).func)
        D.T.emitfin(out)
    except D.BackendError as e:
        raise CompileError(str(e)) from None
    return out.getvalue()


def _frontend(pp, filename, **kw):
    """Preprocessed C to (QBE IR text, parameter info for the middle end)."""
    from .frontend import compile as cproc_compile
    from .frontend import qbe as _fq
    _fq.paraminfo = {}
    try:
        ssa = cproc_compile(pp, filename, target='aarch64-apple' if _APPLE else 'aarch64', **kw)
        return ssa, _fq.paraminfo
    finally:
        _fq.paraminfo = None


def _system_include_dirs():
    """The host's C header directories (for system_headers=True)."""
    if _APPLE:
        import subprocess
        try:
            sdk = subprocess.check_output(['xcrun', '--show-sdk-path'],
                                          universal_newlines=True).strip()
        except (OSError, subprocess.CalledProcessError):
            sdk = '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'
        dirs = [os.path.join(sdk, 'usr', 'include')]
    else:
        import sysconfig
        multiarch = sysconfig.get_config_var('MULTIARCH') or 'aarch64-linux-gnu'
        dirs = ['/usr/local/include', '/usr/include/' + multiarch, '/usr/include']
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise CompileError('system_headers=True: no system header directory found')
    return dirs


def _build(src, filename='<string>', include_dirs=(), system_headers=False):
    """C source -> (assembly text, signatures), cached by a hash of the preprocessed source."""
    from .preprocessor import preprocess, PreprocessError
    from .frontend import compile as cproc_compile, FrontendError
    if platform.machine().lower() not in ('arm64', 'aarch64'):
        raise CompileError('cimo generates arm64 code only; this machine is %s'
                           % platform.machine())
    if system_headers:
        include_dirs = tuple(include_dirs) + tuple(_system_include_dirs())
    try:
        pp = preprocess(_PRELUDE + '#line 1\n' + src, filename, include_dirs=include_dirs,
                        target='arm64_apple' if _APPLE else 'arm64')
    except PreprocessError as e:
        raise CompileError(str(e)) from None
    key = hashlib.sha256(_VERSION + (b'apple' if _APPLE else b'') + (b'm' if MIDDLE else b'') + (b'f' if FMA else b'')
                         + pp.encode()).hexdigest()[:24]
    path = os.path.join(_cache_dir(), key + '.s')
    sig = path[:-2] + '.sig'
    if os.path.exists(path) and os.path.exists(sig):
        with open(path) as f, open(sig) as g:
            return f.read(), g.read()
    try:
        ssa, params = _frontend(pp, filename, fma=FMA, builtins=True)
    except FrontendError as e:
        raise CompileError(str(e)) from None
    text = _qbe(ssa, params, stkchk=_exe.stack_checked(ssa))
    sigs = '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                   ','.join(a.split()[0] for a in m.group(3).split(',')
                                            if a.strip()))
                     for m in _SIG.finditer(ssa))
    for p, data in ((sig, sigs), (path, text)):  # .s last: it marks the entry complete
        with open(p + '.tmp', 'w') as f:
            f.write(data)
        os.replace(p + '.tmp', p)
    return text, sigs


def _address(x):
    """Pointer argument: a buffer (numpy array, bytearray, memoryview, ctypes array) or int."""
    if isinstance(x, int):
        return x
    ai = getattr(x, '__array_interface__', None)
    if ai is not None:
        return ai['data'][0]
    mv = memoryview(x)
    if mv.readonly:
        return ctypes.cast(ctypes.c_char_p(mv.tobytes()), ctypes.c_void_p).value
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class Kernel:
    """A C function. Calls go through its entry trampoline (cimo.execution), so abort(),
    assert and traps raise KernelError; `address` is the function itself."""

    def __init__(self, image, name, ret, params, state, out=None):
        self._image = image  # keeps the code mapped while the kernel is reachable
        self._state = state
        self._out = out  # the module's stdout buffer (cimo.linker.Stdout), if it has one
        self.address = image.address(name)
        self._f = ctypes.CFUNCTYPE(_CTYPE.get(ret), *[_CTYPE[k] for k in params])(
            image.address('__cimo_entry_' + name))
        self.__name__ = name
        self._params = params

    def __call__(self, *args):
        if len(args) != len(self._params):
            raise TypeError('%s() takes %d arguments (%d given)'
                            % (self.__name__, len(self._params), len(args)))
        args = [_address(a) if k == 'l' and not isinstance(a, int) else a
                for a, k in zip(args, self._params)]
        ctx = self._state.ctx()
        try:
            r = self._f(*args)
            if ctx.err:
                self._state.error(ctx, self.__name__)
            return r
        finally:
            if self._out is not None:
                self._out.flush()  # C output reaches sys.stdout before Python's next print

    def __repr__(self):
        return '<cimo kernel %s>' % self.__name__


class Module:
    """Compiled C. Each non-static function is an attribute. Calling the module calls `main`,
    or the only function if there is just one."""

    def __init__(self, text, sigs, link=()):
        names = [line.split(' ')[0] for line in sigs.splitlines()]
        try:
            obj = _asm.assemble(text + _exe.glue(names, _APPLE), 'macho' if _APPLE else 'elf')
            obj, resolve = _link.link(obj, libs=link)
            self._image = _jit.Image(obj, resolve)
        except (_asm.AssemblerError, _jit.LoadError, _link.LinkError) as e:
            raise CompileError(str(e)) from None
        self._state = _exe.State(self._image)
        self._out = _link.Stdout.attach(self._image)
        self._kernels = {}
        for line in sigs.splitlines():
            name, ret, params = line.split(' ')
            self._kernels[name] = Kernel(self._image, name, None if ret == '-' else ret,
                                         params.split(',') if params else [], self._state,
                                         self._out)

    def reset_heap(self):
        """Free every heap block of the module (all threads' heaps) at once, e.g. between calls
        that use the heap as a scratch arena. Pointers into the heap become invalid. Must not
        be called while a kernel of the module is running."""
        self._state.reset_heap()

    def free(self, ptr):
        """free() a pointer that a kernel returned from malloc/calloc/realloc/aligned_alloc."""
        self._state.free(ptr)

    def heap_bytes(self):
        """Bytes of memory currently mapped for the module's heap."""
        return self._state.heap_bytes()

    def __getattr__(self, name):
        try:
            return self.__dict__['_kernels'][name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return list(self._kernels)

    def __call__(self, *args):
        if 'main' in self._kernels:
            return self._kernels['main'](*args)
        if len(self._kernels) == 1:
            return next(iter(self._kernels.values()))(*args)
        raise TypeError('module has no main(); call one of: %s' % ', '.join(self._kernels))


def compile(src, include_dirs=(), link=(), system_headers=False):
    """Compile C source and return a Module of callables.

    The module is linked with cimo's runtime library (string.h, math.h, printf, malloc, ...)
    and nothing else. link: host libraries whose symbols may resolve what the runtime does not
    provide: 'libc', 'libm', 'python' (the Python C API), a soname ('libz.so.1'), a short name
    ('z') or a path. system_headers: also search the host's C headers (after include_dirs and
    before cimo's bundled ones), to use declarations of linked libraries."""
    if isinstance(link, str):
        link = [link]
    return Module(*_build(src, '<string>', tuple(include_dirs), system_headers),
                  link=tuple(link))


def load(path, include_dirs=(), link=(), system_headers=False):
    """Compile a .c file; its directory is searched for "..." includes. Options as compile()."""
    with open(path) as f:
        src = f.read()
    if isinstance(link, str):
        link = [link]
    return Module(*_build(src, path, tuple(include_dirs), system_headers), link=tuple(link))
