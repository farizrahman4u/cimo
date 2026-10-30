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

# Declarations every module sees: GNU builtins that are runtime functions (runtime/builtins.c)
# or macros over cproc's own builtins.
_PRELUDE = '''_Noreturn void __builtin_trap(void);
int __builtin_popcount(unsigned); int __builtin_popcountl(unsigned long);
int __builtin_popcountll(unsigned long long);
int __builtin_clz(unsigned); int __builtin_clzl(unsigned long); int __builtin_clzll(unsigned long long);
int __builtin_ctz(unsigned); int __builtin_ctzl(unsigned long); int __builtin_ctzll(unsigned long long);
int __builtin_ffs(int); int __builtin_ffsl(long); int __builtin_ffsll(long long);
int __builtin_parity(unsigned); int __builtin_parityl(unsigned long);
int __builtin_parityll(unsigned long long);
int __builtin_clrsb(int); int __builtin_clrsbl(long); int __builtin_clrsbll(long long);
unsigned short __builtin_bswap16(unsigned short); unsigned __builtin_bswap32(unsigned);
unsigned long __builtin_bswap64(unsigned long);
int __builtin_isnan(double); int __builtin_isinf(double); int __builtin_isfinite(double);
int __builtin_signbit(double);
#define __builtin_huge_val() ((double)__builtin_inff())
#define __builtin_huge_valf() __builtin_inff()
#define __builtin_inf() ((double)__builtin_inff())
#define __builtin_nan(s) ((double)__builtin_nanf(s))
'''
_CTYPE = {'w': ctypes.c_int32, 'l': ctypes.c_int64, 's': ctypes.c_float, 'd': ctypes.c_double}
_SIG = re.compile(r'^export\s*\nfunction\s+(?:([wlsd]|:[\w.]+)\s+)?\$(\w+)\(([^)]*)\)', re.M)


class CompileError(Exception):
    pass


def _cache_dir():
    """The module cache directory, or None if it is not writable (see _cache_get)."""
    return _link.user_cache('CIMO_CACHE_DIR')


_memory_cache = {}  # key -> (assembly, signatures), when there is no writable cache directory


def _cache_get(key):
    d = _cache_dir()
    if d is None or key in _memory_cache:
        return _memory_cache.get(key)
    path = os.path.join(d, key + '.s')
    sig = path[:-2] + '.sig'
    if os.path.exists(path) and os.path.exists(sig):
        with open(path) as f, open(sig) as g:
            return f.read(), g.read()
    return None


def _cache_put(key, text, sigs):
    d = _cache_dir()
    if d is not None:
        path = os.path.join(d, key + '.s')
        try:
            for p, data in ((path[:-2] + '.sig', sigs), (path, text)):  # .s last: it marks
                with open(p + '.tmp', 'w') as f:                       # the entry complete
                    f.write(data)
                os.replace(p + '.tmp', p)
            return
        except OSError:
            pass
    _memory_cache[key] = (text, sigs)


_APPLE = sys.platform == 'darwin'
_VERSION = b'cimo-1'  # cache entry format; the compiler's own sources are hashed too (linker.compiler_key)


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


# hosted libc names that cimo's runtime does not provide (see _hint)
_HOSTED = frozenset('''FILE stdin stdout stderr errno fopen freopen fclose fflush fread fwrite
    fgets fputs fgetc fputc getc putc ungetc getchar gets fprintf vfprintf fscanf vfscanf scanf
    vscanf sscanf vsscanf fseek ftell rewind fgetpos fsetpos feof ferror clearerr remove rename
    tmpfile tmpnam perror setvbuf setbuf getline getenv setenv unsetenv system atexit at_quick_exit
    quick_exit _Exit time clock difftime mktime localtime gmtime strftime asctime ctime
    clock_gettime gettimeofday nanosleep sleep usleep open openat creat read write close lseek
    pread pwrite unlink access getpid getppid fork execv execvp execl pipe dup dup2 isatty
    getcwd chdir mkdir rmdir stat fstat lstat mmap munmap mprotect signal raise kill sigaction
    opendir readdir closedir setlocale localeconv sysconf ioctl fcntl poll select socket
    connect bind listen accept send recv dlopen dlsym'''.split())


def _hint(msg, host_headers):
    """A hint for errors about the hosted C library, which cimo does not bundle."""
    host = ("pass host_headers=True to use the host's C headers (and link=['libc'] to "
            "call their functions)")
    m = re.search(r'([\w/.+-]+\.h): No such file or directory', msg)
    if m and not host_headers:
        if m.group(1) == 'setjmp.h':
            return msg + '; setjmp/longjmp are not supported by cimo'
        return msg + '; %s is not bundled with cimo: %s' % (m.group(1), host)
    m = re.search(r"(?:undeclared identifier: |unknown type name ')(\w+)", msg)
    if m and not host_headers and (m.group(1) in _HOSTED or m.group(1).startswith('pthread_')):
        return msg + "; %s is not in cimo's runtime: %s" % (m.group(1), host)
    return msg


def _host_include_dirs():
    """The host's C header directories (for host_headers=True)."""
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
        raise CompileError('host_headers=True: no system header directory found')
    return dirs


def _check_machine():
    if platform.machine().lower() not in ('arm64', 'aarch64'):
        raise CompileError('cimo generates arm64 code only; this machine is %s'
                           % platform.machine())


def _preprocess(src, filename='<string>', include_dirs=(), host_headers=False, defines=()):
    """C source -> preprocessed text, with the prelude. defines: (name, value or None) pairs."""
    from .preprocessor import preprocess, PreprocessError
    if host_headers:
        include_dirs = tuple(include_dirs) + tuple(_host_include_dirs())
    try:
        return preprocess(_PRELUDE + '#line 1\n' + src, filename, include_dirs=include_dirs,
                          defines=list(defines) or None,
                          target='arm64_apple' if _APPLE else 'arm64', keep_attributes=True)
    except PreprocessError as e:
        raise CompileError(_hint(str(e), host_headers)) from None


def _ir(pp, filename='<string>', host_headers=False):
    """Preprocessed text -> (QBE IR text, parameter info for the middle end)."""
    from .frontend import FrontendError
    try:
        return _frontend(pp, filename, fma=FMA, builtins=True, diagnostics=True, prune=True)
    except FrontendError as e:
        raise CompileError(_hint(str(e), host_headers)) from None


def _build(src, filename='<string>', include_dirs=(), host_headers=False, defines=()):
    """C source -> (assembly text, signatures), cached by a hash of the preprocessed source."""
    _check_machine()
    pp = _preprocess(src, filename, include_dirs, host_headers, defines)
    key = hashlib.sha256(_VERSION + _link.compiler_key().encode() + (b'apple' if _APPLE else b'') + (b'm' if MIDDLE else b'') + (b'f' if FMA else b'')
                         + pp.encode()).hexdigest()[:24]
    hit = _cache_get(key)
    if hit is not None:
        return hit
    ssa, params = _ir(pp, filename, host_headers)
    text = _qbe(ssa, params, stkchk=_exe.stack_checked(ssa))
    sigs = _sigs(ssa)
    _cache_put(key, text, sigs)
    return text, sigs


def _sigs(ssa):
    """'name ret params' lines for the exported functions of IR text."""
    return '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                   ','.join(a.split()[0] for a in m.group(3).split(',')
                                            if a.strip()))
                     for m in _SIG.finditer(ssa))


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
        self.__name__ = name
        self._params = params
        self._ret = ret
        self._f = None
        if '...' in params:
            self._unsupported = 'it is variadic'
        elif any(k not in _CTYPE for k in params) or (ret is not None and ret not in _CTYPE):
            self._unsupported = 'it passes or returns a struct or union by value'
        else:
            self._f = ctypes.CFUNCTYPE(_CTYPE.get(ret), *[_CTYPE[k] for k in params])(
                image.address('__cimo_entry_' + name))

    def __call__(self, *args):
        if self._f is None:
            raise TypeError('%s() cannot be called from Python: %s (C callers in the module '
                            'can still use it)' % (self.__name__, self._unsupported))
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
                self._out.flush(ctx)  # C output reaches sys.stdout before Python's next print

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


def compile(src, include_dirs=(), link=(), host_headers=False, defines=()):
    """Compile C source and return a Module of callables.

    The module is linked with cimo's runtime library (string.h, math.h, printf, malloc, ...)
    and nothing else. link: host libraries whose symbols may resolve what the runtime does not
    provide: 'libc', 'libm', 'python' (the Python C API), a soname ('libz.so.1'), a short name
    ('z') or a path. host_headers: also search the host's C headers (after include_dirs and
    before cimo's bundled ones), to use declarations of linked libraries. defines: macros, as
    a dict or (name, value) pairs (value None means 1)."""
    return _module(src, '<string>', include_dirs, link, host_headers, defines)


def load(path, include_dirs=(), link=(), host_headers=False, defines=()):
    """Compile a .c file; its directory is searched for "..." includes. Options as compile()."""
    with open(path) as f:
        src = f.read()
    return _module(src, path, include_dirs, link, host_headers, defines)


def _module(src, filename, include_dirs, link, host_headers, defines):
    if isinstance(link, str):
        link = [link]
    if hasattr(defines, 'items'):
        defines = defines.items()
    return Module(*_build(src, filename, tuple(include_dirs), host_headers, tuple(defines)),
                  link=tuple(link))
