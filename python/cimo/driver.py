"""cimo.compile(): C source in, Python callables out.

Pipeline, all pure Python, in this process:

    C --[cimo.preprocessor]--> preprocessed C --[cimo.frontend]--> QBE IR --[cimo.backend]--> arm64 asm
      --[cimo.assembler]--> object --[cimo.loader]--> executable memory

The module is then linked statically with the parts of cimo's runtime library it uses
(cimo.linker): nothing resolves from the host process unless libraries are named in `link=`.
The linked image is cached by a hash of the preprocessed source (cimo.loader.Image.snapshot);
loading it takes milliseconds.
"""

import ctypes
import hashlib
import io
import os
import platform
import re
import struct
import sys
import warnings

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
#define __cimo_s(x) __cimo_s2(x)
#define __cimo_s2(x) #x
#define __cimo_lib(n) __asm__(__cimo_s(__USER_LABEL_PREFIX__) #n)
_Noreturn void __builtin_abort(void) __cimo_lib(abort);
_Noreturn void __builtin_exit(int) __cimo_lib(exit);
void *__builtin_memcpy(void *, const void *, unsigned long) __cimo_lib(memcpy);
void *__builtin_memmove(void *, const void *, unsigned long) __cimo_lib(memmove);
void *__builtin_memset(void *, int, unsigned long) __cimo_lib(memset);
int __builtin_memcmp(const void *, const void *, unsigned long) __cimo_lib(memcmp);
void *__builtin_memchr(const void *, int, unsigned long) __cimo_lib(memchr);
unsigned long __builtin_strlen(const char *) __cimo_lib(strlen);
int __builtin_strcmp(const char *, const char *) __cimo_lib(strcmp);
int __builtin_strncmp(const char *, const char *, unsigned long) __cimo_lib(strncmp);
char *__builtin_strcpy(char *, const char *) __cimo_lib(strcpy);
char *__builtin_strncpy(char *, const char *, unsigned long) __cimo_lib(strncpy);
char *__builtin_strcat(char *, const char *) __cimo_lib(strcat);
char *__builtin_strncat(char *, const char *, unsigned long) __cimo_lib(strncat);
char *__builtin_strchr(const char *, int) __cimo_lib(strchr);
char *__builtin_strrchr(const char *, int) __cimo_lib(strrchr);
char *__builtin_strstr(const char *, const char *) __cimo_lib(strstr);
int __builtin_printf(const char *, ...) __cimo_lib(printf);
int __builtin_sprintf(char *, const char *, ...) __cimo_lib(sprintf);
int __builtin_snprintf(char *, unsigned long, const char *, ...) __cimo_lib(snprintf);
int __builtin_puts(const char *) __cimo_lib(puts);
int __builtin_putchar(int) __cimo_lib(putchar);
void *__builtin_malloc(unsigned long) __cimo_lib(malloc);
void *__builtin_calloc(unsigned long, unsigned long) __cimo_lib(calloc);
void *__builtin_realloc(void *, unsigned long) __cimo_lib(realloc);
void __builtin_free(void *) __cimo_lib(free);
int __builtin_abs(int) __cimo_lib(abs);
long __builtin_labs(long) __cimo_lib(labs);
long long __builtin_llabs(long long) __cimo_lib(llabs);
double __builtin_fabs(double) __cimo_lib(fabs);
float __builtin_fabsf(float) __cimo_lib(fabsf);
double __builtin_copysign(double, double) __cimo_lib(copysign);
float __builtin_copysignf(float, float) __cimo_lib(copysignf);
double __builtin_sqrt(double) __cimo_lib(sqrt);
float __builtin_sqrtf(float) __cimo_lib(sqrtf);
double __builtin_floor(double) __cimo_lib(floor);
double __builtin_ceil(double) __cimo_lib(ceil);
double __builtin_fmin(double, double) __cimo_lib(fmin);
double __builtin_fmax(double, double) __cimo_lib(fmax);
#undef __cimo_lib
#undef __cimo_s
#undef __cimo_s2
#define __builtin_prefetch(...) ((void)(__VA_ARGS__))
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
    if d is not None and _link.cache_writes.get():
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


def _image_get(key):
    """A cached linked image (signatures + cimo.loader snapshot), or None."""
    blob = _memory_cache.get(key + '.img')
    if blob is not None:
        return blob
    d = _cache_dir()
    if d is not None:
        try:
            with open(os.path.join(d, key + '.img'), 'rb') as f:
                return f.read()
        except OSError:
            pass
    return None


def _image_put(key, blob):
    d = _cache_dir()
    if d is not None and _link.cache_writes.get():
        p = os.path.join(d, key + '.img')
        try:
            with open(p + '.tmp%d' % os.getpid(), 'wb') as f:
                f.write(blob)
            os.replace(p + '.tmp%d' % os.getpid(), p)
            return
        except OSError:
            pass
    _memory_cache[key + '.img'] = blob


_APPLE = sys.platform == 'darwin'
_WINDOWS = sys.platform == 'win32'
# This machine's target, as the preprocessor, backend and linker name it: Linux, Apple's ABI
# variant, or Windows (LP64 like the others, with Windows' variadic convention).
TARGET = 'arm64_apple' if _APPLE else 'arm64_win' if _WINDOWS else 'arm64'
_FRONTEND = {'arm64': 'aarch64', 'arm64_apple': 'aarch64-apple', 'arm64_win': 'aarch64-windows'}
_PYODIDE = sys.platform == 'emscripten'  # cimo.wasm: an Emscripten side module, not arm64
_VERSION = b'cimo-1'  # cache entry format; the compiler's own sources are hashed too (linker.compiler_key)


# the middle end's loop optimizations (cimo.middle); CIMO_MIDDLE=0 turns them off
MIDDLE = os.environ.get('CIMO_MIDDLE', '1') != '0'
# fused multiply-add where C allows contraction (within one expression, like clang's
# -ffp-contract=on); CIMO_FMA=0 turns it off, for results identical to -ffp-contract=off
FMA = os.environ.get('CIMO_FMA', '1') != '0'


def _qbe(ssa, params=None, middle=None, stkchk=None, target=None):
    """QBE IR text to assembly text, with the Python QBE port, in this process.
    params: the frontend's parameter info (restrict flags) for the middle end.
    stkchk: names of the functions that get the stack guard (cimo.execution).
    target: a backend target name (default: this machine's)."""
    from .backend import defs as D
    from .backend.main import _Driver, _targets
    from .backend.parse import parse
    from . import middle as _middle
    from .backend import ext as _ext
    D.T = next(t for t in _targets() if t.name == (target or TARGET))
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


def _qbe_wasm(ssa, params=None, middle=None):
    """QBE IR text to a wasm object (cimo.backend.wasm.obj.Object)."""
    from .backend import defs as D
    from .backend.main import _Driver
    from .backend.parse import parse
    from .backend.wasm.targ import T_wasm
    from .backend.wasm.obj import Object
    from . import middle as _middle
    from .backend import ext as _ext
    D.T = T_wasm
    obj = Object()
    try:
        with _middle.options(enabled=MIDDLE if middle is None else middle,
                             params=params or {}), _ext.options(math=True, fma=True):
            drv = _Driver(obj, 0)
            parse(ssa.encode('latin-1'), '<cimo>', drv.dbgfile, drv.data, drv.func)
    except D.BackendError as e:
        raise CompileError(str(e)) from None
    return obj


def _frontend(pp, filename, target=None, **kw):
    """Preprocessed C to (QBE IR text, parameter info for the middle end). target: as _qbe."""
    from .frontend import compile as cproc_compile
    from .frontend import qbe as _fq
    _fq.paraminfo = {}
    try:
        ssa = cproc_compile(pp, filename, target=_FRONTEND[target or TARGET], **kw)
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
    if _WINDOWS:
        raise CompileError("host_headers=True is not available on Windows (the Windows SDK's "
                           "headers need MSVC extensions)")
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


def _preprocess(src, filename='<string>', include_dirs=(), host_headers=False, defines=(),
                target=None):
    """C source -> preprocessed text, with the prelude. defines: (name, value or None) pairs.
    target: as _qbe."""
    from .preprocessor import preprocess, PreprocessError
    if host_headers:
        include_dirs = tuple(include_dirs) + tuple(_host_include_dirs())
    try:
        return preprocess(_PRELUDE + '#line 1\n' + src, filename, include_dirs=include_dirs,
                          defines=list(defines) or None,
                          target=target or TARGET, keep_attributes=True)
    except PreprocessError as e:
        raise CompileError(_hint(str(e), host_headers)) from None


def _ir(pp, filename='<string>', host_headers=False):
    """Preprocessed text -> (QBE IR text, parameter info for the middle end)."""
    from .frontend import FrontendError
    try:
        return _frontend(pp, filename, fma=FMA, builtins=True, diagnostics=True, prune=True)
    except FrontendError as e:
        raise CompileError(_hint(str(e), host_headers)) from None


def _key(pp):
    """Cache key of preprocessed source: also the compiler's and the runtime's sources (a cached
    image contains the runtime members it uses), the target and the code generation options."""
    return hashlib.sha256(
        _VERSION + _link.compiler_key().encode() + _link._key(TARGET).encode() + TARGET.encode()
        + (b'm' if MIDDLE else b'') + (b'f' if FMA else b'')
        + pp.encode('utf-8', 'surrogateescape')).hexdigest()[:24]


def _build(src, filename='<string>', include_dirs=(), host_headers=False, defines=()):
    """C source -> (assembly text, signatures), cached by a hash of the preprocessed source."""
    _check_machine()
    pp = _preprocess(src, filename, include_dirs, host_headers, defines)
    key = _key(pp)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    text, sigs = _compile(pp, filename, host_headers)
    _cache_put(key, text, sigs)
    return text, sigs


def _compile(pp, filename, host_headers):
    """Preprocessed source -> (assembly text, signatures)."""
    ssa, params = _ir(pp, filename, host_headers)
    text = _qbe(ssa, params, stkchk=_exe.stack_checked(ssa))
    sigs = _sigs(ssa)
    if _WINDOWS:
        from .frontend.diag import longexterns
        names = longexterns()
        if names:  # _module warns if one resolves to a link= library
            sigs += '\n#long ' + ' '.join(names)
    return text, sigs


def _sigs(ssa):
    """'name ret params' lines for the exported functions of IR text."""
    return '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                   ','.join(a.split()[0] for a in m.group(3).split(',')
                                            if a.strip()))
                     for m in _SIG.finditer(ssa))


def _address(x, keep):
    """Pointer argument: a buffer (numpy array, bytearray, memoryview, ctypes array, bytes) or
    int. Objects that must outlive the call are appended to `keep`. bytes are passed in place
    (NUL-terminated, as ctypes does); other read-only buffers as a copy."""
    if isinstance(x, int):
        return x
    ai = getattr(x, '__array_interface__', None)
    if ai is not None:
        return ai['data'][0]
    if isinstance(x, bytes):
        p = ctypes.c_char_p(x)  # points into the bytes object, which keeps its trailing NUL
        keep.append((x, p))
        return ctypes.cast(p, ctypes.c_void_p).value or 0
    mv = memoryview(x)
    if mv.readonly:
        buf = ctypes.create_string_buffer(mv.tobytes())  # NUL-terminated copy
        keep.append(buf)
        return ctypes.addressof(buf)
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
        keep = []  # read-only buffers passed by pointer: alive until the call returns
        args = [_address(a, keep) if k == 'l' and not isinstance(a, int) else a
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

    def __init__(self, text, sigs, link=(), snapshot=False):
        """snapshot: keep the linked image for _image_bytes()."""
        names = [l.split(' ')[0] for l in sigs.splitlines() if not l.startswith('#')]
        libsyms = {}
        try:
            obj = _asm.assemble(text + _exe.glue(names, _APPLE, _WINDOWS),
                                'macho' if _APPLE else 'elf')
            obj, resolve = _link.link(obj, libs=link, libsyms=libsyms)
            image = _jit.Image(obj, resolve, snapshot=snapshot)
        except (_asm.AssemblerError, _jit.LoadError, _link.LinkError) as e:
            raise CompileError(str(e)) from None
        self._setup(image, sigs, libsyms)

    @classmethod
    def _from_image(cls, blob, link=()):
        """A module from _image_bytes(): no assembling or linking."""
        n, = struct.unpack_from('<I', blob)
        sigs = blob[4:4 + n].decode('utf-8')
        libsyms = {}
        try:
            image = _jit.Image.load(memoryview(blob)[4 + n:], lambda ext: _link.resolver(
                ext, '_' if _APPLE else '', libs=link, libsyms=libsyms))
        except (_jit.LoadError, _link.LinkError) as e:
            raise CompileError(str(e)) from None
        self = cls.__new__(cls)
        self._setup(image, sigs, libsyms)
        return self

    def _image_bytes(self):
        """The linked image with the signatures, for _from_image(); None if the module was not
        built with snapshot=True or its image cannot be relocated (cimo.loader.Image.snapshot).
        Once only."""
        snap = self._image.snapshot()
        if snap is None:
            return None
        sigs = self._sigs.encode('utf-8')
        return struct.pack('<I', len(sigs)) + sigs + snap

    def _setup(self, image, sigs, libsyms):
        self._image = image
        self._sigs = sigs
        lines = sigs.splitlines()
        # functions declared with long that resolved to a link= library (see _module)
        self._longs = sorted(set(next((l.split()[1:] for l in lines if l.startswith('#long ')),
                                      ())) & set(libsyms))
        self._libsyms = libsyms
        sigs = '\n'.join(l for l in lines if not l.startswith('#'))
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


def compile(src, include_dirs=(), link=(), host_headers=False, defines=(), cache=True):
    """Compile C source and return a Module of callables.

    The module is linked with cimo's runtime library (string.h, math.h, printf, malloc, ...)
    and nothing else. link: host libraries whose symbols may resolve what the runtime does not
    provide: 'libc', 'libm', 'python' (the Python C API), a soname ('libz.so.1'), a short name
    ('z') or a path. host_headers: also search the host's C headers (after include_dirs and
    before cimo's bundled ones), to use declarations of linked libraries. defines: macros, as
    a dict or (name, value) pairs (value None means 1). cache=False: read the compile cache
    but write nothing to disk (no cache entries, no directories, no intermediate files);
    whatever is compiled, including runtime library members, is kept in memory."""
    return _module(src, '<string>', include_dirs, link, host_headers, defines, cache)


def load(path, include_dirs=(), link=(), host_headers=False, defines=(), cache=True):
    """Compile a .c file; its directory is searched for "..." includes. Options as compile()."""
    from .resources import read_text
    return _module(read_text(path), path, include_dirs, link, host_headers, defines, cache)


def _module(src, filename, include_dirs, link, host_headers, defines, cache=True):
    token = _link.cache_writes.set(bool(cache))
    try:
        return _module1(src, filename, include_dirs, link, host_headers, defines)
    finally:
        _link.cache_writes.reset(token)


def _module1(src, filename, include_dirs, link, host_headers, defines):
    if isinstance(link, str):
        link = [link]
    if hasattr(defines, 'items'):
        defines = defines.items()
    if _PYODIDE:
        if link or host_headers:
            raise CompileError('link= and host_headers= are not available in Pyodide (there is '
                               'no host C library)')
        from . import wasm as _wasm
        from .wasm.pyodide import WasmModule
        return WasmModule(*_wasm.build(src, filename, tuple(include_dirs), tuple(defines)))
    _check_machine()
    pp = _preprocess(src, filename, tuple(include_dirs), host_headers, tuple(defines))
    key = _key(pp)
    blob = _image_get(key)
    if blob is not None:
        m = Module._from_image(blob, tuple(link))
    else:
        hit = _cache_get(key)  # assembly: for modules whose image cannot be cached
        text, sigs = hit or _compile(pp, filename, host_headers)
        m = Module(text, sigs, tuple(link), snapshot=hit is None)
        if hit is None:
            blob = m._image_bytes()
            if blob is not None:
                _image_put(key, blob)
            else:
                _cache_put(key, text, sigs)
    for c in m._longs:
        warnings.warn("'%s' (from %s) is declared with long, which is 64-bit in cimo but "
                      "32-bit in Windows DLLs; declare it with int (or long long where the "
                      "DLL uses a 64-bit type)" % (c, m._libsyms[c]), stacklevel=4)
    return m
