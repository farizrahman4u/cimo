"""cimo.compile(): C source in, Python callables out.

Pipeline, all pure Python, in this process:

    C --[cimo.preprocessor]--> preprocessed C --[cimo.frontend]--> QBE IR --[cimo.backend]--> arm64 asm
      --[cimo.assembler]--> object --[cimo.loader]--> executable memory

The module is then linked statically with the parts of cimo's runtime library it uses
(cimo.linker): nothing resolves from the host process unless libraries are named in `link=`.
The linked image is cached by a hash of the preprocessed source (cimo.loader.Image.snapshot);
loading it takes milliseconds.
"""

from __future__ import annotations

import contextvars
import ctypes
import hashlib
import io
import os
import platform
import re
import struct
import sys
import threading
import warnings

from .introspect import Introspection
import zlib

from . import assembler as _asm
from . import execution as _exe
from . import linker as _link
from . import loader as _jit
from .execution import KernelError
from .frontend.main import _Recursion

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Tuple, Union

if TYPE_CHECKING:
    from os import PathLike

    # what compile() and load() accept
    PathArg = Union[str, 'PathLike[str]']
    Defines = Union[Mapping[str, Optional[object]], Iterable[Tuple[str, Optional[object]]]]

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
double __builtin_fma(double, double, double) __cimo_lib(fma);
float __builtin_fmaf(float, float, float) __cimo_lib(fmaf);
float __builtin_fminf(float, float) __cimo_lib(fminf);
float __builtin_fmaxf(float, float) __cimo_lib(fmaxf);
double __builtin_pow(double, double) __cimo_lib(pow);
float __builtin_powf(float, float) __cimo_lib(powf);
float __builtin_floorf(float) __cimo_lib(floorf);
float __builtin_ceilf(float) __cimo_lib(ceilf);
double __builtin_trunc(double) __cimo_lib(trunc);
float __builtin_truncf(float) __cimo_lib(truncf);
double __builtin_round(double) __cimo_lib(round);
float __builtin_roundf(float) __cimo_lib(roundf);
double __builtin_exp(double) __cimo_lib(exp);
double __builtin_log(double) __cimo_lib(log);
char *__builtin_strdup(const char *) __cimo_lib(strdup);
int __builtin_iscntrl(int) __cimo_lib(iscntrl);
int __builtin_isdigit(int) __cimo_lib(isdigit);
int __builtin_isalpha(int) __cimo_lib(isalpha);
int __builtin_isalnum(int) __cimo_lib(isalnum);
int __builtin_isspace(int) __cimo_lib(isspace);
int __builtin_isupper(int) __cimo_lib(isupper);
int __builtin_islower(int) __cimo_lib(islower);
int __builtin_isprint(int) __cimo_lib(isprint);
int __builtin_isxdigit(int) __cimo_lib(isxdigit);
int __builtin_ispunct(int) __cimo_lib(ispunct);
int __builtin_isgraph(int) __cimo_lib(isgraph);
int __builtin_toupper(int) __cimo_lib(toupper);
int __builtin_tolower(int) __cimo_lib(tolower);
#undef __cimo_lib
#undef __cimo_s
#undef __cimo_s2
#define __builtin_prefetch(...) ((void)(__VA_ARGS__))
#define __builtin_assume_aligned(p, ...) ((void *)(p))
int __builtin_isinff(float);
double __builtin_nans(const char *); float __builtin_nansf(const char *);
int __cimo_issignalingf(float); int __cimo_issignaling(double);
#ifndef __CIMO_STOCK_CPROC /* tests/extcompile.py: cproc's own mode has no _Float16 or __int128 */
int __cimo_issignalingh(_Float16); int __cimo_issignalingbf(__bf16);
#define __builtin_issignaling(x) _Generic((x), float: __cimo_issignalingf, _Float16: __cimo_issignalingh, \
	__bf16: __cimo_issignalingbf, default: __cimo_issignaling)(x)
#else
#define __builtin_issignaling(x) _Generic((x), float: __cimo_issignalingf, default: __cimo_issignaling)(x)
#endif
#define __builtin_nansf32(s) __builtin_nansf(s)
#define __builtin_nansf64(s) __builtin_nans(s)
#define __builtin_nansf32x(s) __builtin_nans(s)
#define __builtin_nanf32x(s) __builtin_nan(s)
#define __builtin_nansf16(s) (((union { unsigned short __u; _Float16 __h; }){ 0x7d00 }).__h)
#define __builtin_nansf16b(s) (((union { unsigned short __u; __bf16 __h; }){ 0x7fa0 }).__h)
#define __builtin_nanf16(s) ((_Float16)__builtin_nanf(s))
#define __builtin_nanf32(s) __builtin_nanf(s)
#define __builtin_nanf64(s) __builtin_nan(s)
#define __builtin_inff16() ((_Float16)__builtin_inff())
#define __builtin_inff32() __builtin_inff()
#define __builtin_inff64() __builtin_inf()
#define __builtin_huge_valf16() ((_Float16)__builtin_inff())
#define __builtin_huge_valf32() __builtin_inff()
#define __builtin_huge_valf64() __builtin_inf()
int __cimo_fcmp(int, double, double);
#define __builtin_isunordered(x, y) __cimo_fcmp(0, x, y)
#define __builtin_isgreater(x, y) __cimo_fcmp(1, x, y)
#define __builtin_isgreaterequal(x, y) __cimo_fcmp(2, x, y)
#define __builtin_isless(x, y) __cimo_fcmp(3, x, y)
#define __builtin_islessequal(x, y) __cimo_fcmp(4, x, y)
#define __builtin_islessgreater(x, y) __cimo_fcmp(5, x, y)
int __cimo_iseqsig(double, double);
#define __builtin_iseqsig(x, y) __cimo_iseqsig(x, y)
int __cimo_overflow(int, unsigned long long, int, unsigned long long, int, void *,
                    unsigned long, int);
#define __cimo_signed(x) ((__typeof__(x))-1 < 0)
#ifndef __CIMO_STOCK_CPROC
int __cimo_overflow128(int, unsigned __int128, int, unsigned __int128, int, void *, unsigned long, int);
#endif
#define __cimo_ovf(op, a, b, r, rs, rsg) (sizeof(a) > 8 || sizeof(b) > 8 || (rs) > 64 \
	? __cimo_overflow128(op, (unsigned __int128)(a), __cimo_signed(a), (unsigned __int128)(b), \
	                     __cimo_signed(b), r, rs, rsg) \
	: __cimo_overflow(op, (unsigned long long)(a), __cimo_signed(a), (unsigned long long)(b), \
	                  __cimo_signed(b), r, rs, rsg))
#define __builtin_add_overflow(a, b, r) __cimo_ovf(0, a, b, r, __cimo_widthof(*(r)), __cimo_signed(*(r)))
#define __builtin_sub_overflow(a, b, r) __cimo_ovf(1, a, b, r, __cimo_widthof(*(r)), __cimo_signed(*(r)))
#define __builtin_mul_overflow(a, b, r) __cimo_ovf(2, a, b, r, __cimo_widthof(*(r)), __cimo_signed(*(r)))
#define __builtin_add_overflow_p(a, b, c) __cimo_ovf(0, a, b, (void *)0, __cimo_widthof(c), __cimo_signed(c))
#define __builtin_sub_overflow_p(a, b, c) __cimo_ovf(1, a, b, (void *)0, __cimo_widthof(c), __cimo_signed(c))
#define __builtin_mul_overflow_p(a, b, c) __cimo_ovf(2, a, b, (void *)0, __cimo_widthof(c), __cimo_signed(c))
#define __cimo_ovft(op, t, a, b, r) __cimo_ovf(op, (t)(a), (t)(b), (t *)(r), __cimo_widthof((t)0), (t)-1 < 0)
#define __builtin_sadd_overflow(a, b, r) __cimo_ovft(0, int, a, b, r)
#define __builtin_saddl_overflow(a, b, r) __cimo_ovft(0, long, a, b, r)
#define __builtin_saddll_overflow(a, b, r) __cimo_ovft(0, long long, a, b, r)
#define __builtin_uadd_overflow(a, b, r) __cimo_ovft(0, unsigned, a, b, r)
#define __builtin_uaddl_overflow(a, b, r) __cimo_ovft(0, unsigned long, a, b, r)
#define __builtin_uaddll_overflow(a, b, r) __cimo_ovft(0, unsigned long long, a, b, r)
#define __builtin_ssub_overflow(a, b, r) __cimo_ovft(1, int, a, b, r)
#define __builtin_ssubl_overflow(a, b, r) __cimo_ovft(1, long, a, b, r)
#define __builtin_ssubll_overflow(a, b, r) __cimo_ovft(1, long long, a, b, r)
#define __builtin_usub_overflow(a, b, r) __cimo_ovft(1, unsigned, a, b, r)
#define __builtin_usubl_overflow(a, b, r) __cimo_ovft(1, unsigned long, a, b, r)
#define __builtin_usubll_overflow(a, b, r) __cimo_ovft(1, unsigned long long, a, b, r)
#define __builtin_smul_overflow(a, b, r) __cimo_ovft(2, int, a, b, r)
#define __builtin_smull_overflow(a, b, r) __cimo_ovft(2, long, a, b, r)
#define __builtin_smulll_overflow(a, b, r) __cimo_ovft(2, long long, a, b, r)
#define __builtin_umul_overflow(a, b, r) __cimo_ovft(2, unsigned, a, b, r)
#define __builtin_umull_overflow(a, b, r) __cimo_ovft(2, unsigned long, a, b, r)
#define __builtin_umulll_overflow(a, b, r) __cimo_ovft(2, unsigned long long, a, b, r)
'''
_CTYPE = {'w': ctypes.c_int32, 'l': ctypes.c_int64, 's': ctypes.c_float, 'd': ctypes.c_double,
          # a result of class w or l of an unsigned C type (see _sigs)
          'W': ctypes.c_uint32, 'L': ctypes.c_uint64,
          # char and short parameters on Apple arm64 (frontend.qbe.abiclass)
          'sb': ctypes.c_int8, 'ub': ctypes.c_uint8, 'sh': ctypes.c_int16, 'uh': ctypes.c_uint16,
          # a pointer parameter (class l; see _sigs): a buffer or an address
          'p': ctypes.c_int64}
_SIG = re.compile(r'^export\s*\nfunction\s+(?:([wlsd]|:[\w.]+)\s+)?\$(\w+)\(([^)]*)\)', re.M)


class CompileError(Exception):
    pass


def _cache_dir():
    """The module cache directory, or None if it is not writable (see _cache_get)."""
    return _link.user_cache('CIMO_CACHE_DIR')


_memory_cache: dict = {}  # key -> (assembly, signatures), when there is no writable cache directory


def _cache_get(key):
    d = _cache_dir()
    if d is None or key in _memory_cache:
        return _memory_cache.get(key)
    path = os.path.join(d, key + '.s')
    sig = path[:-2] + '.sig'
    if os.path.exists(path) and os.path.exists(sig):
        try:
            with open(path) as f, open(sig) as g:
                return f.read(), g.read()
        except OSError:  # Windows: another process replacing it (PermissionError): a miss
            pass
    return None


def _cache_put(key, text, sigs):
    d = _cache_dir()
    if d is not None and _link.cache_writes.get():
        path = os.path.join(d, key + '.s')
        try:
            for p, data in ((path[:-2] + '.sig', sigs), (path, text)):  # .s last: it marks
                tmp = p + '.tmp%d' % os.getpid()                       # the entry complete
                with open(tmp, 'w') as f:  # a temporary per process: two writing one entry don't mix
                    f.write(data)
                os.replace(tmp, p)
            return
        except OSError:
            pass
    _memory_cache[key] = (text, sigs)


def _image_get(key):
    """A cached linked image (signatures + cimo.loader snapshot), or None. A file whose
    checksum does not match (truncated, corrupted) is deleted and counts as a miss."""
    blob = _memory_cache.get(key + '.img')
    if blob is not None:
        return blob
    d = _cache_dir()
    if d is not None:
        p = os.path.join(d, key + '.img')
        try:
            with open(p, 'rb') as f:
                data = f.read()
        except OSError:
            return None
        if len(data) >= 4 and struct.unpack_from('<I', data)[0] == zlib.crc32(data[4:]):
            return data[4:]
        try:
            os.remove(p)
        except OSError:
            pass
    return None


def _image_put(key, blob):
    d = _cache_dir()
    if d is not None and _link.cache_writes.get():
        p = os.path.join(d, key + '.img')
        try:
            with open(p + '.tmp%d' % os.getpid(), 'wb') as f:
                f.write(struct.pack('<I', zlib.crc32(blob)) + blob)
            os.replace(p + '.tmp%d' % os.getpid(), p)
            return
        except OSError:
            pass
    _memory_cache[key + '.img'] = blob


_APPLE = sys.platform == 'darwin'
_WINDOWS = sys.platform == 'win32'
_AMD64 = _jit._X86
# This machine's target, as the preprocessor, backend and linker name it: Linux, Apple's ABI
# variant, or Windows (LP64 like the others, with Windows' variadic convention), on arm64; Linux
# (System V ABI) or Windows on x86-64.
if _AMD64:
    TARGET = 'amd64_windows' if _WINDOWS else 'amd64_sysv'
else:
    TARGET = 'arm64_apple' if _APPLE else 'arm64_win' if _WINDOWS else 'arm64'
_FRONTEND = {'arm64': 'aarch64', 'arm64_apple': 'aarch64-apple', 'arm64_win': 'aarch64-windows',
             'amd64_sysv': 'x86_64-sysv', 'amd64_windows': 'x86_64-windows'}
# va_arg of a struct (cimo.frontend vastruct=): each ABI's own
_VASTRUCT = {'arm64': 'aapcs', 'arm64_apple': 'slots', 'arm64_win': 'slots',
             'amd64_sysv': 'sysv', 'amd64_windows': 'win64'}
# Calls and builtins that become one instruction (cimo.frontend builtins=, bitops=): on x86-64
# what SSE2, the baseline, has, and what this CPU adds (cimo.cpu): contraction with FMA, POPCNT;
# fma() with FMA; no SSE4.1 rounding (floor, ceil, ...) yet, and fmin/fmax are C's, not minsd/maxsd.
_X86_BUILTINS = frozenset(('fsqrt', 'fabs'))
_X86_BITOPS = frozenset(('bclz', 'bctz', 'bswap', 'bswap16'))


def _is_amd64(target=None):
    return (target or TARGET).startswith('amd64')


def _x86(target=None):
    """The x86-64 features code for target may use: this CPU's (cimo.cpu) when target is this
    machine's, none for another x86-64 target or for arm64."""
    if not _is_amd64(target) or (target or TARGET) != TARGET:
        return frozenset()
    from . import cpu
    return cpu.x86_features()
_PYODIDE = sys.platform == 'emscripten'  # cimo.wasm: an Emscripten side module, not arm64
_VERSION = b'cimo-2'  # cache entry format; the compiler's own sources are hashed too (linker.compiler_key)


# the middle end's loop optimizations (cimo.middle); CIMO_MIDDLE=0 turns them off
MIDDLE = os.environ.get('CIMO_MIDDLE', '1') != '0'
# fused multiply-add where C allows contraction (within one expression, like clang's
# -ffp-contract=on); CIMO_FMA=0 turns it off, for results identical to -ffp-contract=off
FMA = os.environ.get('CIMO_FMA', '1') != '0'
_fma = contextvars.ContextVar('cimo_fma', default=None)  # compile(fma=): this call's choice
_wrapv = contextvars.ContextVar('cimo_wrapv', default=None)  # compile(wrapv=): this call's choice
WRAPV = os.environ.get('CIMO_WRAPV', '0') not in ('', '0')
# the frontend and backend ports keep their state in module globals, as the C programs do: one
# compilation at a time in a process. Taken only around them, and they take no other lock.
_compile_lock = threading.RLock()
# functions of at most this many IR instructions are copied into the callers that follow them
# (cimo.backend.inline); CIMO_INLINE=0 turns it off
INLINE = int(os.environ.get('CIMO_INLINE', '40'))


def _backend_ext(target=None):
    """The cimo.backend.ext flags for a target: the arm64 selection extensions (fma, ldp,
    layout, isel) only on arm64."""
    arm = not _is_amd64(target)
    x86 = _x86(target)
    return dict(fma=arm or 'fma' in x86, ldp=arm, layout=arm, math=True, trap=True, volatile=True,
                vla=True, cse=True, isel=arm, nowrap=True, vector=True, x86=x86)


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
    out = io.StringIO()
    try:
        with _compile_lock, _middle.options(enabled=MIDDLE if middle is None else middle,
                             params=params or {}), \
                _ext.options(**_backend_ext(target), bigcopy=256, inline=INLINE,
                             stkchk=stkchk), _Recursion():
            D.T = next(t for t in _targets() if t.name == (target or TARGET))
            drv = _Driver(out, 0)
            parse(ssa.encode('latin-1'), '<cimo>', drv.dbgfile, drv.data, drv.func)
            drv.finish()
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
    obj = Object()
    try:
        with _compile_lock, _middle.options(enabled=MIDDLE if middle is None else middle,
                             params=params or {}), _ext.options(math=True, fma=True, volatile=True, vla=True, cse=True,
                                        inline=INLINE), \
                _Recursion():
            D.T = T_wasm
            drv = _Driver(obj, 0)
            parse(ssa.encode('latin-1'), '<cimo>', drv.dbgfile, drv.data, drv.func)
            drv.finish()
    except D.BackendError as e:
        raise CompileError(str(e)) from None
    return obj


def _frontend(pp, filename, target=None, **kw):
    """Preprocessed C to (QBE IR text, parameter info for the middle end). target: as _qbe."""
    from .frontend import compile as cproc_compile
    from .frontend import qbe as _fq
    kw.setdefault('diagnostics', True)  # cimo's mode (the prelude needs it); cproc's own is frontend.compile's
    kw.setdefault('volatile', True)
    kw.setdefault('vlafree', True)
    kw.setdefault('bitops', True)
    kw.setdefault('bigcopy', 256)
    kw.setdefault('vastruct', _VASTRUCT[target or TARGET])
    if _is_amd64(target):
        x86 = _x86(target)
        if kw.get('builtins') is True:  # fma() as one vfmadd with FMA (never split: fmachain only splits contractions)
            kw['builtins'] = _X86_BUILTINS | ({'fma'} if 'fma' in x86 else set())
        if kw.get('bitops') is True:
            kw['bitops'] = _X86_BITOPS | ({'bpopcnt'} if 'popcnt' in x86 else set())
        if 'fma' not in x86:
            kw['fma'] = False
    with _compile_lock:
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
        multiarch = sysconfig.get_config_var('MULTIARCH') or (
            'x86_64-linux-gnu' if _AMD64 else 'aarch64-linux-gnu')
        dirs = ['/usr/local/include', '/usr/include/' + multiarch, '/usr/include']
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise CompileError('host_headers=True: no system header directory found')
    return dirs


def _check_machine():
    if not _jit._64BIT:
        raise CompileError('cimo needs a 64-bit Python; this one is 32-bit')
    if _AMD64:
        if _APPLE:
            raise CompileError('cimo does not support Intel Macs (x86-64 macOS); it runs on '
                               'Apple silicon, and on x86-64 Linux and Windows')
        return
    if not _jit._ARM64:
        raise CompileError('cimo generates arm64 and x86-64 code only; this machine is %s'
                           % platform.machine())


def _preprocess(src, filename='<string>', include_dirs=(), host_headers=False, defines=(),
                target=None):
    """C source -> preprocessed text, with the prelude. defines: (name, value or None) pairs.
    target: as _qbe."""
    from .preprocessor import preprocess, PreprocessError
    warn = None
    if host_headers:
        include_dirs = tuple(include_dirs) + tuple(_host_include_dirs())
    if (target or TARGET) == 'arm64_apple':
        def warn(msg):  # the SDK's sys/cdefs.h warns about every compiler but gcc and clang
            if not msg.endswith('#warning "Unsupported compiler detected"'):
                sys.stderr.write(msg + '\n')
    try:
        return preprocess(_PRELUDE + '#line 1\n' + src, filename, include_dirs=include_dirs,
                          defines=list(defines) or None, warn=warn,
                          target=target or TARGET, keep_attributes=True)
    except PreprocessError as e:
        raise CompileError(_hint(str(e), host_headers)) from None


def _vecq(target=None):
    """Bytes in the backend's class q for a native target (GNU vector types of that size or twice it compute in
    vector registers): 32 on x86-64 with AVX2 (ext.qbytes), else 16."""
    return 32 if _is_amd64(target) and 'avx2' in _x86(target) else 16


def _ir(pp, filename='<string>', host_headers=False):
    """Preprocessed text -> (QBE IR text, parameter info for the middle end)."""
    from .frontend import FrontendError
    try:
        return _frontend(pp, filename, fma=_contract(), nowrap=not _wraps(), builtins=True,
                         diagnostics=True, prune=True, vecq=_vecq(),
                         vecops=() if _is_amd64() else ('convert', 'shuffle'))
    except FrontendError as e:
        raise CompileError(_hint(str(e), host_headers)) from None


def _contract():
    """Whether this compile contracts a*b+c: its fma= argument, else the process default. Never
    on an x86-64 CPU without FMA."""
    if _is_amd64() and 'fma' not in _x86():
        return False
    v = _fma.get()
    return FMA if v is None else v


def _vectorizes():
    from . import middle as _middle
    return _middle.vectorizing


def _wraps():
    """Whether signed overflow wraps in this compile: its wrapv= argument, else the process default."""
    v = _wrapv.get()
    return WRAPV if v is None else v


# a line marker of the preprocessor's output: '# 12 "file.c" 2'
_MARKER = re.compile(r'#\s*(?:line\s+)?\d+(?:\s+"(?:[^"\\]|\\.)*")?(?:\s+\d+)*$')


def _code(pp):
    """Preprocessed text as far as code generation sees it: no line markers, blank lines or
    indentation. Those only place compile diagnostics, which a cache hit never prints;
    __LINE__ and __FILE__ are already literals in the text. A string can't span lines here."""
    return '\n'.join(l for l in (l.strip() for l in pp.splitlines()) if l and not _MARKER.match(l))


def _key(pp):
    """Cache key of preprocessed source (_code: moving code or renaming its file still hits):
    also the compiler's and the runtime's sources (a cached image contains the runtime members
    it uses), the target and the code generation options."""
    return hashlib.sha256(
        _VERSION + _link.compiler_key().encode() + _link._key(TARGET).encode() + TARGET.encode()
        + b'p%d' % _jit.PAGE
        + b'x' + ','.join(sorted(_x86())).encode()
        + (b'm' if MIDDLE else b'') + (b'f' if _contract() else b'') + (b'v' if _wraps() else b'') + (b'V' if _vectorizes() else b'') + b'i%d' % INLINE
        + _code(pp).encode('utf-8', 'surrogateescape')).hexdigest()[:24]


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
    with _compile_lock:  # longexterns() reads what this compilation's frontend left
        ssa, params = _ir(pp, filename, host_headers)
        text = _qbe(ssa, params, stkchk=_exe.stack_checked(ssa))
        sigs = _sigs(ssa, params)
        if _WINDOWS:
            from .frontend.diag import longexterns
            names = longexterns()
            if names:  # _module warns if one resolves to a link= library
                sigs += ('\n' if sigs else '') + '#long ' + ' '.join(names)
    return text, sigs


def _sigs(ssa, params=None):
    """'name ret params' lines for the exported functions of IR text. params: the frontend's
    parameter info, which says whose w or l result is unsigned (ret W, L), and which parameters are
    pointers (p, where the IR says l: a buffer or an address; an l is an integer, a numpy one too)."""
    unsigned = (params or {}).get('#unsigned', ())
    ctypes_ = (params or {}).get('#signatures', {})

    def classes(name, text):
        ks = [a.split()[0] for a in text.split(',') if a.strip()]
        c = ctypes_.get(name)
        if c is not None and len(c[1]) == len(ks):
            ks = ['p' if k == 'l' and t.endswith('*') else k for k, t in zip(ks, c[1])]
        return ','.join(ks)
    return '\n'.join('%s %s %s' % (m.group(2), (m.group(1) or '-').upper() if m.group(2) in unsigned
                                   else m.group(1) or '-', classes(m.group(2), m.group(3)))
                     for m in _SIG.finditer(ssa))


def _address(x, keep):
    """Pointer argument: a buffer (numpy array, bytearray, memoryview, ctypes array, bytes) or
    int. Objects that must outlive the call are appended to `keep`. bytes are passed in place
    (NUL-terminated, as ctypes does); other read-only buffers as a copy."""
    if isinstance(x, int):
        return x
    try:  # a writable buffer of at least a byte, contiguous: the quick way (numpy arrays, bytearray)
        return ctypes.addressof(ctypes.c_char.from_buffer(x))
    except (TypeError, ValueError, BufferError):
        pass
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


def _long(x, keep):
    """An l parameter: an integer, numpy's too. Where the signature has no C types (a module built from IR) a
    pointer is l as well, so a buffer is still taken as one."""
    if getattr(x, 'ndim', 0) == 0 and hasattr(x, '__index__'):
        return x.__index__()
    return _address(x, keep)


def describe(src, filename, options):
    """(signatures, data that isn't const) of C source, from the front end alone (cimo.introspect)."""
    target = ('arm64',) if _PYODIDE else ()
    pp = _preprocess(src, filename, tuple(options.get('include_dirs', ())), options.get('host_headers', False),
                     tuple(options.get('defines', ())), *target)
    with _compile_lock:
        _, params = _ir(pp, filename, options.get('host_headers', False))
    return params.get('#signatures', {}), params.get('#mutable', [])


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
            self._specialize()

    address: int
    __name__: str

    def __call__(self, *args: Any) -> Union[int, float, None]:
        """Calls the C function: ints and floats as C scalars; for pointer parameters, a
        buffer (bytes, bytearray, memoryview, numpy array: anything exposing its memory)
        or an int address. Returns an int, a float or None (void)."""
        if self._f is None:
            raise TypeError('%s() cannot be called from Python: %s (C callers in the module '
                            'can still use it)' % (self.__name__, self._unsupported))
        raise AssertionError('a kernel calls through its own class')

    def _specialize(self):
        """A class of its own whose __call__ is written for this kernel's parameters: pointers converted where
        they are, and the error word and the stdout buffer read from the thread's context as words, with no ctypes
        object made. The general version cost about 450 ns a call over ctypes (scramjit)."""
        n = len(self._params)
        ptrs = [i for i, k in enumerate(self._params) if k == 'p']
        longs = [i for i, k in enumerate(self._params) if k == 'l']
        lines = ['def __call__(self, *args):',
                 '    if len(args) != %d:' % n,
                 '        raise TypeError("%%s() takes %d arguments (%%d given)" %% (name, len(args)))' % n,
                 '    try:',
                 '        w = tls.words',
                 '    except AttributeError:',
                 '        state.ctx()',
                 '        w = tls.words']
        if ptrs or longs:
            lines.append('    keep = []  # read-only buffers passed by pointer: alive until the call returns')
            names = ['a%d' % i for i in range(n)]
            lines.append('    %s, = args' % ', '.join(names))
            for i in ptrs:
                lines.append('    if a%d.__class__ is not int: a%d = address(a%d, keep)' % (i, i, i))
            for i in longs:
                lines.append('    if a%d.__class__ is not int: a%d = long_(a%d, keep)' % (i, i, i))
            lines.append('    r = f(%s)' % ', '.join(names))
        else:
            lines.append('    r = f(*args)')
        if self._out is not None:  # C output reaches sys.stdout before Python's next print, also before an error
            lines.append('    if w[%d]: flush()' % (_exe.CTX_OUT // 8))
        lines += ['    if w[0]:', '        state.error(tls.ctx, name)', '    return r']
        env = {'f': self._f, 'tls': self._state._tls, 'state': self._state, 'name': self.__name__,
               'address': _address, 'long_': _long, 'flush': self._out._flush if self._out is not None else None}
        exec('\n'.join(lines), env)
        self.__class__ = type('Kernel', (Kernel,), {'__call__': env['__call__'], '__slots__': ()})

    def __repr__(self) -> str:
        return '<cimo kernel %s>' % self.__name__


class Module(Introspection):
    """Compiled C. Each non-static function is an attribute. Calling the module calls `main`,
    or the only function if there is just one."""

    def __init__(self, text, sigs, link=(), snapshot=False):
        """snapshot: keep the linked image for _image_bytes()."""
        names = [l.split(' ')[0] for l in sigs.splitlines() if l and not l.startswith('#')]
        libsyms = {}
        try:
            obj = _link.assemble(text + _exe.glue(names, _APPLE, _WINDOWS, TARGET), TARGET)
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
        except _jit.StaleImage:
            raise
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
        sigs = '\n'.join(l for l in lines if l and not l.startswith('#'))
        self._state = _exe.State(self._image)
        self._out = _link.Stdout.attach(self._image)
        self._kernels = {}
        for line in sigs.splitlines():
            name, ret, params = line.split(' ')
            self._kernels[name] = Kernel(self._image, name, None if ret == '-' else ret,
                                         params.split(',') if params else [], self._state,
                                         self._out)

    def reset_heap(self) -> None:
        """Free every heap block of the module (all threads' heaps) at once, e.g. between calls
        that use the heap as a scratch arena. Pointers into the heap become invalid. Must not
        be called while a kernel of the module is running."""
        self._state.reset_heap()

    def free(self, ptr: int) -> None:
        """free() a pointer that a kernel returned from malloc/calloc/realloc/aligned_alloc."""
        self._state.free(ptr)

    def heap_bytes(self) -> int:
        """Bytes of memory currently mapped for the module's heap."""
        return self._state.heap_bytes()

    def __getattr__(self, name: str) -> Kernel:
        try:
            return self.__dict__['_kernels'][name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self) -> list:
        return list(self._kernels)

    def __call__(self, *args: Any) -> Union[int, float, None]:
        if 'main' in self._kernels:
            return self._kernels['main'](*args)
        if len(self._kernels) == 1:
            return next(iter(self._kernels.values()))(*args)
        raise TypeError('module has no main(); call one of: %s' % ', '.join(self._kernels))


def compile(src: str, include_dirs: Iterable[PathArg] = (), link: Union[str, Iterable[str]] = (),
            host_headers: bool = False, defines: Defines = (), cache: bool = True,
            fma: Optional[bool] = None, wrapv: Optional[bool] = None) -> Module:
    """Compile C source and return a Module of callables.

    The module is linked with cimo's runtime library (string.h, math.h, printf, malloc, ...)
    and nothing else. link: host libraries whose symbols may resolve what the runtime does not
    provide: 'libc', 'libm', 'python' (the Python C API), a soname ('libz.so.1'), a short name
    ('z') or a path. host_headers: also search the host's C headers (after include_dirs and
    before cimo's bundled ones), to use declarations of linked libraries. defines: macros, as
    a dict or (name, value) pairs (value None means 1). cache=False: read the compile cache
    but write nothing to disk (no cache entries, no directories, no intermediate files);
    whatever is compiled, including runtime library members, is kept in memory. fma: contract
    a*b+c within one expression into a fused multiply-add (one rounding), like
    -ffp-contract=on; False gives results identical to -ffp-contract=off, and None the process
    default (on, or off with CIMO_FMA=0). Calls of fma() are always fused. wasm never
    contracts. wrapv: signed integer overflow wraps, like -fwrapv; otherwise it is undefined, as in
    C, and the compiler assumes it doesn't happen (None: the process default, off, or on with
    CIMO_WRAPV=1)."""
    return _module(src, '<string>', include_dirs, link, host_headers, defines, cache, fma, wrapv)


def load(path: PathArg, include_dirs: Iterable[PathArg] = (), link: Union[str, Iterable[str]] = (),
         host_headers: bool = False, defines: Defines = (), cache: bool = True,
         fma: Optional[bool] = None, wrapv: Optional[bool] = None) -> Module:
    """Compile a .c file; its directory is searched for "..." includes. Options as compile()."""
    from .resources import read_text
    path = os.fspath(path)
    return _module(read_text(path), path, include_dirs, link, host_headers, defines, cache, fma, wrapv)


def _module(src, filename, include_dirs, link, host_headers, defines, cache=True, fma=None, wrapv=None):
    token = _link.cache_writes.set(bool(cache))
    ftoken = _fma.set(None if fma is None else bool(fma))
    vtoken = _wrapv.set(None if wrapv is None else bool(wrapv))
    try:
        m = _module1(src, filename, include_dirs, link, host_headers, defines)
        options = {k: v for k, v in (('include_dirs', tuple(include_dirs)), ('link', (link,) if isinstance(link, str) else tuple(link)),
                                     ('host_headers', host_headers or ()), ('defines', tuple(defines.items() if hasattr(defines, 'items') else defines)),
                                     ('fma', () if fma is None else fma), ('wrapv', () if wrapv is None else wrapv)) if v != ()}
        m._record(src, filename, options)
        return m
    finally:
        _wrapv.reset(vtoken)
        _fma.reset(ftoken)
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
    m = None
    if blob is not None:
        try:
            m = Module._from_image(blob, tuple(link))
        except _jit.StaleImage:
            pass  # compiled again, below
    if m is None:
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
