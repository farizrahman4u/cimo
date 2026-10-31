"""Windows arm64 (target arm64_win): types (LP64 as everywhere in cimo, with Windows' signed
char, 16-bit wchar_t and 8-byte long double), Windows' variadic convention, x18 left alone,
stack probes. The code generation checks run on any host; the rest needs Windows on arm64:
calls into Microsoft-compiled variadic functions (msvcrt.dll), and, when an arm64 clang is
found (Visual Studio's LLVM, or CIMO_WIN_CLANG), calls in both directions with clang-built
code."""

import ctypes
import glob
import os
import platform
import subprocess
import sys

import pytest

import cimo
import extcompile as X
from cimo import frontend
from cimo.preprocessor import preprocess

WIN_ARM64 = sys.platform == 'win32' and platform.machine().lower() in ('arm64', 'aarch64')
needs_win = pytest.mark.skipif(not WIN_ARM64, reason='needs Windows on arm64')


def asm(src):
    pp = preprocess(src, 't.c', target='arm64_win')
    return X.backend(frontend.compile(pp, 't.c', target='aarch64-windows'), target='arm64_win')


def ir(src):
    return frontend.compile(preprocess(src, 't.c', target='arm64_win'), 't.c',
                            target='aarch64-windows')


def test_types():
    src = r'''
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <inttypes.h>
#define CHECK(c) _Static_assert(c, #c);
CHECK(sizeof(long) == 8 && sizeof(size_t) == 8 && sizeof(void *) == 8)  /* LP64 */
CHECK(LONG_MAX == 9223372036854775807L && __LP64__)
CHECK(sizeof(wchar_t) == 2 && sizeof(L'x') == 2 && (wchar_t)-1 > 0)
CHECK(sizeof(long double) == 8 && (char)-1 < 0 && CHAR_MIN < 0)
CHECK(_WIN32 && _WIN64 && _M_ARM64 && __aarch64__)
#if L'\xffff' < 0 || '\xff' > 0
#error character constants in #if
#endif
int f(void) { return sizeof(PRId64); }  /* "lld": int64_t is long long, as in MSVC */
long g(long a, unsigned long b) { return a + b; }
'''
    out = ir(src)
    assert 'function l $g(l %' in out
    assert 'ret 4\n' in out


def test_variadic_call_passes_floats_in_integer_registers():
    s = asm('double v(int, ...);\ndouble f(double d, float x) { return v(3, d, x, 7); }\n')
    assert 'fmov\tx1, d0' in s and 'fmov\tx2, d1' in s and 'mov\tw3, #7' in s


def test_variadic_definition_saves_x0_x7_only():
    s = asm('double named(double a, ...) { return a; }\n'
            'double defv(int n, ...) { __builtin_va_list ap; __builtin_va_start(ap, n); '
            'double s = __builtin_va_arg(ap, double); __builtin_va_end(ap); return s; }\n')
    assert 'stp\tx0, x1, [sp, -16]!' in s and 'str\tq0' not in s
    assert 'fmov\td0, x0' in s  # a named double arrives in x0


def test_no_x18_and_stack_probes():
    s = asm('void big(double *p, double x) { double a[5000]; a[4999] = x; a[0] = 1; '
            'p[0] = a[4999] + a[0]; }\n'
            'double vla(int n) { double a[n]; for (int i = 0; i < n; i++) a[i] = i; '
            'return a[n - 1]; }\n')
    assert 'x18' not in s and 'w18' not in s
    assert s.count('ldr\txzr, [x17]') == 1  # the big frame
    assert s.count('ldr\txzr, [sp]') == 1  # the VLA
    small = asm('double f(double *p) { double a[400]; a[3] = *p; return a[3]; }\n')
    assert 'xzr' not in small


@needs_win
def test_frames_larger_than_the_guard_page():
    m = cimo.compile(r'''
double big(int n) { double a[200000]; for (int i = 0; i < n; i++) a[i] = i;
  double s = 0; for (int i = 0; i < n; i++) s += a[n - 1 - i]; return s; }
double vla(int n) { double a[n]; for (int i = 0; i < n; i++) a[i] = i;
  double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }
''')
    import threading
    out = []
    t = threading.Thread(target=lambda: out.append((m.big(200000), m.vla(150000))))
    t.start()
    t.join()
    assert out == [(sum(range(200000)), sum(range(150000)))]
    assert m.big(1000) == sum(range(1000))


@needs_win
def test_calls_into_msvcrt_variadics():
    m = cimo.compile(r'''
int _snprintf(char *, unsigned long long, const char *, ...);
int sscanf(const char *, const char *, ...);
int fmt(char *out) {
  return _snprintf(out, 200, "%d|%.3f|%s|%I64d|%c|%g|%g|%g|%g|%g|%d", 42, 3.14159, "str",
                   -1234567890123LL, 'z', 1.0, 2.5f, 3.0, 4.0, 5.5, 99);
}
double scan(void) {
  int i; double d; char s[16]; long long q;
  int n = sscanf("17 2.5 word 9000000000", "%d %lf %15s %I64d", &i, &d, s, &q);
  return n * 1e12 + i * 1e9 + d * 1e6 + (s[0] == 'w') * 1000 + q / 1000000000;
}
''', link=['msvcrt.dll'])
    buf = ctypes.create_string_buffer(200)
    m.fmt(buf)
    assert buf.value == b'42|3.142|str|-1234567890123|z|1|2.5|3|4|5.5|99'
    assert m.scan() == 4e12 + 17e9 + 2.5e6 + 1000 + 9


def _clang():
    c = os.environ.get('CIMO_WIN_CLANG')
    if c:
        return c
    pf = os.environ.get('ProgramFiles', r'C:\Program Files')
    found = glob.glob(os.path.join(pf, 'Microsoft Visual Studio', '*', '*', 'VC', 'Tools',
                                   'Llvm', 'ARM64', 'bin', 'clang.exe'))
    return found[0] if found else None


REF = r'''
#include <stdarg.h>
#define API __declspec(dllexport)
typedef struct { float x, y, z; } V3;
API double vsum(int n, ...) { va_list ap; va_start(ap, n); double s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, double) * (i + 1); va_end(ap); return s; }
API long long vmix(const char *t, ...) { va_list ap; va_start(ap, t); long long s = 0;
  for (; *t; t++) s = s * 100 + (*t == 'd' ? (long long)va_arg(ap, double)
                                 : *t == 'q' ? va_arg(ap, long long) : va_arg(ap, int));
  va_end(ap); return s; }
API V3 v3add(V3 a, V3 b) { V3 r = {a.x + b.x, a.y + b.y, a.z + b.z}; return r; }
API double mixed(int a, double b, int c, float d, long long e, short f, char g, double h,
                 float i, int j, double k) { return a + b + c + d + e + f + g + h + i + j + k; }
API double callv(double (*f)(int, ...)) { return f(4, 1.0, 2.5, 4.0, -0.25); }
API double callv10(double (*f)(int, ...)) { return f(10, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                                      8.0, 9.0, 10.5); }
API long long callmix(long long (*f)(const char *, ...)) { return f("idqdi", 7, 1.5, 3LL, 2.0f, 9); }
'''


@needs_win
def test_abi_with_clang(tmp_path):
    clang = _clang()
    if not clang:
        pytest.skip('no arm64 clang (set CIMO_WIN_CLANG)')
    (tmp_path / 'ref.c').write_text(REF)
    dll = str(tmp_path / 'ref.dll')
    subprocess.check_call([clang, '--target=aarch64-pc-windows-msvc', '-O2', '-shared',
                           '-nostdlib', '-fuse-ld=lld', '-Wl,/noentry', '-o', dll,
                           str(tmp_path / 'ref.c')])
    m = cimo.compile(r'''
#include <stdarg.h>
typedef struct { float x, y, z; } V3;
double vsum(int n, ...); long long vmix(const char *t, ...); V3 v3add(V3 a, V3 b);
double mixed(int a, double b, int c, float d, long long e, short f, char g, double h,
             float i, int j, double k);
double callv(double (*f)(int, ...)); double callv10(double (*f)(int, ...));
long long callmix(long long (*f)(const char *, ...));
static double mysum(int n, ...) { va_list ap; va_start(ap, n); double s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, double) * (i + 1); va_end(ap); return s; }
static long long mymix(const char *t, ...) { va_list ap; va_start(ap, t); long long s = 0;
  for (; *t; t++) s = s * 100 + (*t == 'd' ? (long long)va_arg(ap, double)
                                 : *t == 'q' ? va_arg(ap, long long) : va_arg(ap, int));
  va_end(ap); return s; }
double t_vsum(void) { return vsum(10, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.5); }
long long t_vmix(void) { return vmix("qdidi", 12LL, 34.9, 56, 78.0, 90); }
double t_v3(void) { V3 a = {1, 2, 3}, b = {10, 20, 30}; V3 r = v3add(a, b);
  return r.x + r.y * 100 + r.z * 10000; }
double t_mixed(void) { return mixed(1, 2.5, -3, 4.25f, 1LL << 40, -7, 9, 0.5, 0.25f, 11, 100); }
double t_callv(void) { return callv(mysum); }
double t_callv10(void) { return callv10(mysum); }
long long t_callmix(void) { return callmix(mymix); }
''', link=[dll])
    weighted = sum((i + 1) * x for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.5]))
    assert m.t_vsum() == weighted
    assert m.t_vmix() == 1234567890
    assert m.t_v3() == 332211
    assert m.t_mixed() == 1 + 2.5 - 3 + 4.25 + (1 << 40) - 7 + 9 + 0.5 + 0.25 + 11 + 100
    assert m.t_callv() == 1 + 2.5 * 2 + 4 * 3 - 0.25 * 4
    assert m.t_callv10() == weighted
    assert m.t_callmix() == 701030209
