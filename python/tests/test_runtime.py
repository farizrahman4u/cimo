"""M5: cimo's bundled runtime library (python/cimo/runtime) and how modules link with it.

The C test programs in tests/runtime/t_*.c each export `long run(char *out, long cap)`, which
writes results as text. Each is built three ways and the outputs compared:

- cimo:  cimo.load(), i.e. cimo's runtime compiled by cimo;
- musl:  gcc -O2 against the same runtime sources (musl as gcc builds it): must be identical,
         so this checks the compiler on the runtime as much as the runtime;
- glibc: gcc against the host libc: identical where the C standard pins the result (string,
         ctype, strto*, qsort, printf except documented implementation-defined cases), within
         a few ulps for math.

Plus: printf ordering with Python's output and capture by redirect_stdout, the math builtins
(single instructions, exact C semantics), link= and the unknown-symbol error, and a check that
the default build resolves nothing from the process.
"""

import contextlib
import ctypes
import io
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys

import pytest

import cimo
from cimo import linker
import hosts

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(HERE, 'runtime')
CIMO_PKG = os.path.dirname(os.path.abspath(cimo.__file__))
_SYM = '_' if cimo.driver.TARGET == 'arm64_apple' else ''  # the target's symbol prefix
# the reference compiler: on Windows a cc on PATH is MinGW for x86-64 (GitHub's arm64 image),
# whose libraries an arm64 Python cannot load, so none there
CC = os.environ.get('CC', 'cc') if sys.platform != 'win32' else '<none on Windows>'
LINUX_ARM64 = sys.platform.startswith('linux') and hosts.RUNS  # Linux, any architecture

pytestmark = pytest.mark.skipif(not hosts.RUNS,
                                reason='arm64 only')
def _is_gcc(cc):
    """The references build the runtime itself, which defines __builtin_popcount and friends:
    gcc accepts that, clang does not."""
    import subprocess
    try:
        out = subprocess.run([cc, '--version'], capture_output=True, text=True).stdout
    except OSError:
        return False
    return 'Free Software Foundation' in out and 'clang' not in out


need_cc = pytest.mark.skipif(not (shutil.which(CC) and LINUX_ARM64 and _is_gcc(CC)),
                             reason='needs gcc on Linux for the references')

TESTS = ['t_string', 't_ctype', 't_stdlib', 't_printf', 't_math']
CAP = 8 << 20
# runtime files that need cimo's module glue (heap context, unwinding) or the stdout hook
NOT_IN_REF = ('heap.c', 'abort.c', os.path.join('stdio', 'stdout.c'),
              os.path.join('string', 'strdup.c'), os.path.join('string', 'strndup.c'))


def _run(f):
    buf = ctypes.create_string_buffer(CAP)
    n = f(buf, CAP)
    return buf.raw[:n].decode('latin-1')


@pytest.fixture(scope='module')
def cimo_out():
    from cimo import driver
    out = {}
    old = driver.FMA
    driver.FMA = False  # the test programs' own arithmetic (inputs) as gcc -ffp-contract=off
    try:
        for t in TESTS:
            m = cimo.load(os.path.join(RT, t + '.c'))
            out[t] = _run(m.run)
    finally:
        driver.FMA = old
    return out


@pytest.fixture(scope='module')
def refs(tmp_path_factory):
    """{'musl': {test: output}, 'glibc': {test: output}} from gcc builds."""
    tmp = tmp_path_factory.mktemp('rtref')
    inc = ['-nostdinc', '-isystem', os.path.join(CIMO_PKG, 'include')]
    # -fno-math-errno: else gcc's __builtin_sqrt calls sqrt for negative arguments, which in
    # the runtime's sqrt.c is itself
    common = ['-O2', '-fPIC', '-ffp-contract=off', '-fno-builtin', '-fno-math-errno', '-w',
              '-std=gnu11', '-U_FORTIFY_SOURCE']
    if hosts.AMD64:  # as cimo's own build of it: else floor, round, ... are calls to themselves
        common.append('-D__CIMO_SOFT_MATH=1')
    objs = []
    for rel in linker.runtime_sources():
        if rel in NOT_IN_REF:
            continue
        o = str(tmp / (rel.replace(os.sep, '_') + '.o'))
        objs.append((os.path.join(linker.RUNTIME_DIR, rel), o))
    # one gcc per directory keeps this to a few seconds
    for src, o in objs:
        subprocess.run([CC, '-c'] + common + inc + ['-I', linker.RUNTIME_INCLUDE, src, '-o', o],
                       check=True)
    lib = str(tmp / 'libcimort.a')
    subprocess.run(['ar', 'rcs', lib] + [o for _, o in objs], check=True)
    out = {'musl': {}, 'glibc': {}}
    for t in TESTS:
        src = os.path.join(RT, t + '.c')
        so = str(tmp / (t + '-musl.so'))
        subprocess.run([CC, '-shared'] + common + inc + ['-I', RT, src, lib, '-o', so,
                                                         '-Wl,-Bsymbolic'], check=True)
        out['musl'][t] = _run(ctypes.CDLL(so).run)
        so = str(tmp / (t + '-glibc.so'))
        subprocess.run([CC, '-shared'] + common + ['-I', RT, src, '-o', so, '-lm'], check=True)
        out['glibc'][t] = _run(ctypes.CDLL(so).run)
    return out


# ---------------------------------------------------------------------------------------------
# differential tests

def _nan_sign(text):
    """x86-64: SSE's default NaN is negative, and which operand's NaN an operation keeps depends on
    the register allocation (gcc folds 0.0/0.0 to a positive one). IEEE 754 leaves the sign open."""
    if not hosts.AMD64:
        return text
    return text.replace('fff8000000000000', '7ff8000000000000').replace('ffc00000', '7fc00000')


def _diff(a, b, n=8):
    la, lb = a.splitlines(), b.splitlines()
    d = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    if len(la) != len(lb):
        d.append(('lines', len(la), len(lb)))
    return d[:n]


@need_cc
@pytest.mark.parametrize('t', TESTS)
def test_same_as_musl_built_by_gcc(t, cimo_out, refs):
    assert len(cimo_out[t]) > 1000
    got, want = _nan_sign(cimo_out[t]), _nan_sign(refs['musl'][t])
    if got != want:  # (not assert ==: pytest would diff megabytes)
        pytest.fail(repr(_diff(got, want)))


@need_cc
@pytest.mark.parametrize('t', ['t_string', 't_ctype', 't_stdlib'])
def test_same_as_glibc(t, cimo_out, refs):
    if cimo_out[t] != refs['glibc'][t]:
        pytest.fail(repr(_diff(cimo_out[t], refs['glibc'][t])))


def _printf_known_difference(ours, theirs):
    """Formats where C leaves the output to the implementation and musl and glibc differ."""
    fmt = ours.split(' => ')[0]
    if fmt in ('%5%',):  # undefined: a conversion spec with flags around %
        return True
    if fmt in ('[%ls] [%lc] [%5ls]',):  # wide characters: glibc converts by the locale
        return True
    if '#' in fmt and fmt[-1] in 'gG' and ours.rstrip().endswith(('e+03', 'E+03', 'e+05', 'E+05')):
        # musl 1.2.5 bug: %#.Ng drops the trailing zeros when rounding carries into a new
        # digit (999.9995 -> "1.e+03", glibc and C: "1.00e+03"); gcc-built musl does the same
        return True
    if 'a' in fmt.lower() and re.search(r'%[-+ #0-9.]*[aA]', fmt):
        # %a: the leading digit of a hex float is unspecified for .0 precision rounding
        # (glibc may print 0x2p+0 where musl prints 0x1p+1)
        return True
    return False


@need_cc
def test_printf_matches_glibc(cimo_out, refs):
    ours, theirs = cimo_out['t_printf'].splitlines(), refs['glibc']['t_printf'].splitlines()
    assert len(ours) == len(theirs) > 20000
    bad = [(a, b) for a, b in zip(ours, theirs) if a != b and not _printf_known_difference(a, b)]
    if bad:
        pytest.fail('%d differences, first: %r' % (len(bad), bad[:10]))


def _ulps(a, b, single):
    """Distance in units in the last place between two results given as bit patterns."""
    fmt, ifmt, bits = ('<f', '<i', 32) if single else ('<d', '<q', 64)
    x = struct.unpack(fmt, struct.pack('<Q' if not single else '<I', a))[0]
    y = struct.unpack(fmt, struct.pack('<Q' if not single else '<I', b))[0]
    if math.isnan(x) or math.isnan(y):
        return 0 if math.isnan(x) and math.isnan(y) else float('inf')
    if x == y:
        return 0

    def ordered(v):
        i = struct.unpack(ifmt, struct.pack(fmt, v))[0]
        return i if i >= 0 else -(1 << (bits - 1)) - i
    return abs(ordered(x) - ordered(y))


# Functions the C standard (Annex F) pins exactly, or that musl and glibc both compute
# correctly rounded: must agree bit for bit (NaN payloads aside).
EXACT = set('''fabs sqrt ceil floor nearbyint rint round trunc fmod remainder copysign nextafter
fdim fmax fmin fma frexp modf ldexp scalbn remquo lrint ilogb logb scalbln'''.split())
# Allowed ulps between musl and glibc otherwise (sum of both libraries' documented errors,
# rounded up); lgamma/tgamma near poles and zeros of the gamma function are worse.
ULPS = {'lgamma': 1 << 20, 'tgamma': 16, 'erfc': 8, 'erf': 4, 'acosh': 4, 'asinh': 4,
        'atanh': 4, 'cbrt': 4, 'pow': 4, 'expm1': 4, 'log1p': 4, 'sinh': 4, 'cosh': 4,
        'tanh': 4, 'hypot': 2}


@need_cc
def test_math_matches_glibc_within_ulps(cimo_out, refs):
    ours = cimo_out['t_math'].splitlines()
    theirs = refs['glibc']['t_math'].splitlines()
    assert len(ours) == len(theirs) > 20000
    bad = []
    for a, b in zip(ours, theirs):
        if a == b:
            continue
        fa, fb = a.split(), b.split()
        name = fa[0]
        base = name[:-1] if name.endswith('f') and name not in ('modf', 'erf') else name
        single = base != name or name in ('modff', 'erff')
        if name in ('erff', 'modff'):
            base = name[:-1]
        if name in ('lrint', 'lrintf', 'ilogb', 'nan'):
            bad.append((a, b))  # integer results: exact
            continue
        for x, y in zip(fa[2:], fb[2:]):
            if len(x) > 8 or len(y) > 8 or single:
                u = _ulps(int(x, 16), int(y, 16), single)
            else:
                u = abs(int(x, 16) - int(y, 16))
            limit = 0 if base in EXACT else ULPS.get(base, 2)
            if u > limit:
                bad.append((a, b, u))
    if bad:
        pytest.fail('%d differences, first: %r' % (len(bad), bad[:10]))


# ---------------------------------------------------------------------------------------------
# stdout

def test_printf_order_and_redirect():
    m = cimo.compile(r'''
#include <stdio.h>
void hello(int n) { printf("hello %d\n", n); }
void partial(void) { printf("no newline;"); putchar('x'); }
void many(int n) { for (int i = 0; i < n; i++) printf("%d,", i); puts("end"); }
void chars(void) { putchar('a'); putchar('\n'); puts("p"); printf("%s", ""); }
''')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        print('py 1')
        m.hello(1)
        print('py 2')
        m.partial()       # no newline: still flushed when the kernel returns
        print('py 3')
        m.many(3000)      # more than the 4 KB buffer
        m.chars()
        m.hello(2)
    big = ','.join(str(i) for i in range(3000)) + ',end\n'
    assert f.getvalue() == ('py 1\nhello 1\npy 2\nno newline;xpy 3\n' + big +
                            'a\np\nhello 2\n')


def test_printf_utf8_and_return_values():
    m = cimo.compile(r'''
#include <stdio.h>
int p(void) { return printf("%s|%5.1f|%x\n", "\xc3\xa9t\xc3\xa9", 2.25, 255); }
int q(void) { return puts("done"); }
''')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        assert m.p() == len('été|  2.2|ff\n'.encode())
        assert m.q() >= 0
    assert f.getvalue() == 'été|  2.2|ff\ndone\n'


UTF8_SPLIT_SRC = r'''
#include <stdio.h>
void byte(int c) { putchar(c); }
void fill(int n) { for (int i = 0; i < n; i++) putchar('a'); putchar(0xc3); putchar(0xa9); }
'''


def utf8_split_run(m):
    """A character split between two kernel calls, then one split by a full buffer (4088 bytes
    of buffer: 4087 'a's + the lead byte fill it, so the next byte flushes between the two)."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        m.byte(0xc3)
        m.byte(0xa9)
        m.fill(4087)
        m.byte(ord('\n'))
    return f.getvalue()


def test_printf_utf8_split_between_writes():
    assert utf8_split_run(cimo.compile(UTF8_SPLIT_SRC)) == 'é' + 'a' * 4087 + 'é\n'


def test_printf_from_threads():
    """Each thread has its own buffer: lines printed concurrently come out whole, none lost,
    and each thread's lines stay in order. Partial lines are flushed when the kernel returns."""
    import threading
    m = cimo.compile(r'''#include <stdio.h>
void p(int t, int n) {
    for (int i = 0; i < n; i++) {
        printf("t%d %d ", t, i);
        for (int k = 0; k < 20; k++) putchar('a' + k);
        puts("");
    }
}
void partial(int t) { printf("[%d", t); printf("]"); }''')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ts = [threading.Thread(target=m.p, args=(t, 2000)) for t in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    lines = f.getvalue().splitlines()
    assert len(lines) == 8 * 2000
    seen = [[] for _ in range(8)]
    for line in lines:
        mt = re.fullmatch(r't(\d) (\d+) abcdefghijklmnopqrst', line)
        assert mt, line
        seen[int(mt.group(1))].append(int(mt.group(2)))
    assert all(s == list(range(2000)) for s in seen)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ts = [threading.Thread(target=m.partial, args=(t,)) for t in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert sorted(re.findall(r'\[\d\]', f.getvalue())) == ['[%d]' % t for t in range(8)]


def test_gnu_bit_builtins():
    m = cimo.compile(r'''
int popc(unsigned long long x) { return __builtin_popcountll(x); }
int popc32(unsigned x) { return __builtin_popcount(x); }
int clz32(unsigned x) { return __builtin_clz(x); }
int clz64(unsigned long long x) { return __builtin_clzll(x); }
int ctz32(unsigned x) { return __builtin_ctz(x); }
int ctz64(unsigned long x) { return __builtin_ctzl(x); }
int ffs32(int x) { return __builtin_ffs(x); }
int ffs64(long long x) { return __builtin_ffsll(x); }
int par(unsigned long long x) { return __builtin_parityll(x); }
int clrsb32(int x) { return __builtin_clrsb(x); }
int clrsb64(long x) { return __builtin_clrsbl(x); }
unsigned bs16(unsigned x) { return __builtin_bswap16(x); }
unsigned bs32(unsigned x) { return __builtin_bswap32(x); }
unsigned long bs64(unsigned long x) { return __builtin_bswap64(x); }
int cls(double x) { return __builtin_isnan(x) | __builtin_isinf(x) << 1
                           | __builtin_isfinite(x) << 2 | __builtin_signbit(x) << 3; }
int clsf(float x) { return cls(x) == (__builtin_isnan(x) | __builtin_isinf(x) << 1
                                      | __builtin_isfinite(x) << 2 | __builtin_signbit(x) << 3); }
double consts(int i) { return i == 0 ? __builtin_huge_val() : i == 1 ? __builtin_inf()
                              : i == 2 ? __builtin_huge_valf() : __builtin_nan(""); }
''')
    rnd = __import__('random').Random(5)
    M64 = (1 << 64) - 1

    def s32(v):
        return v - (1 << 32) if v >> 31 else v

    vals = [1, 2, 3, 0x80000000, 0xffffffff, 1 << 63, M64] + [
        rnd.getrandbits(rnd.randrange(1, 65)) | 1 << rnd.randrange(64) for _ in range(300)]
    for v in vals:
        lo = v & 0xffffffff
        sv = v - (1 << 64) if v >> 63 else v
        assert m.popc(sv) == bin(v).count('1')
        assert m.par(sv) == bin(v).count('1') & 1
        assert m.clz64(sv) == 64 - v.bit_length()
        assert m.ctz64(sv) == (v & -v).bit_length() - 1
        assert m.ffs64(sv) == (v & -v).bit_length()
        assert m.clrsb64(sv) == 63 - (sv if sv >= 0 else ~sv).bit_length()
        assert m.bs64(sv) & M64 == int.from_bytes(v.to_bytes(8, 'little'), 'big')
        if lo:
            assert m.popc32(s32(lo)) == bin(lo).count('1')
            assert m.clz32(s32(lo)) == 32 - lo.bit_length()
            assert m.ctz32(s32(lo)) == (lo & -lo).bit_length() - 1
        assert m.ffs32(s32(lo)) == (lo & -lo).bit_length()
        assert m.clrsb32(s32(lo)) == 31 - (s32(lo) if s32(lo) >= 0 else ~s32(lo)).bit_length()
        assert m.bs32(s32(lo)) & 0xffffffff == int.from_bytes(lo.to_bytes(4, 'little'), 'big')
        assert m.bs16(lo & 0xffff) == int.from_bytes((lo & 0xffff).to_bytes(2, 'little'), 'big')
    inf, nan = float('inf'), float('nan')
    for x in (0.0, -0.0, 1.5, -2.0, 5e-324, -1e308, inf, -inf, nan, -nan):
        want = (x != x) | (abs(x) == inf) << 1 | (abs(x) < inf) << 2 | (math.copysign(1, x) < 0) << 3
        assert m.cls(x) == want, x
        assert m.clsf(x) == 1
    assert [m.consts(i) for i in range(3)] == [inf] * 3 and math.isnan(m.consts(3))


def test_rand_exit_inttypes():
    m = cimo.compile(r'''#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
int r(unsigned s, int n) { srand(s); int x = 0; while (n--) x = rand(); return x; }
void quit(int s) { exit(s); }
void quit2(void) { _Exit(EXIT_FAILURE); }
long pri(void) {
    int64_t a = INT64_MIN; uint64_t b = UINT64_MAX; intptr_t c = -1; uint_fast16_t d = 65535;
    char buf[200];
    snprintf(buf, sizeof buf, "%" PRId64 " %" PRIu64 " %" PRIxPTR " %" PRIuFAST16 " %" PRIdMAX,
             a, b, (uintptr_t)c, d, (intmax_t)7);
    puts(buf);
    char *e;
    return strtoimax(" -42x", &e, 10) * 10 + (*e == 'x') + imaxabs(-100) * 1000 + imaxdiv(7, 2).rem;
}''')
    # musl's LCG: same sequence as musl
    seed = 41
    for _ in range(3):
        seed = (6364136223846793005 * seed + 1) & ((1 << 64) - 1)
    assert m.r(42, 3) == seed >> 33
    with pytest.raises(cimo.KernelError) as e:
        m.quit(3)
    assert e.value.kind == 'exit' and e.value.status == 3 and 'exit(3)' in str(e.value)
    with pytest.raises(cimo.KernelError) as e:
        m.quit2()
    assert e.value.status == 1
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        assert m.pri() == -420 + 1 + 100000 + 1
    assert f.getvalue() == '%d %d ffffffffffffffff 65535 7\n' % (-2**63, 2**64 - 1)


def test_unsupported_signatures_raise_on_call():
    m = cimo.compile(r'''#include <stdarg.h>
struct s { int a; };
int va(int n, ...) { va_list ap; va_start(ap, n); int r = va_arg(ap, int); va_end(ap); return r; }
struct s sv(struct s x) { return x; }
int f(void) { return va(1, 41) + sv((struct s){1}).a; }''')
    assert m.f() == 42
    with pytest.raises(TypeError, match='variadic'):
        m.va(1, 2)
    with pytest.raises(TypeError, match='struct'):
        m.sv(1)


# ---------------------------------------------------------------------------------------------
# math builtins

BUILTIN_SRC = r'''
#include <math.h>
double b_sqrt(double x) { return sqrt(x); }
double b_fabs(double x) { return fabs(x); }
double b_floor(double x) { return floor(x); }
double b_ceil(double x) { return ceil(x); }
double b_trunc(double x) { return trunc(x); }
double b_round(double x) { return round(x); }
double b_rint(double x) { return rint(x); }
double b_nearbyint(double x) { return nearbyint(x); }
double b_fmin(double x, double y) { return fmin(x, y); }
double b_fmax(double x, double y) { return fmax(x, y); }
double b_fma(double x, double y, double z) { return fma(x, y, z); }
float b_sqrtf(float x) { return sqrtf(x); }
float b_roundf(float x) { return roundf(x); }
float b_fminf(float x, float y) { return fminf(x, y); }
float b_floorf(float x) { return floorf(x); }
float b_rintf(float x) { return rintf(x); }
float b_nearbyintf(float x) { return nearbyintf(x); }
/* through a function pointer: the runtime's out-of-line definitions */
static double (*volatile p1[])(double) = { sqrt, fabs, floor, ceil, trunc, round, rint,
                                             nearbyint };
static double (*volatile p2[])(double, double) = { fmin, fmax };
double o1(int k, double x) { return p1[k](x); }
double o2(int k, double x, double y) { return p2[k](x, y); }
'''

SPECIALS = [0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994,
            4503599627370495.5, -4503599627370495.5, 1e300, -1e-300, 5e-324, 3.7, -3.7, 1e16 + 1,
            float('inf'), float('-inf'), float('nan'), 2.0, 1e-310]


def _bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def _c_round(x):
    """C round(): halfway cases away from zero."""
    if math.isnan(x) or math.isinf(x) or x == 0:
        return x
    r = math.floor(abs(x) + 0.5) if abs(x) < 2 ** 52 else abs(x)
    if abs(x) < 2 ** 52 and r - abs(x) == 0.5 and (abs(x) + 0.5) != r:
        pass
    # exact: floor(|x| + 0.5) can round up in the addition for 0.49999999999999994
    if abs(x) < 2 ** 52:
        t = math.floor(abs(x))
        r = t + 1 if abs(x) - t >= 0.5 else t
    return math.copysign(r, x)


def _c_rint(x):
    """C rint() in the default rounding mode: halfway cases to even."""
    return x if not math.isfinite(x) else math.copysign(float(round(x)), x)


def _c_fmin(x, y, mx=False):
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x == y == 0:
        neg = (math.copysign(1, x) < 0, math.copysign(1, y) < 0)
        return (0.0 if not all(neg) else -0.0) if mx else (-0.0 if any(neg) else 0.0)
    return max(x, y) if mx else min(x, y)


def test_math_builtins_are_instructions():
    from cimo import driver
    text, _ = driver._build(BUILTIN_SRC)
    body = {}
    if driver.TARGET.startswith('amd64'):
        # x86-64's baseline (SSE2) has sqrt and fabs (andpd with a mask); the others, fma, the
        # SSE4.1 roundings and C's fmin/fmax, are calls to the runtime
        for name, insn in (('b_sqrt', 'sqrtsd'), ('b_fabs', 'andpd'), ('b_sqrtf', 'sqrtss')):
            m = re.search(r'^%s:\n(.*?)\n\s*ret\n' % name, text, re.S | re.M)
            assert m and insn in m.group(1) and 'call' not in m.group(1), (name, m and m.group(1))
        return
    for name in ('b_sqrt', 'b_fabs', 'b_floor', 'b_ceil', 'b_trunc', 'b_round', 'b_fmin',
                 'b_fmax', 'b_fma', 'b_sqrtf', 'b_roundf', 'b_fminf', 'b_floorf', 'b_rint',
                 'b_nearbyint', 'b_rintf', 'b_nearbyintf'):
        m = re.search(r'^%s%s:\n(.*?)\n\s*ret\n' % (_SYM, name), text, re.S | re.M)
        assert m, name
        body[name] = m.group(1)
        assert 'bl\t' not in body[name] and 'blr' not in body[name], (name, body[name])
    for name, insn in (('b_sqrt', 'fsqrt\td'), ('b_fabs', 'fabs\td'), ('b_floor', 'frintm\td'),
                       ('b_ceil', 'frintp\td'), ('b_trunc', 'frintz\td'),
                       ('b_round', 'frinta\td'), ('b_fmin', 'fminnm\td'),
                       ('b_fmax', 'fmaxnm\td'), ('b_fma', 'fmadd\td'), ('b_sqrtf', 'fsqrt\ts'),
                       ('b_roundf', 'frinta\ts'), ('b_fminf', 'fminnm\ts'),
                       ('b_floorf', 'frintm\ts'), ('b_rint', 'frintx\td'),
                       ('b_nearbyint', 'frinti\td'), ('b_rintf', 'frintx\ts'),
                       ('b_nearbyintf', 'frinti\ts')):
        assert insn in body[name], (name, body[name])


def test_math_builtins_semantics():
    m = cimo.compile(BUILTIN_SRC)
    ops = [('sqrt', m.b_sqrt, lambda x: math.sqrt(x) if x >= 0 or math.isnan(x) else float('nan'), 0),
           ('fabs', m.b_fabs, abs, 1), ('floor', m.b_floor, lambda x: x if not math.isfinite(x) else math.copysign(math.floor(x), x), 2),
           ('ceil', m.b_ceil, lambda x: x if not math.isfinite(x) else math.copysign(math.ceil(x), x), 3),
           ('trunc', m.b_trunc, lambda x: x if not math.isfinite(x) else math.copysign(math.trunc(x), x), 4),
           ('round', m.b_round, _c_round, 5), ('rint', m.b_rint, _c_rint, 6),
           ('nearbyint', m.b_nearbyint, _c_rint, 7)]
    for name, f, ref, k in ops:
        for x in SPECIALS + [-s for s in SPECIALS]:
            want = ref(x)
            for got in (f(x), m.o1(k, x)):
                if math.isnan(want):
                    assert math.isnan(got), (name, x, got)
                else:
                    assert _bits(got) == _bits(want), (name, x, got, want)
    vals = [0.0, -0.0, 1.0, -1.0, float('nan'), float('inf'), float('-inf'), 5e-324]
    for x in vals:
        for y in vals:
            for k, (f, mx) in enumerate(((m.b_fmin, False), (m.b_fmax, True))):
                want = _c_fmin(x, y, mx)
                for got in (f(x, y), m.o2(k, x, y)):
                    if math.isnan(want):
                        assert math.isnan(got)
                    else:
                        assert _bits(got) == _bits(want), (k, x, y, got, want)
    assert m.b_fma(0.1, 10.0, -1.0) == 5.551115123125783e-17  # fused: one rounding
    assert m.b_roundf(2.5) == 3.0 and m.b_roundf(-0.5) == -1.0
    assert m.b_fminf(float('nan'), 2.0) == 2.0
    assert m.b_rintf(2.5) == 2.0 and m.b_nearbyintf(-3.5) == -4.0 and m.b_rintf(1.25) == 1.0
    assert m.b_sqrtf(2.0) == struct.unpack('<f', struct.pack('<f', math.sqrt(2)))[0]


def test_user_function_named_like_a_builtin():
    """A static function (or one with another prototype) is not the library function."""
    m = cimo.compile('static double sqrt(double x) { return x + 1; }\n'
                     'double f(double x) { return sqrt(x); }\n')
    assert m.f(4.0) == 5.0


def test_math_in_loops_is_optimized():
    """sqrt in a loop no longer blocks the middle end (nbody_acc's inner loop)."""
    from cimo import driver
    text, _ = driver._build(r'''
#include <math.h>
void f(int n, const double *restrict x, double *restrict y) {
    for (int i = 0; i < n; i++) y[i] = 1.0 / sqrt(x[i] * x[i] + 1.0);
}''')
    body = re.split(r'^_?f:$', text, flags=re.M)[1]  # Apple's symbols have a _
    if driver.TARGET.startswith('amd64'):
        assert body.count('sqrtsd') >= 4 and 'call' not in body  # unrolled by 4
        return
    assert text.count('fsqrt') >= 4  # unrolled by 4
    assert '\tbl\t' not in body


# ---------------------------------------------------------------------------------------------
# linking

def test_default_build_resolves_nothing_from_the_process(monkeypatch, tmp_path):
    """A module using the whole runtime links with no symbol left for the host: the merged
    object has no undefined symbols at all, and the loader's dlsym is never asked, neither on
    the first build nor when the cached image is loaded."""
    from cimo import loader, driver
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(driver, '_memory_cache', {})

    def no_dlsym(name):
        raise AssertionError('looked up %s in the process' % name)
    monkeypatch.setattr(loader, '_dlsym', no_dlsym)
    seen = []
    real_link = linker.link

    def spy(obj, hooks=None, libs=(), libsyms=None):
        merged, resolve = real_link(obj, hooks, libs, libsyms)
        seen.append(merged)
        return merged, resolve
    monkeypatch.setattr(linker, 'link', spy)
    src = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
static int cmp(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
double all(int n) {
    char buf[64];
    int *v = malloc(n * sizeof *v);
    for (int i = 0; i < n; i++) v[i] = (i * 7919) % n;
    qsort(v, n, sizeof *v, cmp);
    snprintf(buf, sizeof buf, "%d %s %g", v[n - 1], "x", sin(1.0));
    double r = strtod("2.5", 0) + strlen(buf) + toupper('a') + atoi("12") + pow(2, 0.5)
               + exp(1) + log(2) + tgamma(4.5) + atan2(1, 2) + fmod(7, 3) + cbrt(27);
    free(v);
    assert(r > 0);
    printf("%s\n", buf);
    return r;
}'''
    for build in range(2):
        m = cimo.compile(src)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            r = m.all(100)
        assert f.getvalue().startswith('99 x 0.841471\n')
        assert r > 0
        monkeypatch.setattr(driver, '_memory_cache', {})
    assert len(seen) == 1  # the second build loaded the image
    merged = seen[-1]
    undefined = [y.name for y in merged.symbols.values() if not y.defined and not y.common]
    assert undefined == []


GETPID = '_getpid' if sys.platform == 'win32' else 'getpid'  # the UCRT's name


def test_unknown_symbol_is_a_compile_error():
    src = 'int getpid(void);\nint f(void) { return getpid(); }\n'.replace('getpid', GETPID)
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile(src)
    msg = str(e.value)
    assert "'%s'" % GETPID in msg and 'link=' in msg
    m = cimo.compile(src, link=['libc'])
    assert m.f() == os.getpid()


def test_link_library_by_soname_and_path():
    src = 'unsigned long zlibCompileFlags(void);\nunsigned long f(void) { return zlibCompileFlags(); }\n'
    import ctypes.util
    so = ctypes.util.find_library('z')
    if not so:
        pytest.skip('no zlib')
    z = ctypes.CDLL(so)
    z.zlibCompileFlags.restype = ctypes.c_ulong
    for spec in (so, 'z'):
        assert cimo.compile(src, link=[spec]).f() == z.zlibCompileFlags()
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile(src, link=['libc'])
    assert 'zlibCompileFlags' in str(e.value)
    with pytest.raises(cimo.CompileError):
        cimo.compile(src, link=['libdoesnotexist.so.9'])


def test_runtime_wins_over_linked_libc():
    """With link=['libc'], functions the runtime provides still come from the runtime (malloc
    from the module heap, printf to sys.stdout); the rest from libc."""
    m = cimo.compile(r'''
#include <stdio.h>
int getpid(void);
int f(void) { return printf("pid %d\n", getpid()); }'''.replace('getpid', GETPID), link=['libc'])
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        m.f()
    assert f.getvalue() == 'pid %d\n' % os.getpid()


@pytest.mark.skipif(not LINUX_ARM64 or not os.path.exists('/usr/include/unistd.h'),
                    reason='needs glibc headers')
def test_host_headers():
    m = cimo.compile(r'''
#include <unistd.h>
#include <string.h>
long f(void) { return getpid() + strlen("abc"); }''', link=['libc'], host_headers=True)
    assert m.f() == os.getpid() + 3


@pytest.mark.skipif(sys.platform == 'win32', reason='host_headers is not available on Windows')
def test_host_data_symbols():
    """External data of linked libraries is reached through GOT slots, so it doesn't matter
    how far from the module the library is mapped."""
    m = cimo.compile(r'''
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#ifdef __APPLE__
#include <crt_externs.h>
#define __environ (*_NSGetEnviron())
#endif
extern char **environ;
int *p = &optind;
long f(void) {
    errno = 0;
    errno = 7;
    return errno * 1000 + (environ == __environ) * 10 + *p + (&optind)[0];
}
int g(void) { return fileno(stdout) * 10 + fileno(stderr); }''', link=['libc'], host_headers=True)
    assert m.f() == 7012
    assert m.g() == 12


def test_user_definition_beats_runtime_and_clash_is_reported():
    m = cimo.compile('#include <string.h>\nsize_t strlen(const char *s) { return 42; }\n'
                     'long f(void) { return strlen("x"); }\n')
    assert m.f() == 42
    # defining a function the runtime also defines in an object that gets pulled in
    # defining puts and calling printf: printf's object (runtime/stdio/stdout.c) defines both
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile('#include <stdio.h>\nint puts(const char *s) { return 0; }\n'
                     'int f(void) { return printf("x"); }\n')
    assert "'puts'" in str(e.value) and 'stdout.c' in str(e.value)


def test_runtime_symbol_table_is_current():
    """runtime/symbols.txt lists exactly what each runtime source defines (regenerate with
    `cimo __update-runtime-symbols`), and every member compiles and links on its own."""
    rt = linker.runtime()
    assert linker.runtime() is rt
    assert sorted(rt.members) == sorted(m.replace(os.sep, '/')
                                        for m in linker.runtime_sources())
    for m in rt.members:
        rt.object(m)  # raises LinkError if symbols.txt is out of date for m


def test_runtime_is_compiled_on_demand(tmp_path, monkeypatch):
    """A link compiles only the members it needs, once; later links reuse them."""
    monkeypatch.setenv('CIMO_RUNTIME_CACHE', str(tmp_path / 'rt'))
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path / 'mods'))
    monkeypatch.setattr(linker, '_runtime', None)
    events = []
    monkeypatch.setattr(linker, 'progress', lambda e, **i: events.append((e, i)))
    m = cimo.compile('#include <string.h>\nlong f(const char *s) { return strlen(s); }')
    assert m.f(bytearray(b'abcd\0')) == 4
    assert [i['name'] for e, i in events if e == 'file'] == ['string/strlen.c']
    assert events[-1][0] == 'done' and events[-1][1]['files'] == 1
    del events[:]
    m = cimo.compile('#include <string.h>\nlong g(const char *s) { return strlen(s) + 1; }')
    assert m.g(bytearray(b'ab\0')) == 3 and events == []
    monkeypatch.setattr(linker, '_runtime', None)  # a new process: from the disk cache
    m = cimo.compile('#include <string.h>\nlong h(const char *s) { return strlen(s) + 2; }')
    assert m.h(bytearray(b'a\0')) == 3 and events == []
    assert os.listdir(str(tmp_path / 'rt'))[0].startswith('runtime-')


def test_cache_false_writes_nothing(tmp_path, monkeypatch):
    """compile(cache=False): no file or directory is created, even with the runtime cold and
    the cache directories writable; the runtime members it builds stay in memory."""
    import builtins
    for d in ('rt', 'mods'):
        (tmp_path / d).mkdir()
    monkeypatch.setenv('CIMO_RUNTIME_CACHE', str(tmp_path / 'rt'))
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path / 'mods'))
    monkeypatch.setattr(linker, '_runtime', None)
    monkeypatch.setattr(linker, 'progress', lambda e, **i: None)
    real_open = builtins.open
    writes = []

    def open_(f, mode='r', *a, **k):
        if any(c in mode for c in 'wax+'):
            writes.append(f)
        return real_open(f, mode, *a, **k)
    monkeypatch.setattr(builtins, 'open', open_)
    for name in ('makedirs', 'mkdir', 'replace', 'rename'):
        monkeypatch.setattr(os, name, lambda *a, _n=name, **k: writes.append((_n, a)))
    src = '#include <string.h>\nlong f(const char *s) { return strlen(s) + 7; }'
    m = cimo.compile(src, cache=False)
    assert m.f(b'abc') == 10
    assert writes == []
    monkeypatch.undo()
    assert os.listdir(str(tmp_path / 'rt')) == os.listdir(str(tmp_path / 'mods')) == []


def test_cache_false_reads_the_cache(tmp_path, monkeypatch):
    """cache=False still uses entries already on disk; with no cache directory it creates
    none."""
    from cimo import driver
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path / 'mods'))
    src = 'long f(long x) { return x * 3 + 1; }'
    cimo.compile(src)
    monkeypatch.setattr(driver, '_memory_cache', {})
    monkeypatch.setattr(driver, '_ir', lambda *a: 1 / 0)  # a miss would raise
    assert cimo.compile(src, cache=False).f(4) == 13
    monkeypatch.undo()
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path / 'none'))
    assert cimo.compile('long g(long x) { return x - 9; }', cache=False).g(10) == 1
    assert not os.path.exists(str(tmp_path / 'none'))


IMAGE_SRC = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static long twice(long x) { return 2 * x; }
static long plus3(long x) { return x + 3; }
long (*table[])(long) = { twice, plus3 };       /* ABS64 pointers into the image */
const char *words[] = { "alpha", "beta" };
long counter;
size_t (*ext_strlen)(const char *) = strlen;
char *getenv(const char *);
char *(*libc_getenv)(const char *) = getenv;  /* ABS64 to a link= library symbol */
long f(long i, long x) { counter++; return table[i](x) + ext_strlen(words[i]); }
long g(int x) { printf("g%d\n", x); return (libc_getenv("CIMO_IMAGE_UNSET") == 0) + x + counter; }
"""


def test_image_cache(tmp_path, monkeypatch, capfd):
    """A module's linked image is cached (.img, no .s) and loaded without compiling,
    assembling or linking; pointers into the image, GOT slots, stubs and stdout work."""
    from cimo import driver, assembler
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(driver, '_memory_cache', {})

    def check(m):
        assert m.f(0, 5) == 15 and m.f(1, 5) == 12
        assert m.g(-4) == -1
        capfd.readouterr()
        m.g(7)
        assert capfd.readouterr().out == 'g7\n'
    check(cimo.compile(IMAGE_SRC, link=['libc']))
    assert sorted(os.path.splitext(n)[1] for n in os.listdir(str(tmp_path))) == ['.img']
    monkeypatch.setattr(driver, '_memory_cache', {})
    monkeypatch.setattr(driver, '_compile', lambda *a: 1 / 0)
    monkeypatch.setattr(assembler, 'assemble', lambda *a: 1 / 0)
    monkeypatch.setattr(linker, 'link', lambda *a, **k: 1 / 0)
    m = cimo.compile(IMAGE_SRC, link=['libc'])
    check(m)
    m2 = cimo.compile(IMAGE_SRC, link=['libc'])  # a second, independent copy
    assert m2.f(0, 1) == 7 and m.g(0) == 3 and m2.g(0) == 2
    with pytest.raises(cimo.CompileError, match="undefined symbol 'getenv'"):
        cimo.compile(IMAGE_SRC)  # the image's externals still resolve against link= only


@pytest.mark.parametrize('damage', ['truncate', 'flip', 'empty'])
def test_corrupt_image_is_recompiled(tmp_path, monkeypatch, damage):
    """A cached image that is truncated, has a flipped byte or is empty is deleted and
    recompiled (a checksum covers it): a flipped byte used to load as wrong code."""
    import glob
    from cimo import driver
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(driver, '_memory_cache', {})
    src = 'long f(long x) { return x * 7 + 3; }\nlong g(long *p) { return p[1]; }'
    cimo.compile(src)
    img, = glob.glob(str(tmp_path / '*.img'))
    with open(img, 'rb') as f:
        b = f.read()
    b = {'truncate': b[:len(b) // 2], 'flip': b[:200] + bytes([b[200] ^ 0xff]) + b[201:],
         'empty': b''}[damage]
    with open(img, 'wb') as f:
        f.write(b)
    monkeypatch.setattr(driver, '_memory_cache', {})
    assert cimo.compile(src).f(5) == 38
    with open(img, 'rb') as f:
        assert f.read() != b


def test_image_cache_fallback(tmp_path, monkeypatch):
    """A module whose image cannot be relocated is cached as assembly."""
    from cimo import driver, loader
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(driver, '_memory_cache', {})
    monkeypatch.setattr(loader.Image, 'snapshot', lambda self: None)
    src = 'long f(long x) { return x * 5; }'
    assert cimo.compile(src).f(2) == 10
    assert sorted(os.path.splitext(n)[1] for n in os.listdir(str(tmp_path))) == ['.s', '.sig']
    monkeypatch.setattr(driver, '_memory_cache', {})
    monkeypatch.setattr(driver, '_compile', lambda *a: 1 / 0)
    assert cimo.compile(src).f(3) == 15


def test_headers_are_provided():
    rt = linker.runtime()
    # every exported C library function of the headers is provided
    names = set(s for names in rt.members.values() for s in names)
    pre = ''
    for hdr in ('string.h', 'stdlib.h', 'ctype.h', 'stdio.h', 'math.h'):
        with open(os.path.join(CIMO_PKG, 'include', hdr)) as f:
            text = f.read()
        for fn in re.findall(r'^[a-z_][\w \*]*?\b(\w+)\(', text, re.M):
            if fn.startswith('__cimo'):
                continue
            assert pre + fn in names, (hdr, fn)


def test_bytes_arguments_stay_alive():
    """bytes and other read-only buffers passed to pointer parameters: valid, NUL-terminated
    memory for the whole call (it used to be a freed temporary)."""
    import gc
    m = cimo.compile('#include <string.h>\nlong f(const char *s) { return strlen(s); }\n'
                     'long g(const char *s, const char *t) { return strcmp(s, t) == 0; }')
    for _ in range(200):
        gc.collect()
        assert m.f(b'abcd') == 4
        assert m.f(memoryview(b'hello')) == 5
        assert m.g(b'x' * 100, memoryview(b'x' * 100)) == 1
    assert m.f(b'') == 0


FMA_SRC = 'double f(double a, double b, double c) { return a * b + c; }\n' \
          '#include <math.h>\ndouble g(double a, double b, double c) { return fma(a, b, c); }\n'
FUSED, SPLIT = 5.551115123125783e-17, 0.0  # 0.1 * 10 - 1, rounded once or twice


def _contracted():
    """What compile(fma=True) gives for a*b + c: fused, except on x86-64, whose baseline has no
    fused multiply-add, so cimo never contracts there (fma() itself is still exact)."""
    from cimo import driver
    return SPLIT if driver.TARGET.startswith('amd64') else FUSED


def test_fma_per_compile():
    """compile(fma=) decides for that call alone: in either order, with the cache on (each
    gets its own entry), without touching the process default. fma() itself stays fused."""
    from cimo import driver
    default = driver.FMA
    src = FMA_SRC + '/* %s */\n' % os.urandom(8).hex()
    for fma, want in ((False, SPLIT), (True, _contracted()), (False, SPLIT), (True, _contracted())):
        m = cimo.compile(src, fma=fma)
        assert m.f(0.1, 10.0, -1.0) == want, fma
        assert m.g(0.1, 10.0, -1.0) == FUSED
        assert driver.FMA == default


def test_fma_none_follows_the_process_default(monkeypatch):
    from cimo import driver
    for flag, want in ((False, SPLIT), (True, _contracted())):
        monkeypatch.setattr(driver, 'FMA', flag)
        assert cimo.compile(FMA_SRC, cache=False).f(0.1, 10.0, -1.0) == want
    monkeypatch.setattr(driver, 'FMA', False)
    assert cimo.compile(FMA_SRC, cache=False, fma=True).f(0.1, 10.0, -1.0) == _contracted()


def test_fma_on_load(tmp_path):
    p = tmp_path / 'k.c'
    p.write_text(FMA_SRC)
    assert cimo.load(p, fma=False).f(0.1, 10.0, -1.0) == SPLIT
    assert cimo.load(p, fma=True).f(0.1, 10.0, -1.0) == _contracted()


def test_compiling_in_threads():
    """Compilations from several threads: each gets its own program and its own fma= (big
    enough, and with switches frequent enough, that unlocked compilations overlap)."""
    from concurrent.futures import ThreadPoolExecutor
    helpers = '\n'.join('int h%d(int *p, int n) { int s = 0; for (int i = 0; i < n; i++) '
                        '{ s += p[i] * %d; if (s > 9) s -= 3; } return s; }' % (j, j + 1)
                        for j in range(40))

    def one(k):
        fma = bool(k % 2)
        src = FMA_SRC + helpers + '\nint id(void) { return %d; }\n/* %s */\n' % (k, os.urandom(4).hex())
        m = cimo.compile(src, cache=False, fma=fma)
        return m.id() == k and m.f(0.1, 10.0, -1.0) == (_contracted() if fma else SPLIT)
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        with ThreadPoolExecutor(4) as ex:
            assert all(ex.map(one, range(16)))
    finally:
        sys.setswitchinterval(old)


def test_unsigned_results_are_not_negative():
    m = cimo.compile('''#include <stdint.h>
    #include <stddef.h>
    unsigned a(void) { return 0xffffffffu; }
    unsigned long b(void) { return 0xffffffffffffffffUL; }
    unsigned long long c(void) { return 0x8000000000000000ULL; }
    uint32_t d(uint32_t x) { return x; }
    uint64_t e(uint64_t x) { return x * 3; }
    size_t g(void) { return (size_t)-2; }
    unsigned char h(void) { return 255; }
    unsigned short i(void) { return 65535; }
    int j(void) { return -1; }
    long k(void) { return -1; }
    signed char l(void) { return -1; }
    short n(void) { return -1; }
    _Bool o(void) { return 1; }
    unsigned q(void) { return 5; }''', cache=False)
    assert m.a() == 2 ** 32 - 1 and m.b() == 2 ** 64 - 1 and m.c() == 2 ** 63
    assert m.d(4000000000) == 4000000000 and m.e(2 ** 63) == 2 ** 63
    assert m.g() == 2 ** 64 - 2 and m.h() == 255 and m.i() == 65535 and m.q() == 5
    assert m.j() == -1 and m.k() == -1 and m.l() == -1 and m.n() == -1 and m.o() == 1


def test_unsigned_results_from_the_cache():
    src = 'unsigned long z(void) { return 0xffffffffffffffffUL; } /* %s */ static int s%s;' % (
        (os.urandom(4).hex(),) * 2)
    assert cimo.compile(src).z() == 2 ** 64 - 1
    assert cimo.compile(src).z() == 2 ** 64 - 1  # the same entry, from memory or disk
