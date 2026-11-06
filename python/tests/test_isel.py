"""arm64 selection beyond QBE's (ext.isel): which instructions come out, and that the results stay
right where each could go wrong (register pressure, spills, widths, signs)."""

import array
import platform
import ctypes
import re

import pytest

import cimo
from cimo import driver

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                                reason='arm64 assembly')


SYM = '_' if driver.TARGET == 'arm64_apple' else ''  # the target's symbol prefix


def body(src, name='f'):
    asm = driver._build(src)[0]
    m = re.search(r'^%s:\n(.*?)^/\* end function' % (SYM + name), asm, re.S | re.M)
    if m is None:
        m = re.search(r'^%s:\n(.*?)^\.type' % (SYM + name), asm, re.S | re.M)
    assert m, name
    return m.group(1)


@pytest.mark.parametrize('src,want', [
    ('long f(long a) { return a << 3; }', r'lsl\s+x0, x0, #3'),
    ('long f(long a) { return a >> 9; }', r'asr\s+x0, x0, #9'),
    ('unsigned f(unsigned a) { return a >> 9; }', r'lsr\s+w0, w0, #9'),
    ('long f(long *p, long i) { return p[i]; }', r'ldr\s+x0, \[x0, x1, lsl #3\]'),
    ('int f(int *p, int i) { return p[i]; }', r'ldr\s+w0, \[x0, w1, sxtw #2\]'),
    ('int f(int *p, unsigned i) { return p[i]; }', r'ldr\s+w0, \[x0, w1, uxtw #2\]'),
    ('int f(unsigned char *p, long i) { return p[i]; }', r'ldrb\s+w0, \[x0, x1\]'),
    ('void f(long *p, long i, long v) { p[i] = v; }', r'str\s+x2, \[x0, x1, lsl #3\]'),
    ('void f(short *p, long i, short v) { p[i] = v; }', r'strh\s+w2, \[x0, x1, lsl #1\]'),
    ('double f(double *p, long i) { return p[i]; }', r'ldr\s+d0, \[x0, x1, lsl #3\]'),
    ('float f(float *p, int i) { return p[i]; }', r'ldr\s+s0, \[x0, w1, sxtw #2\]'),
    ('long f(long a, long b, long c) { return a + b * c; }', r'madd\s'),
    ('long f(long a, long b, long c) { return a - b * c; }', r'msub\s'),
    ('int f(int a, int b, int c) { return b * c + a; }', r'madd\s+w'),
    ('long f(long a, long b) { if (a == 0) return b; return b + 7; }', r'cbn?z\s+x0'),
    ('int f(int a, int b) { if (a != 0) return b; return b + 7; }', r'cbn?z\s+w\d'),
])
def test_selected(src, want):
    b = body(src)
    assert re.search(want, b), b


def test_a_shared_address_or_product_stays_usable():
    src = '''long f(long *p, long i) { long *q = p + i; long a = *q; g = q; return a; }'''
    m = cimo.compile('long *g;\nlong h(void) { return *g; }\n' + src, cache=False)
    x = array.array('q', [5, 6])
    assert m.f(x, 1) == 6 and m.h() == 6
    src = 'long r; long f(long a, long b, long c) { long m = b * c; r = m; return a + m; }'
    assert 'madd' not in body(src)
    assert cimo.compile(src, cache=False).f(1, 2, 3) == 7


def test_leaf_has_no_frame_but_others_do():
    assert 'x29' not in body('long f(long a) { return a + 1; }')
    assert 'x29' in body('long g(long); long f(long a) { return g(a) + 1; } long g(long a) { return a > 5 ? a : f(a + 1); }')
    assert 'x29' in body('long f(long a) { long b[4]; long *volatile p = b; p[1] = a; return p[1]; }')
    assert 'x29' in body('long f(long a, long b, long c, long d, long e, long g, long h, long i, long j) { return j; }')


def test_a_leaf_using_many_registers():
    """x30 is allocatable: a frameless function must not lose its return address, and a leaf that
    uses callee-saved registers must restore them."""
    n = 40
    names = ['v%d' % i for i in range(n)]
    decl = ' '.join('long %s = p[%d] * (a + %d);' % (v, i, i) for i, v in enumerate(names))
    use = ' + '.join('%s * %s' % (names[i], names[n - 1 - i]) for i in range(n))
    m = cimo.compile('long f(long *p, long a) { %s return %s; }\n'
                     'long g(long *p, long a) { long t = a * 3; long u = f(p, a); return t + u; }' % (decl, use),
                     cache=False)
    p = array.array('q', range(1, n + 1))
    vals = [p[i] * (2 + i) for i in range(n)]
    want = sum(vals[i] * vals[n - 1 - i] for i in range(n))
    assert m.f(p, 2) == want and m.g(p, 2) == want + 6


def test_register_offsets_under_pressure():
    """Spilled bases and indices in register-offset loads and stores, of each width and sign."""
    n = 26
    idx = ' '.join('long i%d = (a + %d) & 3;' % (k, k) for k in range(n))
    reads = ' + '.join('(long)q[i%d] + (long)h[i%d] + (long)c[i%d] + (long)u[i%d]' % (k, k, k, k)
                       for k in range(n))
    writes = ' '.join('o[i%d] = i%d;' % (k, k) for k in range(n))
    src = ('long f(long *q, short *h, signed char *c, unsigned char *u, long *o, long a) '
           '{ %s %s return %s; }' % (idx, writes, reads))
    q = array.array('q', [-1, 2, -3, 4])
    h = array.array('h', [-5, 6, -7, 8])
    c = array.array('b', [-9, 10, -11, 12])
    u = array.array('B', [200, 201, 202, 203])
    o = array.array('q', [0] * 4)
    a = 5
    want = 0
    for k in range(n):
        i = (a + k) & 3
        want += q[i] + h[i] + c[i] + u[i]
    assert cimo.compile(src, cache=False).f(q, h, c, u, o, a) == want
    assert list(o) == [0, 1, 2, 3]


@pytest.mark.parametrize('ty,code,vals', [
    ('int', 'i', [0, 1, 2, -1]),
    ('unsigned', 'I', [0, 1, 2, 3]),
    ('long', 'q', [0, 1, 2, 3]),
])
def test_index_types(ty, code, vals):
    m = cimo.compile('long f(long *p, %s i) { return p[i]; }\nvoid g(int *p, %s i, int v) { p[i] = v; }'
                     % (ty, ty), cache=False)
    x = array.array('q', [10, 20, 30, 40])
    w = array.array('i', [0, 0, 0, 0])
    base = ctypes_address(x)
    for i in vals:
        if i >= 0:
            assert m.f(x, i) == x[i]
            m.g(w, i, i + 7)
            assert w[i] == i + 7
    if ty == 'int':
        assert m.f(base + 8, -1) == 10  # sign extended, not zero extended


def ctypes_address(a):
    return a.buffer_info()[0]


def test_unsigned_index_wraps_in_32_bits():
    m = cimo.compile('long f(long *p, unsigned i) { return p[i + 1u]; }', cache=False)
    x = array.array('q', [11, 22])
    assert m.f(x, 0xffffffff) == 11


def test_shift_amounts_keep_the_width():
    m = cimo.compile('''int f(int a) { return a << 31; }
        int g(int a) { return a >> 31; }
        unsigned h(unsigned a) { return a >> 31; }
        long i(long a) { return a << 63; }
        long j(long a) { return a >> 63; }''', cache=False)
    assert m.f(1) == -2 ** 31 and m.g(-1) == -1 and m.h(0x80000000) == 1
    assert m.i(1) == -2 ** 63 and m.j(-5) == -1


def test_madd_wraps_in_its_width():
    m = cimo.compile('''int f(int a, int b, int c) { return (int)((unsigned)a + (unsigned)b * (unsigned)c); }
        long g(long a, int b, int c) { return a + (long)b * c; }''', cache=False)
    assert m.f(1, 0x10000, 0x10000) == 1 and m.g(1, 0x10000, 0x10000) == 2 ** 32 + 1


def test_cbz_with_every_loop_shape():
    m = cimo.compile('''long f(long *p, long n) { long s = 0, i = n;
        while (i != 0) { i--; if (p[i] == 0) continue; s += p[i]; }
        return s; }
        int g(unsigned n) { int c = 0; do { c++; n >>= 1; } while (n != 0); return c; }
        int h(long x) { return (int)x == 0; }
        int k(long x) { if ((int)x) return 1; return 2; }''', cache=False)
    assert m.f(array.array('q', [1, 0, 3]), 3) == 4 and m.g(5) == 3
    assert m.h(1 << 32) == 1 and m.k(1 << 32) == 2  # a w test sees only the low half


@pytest.mark.parametrize('src', [
    'long f(long a, long b) { return a > b ? a : b; }',
    'long f(long a, long b, long c) { return a < b ? c : a; }',
    'int f(int a) { return a < 0 ? -a : a; }',
    'double f(double a, double b, long c) { return c > 0 ? a : b; }',
    'double f(double a, double b) { return a < b ? a : b; }',
    'long f(double a, long b, long c) { return a < 0.5 ? b : c; }',
])
def test_selects_have_no_branch(src):
    b = body(src)
    assert re.search(r'\bf?cs(el|inc|inv|neg)\b', b), b
    assert not re.search(r'\bb(eq|ne|lt|le|gt|ge|cc|cs|hi|ls|mi|pl)\b|\bcbn?z\b', b), b


def test_selects_compute_the_same():
    nan = float('nan')
    m = cimo.compile('''long a(long x, long y) { return x > y ? x : y; }
        unsigned long b(unsigned long x, unsigned long y) { return x > y ? x : y; }
        double c(double x, double y) { return x < y ? x : y; }
        double d(double x, double y) { return x >= y ? x : y; }
        double e(double x, double y) { return x != y ? x : y; }
        long g(double x, long p, long q) { return x == x ? p : q; }
        float h(float x, float y, int k) { return k ? x : y; }
        int i(int c, int x) { return c ? x + 1 : x - 1; }
        int j(long c, int x) { if (c) x = x + 5; return x; }
        int k(long x) { int r = 0, s = 0; if (x > 3) { r = 1; s = 2; } return r * 10 + s; }''', cache=False)
    assert m.a(1, 2) == 2 and m.a(-1, -2) == -1
    assert m.b(1, 2 ** 64 - 1) == 2 ** 64 - 1
    assert m.c(1.0, 2.0) == 1.0 and m.c(nan, 2.0) == 2.0 and m.c(1.0, nan) != m.c(1.0, nan)
    assert m.d(nan, 2.0) == 2.0 and m.d(3.0, 2.0) == 3.0
    assert m.e(nan, 2.0) != m.e(nan, 2.0) and m.e(2.0, 2.0) == 2.0
    assert m.g(nan, 1, 2) == 2 and m.g(0.0, 1, 2) == 1
    assert m.h(1.5, 2.5, 1) == 1.5 and m.h(1.5, 2.5, 0) == 2.5
    assert m.i(1, 5) == 6 and m.i(0, 5) == 4 and m.i(-1, 5) == 6
    assert m.j(1 << 40, 1) == 6 and m.j(0, 1) == 1  # the whole long is tested
    assert m.k(9) == 12 and m.k(1) == 0


def test_a_select_does_not_run_a_division_for_nothing():
    b = body('long f(long a, long b, long c) { return c ? a / b : a; }')
    assert re.search(r'\bcbn?z\b', b), b


def test_a_loop_invariant_condition_keeps_its_branch():
    """It predicts perfectly (LLVM's EarlyIfConverter::shouldConvertIf)."""
    src = '''long f(long *p, long n, long k) { long s = 0;
        for (long i = 0; i < n; i++) { long v = p[i]; if (k > 0) v = v + 1; s += v * v; } return s; }'''
    m = cimo.compile(src, cache=False)
    x = array.array('q', [1, 2, 3])
    assert m.f(x, 3, 1) == 4 + 9 + 16 and m.f(x, 3, 0) == 1 + 4 + 9


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17])
def test_selects_in_unrolled_loops(n):
    """The loop passes copy a select around induction values they compute on demand."""
    m = cimo.compile('''long f(int *x, int n) { int mi = -1, max = -100, cnt = 0;
        for (int i = 0; i < n; i++) { if (x[i] > max) { max = x[i]; mi = i; cnt = i * 2 + 1; } }
        return (long)mi * 1000000 + (long)max * 1000 + cnt; }''', cache=False)
    vals = [(k * 37) % 23 - 7 for k in range(n)]
    x = array.array('i', vals or [0])
    mi, mx, cnt = -1, -100, 0
    for i, v in enumerate(vals):
        if v > mx:
            mx, mi, cnt = v, i, i * 2 + 1
    assert m.f(x, n) == mi * 1000000 + mx * 1000 + cnt


@pytest.mark.parametrize('src,want', [
    ('long f(long a, long b) { return a + (b << 4); }', r'add\s+x\d+, x\d+, x\d+, lsl #4'),
    ('long f(long a, long b) { return (b << 4) + a; }', r'add\s+x\d+, x\d+, x\d+, lsl #4'),
    ('long f(long a, long b) { return a - (b >> 9); }', r'sub\s+x\d+, x\d+, x\d+, asr #9'),
    ('unsigned f(unsigned a, unsigned b) { return a + (b >> 5); }', r'add\s+w\d+, w\d+, w\d+, lsr #5'),
    ('long f(long a, int b) { return a + b; }', r'add\s+x\d+, x\d+, w\d+, sxtw'),
    ('long f(long a, unsigned b) { return a + b; }', r'add\s+x\d+, x\d+, w\d+, uxtw'),
    ('long f(long a, int b) { return a + ((long)b << 3); }', r'add\s+x\d+, x\d+, w\d+, sxtw #3'),
    ('long f(long a, long b) { return a & (b << 2); }', r'and\s+x\d+, x\d+, x\d+, lsl #2'),
    ('long f(long a, long b) { return a | (b >> 2); }', r'orr\s+x\d+, x\d+, x\d+, asr #2'),
    ('long f(long a, long b) { return a ^ (b * 8); }', r'eor\s+x\d+, x\d+, x\d+, lsl #3'),
    ('long f(long a, long b) { return a & ~b; }', r'bic\s'),
    ('long f(long a, long b) { return a | ~b; }', r'orn\s'),
    ('long f(long a, long b) { return a ^ ~b; }', r'eon\s'),
    ('long f(long a) { return ~a; }', r'orn\s+x\d+, xzr, x\d+'),
    ('int f(int a) { return ~a; }', r'orn\s+w\d+, wzr, w\d+'),
    ('long f(long a) { return (a >> 8) & 0xff; }', r'ubfx\s+x\d+, x\d+, #8, #8'),
    ('unsigned f(unsigned a) { return (a >> 3) & 31; }', r'ubfx\s+w\d+, w\d+, #3, #5'),
    ('long f(long a) { return (a << 20) >> 40; }', r'sbfx\s+x\d+, x\d+, #20, #24'),
    ('unsigned long f(unsigned long a) { return (a << 20) >> 40; }', r'ubfx\s+x\d+, x\d+, #20, #24'),
    ('unsigned long f(unsigned long a) { return (a << 13) | (a >> 51); }', r'ror\s+x\d+, x\d+, #51'),
    ('unsigned f(unsigned a) { return (a >> 7) | (a << 25); }', r'ror\s+w\d+, w\d+, #7'),
    ('long f(long a, long b) { return -(a * b); }', r'msub\s+x\d+, x\d+, x\d+, xzr'),
    ('double f(double a, double b) { return -(a * b); }', r'fnmul\s+d'),
    ('float f(float a, float b) { return -(a * b); }', r'fnmul\s+s'),
    ('void f(long *p) { *p = 0; }', r'str\s+xzr'),
    ('void f(int *p, long i) { p[i] = 0; }', r'str\s+wzr, \[x\d+, x\d+, lsl #2\]'),
    ('void f(char *p) { *p = 0; }', r'strb\s+wzr'),
    ('void f(double *p) { *p = 0.0; }', r'str\s+xzr'),
    ('double f(double a) { return a * 3.0; }', r'fmov\s+d\d+, #3\.0'),
    ('float f(float a) { return a * 0.5f; }', r'fmov\s+s\d+, #0\.5'),
    ('double f(double a) { return a + 0.0; }', r'fmov\s+d\d+, xzr'),
])
def test_selected_batch2(src, want):
    b = body(src)
    assert re.search(want, b), b


def test_negative_zero_is_not_stored_as_zero():
    m = cimo.compile('void f(double *p) { *p = -0.0; }\nvoid g(float *p) { *p = -0.0f; }', cache=False)
    d = array.array('d', [1.0])
    s = array.array('f', [1.0])
    m.f(d)
    m.g(s)
    import math
    assert math.copysign(1, d[0]) == -1 and math.copysign(1, s[0]) == -1 and d[0] == 0 and s[0] == 0


FOLDS = '''
long a1(long a, long b) { return a + (b << 4); }
long a2(long a, long b) { return a - (b >> 3); }
unsigned long a3(unsigned long a, unsigned long b) { return a - (b >> 3); }
int a4(int a, int b) { return a + (b << 28); }
int a5(int a, int b) { return a - (b >> 2); }
unsigned a6(unsigned a, unsigned b) { return a + (b >> 2); }
long a7(long a, int b) { return a + b; }
long a8(long a, unsigned b) { return a + b; }
long a9(long a, int b) { return a - ((long)b << 3); }
long a10(long a, unsigned b) { return a + ((long)b << 2); }
long b1(long a, long b) { return a & ~b; }
long b2(long a, long b) { return a | ~b; }
long b3(long a, long b) { return a ^ ~b; }
int b4(int a, int b) { return a & ~b; }
int b5(int a) { return ~a; }
long b6(long a) { return ~a; }
long c1(long a) { return (a >> 8) & 0xff; }
long c2(long a) { return (a >> 56) & 0xff; }
unsigned c3(unsigned a) { return (a >> 3) & 31; }
int c4(int a) { return (a >> 28) & 15; }
long c5(long a) { return (a << 20) >> 40; }
unsigned long c6(unsigned long a) { return (a << 20) >> 40; }
int c7(int a) { return (a << 24) >> 24; }
unsigned c8(unsigned a) { return (a << 24) >> 24; }
int c9(int a) { return (a << 4) >> 28; }
unsigned long d1(unsigned long a) { return (a << 13) | (a >> 51); }
unsigned long d2(unsigned long a) { return (a >> 13) | (a << 51); }
unsigned d3(unsigned a) { return (a << 7) | (a >> 25); }
unsigned long d4(unsigned long a) { return (a << 13) | (a >> 50); }
long e1(long a, long b) { return -(a * b); }
int e2(int a, int b) { return -(a * b); }
double e3(double a, double b) { return -(a * b); }
long f1(long a, long b) { long s = b << 4; return (a + s) * s; }
long f2(long a, long b) { long n = ~b; return (a & n) + n; }
long f3(long a) { long s = a >> 8; return (s & 0xff) + s; }
unsigned long f4(unsigned long a) { unsigned long l = a << 13; return ((l | (a >> 51)) ^ l); }
'''
M64 = 2 ** 64 - 1


def _s(v, bits=64):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _i32(v):
    return _s(v, 32)


def _sar(v, n, bits=64):
    return _s(v, bits) >> n


def test_folds_compute_the_same():
    m = cimo.compile(FOLDS, cache=False)
    vals = [0, 1, -1, 2, 7, 255, 256, 2 ** 31 - 1, -2 ** 31, 2 ** 32 + 5, 2 ** 63 - 1, -2 ** 63,
            0x0123456789abcdef, -0x0123456789abcdef, 0x5a5a5a5a5a5a5a5a]
    w = [_i32(v) for v in vals]
    u = lambda v: v & M64
    uw = lambda v: v & 0xffffffff
    for x in vals:
        assert m.b6(x) == _s(~x)
        assert m.c1(x) == (x >> 8) & 0xff and m.c2(x) == (x >> 56) & 0xff
        assert m.c5(x) == _sar(_s(x << 20), 40)
        assert _s(m.d1(u(x))) == _s(u(x) << 13 | u(x) >> 51)
        assert _s(m.d2(u(x))) == _s(u(x) >> 13 | u(x) << 51)
        assert _s(m.d4(u(x))) == _s(u(x) << 13 | u(x) >> 50)
        assert _s(m.c6(u(x))) == _s((u(x) << 20 & M64) >> 40)
        assert m.f3(x) == ((x >> 8) & 0xff) + (x >> 8)
        assert _s(m.f4(u(x))) == _s(((u(x) << 13) | (u(x) >> 51)) ^ (u(x) << 13))
        for y in vals:
            assert m.a1(x, y) == _s(x + (y << 4)) and m.a2(x, y) == _s(x - (y >> 3))
            assert _s(m.a3(u(x), u(y))) == _s(u(x) - (u(y) >> 3))
            assert m.a7(x, _i32(y)) == _s(x + _i32(y)) and m.a8(x, uw(y)) == _s(x + uw(y))
            assert m.a9(x, _i32(y)) == _s(x - (_i32(y) << 3)) and m.a10(x, uw(y)) == _s(x + (uw(y) << 2))
            assert m.b1(x, y) == x & ~y and m.b2(x, y) == x | ~y and m.b3(x, y) == x ^ ~y
            assert m.e1(x, y) == _s(-(x * y))
            assert m.f1(x, y) == _s((x + _s(y << 4)) * _s(y << 4))
            assert m.f2(x, y) == _s((x & ~y) + ~y)
    for x in w:
        assert m.b5(x) == ~x and m.c4(x) == (x >> 28) & 15
        assert m.c7(x) == _s(x, 8) and m.c9(x) == _i32(x << 4) >> 28
        assert uw(m.c3(uw(x))) == (uw(x) >> 3) & 31 and uw(m.c8(uw(x))) == uw(x) & 0xff
        assert uw(m.d3(uw(x))) == uw(uw(x) << 7 | uw(x) >> 25)
        for y in w:
            assert m.a4(x, y) == _i32(x + (y << 28)) and m.a5(x, y) == _i32(x - (y >> 2))
            assert uw(m.a6(uw(x), uw(y))) == uw(uw(x) + (uw(y) >> 2))
            assert m.b4(x, y) == x & ~y and m.e2(x, y) == _i32(-(x * y))
    assert m.e3(2.0, 3.0) == -6.0


def test_folds_under_register_pressure():
    n = 30
    decl = ' '.join('long v%d = (a + %d) << %d;' % (i, i, i % 7 + 1) for i in range(n))
    use = ' + '.join('((b - (v%d >> 2)) ^ (v%d & ~b) ^ ((v%d >> 3) & 0xff))' % (i, n - 1 - i, i) for i in range(n))
    m = cimo.compile('long f(long a, long b) { %s return %s; }' % (decl, use), cache=False)
    a, b = 12345, 67890
    v = [(a + i) << (i % 7 + 1) for i in range(n)]
    want = sum((b - (v[i] >> 2)) ^ (v[n - 1 - i] & ~b) ^ ((v[i] >> 3) & 0xff) for i in range(n))
    assert m.f(a, b) == want


def test_no_shift_folded_on_a_loop_chain():
    """A folded shift takes 2 cycles from every input: on an accumulator, twice the time."""
    b = body('long f(int *x, long n) { long s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }')
    assert not re.search(r'add\s+(x\d+), \1, w\d+, sxtw', b), b
    b = body('long f(long *x, long n) { long s = 0; for (long i = 0; i < n; i++) s += x[i] << 3; return s; }')
    assert not re.search(r'add\s+(x\d+), \1, x\d+, lsl', b), b


@pytest.mark.parametrize('src,want', [
    ('long f(int *p) { return *p; }', r'ldrsw\s+x0, \[x0\]'),
    ('long f(int *p, long i) { return p[i]; }', r'ldrsw\s+x0, \[x0, x1, lsl #2\]'),
    ('long f(signed char *p) { return *p; }', r'ldrsb\s+x0'),
    ('long f(short *p) { return *p; }', r'ldrsh\s+x0'),
    ('unsigned long f(unsigned *p) { return *p; }', r'ldr\s+w0, \[x0\]'),
])
def test_extension_in_the_load(src, want):
    b = body(src)
    assert re.search(want, b) and not re.search(r'\b(sxt|uxt)', b), b


def test_extending_loads_compute_the_same():
    m = cimo.compile('''long a(int *p) { return *p; }
        unsigned long b(unsigned *p) { return *p; }
        long c(signed char *p) { return *p; }
        long d(short *p) { return *p; }
        long e(unsigned char *p) { return *p; }
        long f(unsigned short *p) { return *p; }
        unsigned long g(int *p) { return (unsigned)*p; }
        long h(unsigned *p) { return (int)*p; }
        long i(int *p, int *q) { int v = *p; *q = 5; return v; }
        long j(int *p) { int v = *p; return (long)v + (v << 1); }
        ''', cache=False)
    w = array.array('i', [-5])
    u = array.array('I', [0xfffffffb])
    assert m.a(w) == -5 and _s(m.b(u)) == 0xfffffffb and m.g(w) == 0xfffffffb and m.h(u) == -5
    assert m.c(array.array('b', [-3])) == -3 and m.d(array.array('h', [-300])) == -300
    assert m.e(array.array('B', [253])) == 253 and m.f(array.array('H', [65000])) == 65000
    x = array.array('i', [7])
    assert m.i(x, x) == 7 and x[0] == 5  # the load stays before the store through the same memory
    assert m.j(array.array('i', [-4])) == -12


def test_no_extension_after_a_narrow_load():
    assert not re.search(r'\b(uxt|sxt)', body('int f(unsigned char *p, long i) { return p[i]; }'))
    assert not re.search(r'\b(uxt|sxt)', body('int f(short *p) { return *p; }'))
    m = cimo.compile('''int f(signed char *p) { return *p; }
        unsigned g(unsigned char *p) { return *p; }
        int h(short *p) { return *p + 1; }
        long i(unsigned short *p) { return *p; }''', cache=False)
    assert m.f(array.array('b', [-1])) == -1 and m.g(array.array('B', [255])) == 255
    assert m.h(array.array('h', [-2])) == -1 and m.i(array.array('H', [65535])) == 65535


CONDFORMS = '''
long a1(long a, long b, long c) { return a < b ? c + 1 : c; }
long a2(long a, long b, long c) { return a < b ? c : c + 1; }
long a3(long a, long b, long c) { return a < b ? -c : c; }
long a4(long a, long b, long c) { return a < b ? c : -c; }
long a5(long a, long b, long c) { return a < b ? ~c : c; }
long a6(long a, long b, long c) { return a < b ? c : ~c; }
long a7(long a, long b, long c) { return a < b ? c : 0; }
long a8(long a, long b, long c) { return a < b ? 0 : c; }
int a9(long a, long b) { return a < b ? -1 : 0; }
long a10(long a) { return a < 0 ? -a : a; }
int a11(int a, int b, int c) { return a < b ? c + 1 : c; }
int a12(int a, int b, int c) { return a == b ? ~c : c; }
long b1(long a, long b, long c) { return c + (a < b); }
long b2(long a, long b, long c) { return (a == b) + c; }
int b3(int a, int b, int c) { return c + (a > b); }
long b4(unsigned long a, unsigned long b, long c) { return c + (a < b); }
long c1(long a, long b) { if (a & 8) return b; return b + 7; }
long c2(long a, long b) { if (a & (1L << 63)) return b; return b + 7; }
long c3(long a, long b) { if ((a & 0xff0) == 0) return b; return b + 7; }
long c4(long a, long b, long c) { if (a & c) return b; return b + 7; }
int c5(int a, int b) { if (a & (1 << 31)) return b; return b + 7; }
long c6(long a) { long m = a & 12; if (m) return m; return 5; }
long d1(long a) { return a * 3; }
long d2(long a) { return a * 7; }
long d3(long a) { return a * 9; }
long d4(long a) { return a * 10; }
long d5(long a) { return a * 14; }
long d6(long a) { return a * -8; }
long d7(long a) { return a * 11; }
int d8(int a) { return a * 15; }
int d9(int a) { return a * -2; }
long d10(long a) { return a * 1025; }
long e1(long a) { return (a & 0xff) << 4; }
int e2(int a) { return (a & 0xf) << 28; }
long e3(long a) { long m = a & 0xff; return (m << 4) + m; }
'''


@pytest.mark.parametrize('name,want', [
    ('a1', r'csinc'), ('a2', r'csinc'), ('a3', r'csneg'), ('a4', r'csneg'), ('a5', r'csinv'), ('a6', r'csinv'),
    ('a7', r'csel\s+x\d+, xzr'), ('a8', r'csel\s+x\d+, xzr'), ('a10', r'csneg'), ('a11', r'csinc\s+w'),
    ('b1', r'csinc'), ('b2', r'csinc'), ('b3', r'csinc\s+w'), ('b4', r'csinc'),
    ('c1', r'tbn?z\s+x\d+, #3'), ('c2', r'tbn?z\s+x\d+, #63'), ('c3', r'tst\s'), ('c4', r'tst\s'), ('c5', r'tbn?z\s+w\d+, #31'),
    ('d1', r'add\s+x0, x0, x0, lsl #1'), ('d3', r'add\s+x0, x0, x0, lsl #3'), ('d6', r'sub\s+x0, xzr, x0, lsl #3'),
    ('e1', r'ubfiz\s+x0, x0, #4, #8'), ('e2', r'ubfiz\s+w0, w0, #28, #4'),
])
def test_selected_batch3(name, want):
    asm = driver._build(CONDFORMS)[0]
    m = re.search(r'^_?%s:\n(.*?)^\.?(?:type|/\*)' % name, asm, re.S | re.M)
    assert re.search(want, m.group(1)), m.group(1)


def test_batch3_computes_the_same():
    m = cimo.compile(CONDFORMS, cache=False)
    vals = [0, 1, -1, 2, 3, 7, 8, 9, 12, 255, 256, 0xff0, 2 ** 31 - 1, -2 ** 31, 2 ** 32, 2 ** 62, 2 ** 63 - 1, -2 ** 63]
    ws = [v for v in vals if -2 ** 31 <= v < 2 ** 31]
    for a in vals:
        for mul, f in ((3, m.d1), (7, m.d2), (9, m.d3), (10, m.d4), (14, m.d5), (-8, m.d6), (11, m.d7), (1025, m.d10)):
            assert f(a) == _s(a * mul), (a, mul)
        assert m.a10(a) == _s(-a if a < 0 else a)
        assert m.e1(a) == (a & 0xff) << 4 and m.e3(a) == ((a & 0xff) << 4) + (a & 0xff)
        assert m.c6(a) == (a & 12 or 5)
        for b in vals:
            assert m.c1(a, b) == (b if a & 8 else _s(b + 7)) and m.c2(a, b) == (b if a < 0 else _s(b + 7))
            assert m.c3(a, b) == (b if a & 0xff0 == 0 else _s(b + 7))
            assert m.a9(a, b) == (-1 if a < b else 0)
            for c in (0, 1, -1, 5, 2 ** 63 - 1, -2 ** 63):
                t = a < b
                assert m.a1(a, b, c) == _s(c + 1 if t else c) and m.a2(a, b, c) == _s(c if t else c + 1)
                assert m.a3(a, b, c) == _s(-c if t else c) and m.a4(a, b, c) == _s(c if t else -c)
                assert m.a5(a, b, c) == (~c if t else c) and m.a6(a, b, c) == (c if t else ~c)
                assert m.a7(a, b, c) == (c if t else 0) and m.a8(a, b, c) == (0 if t else c)
                assert m.b1(a, b, c) == _s(c + t) and m.b2(a, b, c) == _s(c + (a == b))
                assert m.b4(a & M64, b & M64, c) == _s(c + ((a & M64) < (b & M64)))
                assert m.c4(a, b, c) == (b if a & c else _s(b + 7))
    for a in ws:
        for mul, f in ((15, m.d8), (-2, m.d9)):
            assert f(a) == _i32(a * mul)
        assert m.e2(a) == _i32((a & 0xf) << 28) and m.c5(a, 3) == (3 if a < 0 else 10)
        for b in ws:
            for c in (0, 1, -1, 2 ** 31 - 1, -2 ** 31):
                assert m.a11(a, b, c) == _i32(c + 1 if a < b else c)
                assert m.a12(a, b, c) == (~c if a == b else c)
                assert m.b3(a, b, c) == _i32(c + (a > b))


def test_identical_compares_in_a_switch_go():
    src = 'int f(int k) { switch (k) { case 0: return 11; case 1: return 22; case 3: return 41; case 5: return 61; case 6: return 71; default: return 0; } }'
    m = cimo.compile(src, cache=False)
    assert [m.f(i) for i in (-1, 0, 1, 2, 3, 4, 5, 6, 7)] == [0, 11, 22, 0, 41, 0, 61, 71, 0]
    b = body(src)
    cmps = re.findall(r'cmp\s+(w\d+, #\d+)', b)
    assert len(cmps) == len(set(cmps)), b


def test_sibling_calls():
    src = '''long g(long a);
    long f(long a) { return g(a + 1); }
    void h(long a);
    void i(long a) { h(a); }
    long j(long a) { return g(a) + 1; }
    long k(long a) { long b[2]; long *volatile p = b; p[0] = a; return g(p[0]); }
    long g(long a) { static volatile long n; BIG return n; }
    void h(long a) { g(a); BIG2 }
    int callsmall(long a); long l(long a) { return callsmall(a); }
    int callsmall(long a) { static volatile int n; BIG ++n; return n; }'''.replace('BIG2', 'static volatile long q; ' + 'q += a; ' * 45).replace('BIG', 'n += 0; ' * 45)
    asm = driver._build(src)[0]
    assert re.search(r'\bb\s+_?g\b', body(src, 'f')), body(src, 'f')
    assert re.search(r'\bb\s+_?h\b', body(src, 'i')), body(src, 'i')
    assert re.search(r'\bbl\s+_?g\b', body(src, 'j')), body(src, 'j')  # work after it
    assert re.search(r'\bbl\s+_?g\b', body(src, 'k')), body(src, 'k')  # a stack slot
    assert re.search(r'\bbl\s+_?callsmall\b', body(src, 'l')), body(src, 'l')  # extends the result
    src = src.replace('n += 0;', 'n += a;', 45)  # g adds a 45 times
    m = cimo.compile(src, cache=False)
    first = m.f(1)
    assert m.f(1) == first + 2 * 45 and m.j(0) == first + 2 * 45 + 1


def test_sibling_call_with_saved_registers():
    src = '''long g(long a, long b) { return a + b * 3; }
    long f(long a, long b, long c) {
        long t = g(a, b); long u = g(t, c); long v = g(u, a);
        return g(t + u + v, c);
    }
    long h(long a, long b, long c, long d, long e, long i, long j, long k, long l, long m) { return a + b + c + d + e + i + j + k + l * m; }
    long n(long a) { return h(a, 1, 2, 3, 4, 5, 6, 7, 8, 9); }
    '''
    m = cimo.compile(src, cache=False)
    g = lambda a, b: a + b * 3
    t = g(5, 6); u = g(t, 7); v = g(u, 5)
    assert m.f(5, 6, 7) == g(t + u + v, 7)
    assert m.n(10) == 10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 * 9


def test_self_tail_recursion_is_a_loop():
    src = '''long fact(long n, long r) { if (n < 2) return r; return fact(n - 1, r * n); }
    long deep(long n, long a) { if (n == 0) return a; return deep(n - 1, a + n); }
    int sw(int a, int b, int n) { if (n <= 0) return a * 10 + b; return sw(b, a, n - 1); }
    long vs(long n, long a) { long v = a * 2; if (n == 0) return v; return vs(n - 1, v + 1); }
    void cnt(long *p, long n) { if (n == 0) return; *p += n; cnt(p, n - 1); }
    double dd(double x, int n) { if (n == 0) return x; return dd(x * 2.0, n - 1); }
    signed char sc(signed char c, int n) { if (n == 0) return c; return sc(c + 100, n - 1); }
    long notail(long n) { if (n == 0) return 0; return n + notail(n - 1); }
    long esc(long n, long *p) { long x = n; if (n == 0) return *p; return esc(n - 1, &x); }'''
    assert 'bl' not in re.findall(r'\b(bl)\s+_?fact', body(src, 'fact'))
    m = cimo.compile(src, cache=False)
    assert m.fact(10, 1) == 3628800
    assert m.deep(3000000, 0) == 3000000 * 3000001 // 2  # no stack
    assert m.sw(1, 2, 3) == 21 and m.sw(1, 2, 4) == 12  # the assignments are parallel
    v, a = 0, 3
    for n in range(4, -1, -1):
        v = a * 2
        a = v + 1
    assert m.vs(4, 3) == v
    import array
    x = array.array('q', [0])
    m.cnt(x, 4)
    assert x[0] == 10
    assert m.dd(1.5, 3) == 12.0 and m.sc(100, 1) == -56 and m.notail(4) == 10
    y = array.array('q', [7])
    assert m.esc(3, y) == 1  # the pointer to a local keeps its own frame: stays a call
    assert re.search(r'\bbl\s+_?esc', body(src, 'esc')), body(src, 'esc')


CCMP = """
long a1(long a, long b, long c, long d, long e) { if (a < b && c < d) return e; return e + 7; }
long a2(long a, long b, long c, long d, long e) { if (a < b || c < d) return e; return e + 7; }
long a3(long a, long c, long e) { if (a > 3 && c == 9) return e; return e + 7; }
long a4(long a, long c, long e) { if (a == 0 || c >= 5) return e; return e + 7; }
long a5(unsigned long a, unsigned long b, unsigned long c, unsigned long d) { if (a <= b && c > d) return 1; return 2; }
long a6(unsigned long a, unsigned long b, unsigned long c, unsigned long d) { if (a >= b || c < d) return 1; return 2; }
long a7(int a, int b, long c, long d) { if (a != b && c <= d) return 1; return 2; }
long a8(long a, long b, long c) { if (a < b && b < c && a != c) return 1; return 2; }
long a9(long a, long b, long c) { if (a < b || b < c || a == c) return 1; return 2; }
long a10(double a, double b, long c, long d) { if (a < b && c < d) return 1; return 2; }
long a11(long a, long b, double c, double d) { if (a < b && c < d) return 1; return 2; }
long a12(long a, long c) { if (a > 1000000 && c > 1000000) return 1; return 2; }
long a13(long *p, long a) { if (a > 0 && *p > 0) return 1; return 2; }
long a14(long a, long b, long c, long d) { if (a < b && c > d) c = 5; else c = 6; return c + d; }
int m1(long a, long b) { return a < b ? -1 : 0; }
int m2(long a, long b) { return a < b ? 0 : -1; }
long m3(long a, long b) { return a == b ? -1 : 0; }
"""


def test_ccmp_is_selected():
    for name in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'):
        m = re.search(r'^_?%s:\n(.*?)^\.?(?:type|/\*)' % name, driver._build(CCMP)[0], re.S | re.M)
        assert 'ccmp' in m.group(1), m.group(1)
    assert re.search(r'csinv\s+w0, wzr, wzr', body(CCMP, 'm1')), body(CCMP, 'm1')


def test_ccmp_computes_the_same():
    import itertools
    m = cimo.compile(CCMP, cache=False)
    vals = [-2, 0, 1, 3, 4, 5, 9, 1000001, -2 ** 63, 2 ** 63 - 1]
    nan = float('nan')
    for a, b, c, d in itertools.product(vals[:7] + vals[-2:], [0, 4, -2 ** 63], [0, 9, 5], [1, -1]):
        e = 100
        assert m.a1(a, b, c, d, e) == (e if a < b and c < d else e + 7)
        assert m.a2(a, b, c, d, e) == (e if a < b or c < d else e + 7)
        assert m.a3(a, c, e) == (e if a > 3 and c == 9 else e + 7)
        assert m.a4(a, c, e) == (e if a == 0 or c >= 5 else e + 7)
        ua, ub, uc, ud = a & M64, b & M64, c & M64, d & M64
        assert m.a5(ua, ub, uc, ud) == (1 if ua <= ub and uc > ud else 2)
        assert m.a6(ua, ub, uc, ud) == (1 if ua >= ub or uc < ud else 2)
        assert m.a8(a, b, c) == (1 if a < b and b < c and a != c else 2)
        assert m.a9(a, b, c) == (1 if a < b or b < c or a == c else 2)
        assert m.a12(a, c) == (1 if a > 1000000 and c > 1000000 else 2)
        assert m.a14(a, b, c, d) == ((5 if a < b and c > d else 6) + d)
        assert m.m1(a, b) == (-1 if a < b else 0) and m.m2(a, b) == (0 if a < b else -1)
        assert m.m3(a, b) == (-1 if a == b else 0)
        if -2 ** 31 <= a < 2 ** 31 and -2 ** 31 <= b < 2 ** 31:
            assert m.a7(a, b, c, d) == (1 if a != b and c <= d else 2)
    for x, y, c, d in itertools.product([0.5, 1.5, nan], [1.0, nan], [1, 2], [2, 1]):
        assert m.a10(x, y, c, d) == (1 if x < y and c < d else 2)
        assert m.a11(c, d, x, y) == (1 if c < d and x < y else 2)
    assert m.a13(array.array('q', [5]), 1) == 1 and m.a13(array.array('q', [-5]), 1) == 2


def test_condition_with_a_load_is_not_speculated():
    """p may be null when a <= 0: its load must stay behind the test."""
    m = cimo.compile('long f(long *p, long a) { if (a > 0 && *p > 0) return 1; return 2; }', cache=False)
    assert m.f(0, 0) == 2


def test_a_way_out_of_a_loop_keeps_its_branch():
    """Bounds checks: their failure path leaves the loop, and predicts perfectly."""
    b = body("""long f(long *p, long n, long a, long b) { long s = 0;
        for (long i = 0; i < n; i++) { long v = p[i]; if (v < 0 || v >= a) return -1; s += v; }
        return s; }""")
    assert 'ccmp' not in b, b
    b = body("""long f(long *p, long n, long a, long b) { long s = 0;
        for (long i = 0; i < n; i++) { long u = p[i]; if (u < a && u > b) s += u * 3; } return s; }""")
    assert 'ccmp' in b and 'cset' not in b, b


def test_ccmp_for_a_select():
    src = """long f(long a, long b, long c, long d, long v) { return a < b && c > d ? v : v + 1; }
        long g(long a, long b, long c, long d, long v, long w) { long x = v, y = w;
            if (a < b || c == d) { x = w; y = v; } return x * 100 + y; }"""
    assert 'ccmp' in body(src, 'f') and 'cset' not in body(src, 'f'), body(src, 'f')
    m = cimo.compile(src, cache=False)
    for a, b, c, d in ((1, 2, 4, 3), (2, 1, 4, 3), (1, 2, 3, 4), (2, 1, 3, 4), (1, 2, 3, 3), (2, 1, 3, 3)):
        assert m.f(a, b, c, d, 10) == (10 if a < b and c > d else 11)
        assert m.g(a, b, c, d, 1, 2) == (201 if a < b or c == d else 102)


SW = """int s7(int k) { switch (k) { case 0: return 11; case 1: return 22; case 2: return 37; case 3: return 41;
    case 4: return 59; case 5: return 61; case 6: return 71; default: return 0; } }
int a7(int k, int a) { switch (k) { case 0: a += 3; break; case 1: a *= 22; break; case 2: a -= 37; break;
    case 3: a ^= 41; break; case 4: a |= 59; break; case 5: a <<= 2; break; case 6: a >>= 1; break; default: a = 0; } return a; }
long l6(long k, long a) { switch (k) { case 1: return a + 1; case 2: return a * 2; case 3: return a - 3; case 5: return a ^ 5;
    case 7: return a | 7; case 8: return a & 8; } return -1; }
int fall(int k) { int s = 0; switch (k) { case 0: s += 1; case 1: s += 2; case 2: s += 4; break; case 3: s += 8; case 4: s += 16; case 5: s += 32; break; default: s = -1; } return s; }
int holes(int k) { switch (k) { case 1: return 1; case 3: return 3; case 5: return 5; case 20: return 20; case 25: return 25; default: return -7; } }
int neg(int k) { switch (k) { case -2: return 1; case -1: return 2; case 0: return 3; case 1: return 4; case 2: return 5; case 3: return 6; default: return 0; } }
int sparse(int k) { switch (k) { case 1: return 1; case 1000: return 2; case 100000: return 3; case 10000000: return 4; default: return 0; } }
int uns(unsigned k) { switch (k) { case 0: return 1; case 1: return 2; case 2: return 3; case 3: return 4; case 4000000000u: return 5; default: return 0; } }
long loop(long *p, long n) { long s = 0; for (long i = 0; i < n; i++) { switch (p[i] & 7) { case 0: s += 1; break; case 1: s -= 2; break; case 2: s *= 3; break; case 3: s ^= 4; break; case 4: s |= 5; break; case 5: s += p[i]; break; case 6: s -= p[i]; break; default: s = s >> 1; } } return s; }
int nest(int a, int b) { switch (a) { case 0: case 1: case 2: case 3: case 4: switch (b) { case 0: return 1; case 1: return 2; case 2: return 3; case 3: return 4; case 4: return 5; default: return 6; } default: return 7; } }
char sc(char c) { switch (c) { case 'a': return 'A'; case 'b': return 'B'; case 'c': return 'C'; case 'd': return 'D'; case 'e': return 'E'; default: return c; } }
"""


def test_jump_tables(monkeypatch):
    big = 'int f(int k) { switch (k) { %s default: return -1; } }' % ' '.join(
        'case %d: return %d;' % (i, i * 7 + 1) for i in range(20))
    big = big.replace('return %d;' % 8, 'k += 3; return k;').replace('case 1: k', 'case 1: k')
    assert re.search(r'\bbr\s+x17', body(big)), body(big)
    assert not re.search(r'\bbr\s', body(SW, 'a7')), body(SW, 'a7')  # few cases: the tree is faster
    m = cimo.compile(big + '\nstatic int mark_big;', cache=False)
    assert [m.f(k) for k in (-1, 0, 1, 7, 19, 20, 2 ** 31 - 1)] == [-1, 1, 4, 50, 134, -1, -1]
    from cimo.backend.arm64 import emit
    monkeypatch.setattr(emit, 'JT_MINCASES', 4)
    src = SW + 'static int mark_small;'
    assert re.search(r'\bbr\s+x17', body(src, 'a7')), body(src, 'a7')
    assert re.search(r'\bbr\s+x17', body(src, 'l6')), body(src, 'l6')
    assert not re.search(r'\bbr\s', body(src, 'sparse')), body(src, 'sparse')


@pytest.mark.parametrize('cached', [False, True])
def test_jump_tables_compute_the_same(cached, monkeypatch, tmp_path):
    from cimo.backend.arm64 import emit
    monkeypatch.setattr(emit, 'JT_MINCASES', 4)  # every switch here that can have a table gets one
    src = SW + '/* %s */ static int mark%s;' % (tmp_path.name, tmp_path.name.replace('-', '_'))
    m = cimo.compile(src, cache=cached)
    if cached:  # again, from the rebased image
        m = cimo.compile(src)
    t7 = {0: 11, 1: 22, 2: 37, 3: 41, 4: 59, 5: 61, 6: 71}
    for k in list(range(-3, 12)) + [2 ** 31 - 1, -2 ** 31, 1000]:
        assert m.s7(k) == t7.get(k, 0), k
        assert m.holes(k) == ({1: 1, 3: 3, 5: 5, 20: 20, 25: 25}.get(k, -7))
        assert m.neg(k) == ({-2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}.get(k, 0)), k
        assert m.sparse(k) == {1: 1, 1000: 2}.get(k, 0)
        for a in (0, 1, 5):
            want = {0: a + 3, 1: a * 22, 2: a - 37, 3: a ^ 41, 4: a | 59, 5: a << 2, 6: a >> 1}.get(k, 0)
            assert m.a7(k, a) == want
            assert m.l6(k, a) == {1: a + 1, 2: a * 2, 3: a - 3, 5: a ^ 5, 7: a | 7, 8: a & 8}.get(k, -1)
        want = {0: 7, 1: 6, 2: 4, 3: 56, 4: 48, 5: 32}.get(k, -1)
        assert m.fall(k) == want
    assert m.l6(2 ** 40 + 1, 5) == -1 and m.l6(-1, 5) == -1  # the whole long is compared
    assert m.uns(4000000000) == 5 and m.uns(3) == 4 and m.uns(5) == 0
    for a, b in ((0, 0), (0, 4), (0, 5), (4, 2), (5, 0), (-1, 0)):
        assert m.nest(a, b) == (7 if not 0 <= a <= 4 else {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}.get(b, 6))
    assert m.sc(ord('c')) == ord('C') and m.sc(ord('z')) == ord('z')
    p = array.array('q', [(i * 13) % 37 for i in range(50)])
    s = 0
    for v in p:
        k = v & 7
        if k == 0:
            s += 1
        elif k == 1:
            s -= 2
        elif k == 2:
            s *= 3
        elif k == 3:
            s ^= 4
        elif k == 4:
            s |= 5
        elif k == 5:
            s += v
        elif k == 6:
            s -= v
        else:
            s >>= 1
    assert m.loop(p, 50) == s


WRITEBACK = r'''
#include <stdint.h>
long lsum(long *p, long n) { long s = 0; while (n--) s += *p++; return s; }
long isum(int *p, long n) { long s = 0; while (n--) s += *p++; return s; }
long usum(unsigned *p, long n) { long s = 0; while (n--) s += *p++; return s; }
long ssum(short *p, long n) { long s = 0; while (n--) s += *p++; return s; }
long hsum(unsigned short *p, long n) { long s = 0; while (n--) s += *p++; return s; }
long bsum(signed char *p, long n) { long s = 0; while (n--) s += *p++; return s; }
long csum(unsigned char *p, long n) { long s = 0; while (n--) s += *p++; return s; }
double dsum(double *p, long n) { double s = 0; while (n--) s += *p++; return s; }
float fsum(float *p, long n) { float s = 0; while (n--) s += *p++; return s; }
long down(long *p, long n) { long s = 0; p += n; while (n--) s += *--p; return s; }
void fill(int *p, long n, int v) { while (n--) *p++ = v; }
void filld(double *p, long n, double v) { while (n--) *p++ = v; }
void copy8(char *d, const char *s, long n) { while (n--) *d++ = *s++; }
long self(long **p) { long *q = *p; long v = *q; q++; *p = q; return v; }
long both(long *p) { long a = *p; p++; return a + (long)p; }
long keep(long *p, long *out) { long a = *p; long *q = p + 1; *out = (long)p; return a + *q; }
'''


def test_post_index():
    for name in ('lsum', 'isum', 'usum', 'ssum', 'hsum', 'bsum', 'csum', 'dsum', 'fsum', 'fill', 'filld', 'copy8'):
        assert re.search(r'\], -?\d+\b', body(WRITEBACK, name)), (name, body(WRITEBACK, name))
    m = cimo.compile(WRITEBACK)
    n = 50
    want = {'lsum': ('q', 'l'), 'isum': ('i', 'i'), 'usum': ('I', 'i'), 'ssum': ('h', 'h'), 'hsum': ('H', 'h'),
            'bsum': ('b', 'b')}
    for name, (tc, _) in want.items():
        a = array.array(tc, [(-1) ** i * i if tc.islower() else i for i in range(n)])
        assert getattr(m, name)(a, n) == sum(a), name
    c = array.array('B', range(n))
    assert m.csum(c, n) == sum(c)
    d = array.array('d', [i / 2 for i in range(n)])
    assert m.dsum(d, n) == sum(d)
    f = array.array('f', [i / 2 for i in range(n)])
    assert m.fsum(f, n) == sum(f)
    q = array.array('q', range(n))
    assert m.down(q, n) == sum(q)
    w = array.array('i', [0] * (n + 1))
    m.fill(w, n, 7)
    assert list(w) == [7] * n + [0]
    v = array.array('d', [0.0] * (n + 1))
    m.filld(v, n, 1.5)
    assert list(v) == [1.5] * n + [0.0]
    src, dst = bytearray(range(n)), bytearray(n + 1)
    m.copy8(dst, bytes(src), n)
    assert dst[:n] == src and dst[n] == 0


BITFIELDS = r'''
unsigned bfi8(unsigned a, unsigned b) { return (a & ~0xff00u) | ((b & 0xff) << 8); }
unsigned bfi8r(unsigned a, unsigned b) { return ((b & 0xff) << 8) | (a & ~0xff00u); }
unsigned bfi_top(unsigned a, unsigned b) { return (a & 0xffffffu) | (b << 24); }
unsigned bfi_low(unsigned a, unsigned b) { return (a & ~0xffu) | (b & 0xff); }
unsigned bfxil8(unsigned a, unsigned b) { return (a & ~0xffu) | ((b >> 8) & 0xff); }
unsigned bfxil_top(unsigned a, unsigned b) { return (a & ~0xffu) | (b >> 24); }
unsigned long bfi64(unsigned long a, unsigned long b) { return (a & ~(0xffffUL << 16)) | ((b & 0xffff) << 16); }
unsigned long bfi64top(unsigned long a, unsigned long b) { return (a & 0xffffffffUL) | (b << 32); }
unsigned swapped(unsigned a, unsigned b) { return (b & ~0xff00u) | ((a & 0xff) << 8); }
unsigned same(unsigned a) { return (a & ~0xff00u) | ((a & 0xff) << 8); }
unsigned twice(unsigned a, unsigned b, unsigned *o) { unsigned x = a & ~0xff00u; *o = x; return x | ((b & 0xff) << 8); }
unsigned notfield(unsigned a, unsigned b) { return (a & ~0xf0f0u) | ((b & 0xf) << 4); }
unsigned overlap(unsigned a, unsigned b) { return (a & ~0xff00u) | ((b & 0xfff) << 8); }
'''


@pytest.mark.parametrize('name,want', [
    ('bfi8', r'bfi\s'), ('bfi8r', r'bfi\s'), ('bfi_top', r'bfi\s'), ('bfi_low', r'bf(i|xil)\s'),
    ('bfxil8', r'bfxil\s'), ('bfxil_top', r'bfxil\s'), ('bfi64', r'bfi\s'), ('bfi64top', r'bfi\s'),
    ('swapped', r'bfi\s'), ('same', r'bfi\s'),
])
def test_bitfield_insert(name, want):
    assert re.search(want, body(BITFIELDS, name)), body(BITFIELDS, name)


def test_bitfield_insert_only_when_it_fits():
    assert 'bf' not in body(BITFIELDS, 'notfield') and 'bf' not in body(BITFIELDS, 'overlap')


def test_bitfield_insert_computes_the_same():
    m = cimo.compile(BITFIELDS)
    M = 0xffffffff
    L = 2 ** 64 - 1
    for a, b in [(0x12345678, 0x9abcdef0), (0, M), (M, 0), (0xdeadbeef, 0xfeedface)]:
        assert m.bfi8(a, b) == (a & ~0xff00 & M) | ((b & 0xff) << 8)
        assert m.bfi8r(a, b) == m.bfi8(a, b)
        assert m.bfi_top(a, b) == (a & 0xffffff) | (b << 24 & M)
        assert m.bfi_low(a, b) == (a & ~0xff & M) | (b & 0xff)
        assert m.bfxil8(a, b) == (a & ~0xff & M) | (b >> 8 & 0xff)
        assert m.bfxil_top(a, b) == (a & ~0xff & M) | (b >> 24)
        assert m.swapped(a, b) == (b & ~0xff00 & M) | ((a & 0xff) << 8)
        assert m.same(a) == (a & ~0xff00 & M) | ((a & 0xff) << 8)
        assert m.notfield(a, b) == (a & ~0xf0f0 & M) | ((b & 0xf) << 4)
        assert m.overlap(a, b) == (a & ~0xff00 & M) | ((b & 0xfff) << 8 & M)
        o = (ctypes.c_uint * 1)()
        assert m.twice(a, b, o) == m.bfi8(a, b) and o[0] == a & ~0xff00 & M
        A, B = a * 0x100000001 & L, b * 0x10001 & L
        assert m.bfi64(A, B) == (A & ~(0xffff << 16) & L) | ((B & 0xffff) << 16)
        assert m.bfi64top(A, B) == (A & 0xffffffff) | (B << 32 & L)


FLAGS = r'''
int andsv(int a, int b, int *o) { int d = a & b; *o = d; return d ? 7 : 9; }
int andslt(int a, int b, int *o) { int d = a & b; *o = d; return d < 0 ? 7 : 9; }
int andsgt(int a, int b, int *o) { int d = a & b; *o = d; return d > 0 ? 7 : 9; }
long subsv(long a, long b, long *o) { long d = a - b; *o = d; return d == 0 ? 7 : 9; }
long addsv(long a, long b, long *o) { long d = a + b; *o = d; return d != 0 ? 7 : 9; }
int subslt(int a, int b, int *o) { int d = a - b; *o = d; return d < 0 ? 7 : 9; }
int subsgt(int a, int b, int *o) { int d = a - b; *o = d; return d > 0 ? 7 : 9; }
int neg(long a) { if (a < 0) return 1; return 2; }
int nonneg(int a) { if (a >= 0) return 1; return 2; }
int aa(int a, int b, int c, int d) { return a > b && c < d; }
int oo(int a, int b, int c, int d) { return a > b || c < d; }
int ok(int a, int b, int c) { int k = a > 0; if (k) k = b > 0; if (k) return c; return 0; }
'''


def test_flags_from_the_operation():
    assert re.search(r'\bands\s', body(FLAGS, 'andsv')) and 'cmp' not in body(FLAGS, 'andsv')
    assert re.search(r'\bands\s', body(FLAGS, 'andsgt'))
    assert re.search(r'\bsubs\s', body(FLAGS, 'subsv')) and 'cmp' not in body(FLAGS, 'subsv')
    assert re.search(r'\badds\s', body(FLAGS, 'addsv')) and 'cmp' not in body(FLAGS, 'addsv')
    # subs overflows where the comparison with 0 does not
    assert 'cmp' in body(FLAGS, 'subsgt')
    assert 'tbnz' in body(FLAGS, 'neg') and 'cmp' not in body(FLAGS, 'neg')
    assert re.search(r'tb(n)?z', body(FLAGS, 'nonneg')) and 'cmp' not in body(FLAGS, 'nonneg')
    for f in ('aa', 'oo', 'ok'):
        assert 'ccmp' in body(FLAGS, f), body(FLAGS, f)
    m = cimo.compile(FLAGS, wrapv=True)  # the values below overflow
    I = ctypes.c_int * 1
    Q = ctypes.c_long * 1
    M, H = 2 ** 31 - 1, -2 ** 31
    for a, b in [(0, 0), (1, 1), (5, 3), (-1, 1), (M, -1), (H, 1), (-1, H), (H, H), (M, M), (6, 3)]:
        assert m.andsv(a, b, I()) == (7 if a & b else 9)
        assert m.andslt(a, b, I()) == (7 if a & b < 0 else 9)
        assert m.andsgt(a, b, I()) == (7 if a & b > 0 else 9)
        wrapped = (a - b + 2 ** 31) % 2 ** 32 - 2 ** 31
        assert m.subslt(a, b, I()) == (7 if wrapped < 0 else 9), (a, b)
        assert m.subsgt(a, b, I()) == (7 if wrapped > 0 else 9), (a, b)
        assert m.subsv(a, b, Q()) == (7 if a == b else 9)
        assert m.addsv(a, b, Q()) == (7 if a + b else 9)
        assert m.neg(a) == (1 if a < 0 else 2) and m.nonneg(a) == (1 if a >= 0 else 2)
        for c, d in [(0, 1), (1, 0)]:
            assert m.aa(a, b, c, d) == int(a > b and c < d) and m.oo(a, b, c, d) == int(a > b or c < d)
        assert m.ok(a, b, 11) == (11 if a > 0 and b > 0 else 0)


LOOKUP = r"""
int full(int k) { switch (k) { case 0: return 3; case 1: return 5; case 2: return 9; case 3: return 17;
    case 4: return 31; case 5: return 2; default: return -7; } }
long wide(long k) { switch (k) { case 0: return 1L << 40; case 1: return -5; case 2: return 9; case 3: return 17;
    default: return 0; } }
int holes(int k) { switch (k) { case 1: return 3; case 3: return 5; case 5: return 9; case 6: return 17;
    default: return 44; } }
int sideeffect(int k, int *p) { switch (k) { case 0: return 3; case 1: *p = 1; return 5; case 2: return 9;
    case 3: return 17; default: return 0; } }
int few(int k) { switch (k) { case 0: return 3; case 1: return 5; default: return 0; } }
unsigned uns(unsigned k) { switch (k) { case 0: return 4000000000u; case 1: return 3; case 2: return 8;
    case 3: return 1; default: return 2; } }
int inloop(const int *p, int n) { int s = 0; for (int i = 0; i < n; i++) { int v; switch (p[i]) {
    case 0: v = 3; break; case 1: v = 5; break; case 2: v = 7; break; case 3: v = 11; break; case 4: v = 13; break;
    default: v = 1; } s += v; } return s; }
"""


def test_value_tables():
    for name in ('full', 'wide', 'holes', 'uns', 'inloop'):
        assert re.search(r'ldr\s+[wx]\d+, \[x17, ', body(LOOKUP, name)), (name, body(LOOKUP, name))
    for name in ('sideeffect', 'few'):
        assert 'x17, ' not in body(LOOKUP, name), body(LOOKUP, name)
    m = cimo.compile(LOOKUP)
    want = [3, 5, 9, 17, 31, 2]
    for k in range(-3, 10):
        assert m.full(k) == (want[k] if 0 <= k < 6 else -7)
    assert [m.wide(k) for k in (-1, 0, 1, 2, 3, 4, 2 ** 40)] == [0, 1 << 40, -5, 9, 17, 0, 0]
    assert [m.holes(k) for k in range(-1, 9)] == [44, 44, 3, 44, 5, 44, 9, 17, 44, 44]
    assert [m.uns(k) for k in (0, 1, 2, 3, 4, 2 ** 32 - 1)] == [4000000000, 3, 8, 1, 2, 2]
    p = (ctypes.c_int * 1)()
    assert m.sideeffect(1, p) == 5 and p[0] == 1 and m.few(1) == 5 and m.few(2) == 0
    data = array.array('i', [0, 1, 2, 3, 4, 5, -1, 2])
    assert m.inloop(data, len(data)) == 3 + 5 + 7 + 11 + 13 + 1 + 1 + 7


@pytest.mark.parametrize('cached', [False, True])
def test_value_tables_from_the_cache(cached, tmp_path):
    src = LOOKUP + 'static int mark%s;' % tmp_path.name.replace('-', '_')
    m = cimo.compile(src, cache=cached)
    if cached:  # again, from the rebased image
        m = cimo.compile(src)
    assert m.full(4) == 31 and m.wide(0) == 1 << 40 and m.full(100) == -7


CONSTS = r"""
long big1(const long *p, int n) { long s = 0; for (int i = 0; i < n; i++) s ^= p[i] + 0x123456789abcL; return s; }
int big2(const long *p, int n) { int c = 0; for (int i = 0; i < n; i++) c += p[i] == 0x123456789abcL; return c; }
int big3(const int *p, int n) { int c = 0; for (int i = 0; i < n; i++) c += p[i] < 0x12345678; return c; }
long G[8] = {1, 2, 3, 4, 5, 6, 7, 8};
static int S[8] = {1, 2, 3, 4, 5, 6, 7, 8};
long sumg(int n) { long s = 0; for (int i = 0; i < n; i++) s += G[i] + S[i]; return s; }
void setg(int n) { for (int i = 0; i < n; i++) { G[i] = i; S[i] = i * 2; } }
long pairs(const long *p, long n) { long s = 0; while (n--) { s += p[0] * p[1]; p += 2; } return s; }
void spairs(long *p, long n) { while (n--) { p[0] = n; p[1] = n + 1; p += 2; } }
unsigned short lowand(unsigned a) { return a & 0xff00; }
unsigned logicw(unsigned a) { return (a & 0xffff00ff) ^ 0x0f0f0f0f; }
unsigned long logicl(unsigned long a) { return (a | 0xff00ff00ff00ff00UL) & 0x0ffffffffffffff0UL; }
long zext1(unsigned a, unsigned b) { return (unsigned long)(a + b); }
long zext2(unsigned a) { return (unsigned long)(a >> 3); }
unsigned long zext3(unsigned a) { return a; }
unsigned long zext4(int a) { return (unsigned)a; }
int shifted(unsigned char a) { return (unsigned char)(a >> 1); }
int shiftedh(unsigned short a) { return (unsigned short)a >> 9; }
"""


def test_constants_out_of_loops():
    b = body(CONSTS, 'big1')
    assert re.search(r'mov[^\n]*#\d+[^\n]*\n[^\n]*movk', b)
    loop = b[b.rindex('.L'):] if '.L' in b else b
    m = cimo.compile(CONSTS)
    p = array.array('q', [0x123456789abc, 5, 0x123456789abc, 0x123456789abc + (1 << 40)])
    want = 0
    for x in p:
        want ^= x + 0x123456789abc
    assert m.big1(p, 4) == want and m.big2(p, 4) == 2
    assert m.big3(array.array('i', [0x12345677, 0x12345678, -1]), 3) == 2
    assert m.sumg(8) == 72
    m.setg(8)
    assert m.sumg(8) == sum(i * 3 for i in range(8))


def test_global_addresses_out_of_loops():
    src = 'long G[8];\nlong f(int n) { long s = 0; for (int i = 0; i < n; i++) s += G[i]; return s; }'
    b = body(src)
    labels = {m.group(1): m.start() for m in re.finditer(r'^(\.?\w+):', b, re.M)}
    loops = 0
    for m in re.finditer(r'^\s+\w+(?:\.\w+)?\s+(?:[wx]\d+, )*(\.?L\w+)$', b, re.M):
        if labels.get(m.group(1), m.start()) < m.start():  # a branch back: the loop's body
            loops += 1
            assert 'adrp' not in b[labels[m.group(1)]:m.start()], b
    assert loops


def test_pairs_with_write_back():
    assert re.search(r'ldp\s+x\d+, x\d+, \[x\d+\], 16', body(CONSTS, 'pairs')), body(CONSTS, 'pairs')
    assert re.search(r'stp\s+x\d+, x\d+, \[x\d+\], 16', body(CONSTS, 'spairs')), body(CONSTS, 'spairs')
    m = cimo.compile(CONSTS)
    p = array.array('q', [2, 3, 4, 5, 6, 7])
    assert m.pairs(p, 3) == 6 + 20 + 42
    q = array.array('q', [0] * 7)
    m.spairs(q, 3)
    assert list(q) == [2, 3, 1, 2, 0, 1, 0]


def test_logic_immediates():
    for name in ('lowand', 'logicw', 'logicl'):
        assert 'mov' not in body(CONSTS, name), body(CONSTS, name)
    m = cimo.compile(CONSTS)
    for a in (0, 0xdeadbeef, 0xffffffff, 0x12345678):
        assert m.lowand(a) == a & 0xff00
        assert m.logicw(a) == (a & 0xffff00ff) ^ 0x0f0f0f0f
        A = a * 0x100000001
        assert m.logicl(A) == (A | 0xff00ff00ff00ff00) & 0x0ffffffffffffff0


def test_redundant_extensions():
    for name in ('zext1', 'zext2'):
        assert 'mov' not in body(CONSTS, name) and 'uxtw' not in body(CONSTS, name), body(CONSTS, name)
    # no extension apart: Apple's callers have done it, so there a plain lsr does
    assert re.search(r'ubfx|lsr', body(CONSTS, 'shifted')) and 'uxtb' not in body(CONSTS, 'shifted')
    m = cimo.compile(CONSTS)
    M = 2 ** 32 - 1
    for a, b in ((M, 1), (M, M), (5, 7), (2 ** 31, 2 ** 31)):
        assert m.zext1(a, b) == (a + b) & M
        assert m.zext2(a) == a >> 3
        assert m.zext3(a) == a
        assert m.shifted(a & 255) == (a & 255) >> 1
        assert m.shiftedh(a & 65535) == (a & 65535) >> 9
    assert m.zext4(-1) == M


LOOPFOLDS = '''
long X[64], Y[64], Z[64]; int W[64];
long acc(long n) { long s = 0; for (long i = 0; i < n; i++) s += X[i & 63] << 3; return s; }
long acc2(long n) { long s = 0; for (long i = 0; i < n; i++) s -= X[i & 63] >> 3; return s; }
long through(long n) { long s = 1; for (long i = 0; i < n; i++) s = X[i & 63] + (s << 3); return s; }
void off(long n) { for (long i = 0; i < n; i++) Z[i & 63] = X[i & 63] + (Y[i & 63] << 3); }
long sext(long n) { long s = 0; for (long i = 0; i < n; i++) s += (long)W[i & 63]; return s; }
long sext1(long i) { return i + (long)W[i]; }
'''


def test_shift_folded_except_onto_the_accumulator():
    assert not re.search(r'(add|sub)\s.*, (lsl|asr) #', body(LOOPFOLDS, 'acc')) and 'lsl' in body(LOOPFOLDS, 'acc')
    assert not re.search(r'(add|sub)\s.*, (lsl|asr) #', body(LOOPFOLDS, 'acc2'))
    assert re.search(r'add\s.*, lsl #3', body(LOOPFOLDS, 'through'))
    assert re.search(r'add\s.*, lsl #3', body(LOOPFOLDS, 'off'))


def test_extension_of_a_load_stays_in_the_load():
    for f in ('sext', 'sext1'):
        assert 'ldrsw' in body(LOOPFOLDS, f) and 'sxtw\n' not in body(LOOPFOLDS, f)
        assert not re.search(r'add\s.*, sxtw', body(LOOPFOLDS, f))


def test_accumulator_stays_on_its_way_round_the_loop():
    """No copy on the chain: one fmov in `s += x[i] * y[i]` cost 52 to 56% on Neoverse."""
    src = 'double f(int n, const double *restrict x, const double *restrict y) { double s = 0.0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s; }'
    text = body(src, 'f')
    loops = [m.group(2) for m in re.finditer(r'^(\.?L\d+):\n((?:(?!^\.?L\d+:).*\n)*?\tb\w*\t\1\n)', text, re.M)]
    # fmul and fadd since fmachain runs on arm64 (the add's 2 cycles on the chain, not fmadd's 4)
    assert any('fadd' in x or 'fmadd' in x for x in loops), text
    assert not any(('fadd' in x or 'fmadd' in x) and 'fmov' in x for x in loops), text
    x = (ctypes.c_double * 9)(*[i + 0.5 for i in range(9)])
    y = (ctypes.c_double * 9)(*[2.0 * i - 1 for i in range(9)])
    m = cimo.compile(src, cache=False)
    assert m.f(9, ctypes.addressof(x), ctypes.addressof(y)) == sum(a * b for a, b in zip(x, y))
