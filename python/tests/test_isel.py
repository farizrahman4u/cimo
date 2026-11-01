"""arm64 selection beyond QBE's (ext.isel): which instructions come out, and that the results stay
right where each could go wrong (register pressure, spills, widths, signs)."""

import array
import platform
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
    ('double f(double a) { return a * 2.0; }', r'fmov\s+d\d+, #2\.0'),
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
