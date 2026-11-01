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


def body(src, name='f'):
    asm = driver._build(src)[0]
    m = re.search(r'^%s:\n(.*?)^/\* end function' % name, asm, re.S | re.M)
    if m is None:
        m = re.search(r'^%s:\n(.*?)^\.type' % name, asm, re.S | re.M)
    assert m, name
    return m.group(1)


@pytest.mark.parametrize('src,want', [
    ('long f(long a) { return a << 3; }', r'lsl\s+x0, x0, #3'),
    ('long f(long a) { return a >> 9; }', r'asr\s+x0, x0, #9'),
    ('unsigned f(unsigned a) { return a >> 9; }', r'lsr\s+w0, w0, #9'),
    ('long f(long *p, long i) { return p[i]; }', r'ldr\s+x0, \[x0, x1, lsl #3\]'),
    ('int f(int *p, int i) { return p[i]; }', r'ldr\s+w0, \[x0, w1, sxtw #2\]'),
    ('int f(int *p, unsigned i) { return p[i]; }', r'ldr\s+w0, \[x0, w1, uxtw #2\]'),
    ('int f(char *p, long i) { return p[i]; }', r'ldrb\s+w0, \[x0, x1\]'),
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
    assert re.search(r'\bf?csel\b', b), b
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
    assert m.b(1, 2 ** 64 - 1) == -1  # the largest: its bits come back as a signed long
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
