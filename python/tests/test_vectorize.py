"""The auto-vectorizer (cimo.middle.vectorize): which loops it takes, and that they compute what the scalar loop does.
What matters most is overlap: a wrong answer there is a silent wrong result, and random programs rarely hit the offsets."""

import ctypes
import random
import re
import struct

import pytest

import cimo
from cimo import driver, middle
import hosts

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='needs a target with q')


def vectorized(src):
    return bool(re.search(r'\t(ldr|str)\tq\d+|\tmovups ', driver._build(src)[0]))


_n = [0]


def both(src):
    """The module with the vectorizer, and without."""
    _n[0] += 1
    on = cimo.compile(src + '\nstatic int mk_on%d;' % _n[0], cache=False)
    old = middle.vectorizing
    middle.vectorizing = False
    try:
        off = cimo.compile(src + '\nstatic int mk_off%d;' % _n[0], cache=False)
    finally:
        middle.vectorizing = old
    return on, off


def buf(ctype, n, seed=1):
    r = random.Random(seed)
    if ctype in (ctypes.c_float, ctypes.c_double):
        return (ctype * n)(*[r.uniform(-100, 100) for _ in range(n)])
    return (ctype * n)(*[r.randrange(-10000, 10000) for _ in range(n)])


def same(src, ctype=ctypes.c_float, sizes=(0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100), pad=32):
    """f(a, n) leaves the same bytes in a, also around the n elements it was told about."""
    on, off = both(src)
    for n in sizes:
        x, y = buf(ctype, n + 2 * pad, n), buf(ctype, n + 2 * pad, n)
        on.f(ctypes.addressof(x) + pad * ctypes.sizeof(ctype), n)
        off.f(ctypes.addressof(y) + pad * ctypes.sizeof(ctype), n)
        assert bytes(x) == bytes(y), n


@pytest.mark.parametrize('d', range(-9, 10))
@pytest.mark.parametrize('ty,ctype', [('float', ctypes.c_float), ('double', ctypes.c_double), ('int', ctypes.c_int), ('long', ctypes.c_int64)])
def test_a_store_and_a_load_any_distance_apart(d, ty, ctype):
    """a[i + d] = a[i] * 3 + 1 for every d around the number of lanes, with the load first."""
    src = 'void f(%s *a, long n) { for (long i = 0; i < n; i++) a[i + %d] = a[i] * 3 + 1; }' % (ty, d)
    same(src, ctype)


@pytest.mark.parametrize('d', range(-9, 10))
def test_the_store_first(d):
    """t = a[i]; a[i] = 5 is order the vector loop must keep; here the store comes before the load."""
    src = 'long f(int *a, long n) { for (long i = 0; i < n; i++) { a[i] = 7; a[i + %d] = a[i + %d] + 1; } return 0; }' % (d, d + 1)
    same(src, ctypes.c_int)


@pytest.mark.parametrize('d', range(-9, 10))
def test_two_stores_any_distance_apart(d):
    if d == 0:
        return
    src = 'void f(int *a, long n) { int *b = a + 8; for (long i = 0; i < n; i++) { b[i] = 3; b[i + %d] = 9; } }' % d
    same(src, ctypes.c_int)


def test_too_many_arrays_to_compare():
    from cimo.middle import vectorize as V
    args = ', '.join('int *a%d' % n for n in range(8))
    body = ' '.join('a%d[i] = a%d[i] + 1;' % (n, n) for n in range(8))
    assert not vectorized('void f(%s, long n) { for (long i = 0; i < n; i++) { %s } }' % (args, body))


def test_which_distances_are_taken():
    """Distances the rule allows are vectorized, so the tests above do test the vector loop. 4 lanes of int."""
    def taken(d):
        return vectorized('void f(int *a, long n) { for (long i = 0; i < n; i++) a[i + %d] = a[i] * 3 + 1; }' % d)
    assert taken(0)
    assert taken(4) and taken(5)  # a vector's worth ahead
    assert not taken(1) and not taken(3)  # reads what the scalar loop just wrote
    assert taken(-1) and taken(-3)  # reads ahead of the write, and the load comes first


@pytest.mark.parametrize('ty,ctype', [('int', ctypes.c_int), ('double', ctypes.c_double)])
@pytest.mark.parametrize('k', list(range(-12, 13)) + [-100, 100])
def test_two_pointers_any_distance_apart_when_it_runs(ty, ctype, k):
    """No restrict, and one array or not is known only when it runs: y = a + k, x = a. Each answer must be the scalar loop's."""
    src = 'void f(%s *y, const %s *x, long n) { for (long i = 0; i < n; i++) y[i] = x[i] * 3 + 1; }' % (ty, ty)
    assert vectorized(src)
    on, off = both(src)
    size = ctypes.sizeof(ctype)
    for n in (0, 1, 3, 4, 5, 8, 16, 17, 40):
        p, q = buf(ctype, 200, n + 1), buf(ctype, 200, n + 1)
        base = 110 if k < 0 else 20
        on.f(ctypes.addressof(p) + (base + k) * size, ctypes.addressof(p) + base * size, n)
        off.f(ctypes.addressof(q) + (base + k) * size, ctypes.addressof(q) + base * size, n)
        assert bytes(p) == bytes(q), n


def test_three_arrays_and_several_offsets():
    src = 'void f(int *y, const int *x, const int *z, long n) { for (long i = 0; i < n - 2; i++) y[i + 1] = x[i] + z[i + 2] - x[i + 1]; }'
    assert vectorized(src)
    on, off = both(src)
    for dy, dx, dz in [(0, 40, 80), (40, 0, 80), (2, 0, 4), (0, 2, 4), (3, 0, 0), (0, 0, 0), (1, 1, 1), (5, 0, 90), (0, 1, 2), (2, 1, 0), (80, 77, 0)]:
        for n in (0, 3, 4, 5, 6, 7, 9, 13, 30):
            p, q = buf(ctypes.c_int, 200, n), buf(ctypes.c_int, 200, n)
            a, b = ctypes.addressof(p) + 16, ctypes.addressof(q) + 16
            on.f(a + 4 * dy, a + 4 * dx, a + 4 * dz, n)
            off.f(b + 4 * dy, b + 4 * dx, b + 4 * dz, n)
            assert bytes(p) == bytes(q), (dy, dx, dz, n)


def test_an_array_inside_the_other():
    """The ranges overlap though both ends of one are inside the other, and the reverse."""
    src = 'void f(int *y, const int *x, long n, long m) { for (long i = 0; i < n; i++) y[i] = x[i * 1] + (int)m; }'
    on, off = both(src)
    for k in (1, 3, 4, 7, 8):
        p, q = buf(ctypes.c_int, 100), buf(ctypes.c_int, 100)
        on.f(ctypes.addressof(p) + 40, ctypes.addressof(p) + 40 + 4 * k, 32, 5)
        off.f(ctypes.addressof(q) + 40, ctypes.addressof(q) + 40 + 4 * k, 32, 5)
        assert bytes(p) == bytes(q), k


def test_store_before_a_load_ahead_is_not_taken():
    src = 'void f(int *a, int *c, long n) { for (long i = 0; i < n; i++) { a[i] = 1; a[i - 8] = a[i + 1]; } }'
    assert not vectorized(src)
    on, off = both(src)
    x, y = buf(ctypes.c_int, 80), buf(ctypes.c_int, 80)
    on.f(ctypes.addressof(x) + 40, 0, 40)
    off.f(ctypes.addressof(y) + 40, 0, 40)
    assert bytes(x) == bytes(y)


@pytest.mark.parametrize('src,taken', [
    ('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }', True),
    ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }', True),
    ('void f(int *restrict y, const int *restrict x, int a, int n) { for (int i = 0; i < n; i++) y[i] = (x[i] << 3) - a; }', True),
    ('void f(float *restrict y, const float *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = x[i] / 2.5f; }', True),
    ('void f(float *y, const float *x, long n) { for (long i = 0; i < n; i++) y[i] = x[i]; }', True),  # compared when it runs
    ('double f(const double *x, long n) { double s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }', False),  # C fixes the order
    ('void f(long *restrict y, const long *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = x[i] * x[i]; }', False),  # no 64-bit mul
    ('void f(int *restrict y, const int *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = x[i] / 3; }', False),
    ('void f(float *restrict y, const float *restrict x, long n) { for (long i = 0; i < n; i += 2) y[i] = x[i]; }', False),  # not consecutive
    ('void f(float *restrict y, const float *restrict x, long n) { for (long i = n; i > 0; i--) y[i] = x[i]; }', False),  # downward
    ('void f(int *restrict y, const int *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = (int)i + x[i]; }', False),  # counter as data
    ('void g(void); void f(int *restrict y, long n) { for (long i = 0; i < n; i++) { y[i] = 1; g(); } }', False),
])
def test_which_loops_are_taken(src, taken):
    assert vectorized(src) == taken, driver._build(src)[0]


KERNELS = {
    'saxpy': ('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }', ctypes.c_float),
    'neg': ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) y[i] = -x[i] * a - y[i]; }', ctypes.c_double),
    'sqrt': ('#include <math.h>\nvoid f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) y[i] = sqrt(fabs(x[i])) + a; }', ctypes.c_double),
    'minmax': ('#include <math.h>\nvoid f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) y[i] = fmaxf(fminf(x[i], a), y[i]); }', ctypes.c_float),
    'ints': ('void f(int *restrict y, const int *restrict x, int a, long n) { for (long i = 0; i < n; i++) y[i] = ((x[i] * a) ^ (y[i] >> 2)) + (x[i] & 255) - (y[i] | 1); }', ctypes.c_int),
    'long': ('void f(long *restrict y, const long *restrict x, long a, long n) { for (long i = 0; i < n; i++) y[i] = ((x[i] + a) ^ (y[i] >> 3)) - ((unsigned long)x[i] >> 7); }', ctypes.c_int64),
    'stencil': ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 2; i < n - 2; i++) y[i] = a * (x[i - 2] + x[i - 1] + x[i] + x[i + 1] + x[i + 2]); }', ctypes.c_double),
    'int_counter': ('void f(float *restrict y, const float *restrict x, float a, int n) { for (int i = 0; i < n; i++) y[i] = x[i] * x[i] + a; }', ctypes.c_float),
}


@pytest.mark.parametrize('name', sorted(KERNELS))
def test_bit_for_bit_the_scalar_loop(name):
    src, ctype = KERNELS[name]
    if name == 'minmax' and hosts.AMD64:
        pytest.skip('fmin and fmax are calls on x86-64')
    assert vectorized(src), driver._build(src)[0]
    on, off = both(src)
    a = 3 if ctype in (ctypes.c_int, ctypes.c_int64) else 1.25
    for n in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200):
        x = buf(ctype, n + 8, n)
        y1, y2 = buf(ctype, n + 8, n + 1), buf(ctype, n + 8, n + 1)
        on.f(ctypes.addressof(y1), ctypes.addressof(x), a, n)
        off.f(ctypes.addressof(y2), ctypes.addressof(x), a, n)
        assert bytes(y1) == bytes(y2), n


CONDITIONS = {
    'relu': ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) { double v = x[i]; y[i] = v > 0.0 ? v : 0.0; } }', ctypes.c_double),
    'clamp': ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) { double v = x[i] > 0.0 ? x[i] : 0.0; y[i] = v < a ? v : a; } }', ctypes.c_double),
    'ne': ('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) y[i] = x[i] != y[i] ? x[i] : a; }', ctypes.c_float),
    'le': ('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) { float v = y[i] * 2.0f; y[i] = x[i] <= a ? v : x[i]; } }', ctypes.c_float),
    'nested': ('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) { float v = x[i]; y[i] = v < 0.0f ? 0.0f : (v > a ? a : v); } }', ctypes.c_float),
    'ge': ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) y[i] = x[i] >= y[i] ? x[i] - a : y[i]; }', ctypes.c_double),
    'eq': ('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) y[i] = x[i] == a ? 1.0f : x[i]; }', ctypes.c_float),
    'int': ('void f(int *restrict y, const int *restrict x, int a, long n) { for (long i = 0; i < n; i++) { int v = x[i], w = y[i]; y[i] = v < a ? v : (v == w ? 0 : w); } }', ctypes.c_int),
    'unsigned': ('void f(unsigned *restrict y, const unsigned *restrict x, unsigned a, long n) { for (long i = 0; i < n; i++) y[i] = x[i] <= y[i] ? x[i] : a; }', ctypes.c_uint),
    'long': ('void f(long *restrict y, const long *restrict x, long a, long n) { for (long i = 0; i < n; i++) y[i] = x[i] > y[i] ? x[i] : y[i] - a; }', ctypes.c_int64),
    'ulong': ('void f(unsigned long *restrict y, const unsigned long *restrict x, unsigned long a, long n) { for (long i = 0; i < n; i++) y[i] = x[i] >= y[i] ? a : x[i]; }', ctypes.c_uint64),
}


def _specials(ctype, n, seed):
    """Values where a condition is easy to get wrong: NaNs, signed zeros, infinities and equal pairs for floats; signs
    and both ends of the range for integers."""
    r = random.Random(seed)
    if ctype in (ctypes.c_float, ctypes.c_double):
        pool = [float('nan'), -0.0, 0.0, float('inf'), -float('inf'), 1.25, -1.25, 3.0]
        return (ctype * n)(*[r.choice(pool) if r.random() < 0.6 else r.uniform(-4, 4) for _ in range(n)])
    bits = 8 * ctypes.sizeof(ctype)
    lo, hi = (0, 2 ** bits - 1) if ctype in (ctypes.c_uint, ctypes.c_uint64) else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    pool = [lo, hi, 0, 1, 3, lo + 1, hi - 1]
    return (ctype * n)(*[r.choice(pool) if r.random() < 0.6 else r.randint(lo, hi) for _ in range(n)])


@pytest.mark.parametrize('name', sorted(CONDITIONS))
def test_conditions_bit_for_bit(name):
    """A comparison and a select per lane: exact, NaNs and signed zeros included, on every lane width."""
    src, ctype = CONDITIONS[name]
    if hosts.AMD64 and ctype in (ctypes.c_float, ctypes.c_double):
        pytest.skip('x86-64 if-converts integer diamonds only: its scalar code has no float select')
    assert vectorized(src), driver._build(src)[0]
    on, off = both(src)
    for n in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200):
        for seed in range(3):
            a = _specials(ctype, 1, seed + 100)[0]
            if ctype in (ctypes.c_float, ctypes.c_double) and seed == 0:
                a = 3.0
            x = _specials(ctype, n + 8, seed)
            y1, y2 = _specials(ctype, n + 8, seed + 50), _specials(ctype, n + 8, seed + 50)
            if n:
                x[0] = y1[0] = y2[0]  # an equal pair
            on.f(ctypes.addressof(y1), ctypes.addressof(x), a, n)
            off.f(ctypes.addressof(y2), ctypes.addressof(x), a, n)
            assert bytes(y1) == bytes(y2), (n, seed)


def test_a_mask_only_feeds_a_select():
    """A comparison whose result is a number (0 or 1), not a choice, has no vector form here."""
    assert not vectorized('void f(int *restrict y, const float *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = x[i] > 1.0f; }')


def test_values_the_loop_leaves_with():
    src = 'long f(float *restrict y, const float *restrict x, long n) { long i; for (i = 0; i < n; i++) y[i] = x[i] + 1; return i; }'
    on, off = both(src)
    for n in (0, 1, 3, 4, 5, 8, 9):
        x, y1, y2 = buf(ctypes.c_float, 16), buf(ctypes.c_float, 16, 2), buf(ctypes.c_float, 16, 2)
        assert on.f(ctypes.addressof(y1), ctypes.addressof(x), n) == off.f(ctypes.addressof(y2), ctypes.addressof(x), n) == n
        assert bytes(y1) == bytes(y2)


def test_nothing_beyond_the_end_is_touched():
    """The vector loop runs whole vectors: 7 elements take one, and 3 scalar steps."""
    src = 'void f(int *restrict y, const int *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = x[i] + 1; }'
    m = cimo.compile(src, cache=False)
    for n in range(0, 20):
        x = buf(ctypes.c_int, 32)
        y = (ctypes.c_int * 32)(*([-1] * 32))
        m.f(ctypes.addressof(y), ctypes.addressof(x), n)
        assert list(y[:n]) == [v + 1 for v in x[:n]]
        assert list(y[n:]) == [-1] * (32 - n), n


REDUCTIONS = {
    'wsum': ('int f(const int *restrict x, long n) { int s = 7; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_int, True),
    'lsum': ('long f(const long *restrict x, long n) { long s = -7; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_int64, True),
    'widen': ('long f(const int *restrict x, long n) { long s = 3; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_int, True),
    'uwiden': ('long f(const unsigned *restrict x, long n) { long s = 3; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_uint, True),
    'int_i': ('long f(const int *restrict x, int n) { long s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_int, True),
    'sub': ('int f(const int *restrict x, long n) { int s = 100; for (long i = 0; i < n; i++) s -= x[i]; return s; }', ctypes.c_int, True),
    'xor': ('int f(const int *restrict x, long n) { int s = 5; for (long i = 0; i < n; i++) s ^= x[i]; return s; }', ctypes.c_int, True),
    'and': ('int f(const int *restrict x, long n) { int s = -1; for (long i = 0; i < n; i++) s &= x[i] | 0xf0; return s; }', ctypes.c_int, True),
    'or': ('long f(const long *restrict x, long n) { long s = 1; for (long i = 0; i < n; i++) s |= x[i]; return s; }', ctypes.c_int64, True),
    'mulsum': ('int f(const int *restrict x, long n) { int s = 0; for (long i = 0; i < n; i++) s += x[i] * x[i]; return s; }', ctypes.c_int, True),
    # reassociated into (s + x * x) + 3, a chain of two: stays scalar for now
    'chain': ('int f(const int *restrict x, long n) { int s = 0; for (long i = 0; i < n; i++) s += x[i] * x[i] + 3; return s; }', ctypes.c_int, False),
    'two': ('int f(const int *restrict x, long n) { int s = 0, t = 1; for (long i = 0; i < n; i++) { s += x[i]; t ^= x[i]; } return s - t; }', ctypes.c_int, True),
    'wraps': ('int f(const unsigned *restrict x, long n) { unsigned s = 4000000000u; for (long i = 0; i < n; i++) s += x[i] * 65536; return (int)s; }', ctypes.c_uint, True),
    # read in the loop: the running value is seen, so lanes won't do
    'seen': ('int f(int *restrict y, const int *restrict x, long n) { int s = 0; for (long i = 0; i < n; i++) { s += x[i]; y[i] = s; } return s; }', ctypes.c_int, False),
    'mul': ('int f(const int *restrict x, long n) { int s = 1; for (long i = 0; i < n; i++) s *= x[i]; return s; }', ctypes.c_int, False),
    # float sums keep C's order: lane by lane, and only where the rest of the loop is worth it, not a plain sum or dot
    'fsum': ('float f(const float *restrict x, long n) { float s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_float, False),
    'dsum': ('double f(const double *restrict x, long n) { double s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_double, False),
    'ddot': ('double f(const double *restrict x, long n) { double s = 0; for (long i = 0; i < n; i++) s += x[i] * x[i]; return s; }', ctypes.c_double, False),
    'fdot': ('float f(const float *restrict x, long n) { float s = 0; for (long i = 0; i < n; i++) s += x[i] * (x[i] + 1); return s; }', ctypes.c_float, False),
    'dheavy': ('#include <math.h>\ndouble f(const double *restrict x, long n) { double s = 0.5; for (long i = 0; i < n; i++) s += 1.0 / sqrt(x[i] * x[i] + 1.0); return s; }', ctypes.c_double, hosts.AMD64),
    'fheavy': ('#include <math.h>\nfloat f(const float *restrict x, long n) { float s = 3; for (long i = 0; i < n; i++) s -= sqrtf(fabsf(x[i]) + 2.0f) / x[i]; return s; }', ctypes.c_float, hosts.AMD64),
    'swapped': ('#include <math.h>\nfloat f(const float *restrict x, long n) { float s = 1e7f; for (long i = 0; i < n; i++) s = 1.0f / sqrtf(x[i] * x[i] + 1.0f) + s; return s; }', ctypes.c_float, hosts.AMD64),
    'fmaheavy': ('#include <math.h>\ndouble f(const double *restrict x, long n) { double s = 0; for (long i = 0; i < n; i++) s += x[i] * (1.0 / sqrt(x[i] * x[i] + 2.0)); return s; }', ctypes.c_double, hosts.AMD64),
    'three': ('#include <math.h>\ndouble f(const double *restrict x, long n) { double a = 0, b = 1, c = 2; for (long i = 0; i < n; i++) { double r = 1.0 / sqrt(x[i] * x[i] + 1e-9); a += x[i] * r; b -= r; c += r * r; } return a + 3 * b + 9 * c; }', ctypes.c_double, hosts.AMD64),
    # cheap arithmetic is worth it on both: the lanes do it at once, and only the adds wait for each other
    'poly': ('float f(const float *restrict x, long n) { float s = 0; for (long i = 0; i < n; i++) { float t = x[i] * 0.01f; s += (((((t * 3 - 1) * t + 2) * t - 5) * t + 7) * t - 11) * t * t + 13 * t; } return s; }', ctypes.c_float, True),
    'dpoly': ('double f(const double *restrict x, long n) { double s = 1; for (long i = 0; i < n; i++) { double t = x[i] * 0.01; s -= (((((t * 3 - 1) * t + 2) * t - 5) * t + 7) * t - 11) * t * t + 13 * t * t * t; } return s; }', ctypes.c_double, True),
    'fseen': ('#include <math.h>\nfloat f(float *restrict y, const float *restrict x, long n) { float s = 0; for (long i = 0; i < n; i++) { s += sqrtf(fabsf(x[i])) / 3.0f; y[i] = s; } return s; }', ctypes.c_float, False),
    'wmin': ('long f(const short *restrict x, long n) { long s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_short, False),
}


@pytest.mark.parametrize('name', sorted(REDUCTIONS))
def test_reductions(name):
    src, ctype, taken = REDUCTIONS[name]
    assert vectorized(src) == taken, driver._build(src)[0]
    on, off = both(src)
    for n in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200):
        x = buf(ctype, n + 8, n)
        y = (ctypes.c_int * (n + 8))()
        if name == 'fseen':
            y = (ctypes.c_float * (n + 8))()
        extra = (ctypes.addressof(y),) if name in ('seen', 'fseen') else ()
        args = extra + (ctypes.addressof(x), n)
        a, b = on.f(*args), off.f(*args)
        assert a == b or (a != a and b != b), (name, n, a, b)


def test_reduction_reaches_the_values_the_loop_leaves_with():
    """Lanes collected, the remainder steps on from them, and the sum leaves the loop right for every remainder."""
    src = 'long f(const int *restrict x, long n, long *count) { long s = 0, i; for (i = 0; i < n; i++) s += x[i]; *count = i; return s; }'
    assert vectorized(src)
    m = cimo.compile(src, cache=False)
    for n in range(0, 40):
        x = buf(ctypes.c_int, 48, n)
        c = ctypes.c_int64()
        assert m.f(ctypes.addressof(x), n, ctypes.addressof(c)) == sum(x[:n])
        assert c.value == n


def _x86(src, target='amd64_sysv'):
    pp = driver._preprocess(src, 'x.c', (), False, ())
    ssa, params = driver._frontend(pp, 'x.c', target=target)
    return driver._qbe(ssa, params, target=target)


_ALLOPS = 'void f(int *restrict c, const int *restrict a, const int *restrict b, long n) { for (long i = 0; i < n; i++) c[i] = (a[i] * b[i]) >> 3; }\n' \
          'long g(const int *a, long n) { long s = 0; for (long i = 0; i < n; i++) s += a[i]; return s; }\n' \
          'unsigned long h(const unsigned *a, long n) { unsigned long s = 0; for (long i = 0; i < n; i++) s += a[i]; return s; }\n' \
          'void k(double *restrict y, const double *restrict x, double a, long n) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }\n' \
          'void m(long *restrict y, const long *restrict x, long n) { for (long i = 0; i < n; i++) y[i] = (x[i] >> 3) + 5; }\n'


@pytest.mark.parametrize('target', ['amd64_sysv', 'amd64_windows'])
def test_x86_code_runs_on_any_x86_64(target, monkeypatch):
    """With no features detected (cimo.cpu), nothing past SSE2; the instructions below are what's easiest to reach for."""
    monkeypatch.setattr(driver, '_x86', lambda t=None: frozenset())
    text = _x86(_ALLOPS, target)
    assert 'movups' in text and 'pmuludq' in text and 'punpckhdq' in text
    later = re.findall(r'\t(pmulld|pmovsx\w+|pmovzx\w+|pcmpgtq|pcmpeqq|pminsd|pmaxsd|pminud|pmaxud|pextr[bdq]|pinsr\w|movddup|haddp\w|phadd\w|pabs\w|blendv\w+|pblend\w+|v\w+|\w*fmadd\w*)\b', text)
    assert not later


def test_x86_back_edge_is_one_jump():
    """The conditional jump goes back, so an iteration doesn't take one jump over another."""
    text = _x86('void f(float *restrict y, const float *restrict x, float a, long n) { for (long i = 0; i < n; i++) y[i] += a * x[i]; }')
    at = text.index('mulps')
    head = re.findall(r'^(\.Lbb\d+):', text[:at], re.M)[-1]
    assert re.search(r'\tj(?!mp)\w+ ' + re.escape(head) + r'\n', text[at:at + 300]), text


def test_x86_windows_keeps_the_scratch_register():
    """xmm15 is the scratch register for 128-bit moves, and on Windows a callee has to keep it, all 128 bits."""
    src = 'int f(const int *a, long n) { int s = 0; for (long i = 0; i < n; i++) s += a[i]; return s; }'
    text = _x86(src, 'amd64_windows')
    assert 'xmm15' in text and re.search(r'movdqu %xmm15, \d+\(%rsp\)', text) and re.search(r'movdqu \d+\(%rsp\), %xmm15', text)


def test_x86_mask_leaves_the_loop():
    """gvn and gcm see what SSE2 lacks already written out: one all-zero register, made before the loop."""
    text = _x86('long g(const int *a, long n) { long s = 0; for (long i = 0; i < n; i++) s += a[i]; return s; }')
    from cimo.middle import vectorize
    assert text.count('pcmpgtd') == vectorize.INTERLEAVE  # one a vector, for its high halves
    head = text[: text.index('pcmpgtd')]
    assert 'pxor' in head and head.rindex('pxor') < head.rindex('.p2align')


def test_x86_float_moves_wait_for_nothing():
    """movss and movsd between registers keep the destination's upper bits, and wait for whatever last wrote them: 6.5x on
    nbody_acc on Intel. The same goes for what's written by a square root or a conversion, so the destination is cleared."""
    text = _x86('double f(double *x, int n) { double s = 1, t = 2; for (int i = 0; i < n; i++) { double u = s * t + i; t = s; s = u / (t + x[i]); } return s + t; }\n'
                'double r(double a, double b) { return __builtin_sqrt(a) + b * a; }')
    assert not re.search(r'\tmovs[sd] %xmm\d+, %xmm\d+', text), text
    assert 'movaps' in text
    for m in re.finditer(r'\t(cvtsi2sd\w*|sqrtsd) (\S+), %(xmm\d+)\n', text):
        if m[2] != '%' + m[3]:
            before = text[: m.start()].rstrip().rsplit('\n', 1)[-1]
            assert 'xorps %%%s, %%%s' % (m[3], m[3]) in before, text


_FLAGS = re.compile(r'\t(add|sub|and|or|xor|neg|inc|dec|shl|shr|sar|imul|cmp|test|ucomis[sd])[a-z]* ')


def _flags_reach_their_jump(text):
    """Nothing that writes flags between a compare and the jump that reads them."""
    lines = text.split('\n')
    for n, l in enumerate(lines):
        if re.match(r'\tj(?!mp)\w+ ', l):
            for back in range(n - 1, max(n - 40, -1), -1):
                m = lines[back]
                if m.endswith(':') or re.match(r'\t(jmp|ret|callq)', m):
                    break
                if _FLAGS.match(m):
                    assert re.match(r'\t(cmp|test|ucomis|add|sub|and|or|xor|neg|shl|shr|sar)', m), m
                    if re.match(r'\t(cmp|test|ucomis)', m):
                        break
                    # an add whose flags the jump means to read has no compare above it in this block
                    for up in range(back - 1, max(back - 40, -1), -1):
                        k = lines[up]
                        if k.endswith(':') or re.match(r'\t(jmp|ret|callq|j)', k):
                            break
                        assert not re.match(r'\t(cmp|test|ucomis)', k), '\n'.join(lines[up:n + 1])
                    break


@pytest.mark.parametrize('target', ['amd64_sysv', 'amd64_windows'])
def test_x86_address_for_a_phi_leaves_the_flags(target):
    """extern_array + 2 as a phi's argument is made at the end of the block before, after the compare the jump reads."""
    src = '''extern short arr[]; extern short brr[]; void use(short *p);
    long f(long n, long m) { short *p = brr + 3; long s = 0;
        for (long i = 0; i < m; i++) { s += *p; p += 1; if (n > 4) { p = arr + 2; continue; } if (i > 9) break; p = brr + 5; }
        use(p); return s; }'''
    text = _x86(src, target)
    assert re.search(r'lea\w* \d+\(%\w+\), %\w+', text), text
    _flags_reach_their_jump(text)


def test_x86_min_in_a_loop_is_cmov():
    """The middle end's if-conversion selects, as QBE's own ifopt does on x86-64: cmov, not a mask of six instructions
    on the chain through cur[j - 1] (edit_distance 2.1 to 2.4x of gcc)."""
    src = '''void f(int m, const unsigned char *restrict a, const unsigned char *restrict b, const int *restrict prev, int *restrict cur) {
        for (int j = 1; j <= m; j++) { int s = prev[j - 1] + (a[0] != b[j - 1]), d = prev[j] + 1, i = cur[j - 1] + 1;
            int v = s < d ? s : d; cur[j] = v < i ? v : i; } }'''
    text = _x86(src)
    assert 'cmov' in text and 'negl' not in text, text
    on, off = both(src)
    for m in (0, 1, 5, 17, 40):
        a, b = (ctypes.c_ubyte * 1)(3), (ctypes.c_ubyte * 48)(*[(k * 7) % 5 for k in range(48)])
        prev = buf(ctypes.c_int, 48, m)
        cur1, cur2 = (ctypes.c_int * 48)(9), (ctypes.c_int * 48)(9)
        on.f(m, a, b, prev, cur1)
        off.f(m, a, b, prev, cur2)
        assert list(cur1) == list(cur2)


_RMWSRC = '''
void hist(int n, const unsigned char *restrict x, int *restrict h) { for (int i = 0; i < n; i++) h[x[i]]++; }
void between(int *a, int *b) { int t = a[0]; b[0] = 7; a[0] = t + 1; }
void back(int *a) { a[1] = 5 - a[1]; }
void ops(long *a, int *b, long v) { a[0] -= v; a[1] &= v; a[2] |= 6; a[3] ^= v; b[0] += (int)v; b[1] = b[1] - 9; }
void vol(volatile int *a) { a[0] += 3; }
void other(int *a) { a[1] = a[0] + 3; }
'''


def test_x86_read_modify_write():
    """A load, one operation and a store to the same address are one x86 instruction, as gcc's (histogram on Zen 4: 8.8 to
    5.9 ms against gcc's 6.3), and a byte extended to an index is loaded so, once. Not past a store that may be to the
    same place, not 5 - x as x - 5, and never on a volatile."""
    text = _x86(_RMWSRC)
    def body(f):
        t = text[text.index('\n%s:\n' % f):]
        return t[:t.index('end function')]
    assert re.search(r'\taddl \$1, \(%\w+, %\w+, 4\)', body('hist')) and not re.search(r'movzbq %\w+, ', body('hist'))
    assert not re.search(r'\taddl [^,]+, \(', body('between'))
    assert not re.search(r'\tsubl [^,]+, \d*\(', body('back'))
    ops = body('ops')
    for mn in ('subq', 'andq', 'orq', 'xorq', 'addl', 'subl'):
        assert re.search(r'\t%s [^,]+, \d*\(%%\w+\)' % mn, ops), (mn, ops)
    assert not re.search(r'\taddl [^,]+, \(', body('vol'))
    assert not re.search(r'\taddl [^,]+, \d*\(', body('other'))


@pytest.mark.skipif(not hosts.AMD64, reason='runs x86-64 code')
def test_x86_read_modify_write_runs():
    m = cimo.compile(_RMWSRC, cache=False)
    a = (ctypes.c_int * 4)(10, 20, 0, 0)
    m.between(a, a)
    assert a[0] == 11
    m.back(a)
    assert a[1] == -15
    la, b = (ctypes.c_int64 * 4)(100, 0xff, 1, 0xf0), (ctypes.c_int * 2)(5, 3)
    m.ops(la, b, 12)
    assert list(la) == [88, 12, 7, 0xf0 ^ 12] and list(b) == [17, -6]
    x = (ctypes.c_ubyte * 300)(*[(k * 37) % 256 for k in range(300)])
    h = (ctypes.c_int * 256)()
    m.hist(300, x, h)
    want = [0] * 256
    for k in range(300):
        want[(k * 37) % 256] += 1
    assert list(h) == want
    v = (ctypes.c_int * 1)(4)
    m.vol(v)
    assert v[0] == 7
    o = (ctypes.c_int * 2)(4, 0)
    m.other(o)
    assert list(o) == [4, 7]


@pytest.mark.parametrize('load,ext', [(l, e) for l in ('int', 'unsigned') for e in ('long', 'unsigned long')] +
                         [(l, e) for l in ('signed char', 'unsigned char', 'short', 'unsigned short') for e in ('long',)])
def test_x86_extended_load(load, ext):
    """The load does the extension, and its signedness is the extension's: (unsigned long)(int)x zero-extends."""
    src = '%s f(const %s *p) { return (%s)p[1]; }' % (ext, load, ext)
    text = _x86(src)
    body = text[text.index('\nf:\n'):]
    assert not re.search(r'\tmov[sz]\w+ %\w+, %\w+', body[:body.index('ret')]), body
    if hosts.AMD64:
        m = cimo.compile(src, cache=False)
        ct = {'int': ctypes.c_int, 'unsigned': ctypes.c_uint, 'signed char': ctypes.c_byte, 'unsigned char': ctypes.c_ubyte,
              'short': ctypes.c_short, 'unsigned short': ctypes.c_ushort}[load]
        a = (ct * 2)(0, -3 if load in ('int', 'signed char', 'short') else 0xfffd & ((1 << 8 * ctypes.sizeof(ct)) - 1) | 0x80)
        v = a[1]
        want = v & ((1 << 64) - 1) if ext == 'unsigned long' else v
        m.f.restype = ctypes.c_uint64 if ext == 'unsigned long' else ctypes.c_int64
        assert m.f(a) == want, (m.f(a), want)


_FEATURES = frozenset(('fma', 'avx', 'avx2', 'popcnt', 'lzcnt', 'bmi1', 'bmi2'))


def _x86with(src, feats, monkeypatch, target='amd64_sysv'):
    monkeypatch.setattr(driver, '_x86', lambda t=None: feats if (t or driver.TARGET).startswith('amd64') else frozenset())
    monkeypatch.setattr(driver, 'TARGET', target)
    pp = driver._preprocess(src, 'x.c', (), False, (), target=target)
    ssa, params = driver._ir(pp, 'x.c')
    return driver._qbe(ssa, params, target=target)


def _body(text, f):
    t = text[text.index('\n%s:\n' % f):]
    return t[:t.index('end function')]


def test_x86_this_cpu_fma_and_bit_instructions(monkeypatch):
    """With FMA, a*b + c is one vfmadd, in loops on vectors too; popcnt, lzcnt and tzcnt are one instruction each,
    their destination cleared first (a false dependency on several Intel cores)."""
    src = ('double f(double a, double b, double c) { return a * b + c; }\n'
           'void saxpy(long n, float a, const float *restrict x, float *restrict y) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }\n'
           'int pc(unsigned x) { return __builtin_popcount(x); }\n'
           'int lz(unsigned long x) { return __builtin_clzl(x); }\n'
           'int tz(unsigned x) { return __builtin_ctz(x); }\n')
    text = _x86with(src, _FEATURES, monkeypatch)
    assert re.search(r'\tvfmadd2[13]{2}sd ', _body(text, 'f')) and 'mulsd' not in _body(text, 'f')
    assert re.search(r'\tvfmadd231ps ', _body(text, 'saxpy'))
    for f, mn in (('pc', 'popcntl'), ('lz', 'lzcntq'), ('tz', 'tzcntl')):
        b = _body(text, f)
        assert re.search(r'\txorl (%\w+), \1\n\t' + mn + ' ', b), b
    base = _x86with(src, frozenset(), monkeypatch)
    assert 'vfmadd' not in base and 'popcnt' not in base and 'lzcnt' not in base and 'tzcnt' not in base


def test_x86_fma_operand_orders(monkeypatch):
    """vfmadd231 when the result shares the addend's register, 213 when a factor's, and a copy first otherwise."""
    src = ('double s(double a, double b, double c) { return a * b + c; }\n'
           'double t(double a, double b, double c) { return c + a * b; }\n'
           'double u(double c, double a, double b) { return a * b + c; }\n'
           'double v(double a, double b, double c, double *o) { *o = a; return a * b + c + a; }\n')
    text = _x86with(src, _FEATURES, monkeypatch)
    assert text.count('vfmadd') >= 4


@pytest.mark.skipif(not hosts.AMD64, reason='runs x86-64 code')
def test_x86_this_cpu_runs(monkeypatch):
    """What runs is right, with what this CPU has, and with nothing past SSE2 (CIMO_X86=baseline)."""
    from cimo import cpu
    feats = cpu.x86_features()
    src = ('double f(double a, double b, double c) { return a * b + c; }\n'
           'void saxpy(long n, float a, const float *restrict x, float *restrict y) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }\n'
           'int pc(unsigned x) { return __builtin_popcount(x); }\n'
           'int lz(unsigned x) { return __builtin_clz(x); }\n'
           'int tz(unsigned long x) { return __builtin_ctzl(x); }\n')
    m = cimo.compile(src, cache=False)
    e = 1.0 + 2 ** -30
    want = 2.0 ** -60 if 'fma' in feats else 0.0  # e*e - (1 + 2^-29): rounds away the 2^-60 unless fused
    assert m.f(e, e, -(1.0 + 2 ** -29)) == want
    for x in (0, 1, 0xff, 0x80000000, 0x12345678):
        assert m.pc(x) == bin(x).count('1')
    for x in (1, 2, 0x80000000, 0x12345678):
        assert m.lz(x) == 32 - x.bit_length()
        assert m.tz(x) == (x & -x).bit_length() - 1
    x = (ctypes.c_float * 37)(*[k * 0.5 for k in range(37)])
    y = (ctypes.c_float * 37)(*[1.0] * 37)
    m.saxpy(37, 3.0, x, y)
    assert list(y) == [1.0 + 3.0 * k * 0.5 for k in range(37)]


def test_x86_fma_negations(monkeypatch):
    """A negated factor or addend is the instruction's: vfnmadd, vfmsub, vfnmsub, with no xorps left on the path."""
    src = ('double s(double a, double b, double c) { return c - a * b; }\n'
           'double t(double a, double b, double c) { return a * b - c; }\n'
           'double u(double a, double b, double c) { return -a * b - c; }\n')
    text = _x86with(src, _FEATURES, monkeypatch)
    for f, mn in (('s', 'vfnmadd'), ('t', 'vfmsub'), ('u', 'vfnmsub')):
        b = _body(text, f)
        assert mn in b and 'xorp' not in b, b


@pytest.mark.skipif(not hosts.AMD64, reason='runs x86-64 code')
def test_x86_fma_negations_round_once():
    from cimo import cpu
    if 'fma' not in cpu.x86_features():
        pytest.skip('a CPU without FMA')
    m = cimo.compile('double s(double a, double b, double c) { return c - a * b; }\n'
                     'double t(double a, double b, double c) { return a * b - c; }\n'
                     'double u(double a, double b, double c) { return -a * b - c; }\n'
                     'double v(double a, double b, double c) { return a * b + c; }\n', cache=False)
    e = 1.0 + 2 ** -30
    c = 1.0 + 2 ** -29
    assert m.v(e, e, -c) == 2.0 ** -60  # fused: the 2^-60 the product alone rounds away
    assert m.t(e, e, c) == 2.0 ** -60
    assert m.s(e, e, c) == -2.0 ** -60
    assert m.u(e, e, -c) == -2.0 ** -60


def test_x86_no_fma_on_a_recurrence_through_its_addend(monkeypatch):
    """As gcc's avoid_fma_chains: an FMA whose addend comes round the loop costs its latency where the add's was
    shorter (dot 2.0x slower on Sapphire Rapids, iir 1.34x); through a factor it saves the add, and stays."""
    src = ('double dot(int n, const double *x, const double *y) { double s = 0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s; }\n'
           'void iir(int n, float b0, float a1, float a2, const float *x, float *y) { float y1 = 0, y2 = 0;\n'
           '  for (int i = 0; i < n; i++) { float v = b0 * x[i] - a1 * y1 - a2 * y2; y[i] = v; y2 = y1; y1 = v; } }\n'
           'double rec(int n, double a, double b) { double y = 1; for (int i = 0; i < n; i++) y = a * y + b; return y; }\n'
           'void saxpy(long n, float a, const float *restrict x, float *restrict y) { for (long i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }\n')
    text = _x86with(src, _FEATURES, monkeypatch)
    assert not re.search(r'\tvf\w+ ', _body(text, 'dot'))
    assert not re.search(r'\tvf\w+ ', _body(text, 'iir'))
    assert re.search(r'\tvfmadd\w+ ', _body(text, 'rec'))  # y as a factor: fused
    assert re.search(r'\tvfmadd231ps ', _body(text, 'saxpy'))  # nothing comes round: fused


def _select_forms():
    out = []
    for ty, ctype in (('int', ctypes.c_int32), ('unsigned', ctypes.c_uint32), ('long', ctypes.c_int64), ('unsigned long', ctypes.c_uint64)):
        for op in ('>', '>=', '<', '<='):
            for cond in ('x %s r', 'r %s x'):
                for arms in ('x : r', 'r : x'):
                    out.append((ty, ctype, 'r = (%s) ? %s;' % (cond % op, arms)))
    return out


SELECT_FORMS = _select_forms()


@pytest.mark.parametrize('k', range(len(SELECT_FORMS)))
def test_reduction_through_a_condition(k):
    """r = x > r ? x : r and every other way to write an integer maximum or minimum: four vectors of lanes, each its
    own, then the lanes into r through the same comparison. Equal integers are the same, so this is the scalar result."""
    ty, ctype, body = SELECT_FORMS[k]
    src = '%s f(const %s *a, long n, %s r) { for (long i = 0; i < n; i++) { %s x = a[i]; %s } return r; }' % (ty, ty, ty, ty, body)
    assert vectorized(src), driver._build(src)[0]
    on, off = both(src)
    bits = 8 * ctypes.sizeof(ctype)
    lo, hi = (0, 2 ** bits - 1) if ty.startswith('unsigned') else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    r = random.Random(k)
    for n in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100):
        for _ in range(3):
            a = (ctype * max(n, 1))(*[r.choice([lo, hi, 0, 1, lo + 1, hi - 1, r.randint(lo, hi)]) for _ in range(n)])
            r0 = r.choice([lo, hi, 0, r.randint(lo, hi)])
            assert on.f(a, n, r0) == off.f(a, n, r0), (n, list(a)[:n], r0)


def test_float_maximum_stays_scalar():
    """Two equal zeros or a NaN: lanes merged in their order needn't pick what the scalar loop does."""
    assert not vectorized('double f(const double *a, long n, double r) { for (long i = 0; i < n; i++) if (a[i] > r) r = a[i]; return r; }')


STRIDED = {
    'max': 'long f(const long *a, long n, long st) { long r = a[0]; for (long i = 1; i < n; i++) if (a[i * st] > r) r = a[i * st]; return r; }',
    'sum': 'long f(const int *a, long n, int st) { long s = 0; for (long i = 0; i < n; i++) s += a[i * st]; return s; }',
}


@pytest.mark.parametrize('name', sorted(STRIDED))
def test_unit_stride_is_versioned(name):
    """a[i * st] gets a vector loop for st == 1 (cimo.middle.stride), and the general loop still runs for other strides."""
    src = STRIDED[name]
    assert vectorized(src), driver._build(src)[0]
    on, off = both(src)
    itemsize = 8 if name == 'max' else 4
    ctype = ctypes.c_int64 if name == 'max' else ctypes.c_int32
    for st in (1, 2, 3, 0, -1):
        for n in (1, 2, 3, 4, 7, 8, 9, 17, 64, 101):
            span = abs(st) * n + 8
            a = buf(ctype, 2 * span + 16, n + st)
            start = ctypes.addressof(a) + itemsize * (span + 4 if st < 0 else 4)
            assert on.f(start, n, st) == off.f(start, n, st), (st, n)


def test_strided_store_is_versioned():
    src = 'void f(double *restrict o, const double *restrict a, long n, long st) { for (long i = 0; i < n; i++) o[i] = a[i * st] * 2; }'
    assert vectorized(src)
    on, off = both(src)
    for st in (1, 2, 0, -1):
        for n in (0, 1, 3, 4, 5, 9, 33):
            span = abs(st) * n + 8
            a = buf(ctypes.c_double, 2 * span + 16, n)
            start = ctypes.addressof(a) + 8 * (span + 4 if st < 0 else 4)
            o1, o2 = buf(ctypes.c_double, n + 1, 7), buf(ctypes.c_double, n + 1, 7)
            on.f(ctypes.addressof(o1), start, n, st)
            off.f(ctypes.addressof(o2), start, n, st)
            assert bytes(o1) == bytes(o2), (st, n)
