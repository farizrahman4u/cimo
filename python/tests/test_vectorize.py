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

pytestmark = pytest.mark.skipif(not hosts.ARM64, reason='arm64 only so far')


def vectorized(src):
    return bool(re.search(r'\t(ldr|str)\tq\d+', driver._build(src)[0]))


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
@pytest.mark.parametrize('ty,ctype', [('float', ctypes.c_float), ('double', ctypes.c_double), ('int', ctypes.c_int), ('long', ctypes.c_long)])
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
    'long': ('void f(long *restrict y, const long *restrict x, long a, long n) { for (long i = 0; i < n; i++) y[i] = ((x[i] + a) ^ (y[i] >> 3)) - ((unsigned long)x[i] >> 7); }', ctypes.c_long),
    'stencil': ('void f(double *restrict y, const double *restrict x, double a, long n) { for (long i = 2; i < n - 2; i++) y[i] = a * (x[i - 2] + x[i - 1] + x[i] + x[i + 1] + x[i + 2]); }', ctypes.c_double),
    'int_counter': ('void f(float *restrict y, const float *restrict x, float a, int n) { for (int i = 0; i < n; i++) y[i] = x[i] * x[i] + a; }', ctypes.c_float),
}


@pytest.mark.parametrize('name', sorted(KERNELS))
def test_bit_for_bit_the_scalar_loop(name):
    src, ctype = KERNELS[name]
    assert vectorized(src), driver._build(src)[0]
    on, off = both(src)
    a = 3 if ctype in (ctypes.c_int, ctypes.c_long) else 1.25
    for n in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200):
        x = buf(ctype, n + 8, n)
        y1, y2 = buf(ctype, n + 8, n + 1), buf(ctype, n + 8, n + 1)
        on.f(ctypes.addressof(y1), ctypes.addressof(x), a, n)
        off.f(ctypes.addressof(y2), ctypes.addressof(x), a, n)
        assert bytes(y1) == bytes(y2), n


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
