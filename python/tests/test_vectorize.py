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
    'fsum': ('float f(const float *restrict x, long n) { float s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_float, False),
    'dsum': ('double f(const double *restrict x, long n) { double s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }', ctypes.c_double, False),
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
        extra = (ctypes.addressof(y),) if name == 'seen' else ()
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
def test_x86_code_runs_on_any_x86_64(target):
    """No CPU check guards these, so nothing past SSE2; the instructions below are what's easiest to reach for."""
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
    assert text.count('pcmpgtd') == 1
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
