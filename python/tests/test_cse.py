"""Common subexpressions QBE leaves behind (cimo.backend.cse, ext.cse): what gets merged, and that
merging keeps the results, on the paths where a merge would be wrong if it went too far."""

import array
import re

import pytest

import cimo
from cimo import driver


def body(src, name):
    asm = driver._build(src)[0]
    m = re.search(r'^%s:\n(.*?)^/\* end function' % name, asm, re.S | re.M)
    assert m, name
    return m.group(1)


def test_repeated_loads_merge():
    b = body('double f(double *p, long i) { return p[i] + p[i] * 3 + p[i]; }', 'f')
    assert len(re.findall(r'\bldr\s+d\d+, \[x', b)) == 1, b  # p[i]; the constant 3 is an fmov


def test_a_store_between_keeps_the_second_load():
    src = 'long f(long *p, long *q, long i) { long a = p[i]; q[0] = 5; return a + p[i]; }'
    assert len(re.findall(r'\bldr\s', body(src, 'f'))) == 2
    import array
    x = array.array('q', [1, 2])
    assert cimo.compile(src, cache=False).f(x, x, 0) == 6  # p and q are the same array


def test_division_and_remainder_share_one_sdiv():
    src = 'long f(long a, long b) { return a % b + a / b + a % b * (a / b); }'
    assert body(src, 'f').count('sdiv') == 1
    m = cimo.compile(src, cache=False)
    for a, b in ((17, 5), (-17, 5), (17, -5), (-17, -5), (0, 3)):
        q = int(a / b)
        r = a - q * b
        assert m.f(a, b) == r + q + r * q, (a, b)


@pytest.mark.parametrize('a,b,want', [(-2 ** 63, -1, -2 ** 63), (5, 0, 5), (2 ** 63 - 1, 1, 2 ** 63 - 1)])
def test_division_edge_cases_on_arm64(a, b, want):
    """What sdiv + msub give: INT_MIN / -1 wraps and b == 0 gives 0 (undefined in C, but it must
    not change between a remainder alone and one sharing a division)."""
    m = cimo.compile('long f(long a, long b) { return a / b + a % b; }', cache=False)
    assert m.f(a, b) == want


def test_sibling_divisions_merge():
    src = '''long f(long a, long b, int c) { long x;
        if (c) x = a / b + 1; else x = a / b + 2;
        return x; }'''
    assert body(src, 'f').count('sdiv') == 1
    m = cimo.compile(src, cache=False)
    assert m.f(9, 3, 1) == 4 and m.f(9, 3, 0) == 5


CHECK = '''long f(long y, long n, long *a) {
    long i = y < 0 ? y + n : y;
    if ((unsigned long)i >= (unsigned long)n) return -1;
    long s = a[i];
    long j = y < 0 ? y + n : y;
    if ((unsigned long)j >= (unsigned long)n) return -2;
    return s + a[j];
}'''


def test_repeated_bounds_check_goes():
    b = body(CHECK, 'f')
    assert not re.search(r'#-2\b', b), b  # the second check's failure path is gone
    import array
    x = array.array('q', [10, 20, 30])
    m = cimo.compile(CHECK, cache=False)
    assert m.f(1, 3, x) == 40 and m.f(-1, 3, x) == 60 and m.f(3, 3, x) == -1


def test_a_join_does_not_decide_a_condition():
    """A branch's successor that other paths reach too says nothing about the condition (this
    miscompiled strtod in the first version)."""
    m = cimo.compile('''long f(long a, long b) {
        int c = a > b;
        long s = 0;
        if (b == 7) goto join;
        if (c) {
    join:
            s += 1;
            if (c) s += 10;
        }
        return s;
    }''', cache=False)
    assert m.f(1, 7) == 1 and m.f(9, 7) == 11 and m.f(9, 2) == 11 and m.f(1, 2) == 0


def test_a_condition_changing_around_a_loop():
    m = cimo.compile('''long f(long *p, long n) {
        long s = 0;
        for (long i = 0; i < n; i++) {
            int c = p[i] > 0;
            if (c) s += 1;
            if (c) s += 10; else s += 100;
        }
        return s;
    }''', cache=False)
    import array
    assert m.f(array.array('q', [1, -1, 2]), 3) == 11 + 100 + 11


def test_selects_differ_by_condition_or_value():
    m = cimo.compile('''long f(long a, long b) {
        long x = a < 0 ? b : a;
        long y = a < 1 ? b : a;
        long z = a < 0 ? a : b;
        return x * 10000 + y * 100 + z;
    }''', cache=False)
    assert m.f(0, 5) == 0 * 10000 + 5 * 100 + 5
    assert m.f(-1, 5) == 5 * 10000 + 5 * 100 - 1


def test_volatile_loads_stay():
    b = body('long f(volatile long *p) { return *p + *p; }', 'f')
    assert len(re.findall(r'\bldr\s', b)) == 2


def test_wasm_keeps_a_division_that_may_not_run():
    """wasm's division traps: one under a condition must not move out of it."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasm'))
    from host import Host, available
    if not available():
        pytest.skip('needs node')
    import cimo.wasm as W
    h = Host()
    try:
        mod, _ = W.build('int f(int a, int b) { int s = 0; if (b) s = a / b; if (b) s += a / b; return s; }')
        h.load(mod)
        assert h.call('f', 'w,w', 9, 3)[0] == 6 and h.call('f', 'w,w', 9, 0)[0] == 0
    finally:
        h.close()


DIVISORS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 100, 125, 127, 128, 255, 256, 641, 1000, 1024, 3600,
            65535, 65536, 65537, 86400, 1000000, 1000000007, 2147483647, 1 << 30]
_EDGE32 = [0, 1, 2, 3, 6, 7, 9, 10, 99, 100, 127, 128, 255, 256, 999, 1000, 65535, 65536, 2 ** 30,
           2 ** 31 - 2, 2 ** 31 - 1]
_EDGE64 = _EDGE32 + [2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 33 + 1, 10 ** 15, 2 ** 62, 2 ** 63 - 2, 2 ** 63 - 1]


def _tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _kernels(ty, name, divisors):
    return ''.join('%s %sd%d(%s a) { return a / %s; }\n%s %sr%d(%s a) { return a %% %s; }\n'
                   % (ty, name, n, ty, lit, ty, name, n, ty, lit) for n, lit in divisors)


@pytest.mark.parametrize('bits,signed', [(32, True), (32, False), (64, True), (64, False)])
def test_division_by_constants(bits, signed):
    """Against Python's integers: each divisor, of both signs, on the edges of each range and on
    multiples of the divisor and their neighbours."""
    import random
    ty = {(32, True): 'int', (32, False): 'unsigned', (64, True): 'long', (64, False): 'unsigned long'}[bits, signed]
    ds = [d for d in DIVISORS] + ([2 ** 62, 2 ** 63 - 1, 10 ** 18, 2 ** 40 + 1] if bits == 64 else [])
    if not signed:
        ds += [2 ** (bits - 1), 2 ** (bits - 1) + 1, 2 ** bits - 1, 2 ** bits - 2, 3 * 2 ** (bits - 2)]
    suffix = 'L' if bits == 64 else ''
    suffix = ('U' + suffix) if not signed else suffix
    items = [(d, '%d%s' % (d, suffix)) for d in ds]
    src = _kernels(ty, 'p', items)
    if signed:
        src += _kernels(ty, 'm', [(d, '(-%d%s)' % (d, suffix)) for d in ds])
    asm = driver._build(src)[0]
    assert not re.search(r'\b[su]div\b', asm), 'a division is left'
    m = cimo.compile(src, cache=False)
    rnd = random.Random(bits * 2 + signed)
    edge = _EDGE64 if bits == 64 else _EDGE32
    lo, hi = (-2 ** (bits - 1), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    wrap = lambda v: ((v + 2 ** (bits - 1)) % 2 ** bits) - 2 ** (bits - 1)  # as the binding returns
    for d in ds:
        xs = set(edge + [hi, hi - 1, lo, lo + 1])
        for _ in range(30):
            xs.add(rnd.randint(lo, hi))
            q = rnd.randint(0, hi // d)
            xs.update((q * d, q * d - 1, q * d + 1))
        if signed:
            xs |= {-x for x in xs}
        for x in xs:
            if not lo <= x <= hi:
                continue
            t = _tdiv(x, d)
            r = x - t * d
            dv, rv = getattr(m, 'pd%d' % d), getattr(m, 'pr%d' % d)
            assert wrap(dv(x)) == wrap(t), (ty, x, d)
            assert wrap(rv(x)) == wrap(r), (ty, x, '%', d)
            if signed and not (x == lo and d == 1):
                assert wrap(getattr(m, 'md%d' % d)(x)) == wrap(_tdiv(x, -d)), (ty, x, -d)
                assert wrap(getattr(m, 'mr%d' % d)(x)) == wrap(x - _tdiv(x, -d) * -d), (ty, x, '%', -d)


def test_a_quotient_and_remainder_share_their_work():
    src = 'long f(long a) { return a / 10 * 1000 + a % 10; }'
    b = body(src, 'f')
    assert 'sdiv' not in b and b.count('smulh') == 1, b
    m = cimo.compile(src, cache=False)
    assert m.f(1234) == 123001 + 3 and m.f(-1234) == -123000 - 4


def test_divisors_that_stay_divisions():
    m = cimo.compile('''long f(long a) { return a / 1; }
        long g(long a) { return a / -1; }
        unsigned h(unsigned a) { return a / 1; }
        long i(long a, long b) { return a / b; }''', cache=False)
    assert m.f(7) == 7 and m.g(7) == -7 and m.g(-2 ** 63) == -2 ** 63 and m.h(9) == 9 and m.i(9, 2) == 4


def test_a_load_moved_up_a_loop_keeps_its_offset():
    """gcc-torture's 20011008-3: loading a field at an offset, the load moved to the loop's
    predecessor read the start of the object (it hung on Windows, where that was garbage)."""
    src = '''struct S { int first; int pad[5]; int n; int *p; };
    int f(struct S *e, unsigned flags) { int val = 0;
        for (int i = 0; i < (!(flags & 1) ? 1 : e->n); i++) {
            val++;
            for (int j = 0; j < e->n - 1; j++) val += 100;
        }
        return val; }
    long g(struct S *e, int k) { long s = 0;
        if (k) s = e->n;
        for (int i = 0; i < 3; i++) s += e->n;
        return s; }'''
    b = body(src, 'f')
    assert not re.search(r'ldr\s+w\d+, \[x\d+\]', b), b  # n is at offset 24, nothing at 0 is read
    import struct
    buf = bytearray(struct.pack('<iiiiiii', 1000000, 0, 0, 0, 0, 0, 3) + bytes(16))
    arr = (__import__('ctypes').c_char * len(buf)).from_buffer(buf)
    m = cimo.compile(src, cache=False)
    assert m.f(arr, 1) == 3 + 3 * 2 * 100 and m.f(arr, 0) == 1 + 2 * 100
    assert m.g(arr, 1) == 12 and m.g(arr, 0) == 9


def test_a_struct_copy_forwarded_at_an_offset():
    """A load from a blit's destination reads the source at the same offset."""
    m = cimo.compile('''struct T { long a, b, c, d; };
    struct T g = {1, 2, 3, 4};
    long f(struct T *q, int k) { struct T t = *q; long s = 0;
        if (k) s = t.b;
        for (int i = 0; i < 2; i++) s += t.c * 10 + t.d;
        return s; }
    long h(int k) { struct T t = g; long s = 0;
        if (k) s = t.b;
        for (int i = 0; i < 2; i++) s += t.c * 10 + t.d;
        return s; }''', cache=False)
    import array
    x = array.array('q', [100, 200, 300, 400])
    assert m.f(x, 1) == 200 + 2 * 3400 and m.f(x, 0) == 2 * 3400
    assert m.h(1) == 2 + 2 * 34 and m.h(0) == 2 * 34


def test_a_large_multiplier_is_built_out_of_the_loop():
    src = 'void f(const long *x, long n, long *y) { for (long i = 0; i < n; i++) y[i] = x[i] / 7; }'
    asm = body(src, 'f')
    first = asm.index('smulh')
    assert 'movk' not in asm[first:], asm  # built before the first division, not per copy
    assert asm.count('movk') == 3, asm
    import array
    x = array.array('q', [i * 13 - 50 for i in range(23)])
    y = array.array('q', [0] * 23)
    cimo.compile(src, cache=False).f(x, 23, y)
    assert list(y) == [_tdiv(v, 7) for v in x]


def test_a_large_multiplier_with_many_loops_and_calls():
    """It lives across calls and in nested loops, and also in functions that never reach it."""
    m = cimo.compile('''long g(long a) { return a + 1; }
    long f(const long *x, long n, int skip) {
        long s = 0;
        if (skip) return -1;
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < 3; j++) s += g(x[i] + j) / 1000003;
            s -= x[i] * 0x123456789abcdefL;
        }
        return s; }''', cache=False)
    import array
    x = array.array('q', [10 ** 12, -7 * 10 ** 9, 5])
    want = 0
    for v in x:
        for j in range(3):
            want += _tdiv(v + j + 1, 1000003)
        want -= v * 0x123456789abcdef
    want = ((want + 2 ** 63) % 2 ** 64) - 2 ** 63
    assert m.f(x, 3, 0) == want and m.f(x, 3, 1) == -1


@pytest.mark.parametrize('p,q,want', [(2, 1, 36 - 8000), (1, 2, 36 + 800)])
def test_a_condition_only_a_select_reads(p, q, want):
    """Its definition stayed: the select's fallback (compare with 0) counts as a use of it."""
    m = cimo.compile('''int g;
    long f(int *a, int n, int p, int q, int x, int y) {
        int c = p < q; if (c) g = 1;
        long s = 0;
        for (int i = 0; i < n; i++) { int v = a[i]; int r; if (c) r = v + x; else r = v - y; s += r; }
        return s; }
    int arr[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    long t(int p, int q) { return f(arr, 8, p, q, 100, 1000); }''', cache=False)
    assert m.t(p, q) == want


def test_a_product_that_is_also_a_select_condition():
    m = cimo.compile('''int f(int x, int y, int z, int a, int b) {
        int c = x * y; int s = z + c; int r;
        if (c) r = a + s; else r = b - s;
        return r; }''', cache=False)
    assert m.f(0, 5, 1, 100, 200) == 199 and m.f(2, 5, 1, 100, 200) == 111


def test_volatile_accesses_keep_their_order():
    b = body('long f(volatile long *a, volatile long *b) { long x = a[0]; long y = b[0]; long z = a[1]; return x + y + z; }', 'f')
    assert 'ldp' not in b, b
    assert re.findall(r'ldr\s+\w+, \[(x\d+)(?:, (\d+))?\]', b)[2][1] == '8', b
    b = body('void g(volatile int *a) { a[0] = 1; a[2] = 2; a[1] = 3; }', 'g')
    assert 'stp' not in b, b
    assert [o or '0' for o in re.findall(r'str\s+\w+, \[x\d+(?:, (\d+))?\]', b)] == ['0', '8', '4'], b


@pytest.mark.parametrize('src,args,want', [
    ('float f(float *p, int n) { float s = 0; int i = 0; do { s += *p; i++; } while (i < n); return s; }',
     (array.array('f', [1.5]), 3), 4.5),
    ('double f(double *p, int n) { double s = 0; int i = 0; do { s += *p; i++; } while (i < n); return s; }',
     (array.array('d', [1.5]), 3), 4.5),
    ('struct V { float x, y; }; float f(struct V *v, int n) { float s = 0; do { s += v->x * v->y; } while (--n > 0); return s; }',
     (array.array('f', [1.5, 2.0]), 3), 9.0),
    ('struct V { double x; float y; }; double f(struct V *v, int n) { double s = 0; for (;;) { s += v->x + v->y; if (--n <= 0) return s; } }',
     (array.array('d', [1.5, 0.0]), 2), 3.0),
])
def test_a_float_load_moved_up_a_loop(src, args, want):
    assert cimo.compile(src, cache=False).f(*args) == want


def test_a_float_load_moved_up_a_loop_on_wasm():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasm'))
    from host import Host, available
    if not available():
        pytest.skip('needs node')
    import cimo.wasm as W
    h = Host()
    try:
        h.load(W.build('float f(float *p, int n) { float s = 0; int i = 0; do { s += *p; i++; } while (i < n); return s; }')[0])
        assert h.call('f', 'l,w', h.array('f', [1.5]), 3)[0] == 4.5
    finally:
        h.close()


BLIT = '''
data $G = { w 7, w 8, w 9, w 10 }
export
function w $f(w %%n) {
@start
	%%a =l alloc8 16
	blit $G, %%a, %d
	storew 99, $G
	jmp @h
@h
	%%i =w phi @start 0, @b %%i1
	%%v =%s load%s %%a
	%%c =w csltw %%i, %%n
	jnz %%c, @b, @e
@b
	%%i1 =w add %%i, 1
	jmp @h
@e
	ret %%i
}
'''


@pytest.mark.parametrize('size,k', [(16, 'w'), (7, 'l'), (5, 'l'), (3, 'w')])
def test_a_load_through_a_blit_in_a_loop(size, k):
    """From IR text (the frontend emits no blit today): no load of the source below the blit, which
    a store has changed, and no load of a size that doesn't exist."""
    from cimo.backend import defs as D, ext, parse
    from cimo.backend.main import _Driver, _targets
    import io
    D.T = next(t for t in _targets() if t.name == 'arm64')
    out = io.StringIO()
    with ext.options(cse=True, isel=True):
        drv = _Driver(out, 0)
        parse.parse((BLIT % (size, k, k)).encode(), 'x', drv.dbgfile, drv.data, drv.func)
        drv.finish()
    assert 'f:' in out.getvalue()
