"""Common subexpressions QBE leaves behind (cimo.backend.cse, ext.cse): what gets merged, and that
merging keeps the results, on the paths where a merge would be wrong if it went too far."""

import re

import pytest

import cimo
from cimo import driver
import hosts

pytestmark = pytest.mark.skipif(not hosts.ARM64, reason='arm64 code generation')


def body(src, name):
    asm = driver._build(src)[0]
    m = re.search(r'^%s:\n(.*?)^/\* end function' % name, asm, re.S | re.M)
    assert m, name
    return m.group(1)


def test_repeated_loads_merge():
    b = body('double f(double *p, long i) { return p[i] + p[i] * 3 + p[i]; }', 'f')
    assert len(re.findall(r'\bldr\s+d\d+, \[x', b)) == 2  # p[i], and the constant 3


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
