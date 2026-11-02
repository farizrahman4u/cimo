"""C's promise that signed +, - and * don't overflow (cimo.backend.nowrap): what it allows, and everything
that must not get it. The fuzzers generate no overflow, so a mark on something that may wrap only shows here."""

import ctypes
import re

import pytest

import cimo
from cimo import driver

INT_MAX = 2**31 - 1
INT_MIN = -2**31
UINT_MAX = 2**32 - 1


def asm(src):
    return driver._build(src)[0]


def body(text, name):
    m = re.search(r'^%s:\n(.*?)^/\* end function' % name, text, re.S | re.M)
    return m.group(1)


@pytest.mark.parametrize('src,folded', [
    ('int f(int x) { return x + 1 > x; }', True),
    ('int f(int x) { return x - 1 < x; }', True),
    ('int f(int x) { return x + 2 <= x + 1; }', True),
    ('long f(long x) { return x + 1 > x; }', True),
    ('int f(int x) { return x + 1 > x + 1 - 1; }', False),  # gvn made this x + 1 > x, without the promise for the sum
])
def test_compare_known(src, folded):
    if not folded:
        return
    assert 'cmp' not in body(asm(src), 'f')


@pytest.mark.parametrize('src,arg,expect', [
    ('int f(unsigned x) { return x + 1 > x; }', UINT_MAX, 0),
    ('int f(int x) { return (int)((unsigned)x + 1) > x; }', INT_MAX, 0),
    ('int f(unsigned x) { return x - 1 < x; }', 0, 0),
    ('int f(unsigned long x) { return x + 1 > x; }', 2**64 - 1, 0),
    ('int f(unsigned x) { return (x * 4) / 4 == x; }', 2**31, 0),
    ('unsigned f(unsigned x) { return (x * 4) / 4; }', 2**31, 0),
    ('int f(int x) { return (int)((unsigned)x * 4) / 4; }', 2**30, 0),
    ('int f(unsigned x) { return x - 1 > 5u; }', 0, 1),
])
def test_unsigned_wraps(src, arg, expect):
    assert cimo.compile(src, cache=False).f(arg) == expect


def test_small_types_compute_in_int():
    m = cimo.compile('''
    int cinc(signed char c) { c++; return c; }
    int cadd(signed char c) { c = c + 1; return c > 0; }
    int sadd(short s) { short t = s + 1; return t > s; }
    int ucmp(unsigned char c) { return c + 1 > c; }
    int bits(int x) { struct { int f : 3; } s; s.f = x; s.f = s.f + 1; return s.f; }
    ''', cache=False)
    assert m.cinc(127) == -128
    assert m.cadd(127) == 0
    assert m.sadd(32767) == 0
    assert m.ucmp(255) == 1  # 256 in int
    assert m.bits(3) == -4


def test_pointers_have_no_promise():
    m = cimo.compile('''
    long diff(char *p, char *q) { return p - q; }
    int high(char *p) { return p + 1 > p; }
    ''', cache=False)
    assert m.diff(1000, 1) == 999


def test_wrapv():
    src = '''int gt(int x) { return x + 1 > x; }
    int d(int x) { return (x * 4) / 4; }
    long w(int *a, int n) { long s = 0; for (int i = n; i != n + 2; i++) s += (long)i; return s; }
    '''
    m = cimo.compile(src, wrapv=True, cache=False)
    assert m.gt(INT_MAX) == 0
    assert m.d(2**30) == 0
    assert m.w(0, INT_MAX) == INT_MAX + INT_MIN
    assert cimo.compile(src, cache=False).gt(INT_MAX) == 1
    assert cimo.compile(src, wrapv=True, cache=False).gt(INT_MAX) == 0  # not the other's cache entry


def test_merged_with_one_without_promise():
    """gvn makes both sums one instruction, and gcm may put it on either path: the promise is gone."""
    m = cimo.compile('''
    int f(int x, int c) {
        if (c) { int y = x + 1; return y > x; }
        int z = (int)((unsigned)x + 1);
        return z > x;
    }
    int g(int x, int c) {
        int z = (int)((unsigned)x + 1);
        if (c) return z > x;
        return x + 1 > x;
    }
    ''', cache=False)
    assert m.f(INT_MAX, 0) == 0
    assert m.g(INT_MAX, 1) == 0


@pytest.mark.parametrize('src,folded', [
    ('int f(int x, int y) { return x + 3 < y + 3; }', True),
    ('int f(int x, int y) { return x - y < 0; }', True),
    ('int f(int x, int y) { return x - y <= 0; }', True),
    ('int f(int x, int y) { return x - y > 0; }', True),
    ('int f(int x, int y) { return x - y >= 0; }', True),
    ('int f(int x, int y) { return x - y < 1; }', True),
    ('int f(int x, int y) { return x - y > -1; }', True),
    ('int f(int x, int y) { return x * 5 < y * 5; }', True),
    ('int f(int x, int y) { return x * -5 < y * -5; }', True),
])
def test_compares_same_as_without_the_arithmetic(src, folded):
    text = body(asm(src), 'f')
    assert 'add' not in text and 'sub\t' not in text and 'mul' not in text and 'madd' not in text
    m = cimo.compile(src, cache=False)
    w = cimo.compile(src, wrapv=True, cache=False)
    for x, y in ((0, 0), (1, 2), (2, 1), (-1, 1), (1, -1), (-100, -3), (50, 50), (1000, -1000)):
        assert m.f(x, y) == w.f(x, y), (x, y)


@pytest.mark.parametrize('expr,args', [
    ('x + 5 < 10', (4, 5, 6, -100, 100)),
    ('x + 5 >= 10', (4, 5, 6, -100, 100)),
    ('10 > x + 5', (4, 5, 6, -100, 100)),
    ('x - 5 > -2147483647', (0, 3, 100)),
    ('x * 4 < 0', (-1, 0, 1)),
    ('x * -4 < 0', (-1, 0, 1)),
    ('x * 4 >= 0', (-1, 0, 1)),
    ('x * 6 / 2', (-7, 0, 7)),
    ('x * 4 / 4', (-7, 0, 7)),
    ('x * 4 % 4', (-7, 0, 7)),
    ('x * 8 >> 3', (-7, 0, 7)),
    ('x * 12 >> 2', (-7, 0, 7)),
    ('x * 7 / 2', (-7, 0, 7)),
    ('x * -3 / 3', (-7, 0, 7)),
    ('x * 3 / -3', (-7, 0, 7)),
])
def test_value(expr, args):
    src = 'int f(int x) { return %s; }' % expr
    m = cimo.compile(src, cache=False)
    w = cimo.compile(src, wrapv=True, cache=False)
    for x in args:
        assert m.f(x) == w.f(x), x


LOOPS = {
    'up': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }',),
    'down': ('long f(int *a, int n) { long s = 0; for (int i = n - 1; i >= 0; i--) s += a[i]; return s; }',),
    'two': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i += 2) s += a[i]; return s; }',),
    'neg': ('long f(int *a, int n) { long s = 0; for (int i = -3; i < n; i++) s += i < 0 ? i : a[i]; return s; }',),
    'next': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n - 1; i++) s += a[i + 1]; return s; }',),
    'left': ('long f(int *a, int n) { int i; long s = 0; for (i = 0; i < n; i++) s += a[i]; return s + i; }',),
    'read': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i++) { s += a[i]; s ^= i * 3; } return s; }',),
    'uns': ('long f(int *a, int n) { long s = 0; for (int i = -2; i < n; i++) s += (unsigned)i; return s; }',),
    'ulong': ('long f(int *a, int n) { long s = 0; for (int i = -2; i < n; i++) s += (long)(unsigned)i >> 3; return s; }',),
    'ucmp': ('long f(int *a, int n) { long s = 0; for (int i = -2; (unsigned)(i + 2) < (unsigned)n; i++) s += (long)i; return s; }',),
    'cont': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i++) { if (a[i] & 1) continue; s += a[i]; } return s; }',),
    'nest': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < 4; i++) for (int j = 0; j < n / 4; j++) s += a[i * (n / 4) + j]; return s; }',),
    'init': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i++) for (int j = i; j < n; j += 3) s += a[j]; return s; }',),
    'ne': ('long f(int *a, int n) { long s = 0; for (int i = 0; i != n; i++) s += a[i]; return s; }',),
    'while': ('long f(int *a, int n) { int i = 0; long s = 0; while (i < n && a[i] != -1) { s += a[i]; i++; } return s; }',),
    'twoiv': ('long f(int *a, int n) { long s = 0; int j = n - 1; for (int i = 0; i < j; i++, j--) s += a[i] * a[j]; return s; }',),
    'char': ('long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i++) s += (signed char)(i * 37) + a[(unsigned char)i % 8]; return s; }',),
    'swap': ('long f(int *a, int n) { long s = 0; for (int i = 0; n > i; i++) s += a[i]; return s; }',),
    'gt': ('long f(int *a, int n) { long s = 0; for (int i = n; i > 0; i--) s += a[i - 1]; return s; }',),
    'storei': ('long f(int *a, int n) { int b[64]; long s = 0; for (int i = 0; i < n && i < 64; i++) b[i] = i; for (int i = 0; i < n && i < 64; i++) s += b[i] * (long)i; return s; }',),
}


@pytest.mark.parametrize('name', sorted(LOOPS))
def test_loops_same_with_and_without_the_promise(name):
    src = LOOPS[name][0]
    data = (ctypes.c_int * 100)(*[(i * 37 + 5) % 23 - 7 for i in range(100)])
    p = ctypes.addressof(data)
    m = cimo.compile(src, cache=False)
    w = cimo.compile(src, wrapv=True, cache=False)
    for n in (0, 1, 2, 3, 4, 7, 8, 9, 16, 17, 63, 64, 100):
        assert m.f(p, n) == w.f(p, n), n


@pytest.mark.parametrize('src', [
    'long f(int *a, int n) { int i = 0; long s = 0; while (a[i] != 0 && i < n) { s += i; i += 2; } return s; }',
    'long f(int *a, int n, long (*g)(long)) { long s = 0; for (int i = 0; i < n; i++) s += g(a[i]) + a[i + 1]; return s; }',
    'long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i++) { if (a[i] > 3) s += a[i + 1]; else s -= a[i + 2]; } return s; }',
])
def test_no_extension_in_the_loop(src):
    text = body(asm(src), 'f')
    start = re.search(r'^\.L\d+:', text, re.M).start()
    assert 'sxtw' not in text[start:]


def test_extension_stays_when_the_step_may_wrap():
    src = '''long f(int *a, int n) { long s = 0; for (int i = 0; i < n; i = (int)((unsigned)i + 1)) s += a[i]; return s; }
    long g(int n) { long s = 0; int i = 2147483646; for (int k = 0; k < n; k++) { s += (long)i; i = (int)((unsigned)i + 1); } return s; }'''
    m = cimo.compile(src, cache=False)
    assert m.g(3) == INT_MAX - 1 + INT_MAX + INT_MIN
