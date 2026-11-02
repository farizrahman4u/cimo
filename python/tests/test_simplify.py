"""The simplifier and known bits (cimo.backend.simplify): the rules that fired, and random expressions
evaluated by compiled code and by Python's integers."""

import random
import re

import pytest
import ctypes
import array

import cimo
from cimo import driver


def body(src, name='f'):
    asm = driver._build(src)[0]
    m = re.search(r'^_?%s:\n(.*?)(?:^\.type|^/\* end function)' % name, asm, re.S | re.M)
    assert m, name
    return m.group(1)


def nins(src, name='f'):
    return len([l for l in body(src, name).split('\n') if l.strip() and not l.strip().startswith(('hint', '.'))])


@pytest.mark.parametrize('src', [
    'long f(long a, long b) { return (a + b) - b; }',
    'long f(long a) { return -(-a); }',
    'long f(long a, long b) { return (a & b) | (a & ~b); }',
    'long f(long a) { return a / 1; }',
    'long f(long a) { return a * 1; }',
    'long f(long a) { return a == 0 ? 0 : a; }',
    'long f(long a, long b) { return a != b ? a : b; }',
    'double f(double a) { return -(-a); }',
    'double f(double a) { return a * 1.0; }',
])
def test_nothing_left(src):
    assert nins(src) == 1, body(src)  # ret


@pytest.mark.parametrize('src', [
    'long f(long a, long b) { return (a + b) - a; }',
    'long f(long a, long b) { return a - (a - b); }',
])
def test_only_a_move_left(src):
    assert nins(src) == 2 and 'mov' in body(src), body(src)


@pytest.mark.parametrize('src,value', [
    ('int f(unsigned char c) { return c < 300; }', 1),
    ('int f(int x) { return (x & 7) < 8; }', 1),
    ('long f(long a) { return (a & 255) >> 9; }', 0),
    ('long f(long a) { return (a | 15) & 15; }', 15),
    ('long f(long a) { return a * 0; }', 0),
    ('long f(long a) { return a % 1; }', 0),
    ('int f(unsigned a) { return a >= 0; }', 1),
    ('int f(unsigned a) { return a < 0; }', 0),
    ('long f(long a) { return a - a; }', 0),
    ('long f(long a) { return a & ~a; }', 0),
    ('long f(long a) { return a | ~a; }', -1),
    ('int f(int a) { return (a & 3) == 4; }', 0),
    ('int f(int a) { return (a | 1) == 0; }', 0),
    ('int f(unsigned short a) { return (a >> 3) <= 8191; }', 1),
])
def test_folds_to_a_constant(src, value):
    assert nins(src) == 2, body(src)  # mov, ret
    m = cimo.compile(src, cache=False)
    assert m.f(*([5] * len(m.f._params))) == value


@pytest.mark.parametrize('src,want', [
    ('long f(long a, long b) { return (a | b) - (a & b); }', r'eor\s'),
    ('int f(long a) { return !(a < 3); }', r'cset\s+w0, ge'),
    ('int f(int a, int b) { return !!(a & b); }', r'tst\s|cset\s+w0, ne'),
    ('int f(long a, long b) { return a + 3 == b + 3; }', r'cmp\s+x0, x1'),
    ('int f(long a) { return a + 3 == 10; }', r'cmp\s+x0, #7'),
    ('long f(unsigned long a) { return (a >> 3) << 3; }', r'and\s'),
    ('long f(long a) { return a + a + a + a; }', r'lsl\s+x0, x0, #2'),
])
def test_rewritten(src, want):
    assert re.search(want, body(src)), body(src)


def test_no_rule_needs_no_overflow():
    """IR integers wrap: these must stay computed, not assume overflow can't happen."""
    m = cimo.compile('''int f(int a) { return (int)((unsigned)a + 1u) > a; }
        int g(long a) { return (long)((unsigned long)a * 2) == 8; }
        int h(long a) { return (long)((unsigned long)a * 2) / 2 == a; }''', cache=False)
    assert m.f(2 ** 31 - 1) == 0 and m.f(5) == 1
    assert m.g(4) == 1 and m.g(5) == 0
    assert m.h(2 ** 62) == 0 and m.h(3) == 1


def test_floats_stay_exact():
    import math
    m = cimo.compile('''double f(double a) { return a + 0.0; }
        double g(double a) { return a - a; }
        int h(double a) { return a == a; }
        double i(double a, double b) { return a == b ? b : a; }''', cache=False)
    assert math.copysign(1, m.f(-0.0)) == 1  # -0 + 0 is +0
    assert math.isnan(m.g(float('inf')))
    assert m.h(float('nan')) == 0
    assert math.copysign(1, m.i(0.0, -0.0)) == -1  # equal, but not the same bits


W = {'int': (32, True), 'unsigned': (32, False), 'long': (64, True), 'unsigned long': (64, False)}


class Gen:
    """A random expression as (C text, Python function of the variables), computed in unsigned
    arithmetic where C would overflow, so it's defined; comparisons and shifts as C has them."""

    def __init__(self, rnd, ty):
        self.r = rnd
        self.ty = ty
        self.bits, self.signed = W[ty]
        self.u = 'unsigned long' if self.bits == 64 else 'unsigned'
        self.mask = (1 << self.bits) - 1

    def val(self, v):
        v &= self.mask
        if self.signed and v >> (self.bits - 1):
            v -= 1 << self.bits
        return v

    def expr(self, depth):
        r = self.r
        if depth == 0 or r.random() < .15:
            if r.random() < .6:
                n = r.choice('abc')
                return n, lambda e, n=n: e[n]
            c = r.choice([0, 1, 2, 3, 7, 8, 15, 16, 255, 256, self.mask >> 1, -1, -2, -8])
            c = self.val(c)
            suffix = ('UL' if not self.signed else 'L') if self.bits == 64 else ''
            lit = '((%s)%d%s)' % (self.ty, c, suffix) if c != -2 ** 63 else '((%s)(-9223372036854775807L - 1))' % self.ty
            return lit, lambda e, c=c: c
        k = r.randrange(11)
        x, fx = self.expr(depth - 1)
        y, fy = self.expr(depth - 1)
        u, val, mask = self.u, self.val, self.mask
        if k <= 3:
            op, f = [('+', lambda a, b: a + b), ('-', lambda a, b: a - b), ('*', lambda a, b: a * b), ('+', lambda a, b: a + b)][k]
            return ('((%s)((%s)%s %s (%s)%s))' % (self.ty, u, x, op, u, y),
                    lambda e, f=f: val(f(fx(e), fy(e))))
        if k <= 6:
            op, f = [('&', lambda a, b: a & b), ('|', lambda a, b: a | b), ('^', lambda a, b: a ^ b)][k - 4]
            return '(%s %s %s)' % (x, op, y), lambda e, f=f: val(f(fx(e), fy(e)))
        if k == 7:
            return '(~%s)' % x, lambda e: val(~fx(e))
        if k == 8:
            n = r.randrange(self.bits)
            if r.random() < .5:
                return ('((%s)((%s)%s << %d))' % (self.ty, u, x, n), lambda e, n=n: val((fx(e) & mask) << n))
            return ('(%s >> %d)' % (x, n), lambda e, n=n: val(fx(e) >> n))
        if k == 9:
            op = r.choice(['==', '!=', '<', '<=', '>', '>='])
            f = {'==': lambda a, b: a == b, '!=': lambda a, b: a != b, '<': lambda a, b: a < b,
                 '<=': lambda a, b: a <= b, '>': lambda a, b: a > b, '>=': lambda a, b: a >= b}[op]
            return ('((%s)(%s %s %s))' % (self.ty, x, op, y), lambda e, f=f: int(f(fx(e), fy(e))))
        z, fz = self.expr(depth - 1)
        return ('(%s ? %s : %s)' % (x, y, z), lambda e: fy(e) if fx(e) else fz(e))


@pytest.mark.parametrize('ty', list(W))
@pytest.mark.parametrize('seed', range(3))
def test_random_expressions(ty, seed):
    rnd = random.Random('%s-%d' % (ty, seed))
    g = Gen(rnd, ty)
    exprs = [g.expr(rnd.randrange(1, 6)) for _ in range(120)]
    src = ''.join('%s e%d(%s a, %s b, %s c) { return %s; }\n' % (ty, n, ty, ty, ty, t)
                  for n, (t, _) in enumerate(exprs))
    m = cimo.compile(src, cache=False)
    special = [0, 1, 2, 3, 7, 8, 255, 256, g.mask >> 1, (g.mask >> 1) + 1, g.mask, g.mask - 1, 0x5555555555555555 & g.mask]
    for n, (t, f) in enumerate(exprs):
        for _ in range(12):
            e = {v: g.val(rnd.choice(special) if rnd.random() < .7 else rnd.getrandbits(g.bits)) for v in 'abc'}
            want = f(e)
            got = getattr(m, 'e%d' % n)(e['a'], e['b'], e['c'])
            assert got == want, (t, e, got, want)


@pytest.mark.parametrize('src,gone', [
    ('long f(long *p, long n, long i) { if (i < 0 || i >= n) return -1; if (i >= n) return -2; return p[i]; }', '#-2'),
    ('long f(long a) { if (a > 5) { if (a > 3) return 1; return 2; } return 3; }', '#2'),
    ('long f(long a, long b) { if (a < b) { if (a >= b) return 1; return 2; } return 3; }', '#1\n'),
    ('long f(long a) { if (a > 0) { long t = a - 1; if (t < 0) return -1; return t; } return 0; }', '#-1'),
    ('long f(unsigned long a, unsigned long b) { if (a < b) { if (a >= b) return 111; return 2; } return 3; }', '#111'),
    ('long f(long a, long b) { if (a == b) { if (a != b) return 111; return 2; } return 3; }', '#111'),
    ('long f(long a) { if (a == 5) { if (a > 7) return 111; return a; } return 3; }', '#111'),
    ('long f(long a, long n) { if (a >= 0 && n >= 0 && a < n) { if ((unsigned long)a >= (unsigned long)n) return 111; return 2; } return 3; }', '#111'),
    ('long f(long n) { long s = 0; for (long i = 0; i < n; i++) { if (i < 0) s += 111; s += i; } return s; }', '#111'),
    ('int f(int n) { int s = 0; for (int i = 3; i < n; i++) { if (i < 2) s += 111; s += i; } return s; }', '#111'),
    ('long f(int a) { if (a >= 0) { long w = a; if (w < 0) return 111; return w; } return 0; }', '#111'),
])
def test_facts_remove_a_check(src, gone):
    assert gone not in body(src), body(src)


def test_facts_dont_overreach():
    """Each of these looks decided, and isn't."""
    M = 2 ** 63 - 1
    m = cimo.compile('''long a(long x) { if (x > 5) { if ((long)((unsigned long)x + 1) > 6) return 1; return 2; } return 3; }
        long b(long x) { if (x >= 0) { if ((long)((unsigned long)x + 1) > 0) return 1; return 2; } return 3; }
        long c(long x) { if (x < 0) { long t = (long)((unsigned long)x - 1); if (t > 0) return 1; return 2; } return 3; }
        long d(long x, long y) { if (x < y) { if ((unsigned long)x < (unsigned long)y) return 1; return 2; } return 3; }
        long e(unsigned long x, unsigned long y) { if (x < y) { if ((long)x < (long)y) return 1; return 2; } return 3; }
        long f(long n) { long s = 0; for (long i = M1; ; i = (long)((unsigned long)i + 1)) { if (i < 0) return s + 1000; if (++s > n) return s; } }
        long g(long n) { long k = 0; for (long i = 9223372036854775806L; k < n; i = (long)((unsigned long)i + 1), k++) { if (i < 0) return 7; } return 8; }
        long h(int x, long y) { if (x > 0) { if (y > 0) return 1; return 2; } return 3; }
        long i(long x) { for (int j = 0; j < 3; j++) { if (x > 5) { x = x - 10; if (x > 5) return 100 + j; } else x += 7; } return x; }
        long j(long x, long y, int k) { if (x < y) goto t; if (k) goto u; return 1; t: if (x >= y) return 2; u: if (x < y) return 3; return 4; }
        unsigned long k(unsigned long x) { if (x > 5) { if (x - 6 > 100) return 1; return 2; } return 3; }'''.replace('M1', '0'), cache=False)
    assert m.a(M) == 2 and m.a(6) == 1 and m.a(1) == 3
    assert m.b(M) == 2 and m.b(0) == 1
    assert m.c(-2 ** 63) == 1 and m.c(-1) == 2
    assert m.d(-1, 1) == 2 and m.d(1, 2) == 1
    assert m.e(1, 2 ** 64 - 1) == 2 and m.e(1, 2) == 1
    assert m.g(5) == 7 and m.g(1) == 8  # a counter that wraps, with no test on itself
    assert m.h(1, 0) == 2 and m.h(0, 0) == 3
    assert m.i(20) == 100 and m.i(3) == 7
    assert m.j(1, 2, 0) == 3 and m.j(2, 1, 1) == 4 and m.j(2, 1, 0) == 1
    assert m.k(7) == 2 and m.k(5) == 3 and m.k(200) == 1


def test_facts_on_a_path_that_rejoins():
    """A fact holds in the arm only, not after the join."""
    m = cimo.compile('''long f(long x) { long s = 0; if (x > 5) s = 1; if (x > 5) s += 10; else s += 20; return s; }
        long g(long x, long y) { long s = 0; if (x < y) { s = 1; } if (x >= y) s += 10; return s; }
        long h(long x) { long s = 0; while (x > 0) { if (x > 0) s++; x -= 2; if (x > 0) s += 100; } return s; }''', cache=False)
    assert m.f(9) == 11 and m.f(1) == 20
    assert m.g(1, 2) == 1 and m.g(2, 1) == 10
    assert m.h(3) == 102 and m.h(0) == 0


@pytest.mark.parametrize('src', [
    'long f(long a, long b) { long x; if (a > 0) x = 1; else x = 0; if (x) return b; return b + 1; }',
    'long f(long a, long b, long c) { int ok = a > 0; if (ok) ok = b > 0; if (ok) return c; return c + 1; }',
    'long f(long a, long b, long c) { if (a > 0 && b > 0) return c; return c + 1; }',
    'long f(long a, long b, long c) { if (a > 0 || b > 0) return c; return c + 1; }',
])
def test_threaded_has_no_flag_left(src):
    b = body(src)
    assert 'cset' not in b, b


THREAD = '''
long t1(long a, long b) { long x; if (a > 0) x = 1; else x = 0; if (x) return b; return b + 1; }
long t2(long a, long b, long c) { int ok = a > 0; if (ok) ok = b > 0; if (ok) return c; return c + 1; }
long t3(long a, long b, long c) { if (a > 0 && b > 0) return c; return c + 1; }
long t4(long a, long b, long c) { if (a > 0 || b > 0) return c; return c + 1; }
long t5(long a, long b, long c) { long v = c; int f = 0; if (a > 0) { v = c * 2; f = 1; } else if (b > 0) { v = c * 3; f = 0; } else { f = 1; v = c * 5; } if (f) return v + 1; return v - 1; }
long t6(long a, long b) { int f = a > 0; long s = 0; if (f) s = 3; if (f) s += b; else s -= b; return s; }
long t7(long n, long k) { long s = 0; for (long i = 0; i < n; i++) { int f = i & 1; if (f) s += k; if (!f) s -= 1; if (f && i > 2) s *= 2; } return s; }
long t8(long a, long b) { int f = 1; if (a) f = 0; long r = f ? b : -b; if (f) r += 1; return r; }
long t9(long a) { int f; if (a == 1) f = 2; else if (a == 2) f = 0; else f = a == 3; if (f == 0) return 10; if (f == 2) return 20; return 30; }
long t10(long a, long b, long c) { int x = a ? (b ? 1 : 0) : (c ? 1 : 0); return x ? a + b : b + c; }
long t11(long n) { long s = 0; int stop = 0; for (long i = 0; i < n && !stop; i++) { s += i; if (s > 20) stop = 1; } return s; }
'''


def test_threading_computes_the_same():
    m = cimo.compile(THREAD, cache=False)
    vals = [-2, -1, 0, 1, 2, 3, 4, 7]
    def py5(a, b, c):
        if a > 0:
            v, f = c * 2, 1
        elif b > 0:
            v, f = c * 3, 0
        else:
            v, f = c * 5, 1
        return v + 1 if f else v - 1

    def py7(n, k):
        s = 0
        for i in range(max(n, 0)):
            f = i & 1
            if f:
                s += k
            if not f:
                s -= 1
            if f and i > 2:
                s *= 2
        return s

    def py8(a, b):
        f = 0 if a else 1
        r = b if f else -b
        return r + 1 if f else r

    def py9(a):
        f = 2 if a == 1 else 0 if a == 2 else int(a == 3)
        return 10 if f == 0 else 20 if f == 2 else 30

    def py11(n):
        s, stop, i = 0, 0, 0
        while i < n and not stop:
            s += i
            if s > 20:
                stop = 1
            i += 1
        return s
    for a in vals:
        assert m.t9(a) == py9(a)
        assert m.t11(a * 3) == py11(a * 3)
        for b in vals:
            assert m.t1(a, b) == (b if a > 0 else b + 1)
            assert m.t6(a, b) == ((3 if a > 0 else 0) + (b if a > 0 else -b))
            assert m.t7(a, b) == py7(a, b)
            assert m.t8(a, b) == py8(a, b)
            for c in vals:
                t = a > 0 and b > 0
                assert m.t2(a, b, c) == m.t3(a, b, c) == (c if t else c + 1)
                assert m.t4(a, b, c) == (c if a > 0 or b > 0 else c + 1)
                assert m.t5(a, b, c) == py5(a, b, c)
                x = (1 if b else 0) if a else (1 if c else 0)
                assert m.t10(a, b, c) == (a + b if x else b + c)


def test_threading_reads_a_long_condition_whole():
    """2**49 has zero low bits, and is true (test_semantics found it). A bare jnz reads a word, a
    comparison its class: "false as a word" says nothing of a long's high half."""
    m = cimo.compile('''long f(double a, long b) { return (a ? 562949953421312L : 2L) && b; }
        long g(long x, long y) { long v; if ((int)x) v = x; else v = x; if (v) return y; return y + 1; }
        long h(long x, long y) { long v = x; if ((int)x) y += 1; if (v) return y; return y + 10; }
        long i(long x) { int s = 0; if ((int)x == 0) { long v = x; if (v != 0) s = 5; else s = 6; } return s; }''', cache=False)
    assert m.f(2.5, 1) == 1 and m.f(0.0, 1) == 1 and m.f(2.5, 0) == 0
    b = 1 << 40
    assert m.g(b, 3) == 3 and m.g(0, 3) == 4 and m.g(1, 3) == 3
    assert m.h(b, 3) == 3 and m.h(0, 3) == 13 and m.h(1, 3) == 4
    assert m.i(b) == 5 and m.i(0) == 6 and m.i(1) == 0


def test_library_calls_with_constants():
    src = """#include <math.h>
    #include <string.h>
    double a(double x) { return pow(x, 2.0); }
    double b(double x) { return pow(x, 2); }
    float c(float x) { return powf(x, 2.0f); }
    double d(double x) { return pow(x, 1.0); }
    double e(double x) { return pow(x, -1.0); }
    double g(double x) { return pow(x, 3.0); }
    double h(double x, double y) { return pow(x, y); }
    unsigned long i(void) { return strlen("hello"); }
    unsigned long j(void) { return strlen("he\\0llo"); }
    unsigned long k(const char *s) { return strlen(s); }
    void *l(char *d, const char *s) { return memcpy(d, s, 16); }
    void *n(char *d) { return memset(d, 0, 16); }
    void *o(char *d) { return memset(d, 0xab, 7); }
    void *p(char *d, char *s, long n) { return memcpy(d, s, n); }
    void *q(char *d, int c) { return memset(d, c, 16); }"""
    asm = driver._build(src)[0]
    for name in 'abcdeijlno':
        assert not re.search(r'\bbl?\s+_?(pow|powf|strlen|memcpy|memset)\b', body(src, name)), body(src, name)
    for name, callee in (('g', 'pow'), ('h', 'pow'), ('k', 'strlen'), ('p', 'memcpy'), ('q', 'memset')):
        assert re.search(r'\bbl?\s+_?%s\b' % callee, body(src, name)), body(src, name)
    m = cimo.compile(src, cache=False)
    assert m.a(3.0) == 9.0 and m.b(-3.0) == 9.0 and m.c(1.5) == 2.25 and m.d(1.25) == 1.25 and m.e(4.0) == 0.25
    assert m.g(2.0) == 8.0 and m.h(2.0, 10.0) == 1024.0
    assert m.i() == 5 and m.j() == 2 and m.k(b"abc") == 3


@pytest.mark.parametrize('n', list(range(0, 66)))
def test_inline_memcpy_and_memset_each_size(n):
    src = """#include <string.h>
    void *f(char *d, const char *s) { return memcpy(d + 3, s + 5, %d); }
    void *g(char *d) { return memset(d + 3, 0x5c, %d); }
    void *h(char *d) { return memset(d + 1, 0, %d); }""" % (n, n, n)
    m = cimo.compile(src, cache=False)
    import ctypes
    size = 96
    s = bytes((i * 7 + 3) & 255 for i in range(size))
    src_ = ctypes.create_string_buffer(s, size)
    d = ctypes.create_string_buffer(b"\xee" * size, size)
    r = m.f(d, src_)
    assert r == ctypes.addressof(d) + 3
    want = bytearray(b"\xee" * size)
    want[3:3 + n] = s[5:5 + n]
    assert bytes(d.raw) == bytes(want)
    d = ctypes.create_string_buffer(b"\xee" * size, size)
    assert m.g(d) == ctypes.addressof(d) + 3
    want = bytearray(b"\xee" * size)
    want[3:3 + n] = b"\x5c" * n
    assert bytes(d.raw) == bytes(want)
    d = ctypes.create_string_buffer(b"\xee" * size, size)
    assert m.h(d) == ctypes.addressof(d) + 1
    want = bytearray(b"\xee" * size)
    want[1:1 + n] = bytes(n)
    assert bytes(d.raw) == bytes(want)


def test_library_names_only_when_they_are_the_library():
    """A function of the same name defined here is that function, not the library's."""
    m = cimo.compile("""static double pow(double x, double y) { return x + y; }
        static unsigned long strlen(const char *s) { return 99; }
        double f(double x) { return pow(x, 2.0); }
        unsigned long g(void) { return strlen("hello"); }""", cache=False)
    assert m.f(3.0) == 5.0 and m.g() == 99


def test_sums_share_their_parts():
    b = body('long f(long *p, long i) { return p[i + 1] + p[i + 2] + p[i + 3]; }')
    assert 'ldp' in b and b.count('lsl') <= 1, b
    b = body('long f(long a, long b, long c) { return (a + 1) + (b + 2) + (c + 3); }')
    assert re.search(r'#6', b), b


def test_reassociation_computes_the_same():
    import array
    src = """long a(long *p, long i) { return p[i + 1] + p[i + 2] + p[i - 1]; }
    long b(long x, long y, long z) { return (x + 1) + (y + 2) + (z - 7); }
    long c(long x) { return (x + 3) * 8; }
    long d(long x) { return (x - 3) << 3; }
    int e(int x, int y) { return (x + 2147483647) + (y + 2147483647); }
    long f(long x) { long t = x + 1; return t * 8 + t; }
    long g(long x, long y) { long t = x + y; return (t + 1) * 4 - t; }
    int h(int x) { return (int)((unsigned)(x + 1) * 3u); }
    long i(int *p, long k) { return p[k + 2] + (long)p[k]; }
    long j(long x) { return (x + 9223372036854775807L) * 2; }
    int k(int x) { return (x + 1) << 31; }"""
    m = cimo.compile(src, cache=False)
    p = array.array('q', range(100, 110))
    addr = p.buffer_info()[0]
    assert m.a(p, 3) == 104 + 105 + 102
    w = array.array('i', range(5))
    assert m.i(w, 1) == 3 + 1
    s = lambda v: _s64(v)
    for x in (0, 1, -1, 2 ** 31, -2 ** 31, 2 ** 62, 2 ** 63 - 1, -2 ** 63):
        assert m.c(x) == s((x + 3) * 8) and m.d(x) == s((x - 3) << 3)
        assert m.f(x) == s((x + 1) * 9) and m.j(x) == s((x + 2 ** 63 - 1) * 2)
        for y in (0, 5, -5, 2 ** 40, -2 ** 63):
            assert m.b(x, y, 7) == s(x + 1 + y + 2 + 7 - 7)
            assert m.g(x, y) == s((x + y + 1) * 4 - (x + y))
    for x in (0, 1, -1, 2 ** 31 - 1, -2 ** 31, 12345):
        assert m.h(x) == _s32((x + 1) * 3) and m.k(x) == _s32((x + 1) << 31)
        for y in (0, 1, -1, 2 ** 31 - 1):
            assert m.e(x, y) == _s32(x + y + 2 * (2 ** 31 - 1))


def _s64(v):
    v &= 2 ** 64 - 1
    return v - 2 ** 64 if v >> 63 else v


def _s32(v):
    v &= 2 ** 32 - 1
    return v - 2 ** 32 if v >> 31 else v


@pytest.mark.parametrize('src,want,wantnot', [
    ('long f(long *p, long a, long c) { long s; if (c) s = p[0] + a; else s = p[0] - a; return s; }', r'ldr', None),
    ('void f(long *p) { *p = 1; *p = 2; }', r'mov\s+x\d+, #2', '#1'),
    ('void f(char *p) { p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0; }', r'str\s+wzr', 'strb'),
    ('void f(short *p) { p[0] = 1; p[1] = 2; }', r'str\s+w', 'strh'),
])
def test_memory_passes_fire(src, want, wantnot):
    b = body(src)
    assert re.search(want, b), b
    if wantnot:
        assert wantnot not in b, b


def test_hoist_and_sink_once():
    b = body('long f(long *p, long a, long c) { long s; if (c) s = p[0] + a; else s = p[0] - a; return s; }')
    assert len(re.findall(r'\bldr\b', b)) == 1, b
    b = body('long f(long *p, long c) { if (c) *p = 1; else *p = 2; return *p; }')
    assert len(re.findall(r'\bstr\b', b)) == 1, b


MEM = """
long a(long *p, long *q, long c) { long s; if (c) { s = p[0] + 1; } else { *q = 7; s = p[0] + 2; } return s; }
long b(long *p, long *q, long c) { long s; if (c) { *q = 7; s = p[0] + 1; } else { s = p[0] + 2; } return s; }
long c(long *p, long c, long v) { if (c) { *p = v; v = 5; } else { *p = v + 1; v = 6; } return *p + v; }
void d(long *p, long *q) { *p = 1; *q = 2; *p = 3; }
long e(long *p, long *q) { *p = 1; long v = *q; *p = 3; return v; }
long g(long *p) { p[0] = 1; long v = p[0]; p[0] = 2; return v; }
void h(char *p) { p[0] = 1; p[1] = 2; p[2] = 3; p[3] = 4; p[4] = 5; p[5] = 6; p[6] = 7; p[7] = 8; }
void i(char *p) { p[1] = 1; p[0] = 2; p[3] = 3; p[2] = 4; }
void j(char *p, char *q) { p[0] = 1; q[0] = 9; p[1] = 2; }
void k(short *p) { p[0] = 1; p[0] = 2; p[1] = 3; }
void l(int *p) { p[0] = -1; p[1] = 0; }
void n(char *p) { p[0] = 1; p[1] = 2; p[2] = 3; }
void o(int *p) { *p = 1; ((short *)p)[0] = 2; }
void q(short *p) { ((int *)p)[0] = 0x01020304; p[0] = 5; }
void r(long *p, long c) { if (c) { *p = 1; *p = 2; } *p = 3; }
long s(long *p, long c) { *p = 1; if (c) return *p; *p = 2; return 0; }
void u(double *p) { p[0] = 1.5; p[1] = 2.5; }
void v(volatile int *p) { *p = 1; *p = 2; }
"""


def test_memory_passes_compute_the_same():
    import array
    m = cimo.compile(MEM, cache=False)
    x, y = array.array('q', [10, 20]), array.array('q', [30])
    assert m.a(x, y, 1) == 11 and m.a(x, y, 0) == 12 and y[0] == 7
    y[0] = 30
    assert m.b(x, y, 0) == 12 and y[0] == 30 and m.b(x, y, 1) == 11 and y[0] == 7
    assert m.a(x, x, 0) == 9 and x[0] == 7  # q is p: the store comes first in that arm
    x[0] = 10
    assert m.b(x, x, 1) == 8 and x[0] == 7
    assert m.c(x, 1, 3) == 3 + 5 and x[0] == 3 and m.c(x, 0, 3) == 4 + 6 and x[0] == 4
    m.d(x, y)
    assert x[0] == 3 and y[0] == 2
    m.d(x, x)
    assert x[0] == 3
    x[0] = 0
    assert m.e(x, x) == 1 and x[0] == 3  # the load reads the first store
    assert m.g(x) == 1 and x[0] == 2
    z = bytearray(9)
    a8 = (__import__('ctypes').c_char * 9).from_buffer(z)
    m.h(a8)
    assert bytes(z) == bytes([1, 2, 3, 4, 5, 6, 7, 8, 0])
    z[:] = bytes(9)
    m.i(a8)
    assert bytes(z[:5]) == bytes([2, 1, 4, 3, 0])
    z[:] = bytes(9)
    m.j(a8, a8)
    assert bytes(z[:3]) == bytes([9, 2, 0])  # the middle store may be to the same place
    z[:] = bytes(9)
    m.n(a8)
    assert bytes(z[:4]) == bytes([1, 2, 3, 0])
    s16 = array.array('h', [0, 0, 0])
    m.k(s16)
    assert list(s16) == [2, 3, 0]
    i32 = array.array('i', [9, 9, 9])
    m.l(i32)
    assert list(i32) == [-1, 0, 9]
    m.o(i32)
    assert i32[0] == 2
    m.q(s16)
    assert list(s16) == [5, 0x0102, 0]
    x[0] = 0
    m.r(x, 1)
    assert x[0] == 3
    assert m.s(x, 1) == 1 and x[0] == 1 and m.s(x, 0) == 0 and x[0] == 2
    d = array.array('d', [0, 0])
    m.u(d)
    assert list(d) == [1.5, 2.5]
    assert 'str' in body('void v(volatile int *p) { *p = 1; *p = 2; }', 'v') and \
        len(re.findall(r'\bstr\b', body('void v(volatile int *p) { *p = 1; *p = 2; }', 'v'))) == 2


UNSW = """
void a(long *x, long n, long k) { for (long i = 0; i < n; i++) { if (k) x[i] += 1; else x[i] -= 1; } }
long b(long *x, long n, long k) { long s = 0; for (long i = 0; i < n; i++) { if (k > 2) s += x[i]; else s -= x[i]; } return s; }
long c(long *x, long n, long k) { long s = 0, i; for (i = 0; i < n; i++) { if (k) s += x[i]; if (x[i] == 7) break; } return s * 100 + i; }
long d(long *x, long n, int k) { long last = -1; for (long i = 0; i < n; i++) { long v = x[i] * 2; if (k) v += 1; last = v; } return last; }
long e(long *x, long n, long k, long j) { long s = 0; for (long i = 0; i < n; i++) { if (k) s += 1; if (j) s += 2; s += x[i]; } return s; }
long f(long *x, long n, long k) { long s = 0; for (long i = 0; i < n; i++) { if (k) { s += x[i]; if (s > 100) return -1; } else s -= 1; } return s; }
long g(long n, long k) { long s = 0; for (long i = 0; i < n; i++) { for (long j = 0; j < 3; j++) { if (k) s += j; else s += 10; } } return s; }
double h(double *x, long n, int k) { double s = 0; for (long i = 0; i < n; i++) { if (k) s += x[i]; else s += 2 * x[i]; } return s; }
long i(long *x, long n, long k) { long s = 0; long i = 0; while (i < n) { if (k) { i += 2; continue; } s += x[i]; i++; } return s + i; }
long l(long *x, long n, long k) { long t = 0; for (long i = 0; i < n; i++) t = k ? x[i] : t + 1; return t; }
"""


@pytest.mark.parametrize('k', [0, 1, 3])
@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 8, 9])
def test_unswitching_computes_the_same(k, n):
    import array
    m = cimo.compile(UNSW, cache=False)
    vals = [3, 7, 1, 9, 2, 100, 5, 4, 6][:n]
    x = array.array('q', vals or [0])
    a = array.array('q', vals or [0])
    m.a(a, n, k)
    assert list(a)[:n] == [v + 1 if k else v - 1 for v in vals]
    assert m.b(x, n, k) == sum(v if k > 2 else -v for v in vals)
    s, i = 0, 0
    while i < n:
        if k:
            s += vals[i]
        if vals[i] == 7:
            break
        i += 1
    assert m.c(x, n, k) == s * 100 + i
    assert m.d(x, n, k) == (-1 if n == 0 else vals[-1] * 2 + (1 if k else 0))
    for j in (0, 1):
        assert m.e(x, n, k, j) == sum(vals) + n * ((1 if k else 0) + (2 if j else 0))
    if k:
        s = 0
        for v in vals:
            s += v
            if s > 100:
                s = -1
                break
    else:
        s = -n
    assert m.f(x, n, k) == s
    assert m.g(n, k) == n * (3 if k else 30)
    d = array.array('d', [float(v) for v in vals] or [0.0])
    assert m.h(d, n, k) == sum(float(v) * (1 if k else 2) for v in vals)
    if k:
        i = 0
        while i < n:
            i += 2
        assert m.i(x, n, k) == i
    else:
        assert m.i(x, n, k) == sum(vals) + n
    t = 0
    for v in vals:
        t = v if k else t + 1
    assert m.l(x, n, k) == t


def test_unswitched_loop_has_no_branch_on_the_flag():
    src = 'void f(long *x, long n, long k) { for (long i = 0; i < n; i++) { if (k) x[i] += 1; else x[i] -= 1; } }'
    b = body(src)
    assert len(re.findall(r'cbn?z\s+x2', b)) <= 1, b


@pytest.mark.parametrize('src,want', [
    ('unsigned long f(unsigned a, unsigned b) { return (unsigned long)a * b; }', r'umull\s+x0, w\d+, w\d+'),
    ('long f(int a, int b) { return (long)a * b; }', r'smull\s+x0, w\d+, w\d+'),
    ('long f(long s, int a, int b) { return s + (long)a * b; }', r'smaddl\s'),
    ('long f(long s, int a, int b) { return s - (long)a * b; }', r'smsubl\s'),
    ('unsigned long f(unsigned long s, unsigned a, unsigned b) { return s + (unsigned long)a * b; }', r'umaddl\s'),
    ('long f(long a) { return a % 2 == 0; }', r'tst\s|and\s|ubfx\s'),
    ('double f(double x) { return x * 2.0; }', r'fadd\s'),
    ('float f(float x) { return 2.0f * x; }', r'fadd\s'),
])
def test_cheaper_forms(src, want):
    b = body(src)
    assert re.search(want, b), b
    assert not re.search(r'\b(sdiv|fmul)\b', b), b
    if 'int' in src:
        assert 'sxtw' not in b, b


def test_cheaper_forms_compute_the_same():
    m = cimo.compile("""#include <math.h>
        long a(int x, int y) { return (long)x * y; }
        unsigned long b(unsigned x, unsigned y) { return (unsigned long)x * y; }
        long c(long s, int x, int y) { return s + (long)x * y; }
        long d(long s, int x, int y) { return s - (long)x * y; }
        unsigned long e(unsigned long s, unsigned x, unsigned y) { return s + (unsigned long)x * y; }
        long f(long s, int x, unsigned y) { return s + (long)x * (long)y; }
        long g(long x) { return x % 8 == 0; }
        long h(long x) { return x % 8 != 0; }
        long i(unsigned long x) { return x % 16 == 0; }
        int j(int x) { return x % 4 == 0; }
        long k(long x) { long r = x % 8; return (r == 0) * 1000 + r; }
        double l(double x) { return x * 2.0; }
        double n(void) { return sqrt(16.0) + floor(-2.5) + ceil(2.5) + trunc(-2.5) + fabs(-1.5); }
        """, cache=False)
    M = 2 ** 31 - 1
    for x in (0, 1, -1, 3, M, -M - 1):
        for y in (0, 1, -1, 3, M, -M - 1):
            assert m.a(x, y) == x * y
            ux, uy = x & 0xffffffff, y & 0xffffffff
            assert m.b(ux, uy) == ux * uy
            assert m.c(7, x, y) == 7 + x * y and m.d(7, x, y) == 7 - x * y
            assert m.e(7, ux, uy) == 7 + ux * uy
            assert m.f(7, x, uy) == 7 + x * uy
    for x in (0, 8, -8, 7, -7, 16, 2 ** 40, -(2 ** 40) + 1):
        assert m.g(x) == int(x % 8 == 0) and m.h(x) == int(x % 8 != 0)
        r = abs(x) % 8 * (1 if x >= 0 else -1)
        assert m.k(x) == (r == 0) * 1000 + r
    assert m.i(2 ** 64 - 16) == 1 and m.i(2 ** 64 - 1) == 0
    assert m.j(-4) == 1 and m.j(-3) == 0
    assert m.l(1.5) == 3.0 and m.l(float('inf')) == float('inf') and m.l(0.0) == 0.0
    assert m.n() == 4.0 - 3.0 + 3.0 - 2.0 + 1.5


def test_signedness_mix_is_not_a_widening_product():
    b = body('long f(int x, unsigned y) { return (long)x * (long)y; }')
    assert not re.search(r'[su]mull', b), b


ONCE = """static long big(long a, long b) { long s = 0; for (long i = 0; i < a; i++) { s += i * b; if (s > 100) s -= 7; s ^= i;
    s += b * 3; s -= i >> 1; s |= 1; s += i * i; s ^= b; s += a; s -= b; s ^= s >> 2; s += 11; s *= 3; s ^= 77; s -= 5; s += a ^ b; } return s; }
long once(long a) { return big(a, 3); }
static long twice(long a, long b) { long s = 0; for (long i = 0; i < a; i++) { s += i * b; if (s > 100) s -= 7; s ^= i;
    s += b * 3; s -= i >> 1; s |= 1; s += i * i; s ^= b; s += a; s -= b; s ^= s >> 2; s += 11; s *= 3; s ^= 77; s -= 5; s += a ^ b; } return s; }
long t1(long a) { return twice(a, 1); }
long t2(long a) { return twice(a, 2); }
static long small(long a) { return a + 1; }
long s1(long a) { return small(a) * 2; }
static long viaptr(long a) { return a + 2; }
long (*fp)(long) = viaptr;
long callfp(long a) { return fp(a); }
static long viaaddr(long a) { return a + 3; }
long getaddr(long a) { long (*g)(long) = viaaddr; return g(a); }
static long rec2(long a); static long rec1(long a) { return a > 0 ? rec2(a - 1) + 1 : 0; }
static long rec2(long a) { return a > 0 ? rec1(a - 1) + 1 : 0; }
long startrec(long a) { return rec1(a); }
static long unused(long a) { return a; }
"""


def test_once_called_functions_inline_and_dead_ones_go():
    asm = driver._build(ONCE)[0]
    names = set(re.findall(r'^_?(\w+):$', asm, re.M))
    assert 'big' not in names and 'small' not in names  # inlined into their only call
    assert 'twice' in names and 'viaptr' in names and 'rec1' in names and 'unused' not in names
    assert not re.search(r'\bbl?\s+_?big\b', body(ONCE, 'once'))
    m = cimo.compile(ONCE, cache=False)
    assert m.s1(4) == 10 and m.callfp(1) == 3 and m.getaddr(1) == 4 and m.startrec(6) == 6
    def big(a, b):
        s = 0
        for i in range(a):
            s += i * b
            if s > 100:
                s -= 7
            s ^= i
            s += b * 3
            s -= i >> 1
            s |= 1
            s += i * i
            s ^= b
            s += a
            s -= b
            s ^= s >> 2
            s += 11
            s *= 3
            s ^= 77
            s -= 5
            s += a ^ b
            s = ((s + 2 ** 63) % 2 ** 64) - 2 ** 63
        return s
    assert m.once(9) == big(9, 3) and m.t1(5) == big(5, 1) and m.t2(5) == big(5, 2)


PURE = """#include <math.h>
static long pure(long a) { long s = 0; for (long i = 0; i < a; i++) s += i * i; return s; }
long side; static long notpure(long a) { side += a; return a; }
static long readsp(long *p) { return *p; }
static void writes(long *p) { *p += 100; }
long a(long *p, long x) { long s = p[0]; long v = pure(x); long t = pure(x + 1); return s + p[0] + v + t; }
double b(double *p, double x) { double s = p[0] + sin(x); return s + p[0]; }
long c(long *p, long x) { long s = p[0]; long v = notpure(x) + notpure(x); return s + p[0] + v; }
long d(long *p) { p[0] = 5; writes(p); writes(p); return p[0]; }
long e(long *p) { long s = *p; writes(p); writes(p); return s + *p; }
long f(long *p) { long s = readsp(p); *p = 9; return s + readsp(p); }
"""


def test_loads_survive_calls_that_touch_no_memory():
    b = body(PURE, 'b')
    assert len(re.findall(r'\bldr\s+d\d+, \[x(?!29)', b)) == 1, b  # x29: restoring a saved register
    import array
    m = cimo.compile(PURE, cache=False)
    x = array.array('q', [7])
    assert m.a(x, 3) == 7 + 7 + (0 + 1 + 4) + (0 + 1 + 4 + 9)
    assert m.c(x, 3) == 7 + 7 + 6
    y = array.array('d', [1.5])
    assert m.b(y, 0.0) == 3.0
    x[0] = 7
    assert m.d(x) == 205 and x[0] == 205  # writes really writes, between the loads
    x[0] = 1
    assert m.e(x) == 1 + 201
    x[0] = 2
    assert m.f(x) == 2 + 9


def test_a_programs_own_sin_is_not_assumed_pure():
    import array
    m = cimo.compile("""double sin(double x) { static double n; extern double g; g += 1; return g + x; }
        double g = 10;
        double f(void) { double a = g; double s = sin(0.0); return a + g + s; }""", cache=False)
    assert m.f() == 10 + 11 + 11


GLOBALS = """static const int tbl[4] = {1, 2, 3, 4};
static int glob = 5;
static int written = 5;
static int taken = 5;
int *keep(void) { return &taken; }
void wr(void) { written = 9; }
static long big[2] = {1L << 40, -2};
static signed char sc[2] = {-1, 2};
static unsigned short us = 65535;
static double d = 1.5;
static float fl = 2.5f;
static char str[] = "hi";
static int zeros[100];
static int stored[3] = {1, 2, 3};
void wr2(long i) { stored[i] = 7; }
static int part[3] = {1, 2, 3};
int a(void) { return tbl[2]; }
int b(void) { return glob; }
int c(void) { return written; }
int e(void) { return taken; }
long f(void) { return big[0] + big[1]; }
int g(void) { return sc[0] * 10 + sc[1]; }
int h(void) { return us; }
double i(void) { return d; }
float j(void) { return fl; }
int k(void) { return str[1]; }
int l(void) { return zeros[50]; }
int n(void) { return stored[1]; }
int o(long q) { return tbl[q]; }
int p(void) { return *(const int *)((const char *)tbl + 4); }
int q(void) { return part[1]; }
int *r(void) { return &part[2]; }
"""


def test_constant_globals_fold():
    for name in 'abfghijkl':
        assert 'adrp' not in body(GLOBALS, name), body(GLOBALS, name)
    for name in 'cenoq':
        assert 'adrp' in body(GLOBALS, name) or 'bl' in body(GLOBALS, name), body(GLOBALS, name)


def test_constant_globals_compute_the_same():
    import ctypes
    m = cimo.compile(GLOBALS, cache=False)
    assert m.a() == 3 and m.b() == 5 and m.f() == 2 ** 40 - 2 and m.g() == -8 and m.h() == 65535
    assert m.i() == 1.5 and m.j() == 2.5 and m.k() == ord('i') and m.l() == 0
    assert [m.o(v) for v in range(4)] == [1, 2, 3, 4] and m.p() == 2
    assert m.c() == 5
    m.wr()
    assert m.c() == 9
    assert m.e() == 5
    ctypes.c_int.from_address(m.keep()).value = 77
    assert m.e() == 77
    assert m.n() == 2
    m.wr2(1)
    assert m.n() == 7
    assert m.q() == 2
    ctypes.c_int.from_address(m.r() - 4).value = 88  # one past where r() points, in part
    assert m.q() == 88


MEMCMP = """#include <string.h>
int a(char *x, char *y) { return memcmp(x, y, 8) == 0; }
int b(char *x, char *y) { return memcmp(x, y, 4) != 0; }
int c(char *x, char *y) { return !memcmp(x, y, 16); }
long d(char *x, char *y, long v) { if (memcmp(x, y, 7)) return v; return v + 1; }
long e(char *x, char *y, long v) { if (memcmp(x, y, 3) == 0) return v; return v + 1; }
int f(char *x, char *y) { return memcmp(x, y, 8) < 0; }
int g(char *x, char *y) { return memcmp(x, y, 8); }
int h(char *x, char *y, long n) { return memcmp(x, y, n) == 0; }
int i(char *x, char *y) { return memcmp(x, y, 0) == 0; }
int j(char *x, char *y) { return memcmp(x, y, 32) == 0; }
int k(char *x, char *y) { return memcmp(x, y, 33) == 0; }
"""


def test_memcmp_equality_has_no_call():
    for name in 'abcdeij':
        assert 'memcmp' not in body(MEMCMP, name), body(MEMCMP, name)
    for name in 'fghk':
        assert 'memcmp' in body(MEMCMP, name), body(MEMCMP, name)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32])
def test_memcmp_each_size(n):
    import ctypes
    m = cimo.compile('#include <string.h>\nint f(char *x, char *y) { return memcmp(x, y, %d) == 0; }\n'
                     'int g(char *x, char *y) { return memcmp(x, y, %d) != 0; }' % (n, n), cache=False)
    base = bytes(range(1, 41))
    x = ctypes.create_string_buffer(base, 40)
    assert m.f(x, ctypes.create_string_buffer(base, 40)) == 1
    for at in range(0, 40):
        changed = bytearray(base)
        changed[at] ^= 0x80
        y = ctypes.create_string_buffer(bytes(changed), 40)
        assert m.f(x, y) == int(at >= n), (n, at)
        assert m.g(x, y) == int(at < n), (n, at)


SCALAR = r'''
struct P { int x, y; };
struct V { double x, y, z; };
union U { int i; float f; };
extern int printf(const char *, ...);
int sroa1(int a, int b) { struct P p; p.x = a; p.y = b; return p.x * p.y; }
double sroa2(double a) { struct V v = {a, a + 1, a + 2}; struct V w = v; return w.x + w.y + w.z; }
int sroa3(const int *q, int n) { struct P p = {0, 0}; for (int i = 0; i < n; i++) { p.x += q[i]; p.y ^= q[i]; } return p.x + p.y; }
int sroa4(int a) { int t[3]; t[0] = a; t[1] = a * 2; t[2] = a * 3; return t[0] + t[1] + t[2]; }
int unionf(float f) { union U u; u.f = f; return u.i; }
float unioni(int i) { union U u; u.i = i; return u.f; }
int escaped(int a) { int t[2]; int *p = t; t[0] = a; p[1] = a + 1; return t[1]; }
int varindex(int a, int i) { int t[2]; t[0] = a; t[1] = a + 1; return t[i & 1]; }
int overlapped(unsigned a) { union { unsigned w; unsigned char b[4]; } u; u.w = a; return u.b[1]; }
int outofslot(int a) { char s[8]; struct P p; p.x = a; p.y = 1; s[0] = 1; return p.x + s[0]; }
int condstore(int c, int a) { struct P p; p.x = 1; if (c) p.x = a; p.y = p.x + 1; return p.y; }
int sccp1(int n) { int x = 1; for (int i = 0; i < n; i++) { if (x != 1) x = 2; } return x; }
int sccp2(int n) { int x = 1, y = 5; for (int i = 0; i < n; i++) { if (x != 1) y = 9; else y = 5; } return y + x; }
int sccp3(int n) { int x = 1; for (int i = 0; i < n; i++) { if (i == 3) x = 2; } return x; }
int dead1(int n) { int x = 0, y = 1; for (int i = 0; i < n; i++) { x = y + 1; y = x * 2; } return n; }
int dead2(const int *p, int n) { int s = 0, t = 0; for (int i = 0; i < n; i++) { if (p[i]) t += p[i]; s += i; } return s; }
int alive(const int *p, int n, int *o) { int t = 0; for (int i = 0; i < n; i++) { if (p[i]) t += p[i]; } *o = t; return n; }
int dist1(int a, int b, int c) { return a * b + a * c; }
int dist2(int a, int b, int c) { return a * c - b * c; }
int dist3(int a, int b, int c, int *o) { int x = a * b; *o = x; return x + a * c; }
int half(int a) { return a / 2; }
long halfl(long a) { return a / 2; }
int digit(int c) { return c >= '0' && c <= '9'; }
int notdigit(int c) { return c < '0' || c > '9'; }
int udigit(unsigned c) { return c >= 48 && c < 58; }
int two(int c) { return c == 7 || c == 8; }
int cased(int c) { return c == 'a' || c == 'A'; }
int empty(int c) { return c >= 9 && c <= 3; }
int wide(long c) { return c > -5 && c < 1000000000000L; }
int minbound(int c) { return c >= -2147483647 - 1 && c <= 5; }
int maxbound(int c) { return c > 2147483647 - 1 || c < 0; }
int edgeout(int c) { return c < -2147483647 - 1 + 1 || c > 2147483646; }
'''


def test_sroa():
    for name in ('sroa1', 'sroa3', 'sroa4', 'condstore'):
        b = body(SCALAR, name)
        assert 'x29' not in b and '[sp' not in b, b
    assert 'ldr' not in body(SCALAR, 'sroa2') and 'fmov\tx' not in body(SCALAR, 'sroa2')
    m = cimo.compile(SCALAR)
    assert m.sroa1(6, 7) == 42 and m.sroa2(1.5) == 1.5 + 2.5 + 3.5 and m.sroa4(5) == 30
    q = array.array('i', [1, 2, 3, 4])
    assert m.sroa3(q, 4) == 10 + (1 ^ 2 ^ 3 ^ 4)
    assert m.unionf(1.0) == 0x3f800000 and m.unioni(0x40000000) == 2.0
    assert m.escaped(3) == 4 and [m.varindex(3, i) for i in (0, 1)] == [3, 4]
    assert m.overlapped(0x11223344) == 0x33 and m.outofslot(4) == 5
    assert m.condstore(0, 9) == 2 and m.condstore(1, 9) == 10


def test_sccp():
    assert 'mov\tw0, #1' in body(SCALAR, 'sccp1').replace(' ', '\t') or re.search(r'mov\s+w0, #1\b', body(SCALAR, 'sccp1'))
    m = cimo.compile(SCALAR)
    for n in (0, 1, 3, 4, 10):
        assert m.sccp1(n) == 1 and m.sccp2(n) == 6
        assert m.sccp3(n) == (2 if n > 3 else 1)


def test_adce():
    assert 'ldr' not in body(SCALAR, 'dead2'), body(SCALAR, 'dead2')
    assert 'mul' not in body(SCALAR, 'dead1') and 'lsl' not in body(SCALAR, 'dead1')
    assert 'ldr' in body(SCALAR, 'alive')
    m = cimo.compile(SCALAR)
    p = array.array('i', [1, 0, 3, 0, 5])
    o = (ctypes.c_int * 1)()
    assert m.dead1(5) == 5 and m.dead2(p, 5) == 10 and m.alive(p, 5, o) == 5 and o[0] == 9


def test_distribute():
    assert len(re.findall(r'\bm(ul|add|sub)\b', body(SCALAR, 'dist1'))) == 1
    assert len(re.findall(r'\bm(ul|add|sub)\b', body(SCALAR, 'dist2'))) == 1
    m = cimo.compile(SCALAR)
    o = (ctypes.c_int * 1)()
    assert m.dist1(3, 4, 5) == 27 and m.dist2(9, 4, 5) == 25
    assert m.dist3(3, 4, 5, o) == 27 and o[0] == 12
    assert m.dist1(2 ** 30, 3, 5) == 0  # wraps the same either way


def test_sign_fixup_of_a_half():
    assert 'asr\tw0, w0, #31' not in body(SCALAR, 'half')
    m = cimo.compile(SCALAR)
    for a in (-7, -2, -1, 0, 1, 7, 2 ** 31 - 1, -2 ** 31):
        assert m.half(a) == int(a / 2)
        assert m.halfl(a * 2 ** 32) == a * 2 ** 31
        assert m.halfl(a) == int(a / 2)


def test_ranges():
    for name in ('digit', 'notdigit', 'udigit', 'two'):
        b = body(SCALAR, name)
        assert b.count('cmp') == 1, b
    assert body(SCALAR, 'cased').count('cmp') == 1
    m = cimo.compile(SCALAR)
    M = 2 ** 31 - 1
    for c in (-M - 1, -M, -6, -5, -4, -1, 0, 3, 6, 7, 8, 9, 47, 48, 57, 58, 65, 97, 96, 1000, M - 1, M):
        assert m.digit(c) == int(48 <= c <= 57), c
        assert m.notdigit(c) == int(c < 48 or c > 57), c
        assert m.two(c) == int(c in (7, 8)), c
        assert m.cased(c) == int(c in (65, 97)), c
        assert m.empty(c) == 0
        assert m.wide(c) == int(-5 < c < 10 ** 12), c
        assert m.minbound(c) == int(c <= 5), c
        assert m.maxbound(c) == int(c > M - 1 or c < 0), c
        assert m.edgeout(c) == int(c < -M or c > M - 1), c
        if c >= 0:
            assert m.udigit(c) == int(48 <= c < 58), c
    assert m.udigit(2 ** 32 - 1) == 0 and m.wide(10 ** 12) == 0 and m.wide(10 ** 12 - 1) == 1 and m.wide(-5) == 0


IPO = r"""
int sink_;
void use(int a) { sink_ += a; }
int ext(int a) { sink_ ^= a; return a + 1; }
#define T4(x) x; x; x; x
#define BIG(a) T4(T4(T4(use(a))))
static int deadarg(int a, int b) { BIG(a); return a; }
int cdead(int a) { return deadarg(a, a * 7) + deadarg(a + 1, 3); }
static long cparam(int a, long b, int c) { BIG(a); return a + b + c; }
long ccon(int a) { return cparam(a, 1L << 40, 7) + cparam(a + 1, 1L << 40, 7); }
static long cdiff(int a, long b) { BIG(a); return a + b; }
long cdif(int a) { return cdiff(a, 1) + cdiff(a, 2); }
static int retk(int a) { BIG(a); return 5; }
int ckret(int a) { return retk(a) + retk(a + 1); }
static int retneg(int a) { BIG(a); if (a) return -1; return -1; }
long ckneg(int a) { return (long)retneg(a) + retneg(a + 1); }
static int rettwo(int a) { BIG(a); if (a) return 1; return 2; }
int cktwo(int a) { return rettwo(a) + rettwo(a + 1); }
static int taken(int a, int b) { BIG(a); return b; }
int (*const taker)(int, int) = taken;
int cktaken(int a) { return taken(a, 3) + taker(a, 4); }
static int floatpar(int a, double d) { BIG(a); return a + (int)d; }
int ckfloat(int a) { return floatpar(a, 2.5) + floatpar(a, 2.5); }
static void structp(int a, int b) { BIG(a); sink_ += b; }
int ckvoid(int a) { structp(a, 2); structp(a, 2); return sink_; }
static int rec(int a, int b) { BIG(a); return a <= 0 ? b : rec(a - 1, b); }
int ckrec(int a) { return rec(a, 9); }
int dse2(int *p, int c) { *p = 1; if (c) *p = 2; else *p = 3; return 0; }
int dse3(int *p, int c, int *q) { *p = 1; if (c) { *q = *p + 3; *p = 2; } else *p = 3; return 0; }
int dse4(int *p, int c) { *p = 1; if (c) *p = 2; return 0; }
int dse5(int *p, int c) { *p = 1; if (c) { ext(1); *p = 2; } else *p = 3; return 0; }
int pre1(int *p, int c) { int a = 0; if (c) a = *p; else ext(1); return a + *p; }
int pre2(int *p, int c, int *q) { int a = 0; if (c) { *p = 5; a = 1; } else { *q = 7; } return a + *p; }
int noloop(int n) { int x = 0; for (int i = 0; i < n; i++) x += i; return n; }
int forever(int n) { for (;;) if (n) break; return 3; }
int usedafter(int n) { int i; for (i = 0; i < n; i++) ; return i; }
long rd(const char *s, long *o) { long a = *o; extern unsigned long strlen(const char *); long n = strlen(s); return a + n + *o; }
int rd2(const int *p, int *o) { int a = *o; extern int memcmp(const void *, const void *, unsigned long); int n = memcmp(p, p + 1, 50); return a + n + *o; }
"""


def test_interprocedural():
    assert not re.search(r'\b[wx]1\b', body(IPO, 'cdead')), body(IPO, 'cdead')
    assert 'mul' not in body(IPO, 'cdead') and 'lsl' not in body(IPO, 'cdead')
    assert re.search(r'w1\b|x1\b', body(IPO, 'cdif')) and re.search(r'mov\s+w0, #10\b', body(IPO, 'ckret'))
    m = cimo.compile(IPO)
    assert m.cdead(2) == 5 and m.ccon(3) == 2 * (1 << 40) + 3 + 4 + 14
    assert m.cdif(4) == 11 and m.ckret(0) == 10 and m.ckneg(0) == -2
    assert m.cktwo(0) == 2 + 1 and m.cktaken(1) == 7 and m.ckfloat(1) == 6
    assert m.ckrec(2) == 9
    before = m.ckvoid(0)
    assert m.ckvoid(0) == before + 4


def test_dead_stores_across_blocks():
    assert re.search(r'mov\s+w\d, #1\b', body(IPO, 'dse2')) is None
    assert re.search(r'mov\s+w\d, #1\b', body(IPO, 'dse4'))  # not overwritten on the way that skips it
    m = cimo.compile(IPO)
    p = (ctypes.c_int * 2)()
    for c, want in ((0, 3), (1, 2)):
        m.dse2(p, c)
        assert p[0] == want
        m.dse5(p, c)
        assert p[0] == want
    q = (ctypes.c_int * 1)()
    m.dse3(p, 1, q)
    assert q[0] == 4 and p[0] == 2
    m.dse4(p, 0)
    assert p[0] == 1
    m.dse4(p, 1)
    assert p[0] == 2


def test_load_pre():
    m = cimo.compile(IPO)
    p = (ctypes.c_int * 1)(11)
    q = (ctypes.c_int * 1)(0)
    assert m.pre1(p, 1) == 22 and m.pre1(p, 0) == 11
    assert m.pre2(p, 1, q) == 6 and p[0] == 5
    p[0] = 8
    assert m.pre2(p, 0, q) == 8 and q[0] == 7
    assert m.pre2(q, 0, q) == 7 and q[0] == 7  # the store on the other way is to the same place


def test_dead_loops():
    assert 'cmp' not in body(IPO, 'noloop') and 'b.' not in body(IPO, 'noloop')
    assert re.search(r'\bb\.?\w*\s|cb', body(IPO, 'forever'))  # it can run forever
    m = cimo.compile(IPO)
    assert m.noloop(5) == 5 and m.forever(1) == 3 and m.usedafter(5) == 5 and m.usedafter(-1) == 0


def test_loads_survive_readonly_calls():
    assert body(IPO, 'rd').count('ldr') <= 2  # the saved register's restore and the one load
    m = cimo.compile(IPO)
    o = (ctypes.c_long * 1)(4)
    assert m.rd(b'abc', o) == 11


def test_dead_store_is_not_one_round_a_loop():
    src = """struct s { struct s *n; } a[4], first;
int f(void) { struct s *next = &first; for (int i = 0; i < 4; i++) { next->n = &a[i]; next = next->n; }
    next->n = 0; int c = 0; for (struct s *p = &first; p; p = p->n) c++; return c; }"""
    assert cimo.compile(src).f() == 5
