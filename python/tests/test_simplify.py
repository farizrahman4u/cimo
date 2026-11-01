"""The simplifier and known bits (cimo.backend.simplify): the rules that fired, and random expressions
evaluated by compiled code and by Python's integers."""

import random
import re

import pytest

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
