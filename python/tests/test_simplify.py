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
