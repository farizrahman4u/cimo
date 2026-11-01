"""Inlining of small functions (cimo.backend.inline): which calls are inlined, and that the
inlined code computes what the call did (each program also runs with inlining off)."""

import re

import pytest

import cimo
from cimo import driver


def body(asm, name):
    m = re.search(r'^%s:\n(.*?)(?=^\S+:\n|\Z)' % re.escape(name), asm, re.S | re.M)
    assert m, name
    return m.group(1)


def calls(asm, name):
    return re.findall(r'\bbl\s+(\w+)', body(asm, name))


@pytest.fixture
def no_inline(monkeypatch):
    def compile_(src):
        monkeypatch.setattr(driver, 'INLINE', 0)
        try:
            return cimo.compile(src, cache=False)
        finally:
            monkeypatch.undo()
    return compile_


HELPERS = r'''
static inline long wsub(long a, long b) { long d = a - b; return d < 0 ? d + 1000 : d; }
static int clz64(unsigned long x) {           /* returns from a loop and from the top */
    if (!x) return 64;
    int n = 0;
    while (!(x >> 63)) { x <<= 1; n++; }
    return n;
}
static signed char sc(signed char c) { return c + 1; }        /* sub-word return */
static unsigned short us(unsigned short a, signed char b) { return a + b; }
static double hyp(double a, double b) { return a * a + b * b; }
static float half(float x) { return x / 2; }
static int g;
static void bump(int k) { if (k < 0) return; g += k; }         /* void, early return */
static int sum3(int *p) { int t[3]; for (int i = 0; i < 3; i++) t[i] = p[i]; return t[0] + t[1] + t[2]; }
static int addr(int x) { int y = x; int *p = &y; *p += 1; return y; }   /* address taken */
static int inner(int x) { return x * 3; }
static int outer(int x) { return inner(x) + inner(x + 1); }  /* inlines inner, then is inlined */
static int fact(int n) { return n < 2 ? 1 : n * fact(n - 1); }   /* calls itself */

long t_wsub(long *x, long n) { long s = 0; for (long i = 0; i < n; i++) s += wsub(x[i], 7); return s; }
int t_clz(unsigned long x) { return x & 1 ? clz64(x) : clz64(x >> 1) + 100; }
int t_sc(int c) { return sc(c); }
int t_us(int a, int b) { return us(a, b); }
double t_hyp(double a, double b) { return hyp(a, b) + half(a); }
int t_bump(int a, int b) { g = 0; bump(a); bump(b); bump(a); return g; }
int t_sum3(int *p, int n) { int s = 0; for (int i = 0; i + 3 <= n; i++) s += sum3(p + i); return s; }
int t_addr(int x) { int s = 0; for (int i = 0; i < x; i++) s += addr(i); return s; }
int t_outer(int x) { return outer(x); }
int t_fact(int n) { return fact(n); }
int t_ptr(int x) { int (*f)(int) = inner; return f(x); }
int t_ignored(long x) { wsub(x, 1); clz64(x); return 5; }
'''


def test_small_helpers_are_inlined():
    asm = driver._build(HELPERS)[0]
    for f in ('t_wsub', 't_clz', 't_sc', 't_us', 't_hyp', 't_bump', 't_sum3', 't_addr',
              't_outer', 't_ignored'):
        assert calls(asm, f) == [], (f, body(asm, f))
    assert calls(asm, 't_fact') == ['fact']  # fact's own recursive call stays a call
    # the definitions are kept
    assert re.search(r'^inner:', asm, re.M) and re.search(r'^wsub:', asm, re.M)


def test_inlined_code_computes_the_same(no_inline):
    import array
    on, off = cimo.compile(HELPERS, cache=False), no_inline(HELPERS)
    xs = array.array('q', [3, 100, -5, 7, 0, 1006, -999, 12])
    ints = array.array('i', [1, -2, 3, 40, 5, -60, 7])
    cases = [('t_wsub', (xs, len(xs))), ('t_sc', (127,)), ('t_sc', (-1,)), ('t_us', (65535, -3)),
             ('t_us', (2, 5)), ('t_hyp', (3.0, 4.0)), ('t_bump', (5, -1)), ('t_bump', (2, 9)),
             ('t_sum3', (ints, len(ints))), ('t_addr', (10,)), ('t_outer', (7,)),
             ('t_fact', (10,)), ('t_ptr', (4,)), ('t_ignored', (9,))]
    cases += [('t_clz', (x,)) for x in (0, 1, 2, 3, 1 << 63, (1 << 64) - 1, 12345, 1 << 40)]
    for name, args in cases:
        assert getattr(on, name)(*args) == getattr(off, name)(*args), (name, args)


def test_callee_defined_later_and_mutual_recursion():
    """Callees defined after their callers inline too; a cycle of calls keeps a call."""
    src = r'''
int later(int);
static int even(unsigned n);
int f(int x) { return later(x) + 1; }
static int odd(unsigned n) { return n == 0 ? 0 : even(n - 1); }
static int even(unsigned n) { return n == 0 ? 1 : odd(n - 1); }
int g(unsigned n) { return even(n); }
int later(int x) { return x * 2; }
'''
    asm = driver._build(src)[0]
    assert calls(asm, 'f') == []
    m = cimo.compile(src, cache=False)
    assert m.f(4) == 9 and m.g(10) == 1 and m.g(7) == 0


def test_big_functions_are_not_inlined():
    big = 'static int big(int x) {\n' + ''.join(
        '    x = x * %d + %d;\n' % (i + 3, i) for i in range(60)) + '    return x;\n}\n'
    asm = driver._build(big + 'int f(int x) { return big(x) + big(x + 1); }\n')[0]
    assert calls(asm, 'f') == ['big', 'big']


def test_large_locals_keep_their_frame():
    """Inlined slots join the caller's frame once per call: these must stay calls."""
    src = r'''
static int buf(int k) { char b[100000]; b[k] = 1; return b[k] + k; }
int f(int k) { return buf(k) + buf(k + 1) + buf(k + 2); }
'''
    assert calls(driver._build(src)[0], 'f') == ['buf', 'buf', 'buf']
    assert cimo.compile(src, cache=False).f(1) == 9


def test_inlined_locals_are_fresh_each_call():
    """A callee's local array keeps no value between calls, even inlined in a loop."""
    m = cimo.compile(r'''
static int first_unset(int k) { int a[4] = {0}; a[k & 3] = 1; for (int i = 0; i < 4; i++) if (!a[i]) return i; return -1; }
int f(int n) { int s = 0; for (int i = 0; i < n; i++) s = s * 4 + first_unset(i); return s; }
''', cache=False)
    assert m.f(8) == int('10001000', 4)


def test_wasm_inlines_too():
    import cimo.wasm as W
    mod, sigs = W.build(HELPERS)
    assert 't_wsub' in sigs
