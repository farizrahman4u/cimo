"""Inlining of small functions (cimo.backend.inline): which calls are inlined, and that the
inlined code computes what the call did (each program also runs with inlining off)."""

import ctypes
import re

import pytest

import cimo
from cimo import driver

SYM = '_' if driver.TARGET == 'arm64_apple' else ''  # the target's symbol prefix


def body(asm, name):
    m = re.search(r'^%s:\n(.*?)(?=^\S+:\n|\Z)' % re.escape(SYM + name), asm, re.S | re.M)
    assert m, name
    return m.group(1)


def calls(asm, name):
    return re.findall(r'\b(?:bl|callq)\s+%s(\w+)' % SYM, body(asm, name))  # arm64, x86-64


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
    # a definition stays while something still needs it: inner has its address taken
    assert re.search(r'^%sinner:' % SYM, asm, re.M) and not re.search(r'^%swsub:' % SYM, asm, re.M)


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


IO = r'''
#include <stdint.h>
#include <string.h>
static void force(int64_t *io, double k, double x) { double r = -k * x; memcpy(&io[0], &r, 8); io[1] = 0; }
double step(double k, double x, double v, double dt, long n) {
    for (long i = 0; i < n; i++) {
        int64_t io[5];
        force(io, k, x);
        double f; memcpy(&f, &io[0], 8);
        v += f * dt; x += v * dt;
    }
    return x + v;
}
'''


def test_inlined_callees_result_array_is_registers():
    """scramjit returns a callee's results through an int64_t io[] written and read with memcpy. Inlined, the callee's
    pointer parameter is a copy of the array's address, which SROA took as the address escaping: the array stayed in
    memory, a store in every iteration. SROA follows copies now, and runs again after promotion."""
    asm = driver._build(IO)[0]
    b = asm[asm.index('\n%sstep:' % SYM):]
    b = b[:b.index('end function')]
    # not the array: Windows x64 saves the callee-saved xmm6-xmm15 it uses on the stack, and reads a fifth argument from
    # above the frame (a positive offset from %rbp)
    b = re.sub(r'\tmovdqu (?:%xmm(?:[6-9]|1[0-5]), -?\d+\(%rsp\)|-?\d+\(%rsp\), %xmm(?:[6-9]|1[0-5]))\n', '', b)
    assert not re.search(r'x29, #|-\d+\(%rbp\)|\(%rsp\)', b), b  # no frame: the array is registers
    k = cimo.compile(IO)
    x, v = 1.0, 0.0
    for _ in range(100):
        v += -2.0 * x * 1e-3
        x += v * 1e-3
    assert k.step(2.0, 1.0, 0.0, 1e-3, 100) == x + v


SPRING = r'''
struct S { double x, v, k; };
void spring(struct S *p, double dt, long n) {
    for (long i = 0; i < n; i++) { p->v += -p->k * p->x * dt; p->x += p->v * dt; }
}
'''


def test_struct_fields_stay_in_registers_through_a_loop(tmp_path, monkeypatch):
    """scramjit keeps a model's state in a struct: p->x and p->v are read and written every iteration, p->k only read.
    They were loaded and stored once per unrolled group (a store-to-load round trip on the loop's dependency chain),
    and p->k reloaded, because offsets of one base were taken as the same memory. Now x and v are state variables of
    the fast loop (loaded before it, stored after it) and k is hoisted, without restrict: p is the only base."""
    from cimo import middle
    old = middle.stats
    middle.stats = {}
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))  # compiled here, not taken from the cache
    try:
        driver._build(SPRING)
        stats = middle.stats
    finally:
        middle.stats = old
    assert stats.get('promoted') == 2, stats
    k = cimo.compile(SPRING, cache=False)
    for n in (0, 1, 3, 4, 5, 100):
        s = (ctypes.c_double * 3)(1.0, 0.0, 2.0)
        k.spring(ctypes.addressof(s), 1e-3, n)
        x, v = 1.0, 0.0
        for _ in range(n):
            v += -2.0 * x * 1e-3
            x += v * 1e-3
        assert abs(s[0] - x) < 1e-12 and abs(s[1] - v) < 1e-12 and s[2] == 2.0, n


def _chain(name, n, spec='static'):
    """a function of 2n arithmetic instructions on one local"""
    return '%s long %s(long x) {\n' % (spec, name) + ''.join(
        '    x = x * %d + %d;\n' % (i + 3, i) for i in range(n)) + '    return x;\n}\n'


def test_size_is_what_the_cleanup_keeps():
    """Locals are stack slots until promotion: their loads and stores, the slots' constant offsets and constants'
    extensions cost nothing once inlined, so they don't count (LLVM's InlineCost does the same). poly is 15
    arithmetic instructions and over 60 before promotion."""
    src = r'''
static double poly(double x) {
    double a = x * 1.5, b = a + x, c = b * a, d = c - b, e = d * x, f = e + c, g = f * d, h = g - a;
    union { double d; unsigned long u; } w;
    w.d = h * b + g * f;
    w.u ^= 1;
    return w.d + a + b + c;
}
double f(double x) { return poly(x) + poly(x + 1.0); }
'''
    assert calls(driver._build(src)[0], 'f') == []
    assert cimo.compile(src, cache=False).f(0.25) == cimo.compile(src.replace('static double', 'static __attribute__((noinline)) double'), cache=False).f(0.25)


def test_declared_inline_gets_more_room():
    """A function declared inline may be HINT times the limit (gcc and LLVM allow those more too)."""
    n = driver.INLINE * 3 // 4                      # 1.5x the limit
    for spec, expect in (('static', ['big', 'big']), ('static inline', [])):
        src = _chain('big', n, spec) + 'long f(long x) { return big(x) + big(x + 1); }\n'
        assert calls(driver._build(src)[0], 'f') == expect, spec
        assert cimo.compile(src, cache=False).f(2) == cimo.compile(src.replace(spec + ' long big', 'long big'), cache=False).f(2)


def test_a_cold_callee_called_once_keeps_its_caller_small():
    """slow is called once, so it inlines at any size, but not into fast, which is small and called from two places:
    fast would grow past the limit and stay a call in both (scramjit's int / int and its long division)."""
    src = _chain('slow', driver.INLINE) + r'''
static long fast(long a) { return a < 100 ? a * 3 : slow(a); }
long f1(long a) { return fast(a) + 1; }
long f2(long a) { return fast(a) + 2; }
'''
    asm = driver._build(src)[0]
    assert calls(asm, 'f1') == ['slow'] and calls(asm, 'f2') == ['slow']
    m = cimo.compile(src, cache=False)
    for a in (5, 99, 100, 12345):
        x = a
        if a >= 100:
            for i in range(driver.INLINE):
                x = (x * (i + 3) + i + 2 ** 63) % 2 ** 64 - 2 ** 63
        else:
            x = a * 3
        assert m.f1(a) == x + 1 and m.f2(a) == x + 2, a


def test_the_budget_goes_to_loops_first():
    """A large function spends CALLER_LIMIT on its deepest loops before its straight-line code, and a callee no bigger
    than its call inlines whatever is left (scramjit's llama2 forward pass: 28 tok/s against 89 with no limit)."""
    from cimo.backend import inline
    helper = ('static long mix%d(long a, long b) { long r = a * 7 + b; for (int k = 0; k < 3; k++) r = (r ^ (r >> 3))'
              ' * 31 + k; return r - b * %d; }\n')
    src = ''.join(helper % (n, n) for n in range(2)) + 'static long w(long a, long b) { return a + b; }\n'
    calls_ = ''.join('    s = mix0(s, %d);\n' % n for n in range(inline.CALLER_LIMIT // 8))
    src += ('long big(const long *x, long n) {\n    long s = 1;\n' + calls_ +
            '    for (long i = 0; i < n; i++) s = w(mix1(s, x[i]), i);\n    return s;\n}\n')
    asm = driver._build(src)[0]
    c = calls(asm, 'big')
    assert 'mix1' not in c and 'w' not in c and 'mix0' in c, sorted(set(c))
    m = cimo.compile(src, cache=False)
    x = (ctypes.c_long * 4)(3, 1, 4, 1)
    old = inline.CALLER_LIMIT
    try:
        inline.CALLER_LIMIT = 1 << 30
        ref = cimo.compile(src + '\nint ref_marker;', cache=False)
    finally:
        inline.CALLER_LIMIT = old
    assert m.big(x, 4) == ref.big(x, 4)
