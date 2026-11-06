"""setjmp and longjmp (C11 7.13), in every module's glue: out of deep recursion, longjmp(env, 0) returning 1,
variables not modified after setjmp keeping their values (the function's loads and stores are volatile, so no
register or shared slot loses them), a jump back into a loop; not inlined; a clear error on WebAssembly."""

import contextlib
import io

import pytest

import cimo
from cimo import driver


def _run(src):
    m = cimo.compile(src)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    return out.getvalue()


SRC = r'''
#include <setjmp.h>
#include <stdio.h>
static jmp_buf env;
static int depth;
static void deep(int n) {
    char pad[100];
    pad[n % 100] = n;
    if (n == 0) longjmp(env, 42);
    deep(n - 1);
    depth += pad[0];
}
static int count_to_three(void) {
    jmp_buf e;
    volatile int count = 0;
    if (setjmp(e) < 3) { count++; longjmp(e, count); }
    return count;
}
static double keep(double a, int k) {
    jmp_buf e;
    double b = a * 1.5, c[4] = { 1, 2, 3, 4 };   /* not modified after setjmp */
    if (setjmp(e)) return a + b + k + c[3];
    {   /* dead before the longjmp on the graph: may not take b's or c's slot */
        double junk[16];
        for (int i = 0; i < 16; i++) junk[i] = i * 3.0;
        printf("%g ", junk[15]);
    }
    longjmp(e, 0);   /* setjmp returns 1 */
}
static long sum_loop(int n) {
    jmp_buf e;
    volatile long s = 0;
    volatile int i = 0;
    setjmp(e);
    if (i < n) { s += i; i++; longjmp(e, 1); }
    return s;
}
int main(void) {
    int r = setjmp(env);
    if (r == 0) { deep(200); puts("not reached"); }
    printf("%d %d %d %g %ld\n", r, depth, count_to_three(), keep(2.0, 7), sum_loop(100));
    return 0;
}
'''


def test_setjmp_longjmp():
    assert _run(SRC) == '45 42 0 3 16 4950\n'


def test_a_function_calling_setjmp_is_not_inlined_and_keeps_variables_in_memory():
    src = ('#include <setjmp.h>\nstatic jmp_buf e;\nstatic int f(int x) { int y = x + 1; if (setjmp(e)) return y; '
           'return 0; }\nint g(int x) { return f(x) + 1; }\n')
    pp = driver._preprocess(src, 'x.c')
    ssa, params = driver._ir(pp, 'x.c')
    assert 'f' in params['#noinline']
    fn = ssa[ssa.index('function w $f'):]
    fn = fn[:fn.index('}')]
    assert 'vstorew' in fn and 'vloadw' in fn and ' storew' not in fn
    text = driver._qbe(ssa, params, target='arm64')
    body = text[text.index('\ng:'):]
    assert 'bl\tf' in body[:body.index('/* end function g */')]


def test_webassembly_says_so():
    from cimo import wasm
    with pytest.raises(cimo.CompileError, match='setjmp/longjmp are not supported on WebAssembly'):
        wasm.build('#include <setjmp.h>\njmp_buf e;\nint f(void) { if (setjmp(e)) return 1; longjmp(e, 1); }\n')
