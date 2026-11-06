"""<fenv.h>: rounding modes and exception flags on the hardware's registers (arm64 FPCR/FPSR, x86-64 MXCSR);
quiet ==/!= and isgreater and the rest, signaling <, <=, >, >= and iseqsig (IEEE 754, C Annex F); each call from
Python starts with the flags clear and leaves Python's rounding mode as it was; `#pragma STDC FENV_ACCESS ON`
keeps inexact operations from folding and floating-point operations in place; noinline and noipa respected."""

import contextlib
import io
import random

import pytest

import cimo
from cimo import assembler, assembler_amd64, driver


def _run(src):
    m = cimo.compile(src)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    return out.getvalue()


FLAGS = r'''
#include <fenv.h>
#include <math.h>
#include <stdio.h>
volatile double nan_ = __builtin_nan(""), one = 1.0, three = 3.0, big = 1e308, zero = 0.0, tiny = 0x1p-1000;
volatile int i;
#define T(e, f) do { feclearexcept(FE_ALL_EXCEPT); e; printf("%d", !!fetestexcept(f)); } while (0)
int main(void) {
    T(i = nan_ == one, FE_INVALID); T(i = nan_ != one, FE_INVALID); T(i = isgreater(nan_, one), FE_INVALID);
    T(i = __builtin_isless(nan_, one), FE_INVALID); T(i = isunordered(nan_, one), FE_INVALID);
    T(i = nan_ < one, FE_INVALID); T(i = nan_ >= one, FE_INVALID); T(i = iseqsig(nan_, one), FE_INVALID);
    T(i = iseqsig(one, one), FE_INVALID);
    printf(" ");
    volatile double d;
    T(d = one / three, FE_INEXACT); T(d = one + one, FE_INEXACT); T(d = big * big, FE_OVERFLOW);
    T(d = one / zero, FE_DIVBYZERO); T(d = tiny * tiny, FE_UNDERFLOW); T(d = zero / zero, FE_INVALID);
    feclearexcept(FE_ALL_EXCEPT);
    printf(" %d %d %d", feraiseexcept(FE_OVERFLOW | FE_INEXACT), fetestexcept(FE_ALL_EXCEPT) == (FE_OVERFLOW | FE_INEXACT),
           fetestexcept(FE_DIVBYZERO));
    fexcept_t saved; fegetexceptflag(&saved, FE_ALL_EXCEPT); feclearexcept(FE_ALL_EXCEPT);
    fesetexceptflag(&saved, FE_OVERFLOW); printf(" %d", fetestexcept(FE_ALL_EXCEPT) == FE_OVERFLOW);
    fenv_t env; feholdexcept(&env); d = one / three; printf(" %d", !!fetestexcept(FE_INEXACT));
    feclearexcept(FE_ALL_EXCEPT); d = big * big; feupdateenv(&env);
    printf(" %d\n", fetestexcept(FE_ALL_EXCEPT) == (FE_OVERFLOW | FE_INEXACT));
    return 0;
}
'''


def test_flags_and_quiet_and_signaling_comparisons():
    assert _run(FLAGS) == '000001110 101111 0 1 0 1 1 1\n'


ROUNDING = r'''
#include <fenv.h>
#include <stdio.h>
volatile double one = 1.0, three = 3.0;
volatile long long big = 0x7fffffffffffffff;
int main(void) {
    int modes[] = { FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO };
    for (int k = 0; k < 4; k++) {
        int r = fesetround(modes[k]);
        volatile double d = one / three, n = -one / three, c = big;
        volatile float f = (float)d;
        printf("%d %d %a %a %a %a\n", r, fegetround() == modes[k], d, n, c, f);
    }
    printf("%d %d\n", fesetround(12345), fegetround() == FE_TOWARDZERO);
    fesetenv(FE_DFL_ENV);
    printf("%d\n", fegetround() == FE_TONEAREST);
    return 0;
}
'''


def test_rounding_modes():
    assert _run(ROUNDING) == (
        '0 1 0x1.5555555555555p-2 -0x1.5555555555555p-2 0x1p+63 0x1.555556p-2\n'
        '0 1 0x1.5555555555556p-2 -0x1.5555555555555p-2 0x1p+63 0x1.555556p-2\n'
        '0 1 0x1.5555555555555p-2 -0x1.5555555555556p-2 0x1.fffffffffffffp+62 0x1.555554p-2\n'
        '0 1 0x1.5555555555555p-2 -0x1.5555555555555p-2 0x1.fffffffffffffp+62 0x1.555554p-2\n'
        '1 1\n1\n')


def test_each_call_starts_clear_and_leaves_python_s_rounding_mode():
    m = cimo.compile('#include <fenv.h>\n#include <stdlib.h>\n'
                     'volatile double tenth = 0.1;\n'
                     'int up(void) { return fesetround(FE_UPWARD); }\n'
                     'void up_then_abort(void) { fesetround(FE_DOWNWARD); abort(); }\n'
                     'int mode(void) { return fegetround(); }\n'
                     'int inexact(void) { volatile double d = tenth * 3; return !!fetestexcept(FE_INEXACT); }\n'
                     'int flags(void) { return fetestexcept(FE_ALL_EXCEPT); }\n')
    assert m.up() == 0 and m.mode() == 0
    assert 1 / 3 == 0.3333333333333333
    with pytest.raises(cimo.KernelError):
        m.up_then_abort()
    assert m.mode() == 0 and 2 / 3 == 0.6666666666666666
    assert m.inexact() == 1 and m.flags() == 0


U64 = r'''
#pragma STDC FENV_ACCESS ON
#include <fenv.h>
#include <stdio.h>
__attribute__((noinline)) static double conv(void) { unsigned long long x = 0x7fffffffffffffff; return x; }
__attribute__((noipa)) static double third(void) { double a = 1, b = 3; return a / b; }
int main(void) {
    fesetround(FE_TONEAREST); double n = conv(), t = third();
    fesetround(FE_DOWNWARD); double d = conv(), u = third();
    unsigned long long x = 0x7fffffffffffffff; double e = x;   /* in main, after its own fesetround */
    double three = 3.0, exact = 6.0 / three;
    fesetround(FE_TONEAREST);
    printf("%a %a %a %a %a %a\n", n, d, e, t, u, exact);
    return 0;
}
'''


def test_fenv_access_keeps_inexact_operations_at_run_time():
    assert _run(U64) == ('0x1p+63 0x1.fffffffffffffp+62 0x1.fffffffffffffp+62 0x1.5555555555555p-2 '
                         '0x1.5555555555555p-2 0x1p+1\n')


def test_noinline_and_noipa_calls_stay_calls():
    src = ('__attribute__((noinline)) static int f(int x) { return x + 1; }\n'
           'static int g(int x) __attribute__((noipa));\nstatic int g(int x) { return x * 2; }\n'
           'static int h(int x) { return x - 1; }\n'
           'int k(void) { return f(1) + g(2) + h(3); }\n')
    pp = driver._preprocess(src, 'x.c')
    ssa, params = driver._ir(pp, 'x.c')
    text = driver._qbe(ssa, params, target='arm64')
    body = text[text.index('\nk:'):]
    body = body[:body.index('/* end function k */')]
    assert 'bl\tf' in body and 'bl\tg' in body and 'bl\th' not in body


def test_int128_converts_in_the_rounding_mode():
    m = cimo.compile('#include <fenv.h>\n'
                     'double sd(long long hi, unsigned long long lo, int k) { int m[] = { FE_TONEAREST, FE_UPWARD,\n'
                     '    FE_DOWNWARD, FE_TOWARDZERO }; __int128 x = (__int128)hi << 64 | lo; fesetround(m[k]);\n'
                     '    return (double)x; }\n'
                     'float uf(unsigned long long hi, unsigned long long lo, int k) { int m[] = { FE_TONEAREST,\n'
                     '    FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO }; unsigned __int128 x = (unsigned __int128)hi << 64 | lo;\n'
                     '    fesetround(m[k]); return (float)x; }\n')
    import math

    def rnd(v, prec, emax, mode):
        neg, a = v < 0, abs(v)
        e = a.bit_length() - prec
        if e <= 0:
            return float(v)
        q, r = divmod(a, 1 << e)
        up = (r > 1 << (e - 1) or r == 1 << (e - 1) and q & 1) if mode == 0 else \
            (r != 0 and not neg) if mode == 1 else (r != 0 and neg) if mode == 2 else False
        val = (q + up) << e
        if val >= 1 << emax:
            if mode == 0 or (mode == 1) != neg and mode != 3:
                return -math.inf if neg else math.inf
            val = ((1 << prec) - 1) << (emax - prec)
        return float(-val if neg else val)
    rng = random.Random(5)
    for _ in range(300):
        v = rng.getrandbits(rng.choice((64, 65, 90, 127))) | 1
        for k in range(4):
            for s in (v, -v):
                hi = s >> 64
                assert m.sd(hi, s & (1 << 64) - 1, k) == rnd(s, 53, 1024, k), (hex(s), k)
        for k in range(4):
            u = v | 1 << 127
            assert m.uf(u >> 64, u & (1 << 64) - 1, k) == rnd(u, 24, 128, k), (hex(u), k)


def test_comparisons_select_quiet_or_signaling_instructions():
    src = ('int lt(double a, double b) { return a < b; }\nint eq(double a, double b) { return a == b; }\n'
           'int ge(float a, float b) { return a >= b; }\nint ne(float a, float b) { return a != b; }\n')
    pp = driver._preprocess(src, 'x.c')
    ssa, params = driver._ir(pp, 'x.c')

    def fns(text):
        out = {}
        for name in ('lt', 'eq', 'ge', 'ne'):
            b = text[text.index('\n%s:' % name):]
            out[name] = b[:b.index('/* end function')]
        return out
    a = fns(driver._qbe(ssa, params, target='arm64'))
    assert 'fcmpe' in a['lt'] and 'fcmpe' in a['ge'] and 'fcmpe' not in a['eq'] + a['ne']
    assert 'fcmp\t' in a['eq'] and 'fcmp\t' in a['ne']
    x = fns(driver._qbe(ssa, params, target='amd64_sysv'))
    assert 'comisd' in x['lt'] and 'ucomisd' not in x['lt'] and 'comiss' in x['ge'] and 'ucomiss' not in x['ge']
    assert 'ucomisd' in x['eq'] and 'ucomiss' in x['ne']


def test_signaling_comparison_is_not_speculated():
    # if-conversion would run x < y on the path where x is a NaN: isgreater's !isunordered(x, y) && x > y
    src = ('int f(double x, double y) { return x == x && y == y && x < y; }\n')
    pp = driver._preprocess(src, 'x.c')
    ssa, params = driver._ir(pp, 'x.c')
    text = driver._qbe(ssa, params, target='arm64')
    body = text[text.index('\nf:'):text.index('/* end function f */')]
    assert body.index('fcmpe') > body.index('bne')  # after the branches on x == x and y == y


def test_register_instructions():
    def a64(line):
        mn, ops = line.split(None, 1)
        return assembler._INSNS[mn](None, mn, [o.strip() for o in ops.split(',')])
    assert [a64(s) for s in ('mrs x0, fpcr', 'mrs x3, fpsr', 'msr fpcr, x1', 'msr fpsr, x2', 'msr fpsr, xzr')] == \
        [0xd53b4400, 0xd53b4423, 0xd51b4401, 0xd51b4422, 0xd51b443f]
    obj = assembler_amd64.assemble('stmxcsr (%rsp)\nldmxcsr 4(%rsp)\nstmxcsr -8(%r12)\ncomisd %xmm1, %xmm0\n'
                                   'comiss %xmm9, %xmm2\n')
    assert bytes(obj.sections[0].data).hex() == '0fae1c240fae542404410fae5c24f8660f2fc1410f2fd1'


def test_x86_register_access_keeps_the_flags_and_the_stack():
    text = driver._qbe('export function w $f(w %x) {\n@s\n\twrfpcr %x\n\t%r =w rdfpsr\n\tret %r\n}\n',
                       target='amd64_sysv')
    assert 'ldmxcsr (%rsp)' in text and 'stmxcsr (%rsp)' in text
    assert text.count('leaq -8(%rsp), %rsp') == 2 and text.count('leaq 8(%rsp), %rsp') == 2
    assert 'subq' not in text and 'addq' not in text


def test_software_fma_rounds_in_the_mode():
    # x86-64 without FMA and wasm use the runtime's integer fma; checked against arm64's fmadd
    import math
    import os
    import struct
    from cimo import loader
    if not loader._ARM64:
        pytest.skip('the reference is the arm64 instruction')
    rt = os.path.join(os.path.dirname(cimo.__file__), 'runtime', 'internal')
    m = cimo.compile('#include <fenv.h>\n#include "%s/softfma.h"\n'
                     'static int md[4] = { FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO };\n'
                     'double sd(double x, double y, double z, int k) { fesetround(md[k]); return __cimo_softfma(x, y, z); }\n'
                     'double hd(double x, double y, double z, int k) { fesetround(md[k]); return __builtin_fma(x, y, z); }\n'
                     'float sf(float x, float y, float z, int k) { fesetround(md[k]); return __cimo_softfmaf(x, y, z); }\n'
                     'float hf(float x, float y, float z, int k) { fesetround(md[k]); return __builtin_fmaf(x, y, z); }\n'
                     % rt.replace(os.sep, '/'))
    rng = random.Random(3)

    def same(a, b):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or a != a and b != b

    def rd():
        if rng.random() < .1:
            return rng.choice([1e308, -1e308, 5e-324, -5e-324, 2.2250738585072014e-308, 1.0, -3.0])
        return rng.uniform(-4, 4) * 2.0 ** rng.randint(-1074, 1023)
    for _ in range(4000):
        x, y = rd(), rd()
        z = -x * y if rng.random() < .1 and math.isfinite(x * y) else rd()
        k = rng.randrange(4)
        assert same(m.sd(x, y, z, k), m.hd(x, y, z, k)), (x, y, z, k)
        f = [struct.unpack('<f', struct.pack('<f', v if abs(v) < 3e38 else 1.0))[0] * 2.0 ** -rng.randint(0, 100)
             for v in (x, y, z)]
        assert same(m.sf(*f, k), m.hf(*f, k)), (f, k)
