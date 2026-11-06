"""float and double _Complex: arithmetic (C11 Annex G: a real operand has no imaginary part; NaN results of
`*` recover infinities; `/` exact on extreme values), conversions, comparisons, calls and returns (a struct of
two parts, by every ABI), static initializers, GNU `__real__`/`__imag__`, `~` and imaginary constants, and
<complex.h> (musl's functions). Each program is checked against the output gcc and glibc give it."""

import contextlib
import io

import pytest

import cimo


CASES = {
    'arithmetic': ('#include <stdio.h>\n#include <math.h>\ntypedef double _Complex cd;\ntypedef float _Complex cf;\ncd add(cd a, cd b) { return a + b; }\ncf mulf(cf a, cf b) { return a * b; }\ncd dv(cd a, cd b) { return a / b; }\nstruct S { int k; cd z; };\nstatic cd gz = 1.5 + 2.5i;\nint main(void) {\n  cd a = 1.0 + 2.0i, b = 3.0 - 1.0i;\n  cd c = add(a, b), m = a * b, q = dv(a, b), n = -a, j = ~a;\n  cf x = 1.0f + 1.0fi, y = mulf(x, x);\n  double r = __real__ m, im = __imag__ m;\n  cd s = 2.0 * a, t = a / 2.0, u = 1.0 / b, v = a + 1;\n  struct S st = { 3, a };\n  __imag__ st.z = 9.0;\n  int eq = (a == 1.0 + 2.0i), ne = (a != b), tr = !a, bz = (_Bool)(0.0 * 1i);\n  cd inf = __builtin_complex(INFINITY, NAN) * (2.0 + 0.0i);\n  printf("%g %g | %g %g | %g %g | %g %g | %g %g | %g %g | %g %g\\n", __real__ c, __imag__ c, r, im,\n         __real__ q, __imag__ q, __real__ n, __imag__ n, __real__ j, __imag__ j, __real__ y, __imag__ y,\n         __real__ gz, __imag__ gz);\n  printf("%g %g | %g %g | %g %g | %g %g | %g %g | %d %d %d %d | %g %g | %zu %zu\\n", __real__ s, __imag__ s, __real__ t, __imag__ t,\n         __real__ u, __imag__ u, __real__ v, __imag__ v, __real__ st.z, __imag__ st.z, eq, ne, tr, bz,\n         __real__ inf, __builtin_fabs(__imag__ inf), sizeof(cd), _Alignof(cf));\n  double d = a; cf w = a; int ok = 0; if (a) ok = 1; while (0.0i) ;\n  printf("%g %g %g %d %g\\n", d, __real__ w, __imag__ w, ok, (double)(a ? 3 : 4));\n  return 0;\n}\n',
        '4 1 | 5 5 | 0.1 0.7 | -1 -2 | 1 -2 | 0 2 | 1.5 2.5\n2 4 | 0.5 1 | 0.3 0.1 | 2 2 | 1 9 | 1 1 0 0 | inf nan | 16 4\n1 1 2 1 3\n'),
    'library': ('#include <stdio.h>\n#include <complex.h>\n#define P(z) printf("%.6g %.6g\\n", creal(z), cimag(z))\nint main(void) {\n  double complex z = 1.5 - 0.5 * I, w = CMPLX(0.25, 2.0);\n  float complex f = 0.5f + 1.0f * I;\n  P(cexp(z)); P(clog(w)); P(csqrt(z)); P(cpow(z, w)); P(csin(z)); P(ccos(w)); P(ctan(z)); P(ctanh(w));\n  P(casin(z)); P(cacos(w)); P(catan(z)); P(casinh(w)); P(cacosh(z)); P(catanh(w)); P(cproj(z)); P(conj(z));\n  printf("%.6g %.6g %.6g %.6g\\n", cabs(z), carg(w), cabsf(f), cargf(f));\n  float complex g = cexpf(f) * csqrtf(f) / f;\n  printf("%.5g %.5g %d\\n", crealf(g), cimagf(g), I * I == -1);\n  return 0;\n}\n',
        '3.93305 -2.14864\n0.700899 1.44644\n1.2412 -0.201419\n1.431 1.58321\n1.1248 -0.0368608\n3.64524 -0.8973\n0.255149 -2.1248\n1.0994 -1.59669\n1.17116 -1.06931\n1.45932 -1.44921\n1.01722 -0.146947\n1.32876 1.42792\n1.06931 -0.399639\n0.0495442 1.11211\n1.5 -0.5\n1.5 0.5\n1.58114 1.44644 1.11803 1.10715\n1.4065 0.6732 1\n'),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_gcc(name):
    src, want = CASES[name]
    m = cimo.compile(src)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    assert out.getvalue() == want


def test_extreme_division_is_accurate():
    # gcc's ieee/cdivchkd.c values: subnormal parts, exponents far apart (within 2^-48 of the true quotient)
    m = cimo.compile('double q(double a, double b, double c, double d, int im) {\n'
                     '    double _Complex z = __builtin_complex(a, b) / __builtin_complex(c, d);\n'
                     '    return im ? __imag__ z : __real__ z; }\n')
    a, b, c, d = -float.fromhex('0x1.16e7fad79e45ep+651'), -float.fromhex('0x1.f7f75b94c6c6ap-860'), -float.fromhex('0x1.2f40d8ff7e55ep+245'), -float.fromhex('0x0.0000000004ebcp-1022')
    assert abs(m.q(a, b, c, d, 0) / float.fromhex('0x1.d6e4b0e282869p+405') - 1) < 2 ** -48
    assert abs(m.q(a, b, c, d, 1) / -float.fromhex('0x1.e9095e311e706p-900') - 1) < 2 ** -48
