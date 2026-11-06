"""GNU vector types (`__attribute__((vector_size(N)))`): arithmetic, shifts and comparisons lane by lane with
scalar broadcast, lanes as lvalues, casts of the bits, calls and struct members, __builtin_shuffle,
__builtin_shufflevector and __builtin_convertvector. Checked against gcc."""

import contextlib
import io

import cimo


SRC = '#include <stdio.h>\ntypedef int v4si __attribute__((vector_size(16)));\ntypedef unsigned char v16qu __attribute__((vector_size(16)));\ntypedef float v4sf __attribute__((vector_size(16)));\ntypedef double v2df __attribute__((vector_size(16)));\ntypedef long v2di __attribute__((vector_size(16)));\ntypedef short __attribute__((vector_size(8))) v4hi;\nv4si addv(v4si a, v4si b) { return a + b; }\nstruct S { char c; v4sf v; };\n#define P4(x) printf("%d %d %d %d\\n", (int)(x)[0], (int)(x)[1], (int)(x)[2], (int)(x)[3])\nint main(void) {\n  v4si a = {1, -2, 3, 40}, b = {5, 6, -7, 8};\n  v4si c = addv(a, b), d = a * b - 3, e = a / 2, f = a % 3, g = a << 2, h = b >> 1, k = ~a, m = -b;\n  P4(c); P4(d); P4(e); P4(f); P4(g); P4(h); P4(k); P4(m);\n  v4si lt = a < b, eq = a == 3; P4(lt); P4(eq);\n  v4sf x = {1.5f, 2.0f, -3.25f, 4.0f}, y = x * 2.0f + x; P4(y);\n  v4si gt = x > 1.75f; P4(gt);\n  v2df p = {0.5, 1.5}; v2df q = p * p; printf("%g %g\\n", q[0], q[1]);\n  v2di cmp = p < 1.0; printf("%ld %ld\\n", cmp[0], cmp[1]);\n  v16qu u = {250, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255}; v16qu w = u + 10;\n  printf("%d %d %d %d\\n", w[0], w[1], w[15], (int)sizeof w);\n  v4hi s4 = {-1, 2, -3, 4}; s4 = s4 * 1000; P4(s4);\n  a[2] = 99; P4(a);\n  v4si r = (v4si)x; printf("%x\\n", r[0]);\n  v4sf back = (v4sf)r; printf("%g\\n", back[0]);\n  long bits = (long)s4; printf("%lx\\n", bits);\n  v4si mask = {3, 0, 1, 2}; v4si sh = __builtin_shuffle(a, mask); P4(sh);\n  v4si m2 = {0, 4, 1, 5}; v4si sh2 = __builtin_shuffle(a, b, m2); P4(sh2);\n  v4si sv = __builtin_shufflevector(a, b, 7, 6, 1, 0); P4(sv);\n  v4sf cv = __builtin_convertvector(a, v4sf); printf("%g %g\\n", cv[2], cv[3]);\n  struct S st = {1, {9, 8, 7, 6}}; st.v += 1; printf("%g %zu %zu\\n", st.v[3], sizeof st, _Alignof(v4si));\n  v4si acc = {0}; for (int i = 0; i < 4; i++) acc += a; P4(acc);\n  return 0;\n}\n'

WANT = '6 4 -4 48\n2 -15 -24 317\n0 -1 1 20\n1 -2 0 1\n4 -8 12 160\n2 3 -4 4\n-2 1 -4 -41\n-5 -6 7 -8\n-1 -1 0 0\n0 0 -1 0\n4 6 -9 12\n0 -1 0 -1\n0.25 2.25\n-1 0\n4 11 9 16\n-1000 2000 -3000 4000\n1 -2 99 40\n3fc00000\n1.5\nfa0f44807d0fc18\n40 1 -2 99\n1 5 -2 6\n8 -7 -2 1\n99 40\n7 32 16\n4 -8 396 160\n'


def test_matches_gcc():
    m = cimo.compile(SRC)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    assert out.getvalue() == WANT


def test_vector_in_type_names_parameters_and_returns():
    m = cimo.compile('#define V __attribute__((vector_size(16)))\n'
                     'V int f(V int t, int a) { ((int *)&t)[1] = a; return t; }\n'
                     'int g(void) { V int r = f((V int){1, 2, 3, 4}, 9); return r[0] * 1000 + r[1] * 100 + r[3]; }\n'
                     'int h(void) { return ((union { long l; int i[2]; })7L).i[0]; }\n')
    assert m.g() == 1904
    assert m.h() == 7
