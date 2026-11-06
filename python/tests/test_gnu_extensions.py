"""GNU extensions cimo accepts, each program checked against the output gcc gives it: statement expressions,
case ranges, range designators, local labels, `mode`, `packed` members and leading `aligned`."""

import contextlib
import io

import pytest

import cimo


CASES = {
    'stmtexpr': ('#include <stdio.h>\n#define max(a, b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })\nint calls;\nint f(int x) { calls++; return x; }\nstruct P { int x, y; };\nint main() {\n  int a = 3, b = 9, n = 0;\n  int m = max(a++, b);                    /* 9, a == 4 */\n  int z = 0 && ({ calls += 100; 1; });    /* short circuit */\n  int c = a > 0 ? ({ int t = a * 2; t; }) : ({ calls += 1000; 0; });\n  for (int i = 0; i < 3; ({ n += 10; i++; })) n++;\n  int w = ({ int s = 0; for (int i = 0; i < 5; i++) { if (i == 3) continue; s += i; } s; });\n  struct P p = ({ struct P q = {1, 2}; q; });\n  ({ ; });\n  double d = (double)({ 7; }) / 2;\n  int g = ({ int r = f(1); r + ({ f(2); }); });\n  int sz = sizeof(({ calls += 10000; 5L; }));\n  printf("%d %d %d %d %d %d %d %d %g %d %d %d\\n", m, a, z, c, n, w, p.x, p.y, d, g, sz, calls);\n  return 0;\n}\n',
        '9 4 0 8 33 7 1 2 3.5 3 8 2\n'),
    'ranges': ('#include <stdio.h>\nstruct S { int x, y; };\nint a[10] = { [2 ... 5] = 7, 8, [0] = 1 };\nint b[] = { [1 ... 3] = 4 };\nstruct S c[5] = { [1 ... 3] = { 5, 6 }, [4].y = 9 };\nint d[3][2] = { [0 ... 2] = { 1, 2 } };\nstruct T { int v[4]; int w; } e = { .v = { [1 ... 2] = 3 }, 11 };\nconst char *names[6] = { [0 ... 5] = "x", [3] = "y" };\nint main(void) {\n  int f[6] = { [1 ... 4] = 5 };\n  for (int i = 0; i < 10; i++) printf("%d ", a[i]);\n  printf("| %zu %d %d %d | ", sizeof b / sizeof *b, b[0], b[1], b[3]);\n  for (int i = 0; i < 5; i++) printf("%d,%d ", c[i].x, c[i].y);\n  printf("| %d %d %d | %d %d %d %d %d | %s%s | ", d[0][1], d[1][0], d[2][1], e.v[0], e.v[1], e.v[2], e.v[3], e.w, names[2], names[3]);\n  for (int i = 0; i < 6; i++) printf("%d", f[i]);\n  printf("\\n");\n  int k = 0;\n  for (int i = -3; i < 12; i++) switch (i) { case 1 ... 3: k += 1; break; case 5: k += 10; break; case 7 ... 9: k += 100; break; case -2 ... -1: k += 1000; break; default: k += 10000; }\n  unsigned u = 0; for (unsigned i = 0; i < 300; i++) switch ((unsigned char)i) { case \'a\' ... \'z\': u++; break; case 200 ... 255: u += 1000; }\n  printf("%d %u\\n", k, u);\n  return 0;\n}\n',
        '1 0 7 7 7 7 8 0 0 0 | 4 0 4 4 | 0,0 5,6 5,6 5,6 0,9 | 2 1 2 | 0 3 3 0 11 | xy | 055550\n62313 56026\n'),
    'labels': ('#include <stdio.h>\n#define FIND(arr, n, x) ({ __label__ found, done; int _i, _r; \\\n  for (_i = 0; _i < (n); _i++) if ((arr)[_i] == (x)) goto found; \\\n  _r = -1; goto done; found: _r = _i; done: _r; })\nint main(void) {\n  int a[] = {4, 8, 15, 16, 23, 42};\n  int r = FIND(a, 6, 15) * 10 + FIND(a, 6, 99);\n  int k = 0;\n  { __label__ out; for (;;) { k++; if (k == 3) goto out; } out: ; }\n  out: k += 100;\n  printf("%d %d\\n", r, k);\n  return 0;\n}\n',
        '19 103\n'),
    'mode': ('#include <stdio.h>\ntypedef int QItype __attribute__ ((mode (QI)));\ntypedef unsigned int UHItype __attribute__ ((mode (HI)));\ntypedef int DItype __attribute__((__mode__(__DI__)));\ntypedef unsigned UWtype __attribute__((mode(word)));\ntypedef float DFtype __attribute__((mode(DF)));\nint main(void) {\n  QItype q = 200; UHItype h = 70000; DItype d = 1; UWtype w = -1; DFtype f = 0.1;\n  int x __attribute__((mode(DI))) = 5;\n  printf("%zu %d %zu %u %zu %zu %zu %zu\\n", sizeof q, q, sizeof h, h, sizeof d, sizeof w, sizeof f, sizeof x);\n  return 0;\n}\n',
        '1 -56 2 4464 8 8 8 8\n'),
    'packed': ('#include <stdio.h>\n#include <stddef.h>\nstruct tv { long s; };\nstruct o { char c; struct tv t __attribute__((packed)); int i; };\nstruct p { char c; int x __attribute__((packed)), y; };\n__attribute__((aligned(32))) static struct { int a; } g;\nint main(void) {\n  printf("%zu %zu %zu %zu | %zu %zu %zu | %d\\n", offsetof(struct o, t), offsetof(struct o, i), sizeof(struct o), _Alignof(struct o),\n         offsetof(struct p, x), offsetof(struct p, y), sizeof(struct p), (int)((unsigned long)&g % 32));\n  return 0;\n}\n',
        '1 12 16 4 | 1 8 12 | 0\n'),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_gcc(name):
    src, want = CASES[name]
    m = cimo.compile(src)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    assert out.getvalue() == want
