"""__int128 and unsigned __int128: arithmetic lowered to 64-bit halves (multiplication by umulh on arm64, 32-bit
halves elsewhere), comparisons, shifts (constant amounts inline), division and float conversions in the runtime,
constants and static data, calls, struct members and arrays, the overflow builtins. Checked against gcc."""

import contextlib
import io

import pytest

import cimo
from cimo.frontend import qbe


SRC = '#include <stdio.h>\ntypedef unsigned __int128 u128;\ntypedef __int128 i128;\nstatic void P(const char *n, u128 v) { printf("%s %016llx%016llx\\n", n, (unsigned long long)(v >> 64), (unsigned long long)v); }\nu128 mulhi_in(unsigned long a, unsigned long b) { return (u128)a * b; }\nunsigned long mulhi(unsigned long a, unsigned long b) { return ((u128)a * b) >> 64; }\ni128 sdiv(i128 a, i128 b) { return a / b; }\nstatic u128 big = ((u128)0x0123456789abcdefULL << 64) | 0xfedcba9876543210ULL;\nstatic i128 neg = -(i128)12345678901234567LL * 1000000000000LL;\nstruct S { char c; u128 v; } s = { 1, 77 };\nint main(void) {\n  u128 a = big, b = ((u128)3 << 100) + 12345;\n  i128 x = neg, y = 987654321;\n  P("add", a + b); P("sub", a - b); P("mul", a * b); P("div", a / 12345); P("mod", a % 999999937);\n  P("div2", a / b); P("mod2", a % b); P("and", a & b); P("or", a | b); P("xor", a ^ b); P("not", ~a);\n  P("shl", a << 7); P("shl70", a << 70); P("shr", a >> 9); P("shr100", a >> 100);\n  int n = 77; P("shlv", a << n); P("shrv", a >> n); P("sarv", (u128)(x >> n)); P("sar", (u128)(x >> 3));\n  P("neg", (u128)-x); P("sdiv", (u128)sdiv(x, y)); P("smod", (u128)(x % y)); P("sdiv2", (u128)sdiv(x, -7));\n  printf("%d %d %d %d %d %d %d\\n", a > b, a < b, x < y, x > 0, a == a, x != y, (i128)-1 < 0);\n  printf("%d %d\\n", (u128)-1 > 0, !a);\n  P("mulhi", mulhi(0xfedcba9876543210UL, 0x0123456789abcdefUL)); P("mulin", mulhi_in(~0UL, ~0UL));\n  printf("%.17g %.17g %.9g %.17g\\n", (double)a, (double)x, (float)b, (double)(u128)-1);\n  P("fromd", (u128)(1e30)); P("fromdn", (u128)(i128)(-3.5e25)); P("fromf", (u128)(i128)2.5f);\n  long l = (long)x; unsigned u = (unsigned)a; int bo = (_Bool)(b >> 120);\n  printf("%ld %u %d\\n", l, u, bo);\n  u128 c = a; c++; ++c; c--; c += 5; c *= 3; c <<= 2; c >>= 1; c /= 7; P("c", c);\n  i128 d = 0; d--; P("d", (u128)d);\n  P("s", s.v); printf("%zu %zu %zu\\n", sizeof(u128), _Alignof(i128), sizeof s);\n  u128 arr[3] = { 1, (u128)1 << 64, ~(u128)0 }; P("arr", arr[0] + arr[1] + arr[2]);\n  for (int i = 0; i < 3; i++) if (arr[i]) printf("%d", i); printf("\\n");\n  return 0;\n}\n'

WANT = 'add 0123459789abcdeffedcba9876546249\nsub 0123453789abcdeffedcba98765401d7\nmul 0da740ddddddda392222222222222590\ndiv 0000060a45f5207db50e4e46eef1631b\nmod 00000000000000000000000023c0ecbd\ndiv2 00000000000000000000000000061172\nmod2 0000000789abcdeffedcba9751b4efae\nand 00000020000000000000000000003010\nor 0123457789abcdeffedcba9876543239\nxor 0123455789abcdeffedcba9876540229\nnot fedcba98765432100123456789abcdef\nshl 91a2b3c4d5e6f7ff6e5d4c3b2a190800\nshl70 b72ea61d950c84000000000000000000\nshr 000091a2b3c4d5e6f7ff6e5d4c3b2a19\nshr100 00000000000000000000000000123456\nshlv 97530eca864200000000000000000000\nshrv 00000000000000000000091a2b3c4d5e\nsarv fffffffffffffffffffffffffffec0df\nsar fffffffffb037c99b72826e3ba289200\nneg 0000000027e41b3246bec8e22ebb7000\nsdiv ffffffffffffffff528714548e73f820\nsmod ffffffffffffffffffffffffca8901e0\nsdiv2 0000000005b2df50533fd38e06ad1000\n1 0 1 0 1 1 1\n1 0\nmulhi 00000000000000000121fa00ad77d742\nmulin fffffffffffffffe0000000000000001\n1.5123660752041709e+36 -1.2345678901234566e+28 3.8029518e+30 3.4028236692093846e+38\nfromd 0000000c9f2c9cd04675000000000000\nfromdn ffffffffffe30c7632b9fb8300000000\nfromf 00000000000000000000000000000002\n-5097732702002180096 1985229328 0\nc 00f9a9342cdc675fff0656cbd32398a5\nd ffffffffffffffffffffffffffffffff\ns 0000000000000000000000000000004d\n16 16 32\narr 00000000000000010000000000000000\n012\n'


@pytest.mark.parametrize("umulh", [True, False])
def test_matches_gcc(umulh, monkeypatch):
    if not umulh:  # the 32-bit-halves product (x86-64, wasm) on any host
        orig = qbe._umulh

        class X86:
            name = 'x86_64-sysv'

        def emulated(f, a, b):
            real = qbe.TG.targ
            qbe.TG.targ = X86
            try:
                return orig(f, a, b)
            finally:
                qbe.TG.targ = real
        monkeypatch.setattr(qbe, '_umulh', emulated)
    m = cimo.compile(SRC, cache=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    assert out.getvalue() == WANT


def test_overflow_builtins_with_int128():
    m = cimo.compile('int f(void) { unsigned __int128 c = (unsigned __int128)1 << 100; unsigned __int128 r;\n'
                     '    return __builtin_mul_overflow(c, c, &r) * 10 + __builtin_add_overflow(c, c, &r); }\n'
                     'int g(void) { long r; return __builtin_add_overflow((__int128)1 << 70, -((__int128)1 << 70), &r) + (int)r; }\n')
    assert m.f() == 10
    assert m.g() == 0
