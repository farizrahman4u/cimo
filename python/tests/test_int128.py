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


def test_shifts_by_a_variable_inline():
    """<<, >> (logical and arithmetic) by a variable 0 to 127, branch-free, no runtime call, on random values."""
    import ctypes
    import random
    from cimo import driver
    src = ('typedef unsigned __int128 u128; typedef __int128 i128;\n'
           'void f(unsigned long lo, unsigned long hi, int s, unsigned long *r) { u128 a = (u128)hi << 64 | lo; i128 b = (i128)a;\n'
           '  u128 x = a << s, y = a >> s; i128 z = b >> s; r[0] = x; r[1] = x >> 64; r[2] = y; r[3] = y >> 64; r[4] = z;\n'
           '  r[5] = (u128)z >> 64; }\n')
    assert '__cimo_i128shift' not in driver._ir(driver._preprocess(src, 'x.c'), 'x.c')[0]
    m = cimo.compile(src)
    rnd = random.Random(1)
    mask = (1 << 128) - 1
    for s in list(range(128)) * 4:
        lo, hi = rnd.getrandbits(64), rnd.getrandbits(64) | (rnd.random() < 0.5) << 63
        r = (ctypes.c_uint64 * 6)()
        m.f(lo, hi, s, r)
        a = hi << 64 | lo
        sa = a - (1 << 128) if a >> 127 else a
        assert [r[0] | r[1] << 64, r[2] | r[3] << 64, r[4] | r[5] << 64] == [(a << s) & mask, a >> s, (sa >> s) & mask], s


def test_no_slot_at_a_call_boundary():
    """A struct or __int128 passed and returned by value is used from its registers (backend abislot): the ABI's slot
    stores and loads are forwarded and go. A slot the body writes to keeps them (and its value)."""
    import ctypes
    from cimo import driver
    src = ('typedef unsigned __int128 u128;\nstruct P { long x, y; };\n'
           'u128 add(u128 a, u128 b) { return a + b; }\nlong sum(struct P p) { return p.x + p.y; }\n'
           'long bump(struct P p) { p.x += 5; return p.x * p.y; }\n'
           '__attribute__((noinline)) static struct P mk(long a) { struct P p = { a, a * 2 }; return p; }\n'
           'long use(long a) { struct P p = mk(a); return p.x - p.y; }\n'
           'unsigned long hi(unsigned long a) { u128 x = add(a, ~0UL); return (unsigned long)(x >> 64); }\n')
    text = driver._qbe(*driver._ir(driver._preprocess(src, 'x.c'), 'x.c'))
    for fn in ('add', 'sum'):
        body = text[text.index('\n%s%s:' % (driver.TARGET == 'arm64_apple' and '_' or '', fn)):]
        body = body[:body.index('end function')]
        assert 'ldr' not in body and 'str' not in body, (fn, body)
    m = cimo.compile(src + 'long t(void) { struct P p = { 3, 4 }; return sum(p) * 1000 + bump(p) * 10 + (p.x == 3); }\n')
    assert m.t() == 7 * 1000 + 32 * 10 + 1 and m.use(10) == -10 and m.hi(5) == 1


def test_slot_stores_that_overlap():
    """A struct result's slot then written in part (a bit-field, a byte): the 8-byte value forwarded from the ABI's
    store must not hide the byte store over it. A sanity check: the shape that failed is gcc's pr108498-1.c (corpora)."""
    src = ('struct T { unsigned char b0 : 1, b1 : 1; char c[7]; };\n'
           '__attribute__((noinline)) static struct T mk(void) { struct T t = { 1, 0, { 5 } }; return t; }\n'
           '__attribute__((noinline)) static int rd(struct T t) { return t.b0 + 2 * t.b1 + 4 * t.c[1] + 100 * t.c[0]; }\n'
           'int f(void) { struct T t = mk(); t.b1 = 1; t.c[1] = 3; return rd(t); }\n')
    assert cimo.compile(src).f() == 1 + 2 + 12 + 500


def test_add_with_carry():
    """An __int128 add is adds and adc on arm64 (emit._carry), carries out of the low word included."""
    import random
    from cimo import driver
    src = ('typedef unsigned __int128 u128;\n__attribute__((noinline)) static u128 add(u128 a, u128 b) { return a + b; }\n'
           'unsigned long f(unsigned long a, unsigned long b, unsigned long c, unsigned long d) {\n'
           '  u128 r = add((u128)b << 64 | a, (u128)d << 64 | c); return (unsigned long)r ^ (unsigned long)(r >> 64) * 3; }\n')
    if not driver._is_amd64():
        text = driver._qbe(*driver._ir(driver._preprocess(src, 'x.c'), 'x.c'))
        body = text[text.index('\n%sadd:' % (driver.TARGET == 'arm64_apple' and '_' or '')):]
        body = body[:body.index('end function')]
        assert '\tadds\t' in body and '\tadc\t' in body and 'csinc' not in body, body
    m = cimo.compile(src)
    rnd = random.Random(1)
    mod = 1 << 64
    for _ in range(300):
        a, b, c, d = [rnd.getrandbits(64) for _ in range(4)]
        if rnd.random() < 0.3:
            a, c = mod - 1, rnd.getrandbits(3)
        r = ((b << 64 | a) + (d << 64 | c)) % (1 << 128)
        assert m.f(a, b, c, d) == (r % mod) ^ ((r >> 64) * 3) % mod
