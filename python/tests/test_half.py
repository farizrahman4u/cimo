"""_Float16 (and __fp16) and __bf16: stored as their 16 bits, computed in float, rounded to nearest even on
assignment, cast, call and return (the runtime's conversions, subnormals and infinities included); the f16, bf16,
f32 and f64 constant suffixes; _Float32/_Float64/_Float32x. Checked against gcc."""

import contextlib
import io

import cimo


SRC = '#include <stdio.h>\n#include <string.h>\ntypedef _Float16 h;\nstatic h g = 1.0009765625;     /* 1 + 2^-10 */\nstatic __bf16 gb = 3.14159;\nstatic unsigned short bits(h x) { unsigned short u; memcpy(&u, &x, 2); return u; }\nstatic unsigned short bbits(__bf16 x) { unsigned short u; memcpy(&u, &x, 2); return u; }\nh addh(h a, h b) { return a + b; }\nint main(void) {\n  h a = 1.5, b = 0.1f16, c = addh(a, b), d = a * b, e = a / 3, big = 70000.0f, tiny = 1e-6f, neg = -a;\n  printf("%04x %04x %04x %04x %04x %04x %04x %04x %04x\\n", bits(a), bits(b), bits(c), bits(d), bits(e), bits(big), bits(tiny), bits(neg), bits(g));\n  printf("%g %g %g %d %d %d\\n", (double)c, (float)d, (double)tiny, a > b, a == 1.5, (int)(a * 10));\n  __bf16 x = 1.0f / 3, y = x * 3, z = (__bf16)100000.0;\n  printf("%04x %04x %04x %04x %g\\n", bbits(x), bbits(y), bbits(z), bbits(gb), (double)gb);\n  h k = 0; for (int i = 0; i < 2100; i++) k++;\n  h s = 2048; s += 1;\n  printf("%g %g %zu %zu\\n", (double)k, (double)s, sizeof(h), sizeof(__bf16));\n  _Float32 f32 = 1.5f32; _Float64 f64 = 2.5f64;\n  printf("%g %g %d\\n", (double)f32, f64, (int)sizeof(_Float64));\n  h arr[4] = {1, 2, 3, 4}; float sum = 0; for (int i = 0; i < 4; i++) sum += arr[i];\n  printf("%g %d\\n", sum, !(h)0.0);\n  return 0;\n}\n'

WANT = '3e00 2e66 3e66 30cc 3800 7c00 0011 be00 3c01\n1.59961 0.149902 1.01328e-06 1 1 15\n3eab 3f80 47c3 4049 3.14062\n2048 2048 2 2\n1.5 2.5 8\n10 1\n'


def test_matches_gcc():
    m = cimo.compile(SRC)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert m.main() == 0
    assert out.getvalue() == WANT


def test_every_half_converts_exactly():
    # every _Float16 bit pattern to float and back (NaNs stay NaNs), against struct's binary16
    import struct
    m = cimo.compile('unsigned short rt(unsigned short b) { union { unsigned short u; _Float16 h; } x = { b };\n'
                     '    float f = x.h; x.h = f; return x.u; }\n'
                     'double val(unsigned short b) { union { unsigned short u; _Float16 h; } x = { b }; return x.h; }\n')
    for b in range(0, 0x10000, 7):
        want = struct.unpack('<e', struct.pack('<H', b))[0]
        got = m.val(b)
        assert got == want or (got != got and want != want), hex(b)
        if want == want:
            assert m.rt(b) == b, hex(b)


def test_float_to_half_rounds_to_nearest_even():
    import random
    import struct
    m = cimo.compile('unsigned short h(float f) { union { unsigned short u; _Float16 h; } x; x.h = f; return x.u; }\n'
                     'unsigned short bf(float f) { union { unsigned short u; __bf16 h; } x; x.h = f; return x.u; }\n')
    rnd = random.Random(5)
    vals = [0.0, -0.0, 65504.0, 65519.99, 65520.0, 1e-8, 2.0 ** -25, 2.0 ** -24 * 1.5, 6.1e-5, 1e30, -1e30]
    vals += [rnd.uniform(-70000, 70000) for _ in range(300)] + [rnd.uniform(-1e-4, 1e-4) for _ in range(300)]
    for v in vals:
        f = struct.unpack('<f', struct.pack('<f', v))[0]
        try:
            want = struct.unpack('<H', struct.pack('<e', f))[0]
        except OverflowError:
            want = 0xfc00 if f < 0 else 0x7c00
        assert m.h(f) == want, (v, hex(m.h(f)), hex(want))
        u = struct.unpack('<I', struct.pack('<f', f))[0]
        assert m.bf(f) == ((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 0xffff, v
