"""Backend edge cases with results computed in Python (runs on every host): 64-bit and
floating-point constants of every materialization form, immediates at the encoding limits,
frames and offsets around the load/store range limits, register pressure with calls, struct
members and globals at large offsets, many stack arguments from a large frame.
"""

import random
import struct

import pytest

import cimo
from cimo import driver

M64 = (1 << 64) - 1


def compile_(src):
    old = driver.FMA
    driver.FMA = False  # the expected values round every operation
    try:
        return cimo.compile(src)
    finally:
        driver.FMA = old


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def dbits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


BITS = '''static unsigned long long db(double x) { union { double f; unsigned long long u; } c; c.f = x; return c.u; }
static unsigned long long fb(float x) { union { float f; unsigned u; } c; c.f = x; return c.u; }
'''


def patterns():
    """64-bit values that take each materialization path."""
    r = random.Random(5)
    out = {0, 1, M64, 0xffff, 0x10000, 0xffffffff, 0x100000000, 1 << 63, (1 << 63) - 1}
    for k in range(4):  # one 16-bit chunk: movz; its complement: movn
        for v in (0x1234, 0xffff, 0x8000, 1):
            out.add(v << (16 * k))
            out.add(~(v << (16 * k)) & M64)
    for size in (2, 4, 8, 16, 32, 64):  # logical immediates: rotated runs of ones
        for ones in (1, size // 2, size - 1):
            if ones < 1 or ones >= size:
                continue
            for rot in (0, 1, size // 3):
                e = ((1 << ones) - 1)
                e = ((e >> rot) | (e << (size - rot))) & ((1 << size) - 1) if rot else e
                v = 0
                for i in range(64 // size):
                    v |= e << (i * size)
                out.add(v)
    for k in range(64):
        out.add((1 << k) - 1)
        out.add((1 << k) + 1 & M64)
    for _ in range(60):
        out.add(r.getrandbits(64))
        out.add(r.getrandbits(32))
        out.add(r.getrandbits(16) << r.choice((0, 16, 32, 48)) | r.getrandbits(16) << r.choice((0, 16, 32, 48)))
    return sorted(out)


def test_integer_constants():
    vals = patterns()
    src = ''.join('unsigned long long k%d(void) { return %#xULL; }\n'
                  'unsigned w%d(void) { return %#xu; }\n' % (i, v, i, v & 0xffffffff)
                  for i, v in enumerate(vals))
    m = compile_(src)
    for i, v in enumerate(vals):
        assert getattr(m, 'k%d' % i)() & M64 == v, hex(v)
        assert getattr(m, 'w%d' % i)() & 0xffffffff == v & 0xffffffff, hex(v)


FLOATS = [0.0, -0.0, 1.0, -1.0, 0.125, 0.5, 2.0, 31.0, 32.0, -31.0, 1.9375, 0.1, 1e300, -1e-300,
          5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 3.0e38, 1.5e-45, 1e-40,
          16777217.0, 9007199254740993.0, 0.3333333333333333, 255.0, 1024.0, 0.0078125,
          4.76837158203125e-07]


def test_float_constants():
    src = BITS + ''.join('unsigned long long d%d(void) { return db(%r); }\n'
                         'unsigned long long f%d(void) { return fb(%rf); }\n'
                         % (i, x, i, x if abs(x) < 3.4e38 else 3.0e38) for i, x in enumerate(FLOATS))
    m = compile_(src)
    for i, x in enumerate(FLOATS):
        assert getattr(m, 'd%d' % i)() & M64 == dbits(x), x
        xf = x if abs(x) < 3.4e38 else 3.0e38
        assert getattr(m, 'f%d' % i)() & 0xffffffff == fbits(f32(xf)), x


IMMS = [0, 1, 4095, 4096, 4097, 0xfff000, 0xfff001, 0x1000000, 0xffffff, -1, -4095, -4096,
        -4097, -0xfff000, 0x7fffffff, -0x80000000, 0xffffffff, 1 << 32, (1 << 63) - 1,
        -(1 << 63), 0x5555555555555555, 0xff00ff00ff00ff00 - (1 << 64)]


def test_arithmetic_immediates():
    ops = ['+', '-', '&', '|', '^', '<', '<=', '==', '!=', '>', '>=', '*']
    lines = []
    for i, k in enumerate(IMMS):
        lit = '(%dLL)' % k if k != -(1 << 63) else '(-9223372036854775807LL - 1)'
        for j, op in enumerate(ops):
            lines.append('long long t%d_%d(long long x) { return x %s %s; }\n' % (i, j, op, lit))
            lines.append('int u%d_%d(int x) { return x %s (int)%s; }\n' % (i, j, op, lit))
    m = compile_(''.join(lines))

    def s(v, bits):
        v &= (1 << bits) - 1
        return v - (1 << bits) if v >> (bits - 1) else v
    for x in (0, 1, -1, 4096, -4096, 0x7fffffff, 1 << 40, -(1 << 62), 123456789):
        for i, k in enumerate(IMMS):
            for j, op in enumerate(ops):
                for bits, name in ((64, 't'), (32, 'u')):
                    a, b = s(x, bits), s(k, bits)
                    want = {'+': a + b, '-': a - b, '&': a & b, '|': a | b, '^': a ^ b,
                            '*': a * b, '<': a < b, '<=': a <= b, '==': a == b, '!=': a != b,
                            '>': a > b, '>=': a >= b}[op]
                    got = getattr(m, '%s%d_%d' % (name, i, j))(a)
                    assert s(got, bits) == s(int(want), bits), (name, x, op, k)


SIZES = [8, 4088, 4096, 4104, 16376, 32760, 32768, 32776, 65528, 65536, 65544, 262144,
         1048576, 1048584, 4194312]


@pytest.mark.parametrize('n', SIZES)
def test_frame_offsets(n):
    # locals of every width beyond a big array: their offsets from sp/fp cross the limits of
    # the scaled 12-bit and unscaled 9-bit load/store immediates
    src = '''
long long f(long long seed) {
    char big[%d];
    signed char c[3]; short h[3]; int w[3]; long long q[3]; float s[3]; double d[3];
    for (int i = 0; i < 3; i++) {
        c[i] = seed + i; h[i] = seed * 3 + i; w[i] = seed * 5 + i; q[i] = seed * 7 + i;
        s[i] = seed + 0.5f * i; d[i] = seed + 0.25 * i;
    }
    big[0] = 1; big[%d] = 2; big[%d] = 3;
    long long r = big[0] + big[%d] * 10 + big[%d] * 100;
    for (int i = 0; i < 3; i++)
        r = r * 31 + c[i] + h[i] + w[i] + q[i] + (long long)(s[i] * 2) + (long long)(d[i] * 4);
    return r;
}
''' % (n, n // 2, n - 1, n // 2, n - 1)
    m = compile_(src)
    import threading
    out = []
    t = threading.Thread(target=lambda: out.append(m.f(9)))  # 4 MB frames: room for them
    threading.stack_size(16 << 20)
    try:
        t.start()
    finally:
        threading.stack_size(0)
    t.join()

    def want(seed):
        r = 1 + 2 * 10 + 3 * 100
        for i in range(3):
            c = (seed + i + 128) % 256 - 128
            h = (seed * 3 + i + 32768) % 65536 - 32768
            r = (r * 31 + c + h + seed * 5 + i + seed * 7 + i + int(2 * (seed + 0.5 * i))
                 + int(4 * (seed + 0.25 * i)))
            r = (r + (1 << 63)) % (1 << 64) - (1 << 63)
        return r
    assert out == [want(9)]


@pytest.mark.parametrize('nvars,seed', [(16, 1), (40, 2), (64, 3)])
def test_register_pressure(nvars, seed):
    # all variables stay live across calls and loop iterations; mixed classes; a big array
    # pushes the spill slots far from sp
    r = random.Random(seed)
    kinds = [r.choice(['unsigned long long', 'unsigned', 'double', 'float']) for _ in range(nvars)]
    decl, upd = [], []
    for i, k in enumerate(kinds):
        decl.append('%s v%d = (%s)(a * %d + %d);' % (k, i, k, i + 1, i))
    for i, k in enumerate(kinds):
        j = (i * 7 + 3) % nvars
        if k in ('double', 'float'):
            upd.append('v%d = v%d * 0.5%s + (%s)(v%d > 1000 ? 1 : 2);' % (i, i, 'f' if k == 'float' else '', k, j))
        else:
            upd.append('v%d = v%d * %d + (%s)v%d + g(v%d > 3);' % (i, i, 2 * i + 3, k, j, i))
    total = ' + '.join('(unsigned long long)(v%d * 3)' % i for i in range(nvars))
    src = '''
static int calls;
int g(int x) { char pad[64]; pad[x & 63] = 1; calls++; return pad[x & 63] + x; }
unsigned long long f(long long a) {
    char big[40000]; big[39999] = 7;
    %s
    for (int it = 0; it < 3; it++) {
        %s
    }
    return %s + big[39999] + calls;
}
''' % ('\n    '.join(decl), '\n        '.join(upd), total)
    m = compile_(src)

    # Python model of the same computation
    def run(a):
        v = []
        for i, k in enumerate(kinds):
            x = a * (i + 1) + i
            v.append(conv(x, k))
        calls = 0
        for _ in range(3):
            for i, k in enumerate(kinds):
                j = (i * 7 + 3) % nvars
                if k == 'double':
                    v[i] = v[i] * 0.5 + (1.0 if cmp(v[j], kinds[j]) else 2.0)
                elif k == 'float':
                    v[i] = f32(f32(v[i] * 0.5) + (1.0 if cmp(v[j], kinds[j]) else 2.0))
                else:
                    calls += 1
                    gv = 1 + int(gt3(v[i]))
                    bits = 64 if k == 'unsigned long long' else 32
                    other = conv(v[j], k)
                    v[i] = (v[i] * (2 * i + 3) + other + gv) & ((1 << bits) - 1)
        tot = 0
        for i, k in enumerate(kinds):
            if k in ('double', 'float'):
                x = v[i] * 3 if k == 'double' else f32(v[i] * 3)
                tot += int(x) & M64
            else:
                tot += (v[i] * 3) & ((1 << (64 if k == 'unsigned long long' else 32)) - 1)
        return (tot + 7 + calls) & M64

    def conv(x, k):
        if k == 'double':
            return float(x)
        if k == 'float':
            return f32(float(x))
        if isinstance(x, float):
            x = int(x)
        return x & (M64 if k == 'unsigned long long' else 0xffffffff)

    def cmp(x, k):
        return x > 1000

    def gt3(x):
        return x > 3
    assert m.f(5) & M64 == run(5)


def test_struct_members_and_globals_at_large_offsets():
    src = '''
struct S { char pad[%d]; signed char c; short h; int w; long long q; float s; double d; char tail[3]; };
static struct S gs;
static long long garr[300000];
static char gbytes[5000000];
long long f(struct S *p, long long x) {
    p->c = x; p->h = x * 3; p->w = x * 5; p->q = x * 7; p->s = x + 0.5f; p->d = x + 0.25;
    p->tail[2] = x + 1;
    gs.c = p->c; gs.h = p->h; gs.w = p->w; gs.q = p->q; gs.s = p->s; gs.d = p->d; gs.tail[2] = p->tail[2];
    garr[0] = x; garr[511] = x + 1; garr[512] = x + 2; garr[4095] = x + 3; garr[299999] = x + 4;
    gbytes[4095] = 1; gbytes[4096] = 2; gbytes[1048576] = 3; gbytes[4999999] = 4;
    return gs.c + gs.h + gs.w + gs.q + (long long)(gs.s * 2) + (long long)(gs.d * 4) + gs.tail[2]
         + garr[0] + garr[511] + garr[512] + garr[4095] + garr[299999]
         + gbytes[4095] + gbytes[4096] + gbytes[1048576] + gbytes[4999999];
}
'''
    for pad in (1, 4093, 4095, 32759, 32767, 65535, 1048575):
        m = compile_(src % pad)
        import ctypes
        buf = ctypes.create_string_buffer(pad + 64)
        x = 11
        want = (x + 3 * x + 5 * x + 7 * x + int(2 * (x + 0.5)) + int(4 * (x + 0.25)) + x + 1
                + 5 * x + 10 + 10)
        assert m.f(buf, x) == want, pad


def test_many_stack_arguments_from_a_large_frame():
    n = 30
    types = ['long long', 'double', 'int', 'float', 'short', 'unsigned char']
    ps = ', '.join('%s p%d' % (types[i % 6], i) for i in range(n))
    body = ' + '.join('(double)p%d * %d' % (i, i + 1) for i in range(n))
    args = ', '.join(str(i * 3 + 1) for i in range(n))
    src = '''
double callee(%s) { return %s; }
double caller(void) { char big[70000]; big[69999] = 1; return callee(%s) + big[69999]; }
''' % (ps, body, args)
    m = compile_(src)
    import threading
    out = []
    t = threading.Thread(target=lambda: out.append(m.caller()))
    t.start()
    t.join()
    want = 0.0
    for i in range(n):
        want += float(i * 3 + 1) * (i + 1)
    assert out == [want + 1]
