"""_Float16 and __bf16 to and from float without runtime calls: arm64 fcvt through h31 (frontend vecops 'half'), and
__bf16 inline on every target (a shift one way; round to nearest even, NaN kept quiet, the other). Against Python's
half conversion (struct 'e') and the runtime's formulas, on normals, subnormals, ties, overflow, infinities and NaNs."""

import ctypes
import math
import random
import struct

import cimo
from cimo import driver

SRC = r'''
unsigned short f2h(float x) { _Float16 h = x; unsigned short u; __builtin_memcpy(&u, &h, 2); return u; }
float h2f(unsigned short u) { _Float16 h; __builtin_memcpy(&h, &u, 2); return h; }
unsigned short f2bf(float x) { __bf16 h = x; unsigned short u; __builtin_memcpy(&u, &h, 2); return u; }
float bf2f(unsigned short u) { __bf16 h; __builtin_memcpy(&h, &u, 2); return h; }
double h2d(unsigned short u) { _Float16 h; __builtin_memcpy(&h, &u, 2); return h; }
'''


def _bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def _f32(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]


def _want_f2h(x):
    b = _bits(x)
    if math.isnan(_f32(b)):
        return (b >> 16) & 0x8000 | 0x7e00 | (b >> 13) & 0x1ff
    try:
        return struct.unpack('<H', struct.pack('<e', _f32(b)))[0]
    except OverflowError:
        return (b >> 16) & 0x8000 | 0x7c00


def _want_f2bf(x):
    b = _bits(x)
    if b & 0x7fffffff > 0x7f800000:
        return b >> 16 | 0x40
    return (b + 0x7fff + (b >> 16 & 1)) >> 16


def _floats():
    r = random.Random(5)
    out = [0.0, -0.0, 1.0, -1.0, 65504.0, 65519.99, 65520.0, 1e10, -1e10, 6e-8, 3e-8, 2.98e-8, 1e-40, float('inf'),
           float('-inf'), float('nan'), _f32(0x7fa00001), _f32(0xffc12345), 1 + 2 ** -11, 1 + 3 * 2 ** -11, 2 ** -24,
           2 ** -25, 1.5 * 2 ** -25, 0.1, 1 / 3]
    out += [_f32(r.getrandbits(32)) for _ in range(300)] + [r.uniform(-70000, 70000) for _ in range(300)]
    out += [r.uniform(-1e-4, 1e-4) for _ in range(300)]
    return [_f32(_bits(x)) if not math.isnan(x) else x for x in out]


def test_conversions():
    m = cimo.compile(SRC)
    pp = driver._preprocess(SRC, 'x.c')
    ir = driver._ir(pp, 'x.c')[0]
    assert '__cimo_f2bf' not in ir and '__cimo_bf2f' not in ir
    if not driver._is_amd64():
        assert '__cimo_f2h' not in ir and '__cimo_h2f' not in ir and ' h2f ' in ir and ' f2h ' in ir
    for x in _floats():
        b = _bits(x) if not math.isnan(x) else None
        xin = ctypes.c_float(x).value if b is None else _f32(b)
        got = m.f2h(ctypes.c_float(xin))
        assert got == _want_f2h(xin), (x, hex(got), hex(_want_f2h(xin)))
        got = m.f2bf(ctypes.c_float(xin))
        assert got == _want_f2bf(xin), (x, hex(got))
    for u in range(65536):
        want = struct.unpack('<e', struct.pack('<H', u))[0]
        got = m.h2f(u)
        assert got == want or (math.isnan(got) and math.isnan(want)), hex(u)
        if not math.isnan(want):
            assert m.h2d(u) == want
        got = m.bf2f(u)
        assert _bits(got) == u << 16 or (math.isnan(got) and u & 0x7fff > 0x7f80), hex(u)
