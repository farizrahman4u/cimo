"""AMX tiles on x86-64 (amx-tile, amx-bf16, amx-int8) through the intrinsics of the bundled immintrin.h: the code,
the errors, results against scalar C where the host has AMX, and tile state across a fault's unwind and Ctrl-C.
Encodings are in data/amd64_forms.s."""

import ctypes
import os
import random
import signal
import struct
import threading
import time

import pytest

import cimo
from cimo import cpu, driver
import hosts

_AMX = frozenset(('popcnt', 'fma', 'avx', 'avx2', 'bmi1', 'bmi2', 'lzcnt', 'f16c', 'avx512f', 'avx512bw', 'avx512dq',
                  'avx512vl', 'avx512bf16', 'amx-tile', 'amx-bf16', 'amx-int8'))

_SRC = r'''
#include <immintrin.h>
#include <stdint.h>
static void config(void) {
    unsigned char cfg[64] = {1};
    for (int t = 0; t < 3; t++) {
        cfg[16 + 2 * t] = 64;
        cfg[48 + t] = 16;
    }
    _tile_loadconfig(cfg);
}
/* C[16][16] += A[16][32] B[32][16], bf16 into float; B in pairs of k ([16][16][2]) */
void bf16(const uint16_t *a, const uint16_t *b, float *c) {
    config();
    _tile_loadd(0, c, 64);
    _tile_loadd(1, a, 64);
    _tile_loadd(2, b, 64);
    _tile_dpbf16ps(0, 1, 2);
    _tile_stored(0, c, 64);
    _tile_release();
}
/* int32 C[16][16] += A[16][64] B[64][16] in fours of k, signed by signed and unsigned by signed */
void s8(const int8_t *a, const int8_t *b, int32_t *c, int unsigned_a) {
    config();
    _tile_zero(0);
    _tile_stream_loadd(1, a, 64);
    _tile_loadd(2, b, 64);
    if (unsigned_a)
        _tile_dpbusd(0, 1, 2);
    else
        _tile_dpbssd(0, 1, 2);
    _tile_stored(0, c, 64);
    _tile_release();
}
/* a fault while the tiles are configured */
void fault(const uint16_t *a, int *p) {
    config();
    _tile_loadd(1, a, 64);
    *p = 1;
    _tile_release();
}
/* tiles busy for a while, for Ctrl-C */
void spin(const uint16_t *a, const uint16_t *b, float *c, long n) {
    config();
    _tile_zero(0);
    _tile_loadd(1, a, 64);
    _tile_loadd(2, b, 64);
    for (long i = 0; i < n; i++)
        _tile_dpbf16ps(0, 1, 2);
    _tile_stored(0, c, 64);
    _tile_release();
}
'''


def _asm(src, feats, monkeypatch):
    monkeypatch.setattr(driver, '_x86', lambda t=None: feats if (t or driver.TARGET).startswith('amd64') else frozenset())
    monkeypatch.setattr(driver, '_cpu', lambda t=None: feats)
    monkeypatch.setattr(driver, 'TARGET', 'amd64_sysv')
    pp = driver._preprocess(src, 'x.c', (), False, (), target='amd64_sysv')
    ssa, params = driver._ir(pp, 'x.c')
    text = driver._qbe(ssa, params, target='amd64_sysv')
    from cimo import assembler_amd64
    assembler_amd64.assemble(text)
    return ssa, text


def test_the_code(monkeypatch):
    ssa, text = _asm(_SRC, _AMX, monkeypatch)
    for mn in ('ldtilecfg (', 'tilezero %tmm0', 'tileloadd (', 'tileloaddt1 (', 'tdpbf16ps %tmm2, %tmm1, %tmm0',
               'tdpbssd %tmm2, %tmm1, %tmm0', 'tdpbusd %tmm2, %tmm1, %tmm0', 'tilestored %tmm0, (', 'tilerelease'):
        assert mn in text, mn
    import re  # a constant stride goes in a register: the memory operand is (base, index, 1)
    mems = re.findall(r'\ttile(?:loadd|loaddt1|stored) [^\n]*', text)
    assert mems and all(re.search(r'\(%r\w+,%r\w+,1\)', m) for m in mems), mems


def test_without_the_features(monkeypatch):
    with pytest.raises(cimo.CompileError, match=r"'_tile_dpbf16ps' needs the CPU features amx-tile and amx-bf16, "
                                               r"which this CPU lacks \(amx-bf16\)"):
        _asm(_SRC, _AMX - {'amx-bf16'}, monkeypatch)


@pytest.mark.parametrize('body, msg', [
    ('void f(int t) { _tile_zero(t); }', 'a tile number must be a constant from 0 to 7'),
    ('void f(void) { _tile_zero(8); }', 'a tile number must be a constant from 0 to 7'),
    ('void f(void) { _tile_dpbf16ps(0, 0, 1); }', 'the destination and the sources must be different tiles'),
])
def test_tile_numbers(body, msg, monkeypatch):
    with pytest.raises(cimo.CompileError, match=msg):
        _asm('#include <immintrin.h>\n' + body, _AMX, monkeypatch)


def test_memory_the_tiles_write_is_read_again(monkeypatch):
    """A store a tile store overwrites isn't forwarded past it to a later load (the passes treat tile ops as calls)."""
    src = r'''
#include <immintrin.h>
float f(float *c, const void *cfg) {
    c[3] = 1.0f;
    _tile_loadconfig(cfg);
    _tile_zero(0);
    _tile_stored(0, c, 64);
    return c[3];
}
'''
    ssa, text = _asm(src, _AMX, monkeypatch)
    body = text[text.index('tilestored'):]
    assert 'movss 12(' in body or 'vmovss 12(' in body, body


def _bf16(x):
    b = struct.unpack('<I', struct.pack('<f', x))[0]
    assert b & 0xffff == 0
    return b >> 16


def _need_amx():
    if not hosts.AMD64 or not {'amx-tile', 'amx-bf16', 'amx-int8'} <= cpu.x86_features():
        pytest.skip('runs AMX code')


def test_permission():
    """The module says it uses tiles, and loading it asks Linux for the tile state."""
    _need_amx()
    pp = driver._preprocess(_SRC, 'x.c')
    assert '#amx' in driver._compile(pp, 'x.c', False)[1].split('\n')
    cimo.compile(_SRC, cache=False)
    assert cpu._tiles == ''


def test_against_scalar():
    _need_amx()
    m = cimo.compile(_SRC, cache=False)
    rnd = random.Random(1)
    A = [[float(rnd.randint(-8, 8)) for _ in range(32)] for _ in range(16)]
    B = [[float(rnd.randint(-8, 8)) for _ in range(16)] for _ in range(32)]
    a = (ctypes.c_uint16 * 512)(*[_bf16(A[i][k]) for i in range(16) for k in range(32)])
    b = (ctypes.c_uint16 * 512)(*[_bf16(B[2 * p + q][j]) for p in range(16) for j in range(16) for q in range(2)])
    c = (ctypes.c_float * 256)(*[float(i) for i in range(256)])
    m.bf16(a, b, c)
    assert list(c) == [16 * i + j + sum(A[i][k] * B[k][j] for k in range(32)) for i in range(16) for j in range(16)]
    S = [[rnd.randint(-128, 127) for _ in range(64)] for _ in range(16)]
    U = [[rnd.randint(0, 255) for _ in range(64)] for _ in range(16)]
    T = [[rnd.randint(-128, 127) for _ in range(16)] for _ in range(64)]
    bt = (ctypes.c_int8 * 1024)(*[T[4 * p + q][j] for p in range(16) for j in range(16) for q in range(4)])
    out = (ctypes.c_int32 * 256)()
    for X, unsigned in ((S, 0), (U, 1)):
        xa = (ctypes.c_int8 * 1024)(*[ctypes.c_int8(X[i][k]).value for i in range(16) for k in range(64)])
        m.s8(xa, bt, out, unsigned)
        assert list(out) == [sum(X[i][k] * T[k][j] for k in range(64)) for i in range(16) for j in range(16)]


def test_a_fault_and_ctrl_c_with_tiles_configured():
    """A fault while tiles are configured raises KernelError, releases them, and the next kernel's tiles work; so
    does Ctrl-C in a loop of tile instructions (the signal frame holds the tile state: our alternate stack is big
    enough for it)."""
    _need_amx()
    m = cimo.compile(_SRC, cache=False)
    a = (ctypes.c_uint16 * 512)(*[_bf16(1.0)] * 512)
    b = (ctypes.c_uint16 * 512)(*[_bf16(2.0)] * 512)
    with pytest.raises(cimo.KernelError, match='memory fault'):
        m.fault(a, 8)
    c = (ctypes.c_float * 256)()
    m.bf16(a, b, c)
    assert list(c) == [64.0] * 256
    if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        pytest.skip('a SIGINT handler of its own')
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    t0 = time.time()
    with pytest.raises(KeyboardInterrupt):
        m.spin(a, b, c, 10 ** 12)
    assert time.time() - t0 < 30
    m.bf16(a, b, c)
    assert list(c) == [64.0 * 2] * 256
