"""512-bit vectors on x86-64 (AVX-512 F, BW, DQ, VL): the q width each function gets, the zmm code, and results
against scalar C where the host runs it. Encodings are in data/amd64_forms.s."""

import ctypes
import random

import pytest

import cimo
from cimo import cpu, driver
import hosts

_AVX512 = frozenset(('popcnt', 'fma', 'avx', 'avx2', 'bmi1', 'bmi2', 'lzcnt', 'f16c', 'avx512f', 'avx512bw', 'avx512dq',
                     'avx512vl'))


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


def _body(text, f):
    t = text[text.index('\n%s:\n' % f):]
    return t[:t.index('end function')]


_SRC = r'''
typedef int v16 __attribute__((vector_size(64)));
typedef int v8 __attribute__((vector_size(32)));
typedef int v4 __attribute__((vector_size(16)));
void add16(v16 *r, const v16 *a, const v16 *b) { *r = *a + *b * 3; }
void add8(v8 *r, const v8 *a, const v8 *b) { *r = *a + *b * 3; }
void mixed(v16 *r, v8 *s, v4 *t, const v16 *a, const v8 *b, const v4 *c) {
    *r = *a > 5;
    *s = *b - (v8){1, 2, 3, 4, 5, 6, 7, 8};
    *t = *c << 2;
}
void loop(long n, float *restrict y, const float *restrict x) { for (long i = 0; i < n; i++) y[i] += 2 * x[i]; }
'''


def test_each_function_its_width(monkeypatch):
    """64 bytes for a function of 64-byte vectors, 32 for one of 32-byte ones; one with none vectorizes its loops
    at what this CPU runs fastest ('zmm': 512 bits)."""
    ssa, text = _asm(_SRC, _AVX512, monkeypatch)
    assert 'export\nvector 64 function $add16' in ssa and 'export\nvector 32 function $add8' in ssa
    assert '%zmm' in _body(text, 'add16') and '%zmm' not in _body(text, 'add8') and '%ymm' in _body(text, 'add8')
    m = _body(text, 'mixed')  # 32- and 16-byte vectors in a function of 64: in a zmm's lower half or quarter
    assert 'vpcmpgtd' in m and '%k1' in m and 'vpmovm2d' in m and 'vbroadcasti64x4' in m and 'vbroadcasti32x4' in m
    assert '%zmm' not in _body(text, 'loop') and '%ymm' in _body(text, 'loop')
    ssa, text = _asm(_SRC, _AVX512 | {'zmm'}, monkeypatch)
    assert 'export\nvector 64 function $loop' in ssa and '%zmm' in _body(text, 'loop')
    ssa, text = _asm(_SRC, _AVX512 - {'avx512bw'}, monkeypatch)  # without BW and DQ: AVX2's 32 bytes
    assert 'vector' not in ssa and '%zmm' not in text


def test_the_width_line_on_the_function(monkeypatch):
    """`vector 64` is on the function's own line, where prune, the signatures and the stack check read it: a static
    function nothing calls goes with it (gcc's 20000502-1.c left `vector 64` lines alone), main stays callable."""
    src = 'static int f(int a) { return a; }\nstatic int g(void) { return f(1); }\nint main(void) { return 0; }\n'
    ssa, text = _asm(src, _AVX512, monkeypatch)
    assert 'export\nvector 32 function w $main()' in ssa and '$f' not in ssa and ssa.count('vector') == 1
    assert driver._sigs(ssa).split()[0] == 'main'
    from cimo import execution
    assert execution.stack_checked('vector 32 function w $r() {\n@s\n\tcall $r()\n\tret 0\n}\n') == {'r'}


def test_parallel_loops_keep_their_width(monkeypatch):
    """A parallel loop's function of its own (omp.outline) has the width of the function it came from."""
    src = r'''
typedef float v16 __attribute__((vector_size(64)));
void f(long n, v16 *a) {
#pragma omp parallel for
    for (long i = 0; i < n; i++) a[i] = a[i] * 2;
}
'''
    ssa, text = _asm(src, _AVX512, monkeypatch)
    assert ssa.count('vector 64 function') == 2 and 'vmulps' in text and '%zmm' in text


def test_no_inlining_across_widths(monkeypatch):
    """A function's q values have its width: one of 32-byte vectors isn't copied into a function of 64."""
    src = r'''
typedef int v16 __attribute__((vector_size(64)));
typedef int v8 __attribute__((vector_size(32)));
static void twice(v8 *p) { *p = *p + *p; }
void f(v16 *r, v8 *s) { *r = *r + *r; twice(s); }
'''
    ssa, text = _asm(src, _AVX512, monkeypatch)
    assert 'callq twice' in _body(text, 'f')


def _need_avx512():
    if not hosts.AMD64 or not driver._zmm():
        pytest.skip('runs AVX-512 code')


def test_512_bit_gnu_vectors_against_scalar():
    _need_avx512()
    m = cimo.compile(_SRC, cache=False)
    rnd = random.Random(2)
    a = (ctypes.c_int * 16)(*[rnd.randint(-100, 100) for _ in range(16)])
    b = (ctypes.c_int * 16)(*[rnd.randint(-100, 100) for _ in range(16)])
    r = (ctypes.c_int * 16)()
    m.add16(r, a, b)
    assert list(r) == [x + 3 * y for x, y in zip(a, b)]
    m.add8(r, a, b)
    assert list(r)[:8] == [x + 3 * y for x, y in zip(a[:8], b[:8])]
    s, t = (ctypes.c_int * 8)(), (ctypes.c_int * 4)()
    m.mixed(r, s, t, a, b, b)
    assert list(r) == [-1 if x > 5 else 0 for x in a]
    assert list(s) == [y - k - 1 for k, y in enumerate(b[:8])]
    assert list(t) == [(y << 2) for y in b[:4]]


def test_512_bit_loops_against_scalar(monkeypatch):
    """Loops vectorized 512 bits at a time (cimo.cpu's zmm), conversions and narrowings among them."""
    _need_avx512()
    monkeypatch.setattr(cpu, '_cached', cpu.x86_features() | {'zmm'})
    src = r'''
#include <stdint.h>
void conv(long n, double *restrict d, const int32_t *restrict a, const uint8_t *restrict b, uint8_t *restrict c,
          float *restrict f, const int64_t *restrict q, uint16_t *restrict h) {
    for (long i = 0; i < n; i++) {
        d[i] = a[i] * 0.5 + (double)q[i];
        c[i] = (uint8_t)(b[i] + a[i]);
        f[i] = (float)(uint32_t)a[i];
        h[i] = (uint16_t)(a[i] >> 3);
    }
}
'''
    m = cimo.compile(src, cache=False)
    n = 203
    rnd = random.Random(4)
    a = (ctypes.c_int32 * n)(*[rnd.randint(-2 ** 31, 2 ** 31 - 1) for _ in range(n)])
    b = (ctypes.c_uint8 * n)(*[rnd.randint(0, 255) for _ in range(n)])
    q = (ctypes.c_int64 * n)(*[rnd.randint(-2 ** 62, 2 ** 62) for _ in range(n)])
    d, c, f, h = (ctypes.c_double * n)(), (ctypes.c_uint8 * n)(), (ctypes.c_float * n)(), (ctypes.c_uint16 * n)()
    m.conv(n, d, a, b, c, f, q, h)
    for i in range(n):
        assert d[i] == a[i] * 0.5 + float(q[i])
        assert c[i] == (b[i] + a[i]) & 0xff
        assert f[i] == ctypes.c_float(float(a[i] & 0xffffffff)).value
        assert h[i] == (a[i] >> 3) & 0xffff
