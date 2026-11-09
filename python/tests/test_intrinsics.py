"""The CPU intrinsics of the bundled arm_neon.h and immintrin.h: dot products as instructions (arm64 bfdot, sdot,
udot, usdot; x86-64 vdpbf16ps, vpdpbusd), checked against scalar C, the error for a CPU without the feature, and
the x86-64 code where the host can't run it. Encodings are in test_assembler.py and data/amd64_forms.s."""

import ctypes
import random
import struct

import pytest

import cimo
from cimo import cpu, driver
import hosts

_X86_ALL = frozenset(('popcnt', 'fma', 'avx', 'avx2', 'bmi1', 'bmi2', 'lzcnt', 'f16c', 'avx512f', 'avx512bw',
                      'avx512dq', 'avx512vl', 'avx512bf16', 'avx512vnni'))


def _bf16(x):
    """The bf16 bits of a float that has an exact bf16 value."""
    b = struct.unpack('<I', struct.pack('<f', x))[0]
    assert b & 0xffff == 0
    return b >> 16


def _x86asm(src, feats, monkeypatch, target='amd64_sysv'):
    monkeypatch.setattr(driver, '_x86', lambda t=None: feats if (t or driver.TARGET).startswith('amd64') else frozenset())
    monkeypatch.setattr(driver, '_cpu', lambda t=None: feats)
    monkeypatch.setattr(driver, 'TARGET', target)
    pp = driver._preprocess(src, 'x.c', (), False, (), target=target)
    ssa, params = driver._ir(pp, 'x.c')
    text = driver._qbe(ssa, params, target=target)
    from cimo import assembler_amd64
    assembler_amd64.assemble(text)
    return text


_ARM = r'''
#include <arm_neon.h>
#include <stddef.h>
float bfdot(const bfloat16_t *a, const bfloat16_t *b, size_t n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vbfdotq_f32(s0, vld1q_bf16(a + i), vld1q_bf16(b + i));
        s1 = vbfdotq_f32(s1, vld1q_bf16(a + i + 8), vld1q_bf16(b + i + 8));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++)
        s += (float)a[i] * (float)b[i];
    return s;
}
void lanes(const bfloat16_t *a, const bfloat16_t *b, float *acc) {
    vst1q_f32(acc, vbfdotq_f32(vld1q_f32(acc), vld1q_bf16(a), vld1q_bf16(b)));
}
int sdot(const int8_t *a, const int8_t *b, size_t n) {
    int32x4_t s = vdupq_n_s32(0);
    for (size_t i = 0; i + 16 <= n; i += 16)
        s = vdotq_s32(s, vld1q_s8(a + i), vld1q_s8(b + i));
    return vaddvq_s32(s);
}
unsigned udot(const uint8_t *a, const uint8_t *b, size_t n) {
    uint32x4_t s = vdupq_n_u32(0);
    for (size_t i = 0; i + 16 <= n; i += 16)
        s = vdotq_u32(s, vld1q_u8(a + i), vld1q_u8(b + i));
    return vaddvq_u32(s);
}
int usdot(const uint8_t *a, const int8_t *b, size_t n) {
    int32x4_t s = vdupq_n_s32(0);
    for (size_t i = 0; i + 16 <= n; i += 16)
        s = vusdotq_s32(s, vld1q_u8(a + i), vld1q_s8(b + i));
    return vaddvq_s32(s);
}
'''


def _need_arm(*feats):
    if not hosts.ARM64:
        pytest.skip('runs arm64 code')
    lack = set(feats) - cpu.arm64_features()
    if lack:
        pytest.skip('a CPU without %s' % ', '.join(sorted(lack)))


def test_bfdot_against_scalar():
    _need_arm('bf16')
    m = cimo.compile(_ARM, cache=False)
    rnd = random.Random(5)
    for n in (0, 1, 15, 16, 17, 32, 100, 257):
        xs = [float(rnd.randint(-64, 64)) for _ in range(n)]  # products and sums exact in float
        ys = [float(rnd.randint(-64, 64)) for _ in range(n)]
        a = (ctypes.c_uint16 * max(n, 1))(*[_bf16(x) for x in xs])
        b = (ctypes.c_uint16 * max(n, 1))(*[_bf16(y) for y in ys])
        assert m.bfdot(a, b, n) == sum(x * y for x, y in zip(xs, ys))
    # lane by lane: lane i + a[2i]*b[2i] + a[2i+1]*b[2i+1]
    xs, ys = [1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 0.5, 0.25], [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 4.0]
    acc = (ctypes.c_float * 4)(100.0, 200.0, 300.0, 400.0)
    m.lanes((ctypes.c_uint16 * 8)(*map(_bf16, xs)), (ctypes.c_uint16 * 8)(*map(_bf16, ys)), acc)
    assert list(acc) == [100 + 8 + 14, 200 + 18 + 20, 300 - 20 + 18, 400 + 1 + 1]


def test_byte_dots_against_scalar():
    _need_arm('dotprod', 'i8mm')
    m = cimo.compile(_ARM, cache=False)
    rnd = random.Random(7)
    for n in (0, 16, 48, 160):
        s8 = [rnd.randint(-128, 127) for _ in range(n)]
        t8 = [rnd.randint(-128, 127) for _ in range(n)]
        u8 = [rnd.randint(0, 255) for _ in range(n)]
        v8 = [rnd.randint(0, 255) for _ in range(n)]
        arr = (lambda t, xs: (t * max(n, 1))(*xs))
        assert m.sdot(arr(ctypes.c_int8, s8), arr(ctypes.c_int8, t8), n) == sum(x * y for x, y in zip(s8, t8))
        assert m.udot(arr(ctypes.c_uint8, u8), arr(ctypes.c_uint8, v8), n) == sum(x * y for x, y in zip(u8, v8))
        assert m.usdot(arr(ctypes.c_uint8, u8), arr(ctypes.c_int8, t8), n) == sum(x * y for x, y in zip(u8, t8))


def test_a_cpu_without_the_feature_is_an_error(monkeypatch):
    """The intrinsic's call is a compile error naming the feature, on a CPU without it."""
    if hosts.ARM64:
        monkeypatch.setattr(cpu, '_cached_arm', frozenset(('dotprod',)))
        with pytest.raises(cimo.CompileError, match=r"'vbfdotq_f32' needs the CPU feature bf16, which this CPU lacks"):
            cimo.compile(_ARM, cache=False)
    elif hosts.AMD64:
        monkeypatch.setattr(cpu, '_cached', frozenset(('avx', 'avx2', 'fma', 'avx512f', 'avx512vl')))
        with pytest.raises(cimo.CompileError, match=r"needs the CPU features avx512bf16 and avx512vl, which this CPU "
                                                   r"lacks \(avx512bf16\)"):
            cimo.compile(_X86, cache=False)
    else:
        pytest.skip('compiles native code')


def test_the_feature_macros(monkeypatch):
    src = ('#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)\n'
           'int f(void) { return 1; }\n#else\nint f(void) { return 0; }\n#endif\n')
    if not hosts.RUNS:
        pytest.skip('compiles native code')
    have = 'bf16' in cpu.features() or 'avx512bf16' in cpu.features()
    assert cimo.compile(src, cache=False).f() == int(have)
    monkeypatch.setattr(cpu, '_cached_arm', frozenset())
    monkeypatch.setattr(cpu, '_cached', frozenset())
    assert cimo.compile(src, cache=False).f() == 0


def test_cpu_features_is_public():
    f = cimo.cpu_features()
    assert f == cpu.features() and f <= cpu.KNOWN | cpu.ARM64_KNOWN
    assert not (hosts.ARM64 and f & cpu.KNOWN) and not (hosts.AMD64 and f & cpu.ARM64_KNOWN)


def test_arm64_override(monkeypatch):
    monkeypatch.setattr(cpu, '_cached_arm', None)
    monkeypatch.setenv('CIMO_ARM64', 'baseline')
    assert cpu.arm64_features() == frozenset()
    monkeypatch.setattr(cpu, '_cached_arm', None)
    monkeypatch.setenv('CIMO_ARM64', 'bf16,i8mm')
    assert cpu.arm64_features() == (frozenset(('bf16', 'i8mm')) if hosts.ARM64 else frozenset())
    monkeypatch.setattr(cpu, '_cached_arm', None)
    monkeypatch.setenv('CIMO_ARM64', 'avx2')
    with pytest.raises(ValueError):
        cpu.arm64_features()
    monkeypatch.setattr(cpu, '_cached_arm', None)


def test_arm64_detection_paths(monkeypatch):
    """Each OS's source of the feature bits, read the way the OS gives them."""
    class Libc:
        def __init__(self, caps):
            self.caps = caps

        @property
        def getauxval(self):
            caps = self.caps

            class F:
                restype = argtypes = None

                def __call__(self, n):
                    return caps.get(n, 0)
            return F()
    monkeypatch.setattr(ctypes, 'CDLL', lambda name=None: Libc({16: 1 << 20, 26: 1 << 14}))
    assert cpu._linux_arm64() == frozenset(('dotprod', 'bf16'))
    monkeypatch.setattr(ctypes, 'CDLL', lambda name=None: Libc({16: 0, 26: 1 << 13}))
    assert cpu._linux_arm64() == frozenset(('i8mm',))


def test_features_change_the_arm64_cache_key(monkeypatch):
    monkeypatch.setattr(driver, 'TARGET', 'arm64')
    monkeypatch.setattr(driver, '_cpu', lambda t=None: frozenset())
    a = driver._key('int f(void) { return 1; }')
    monkeypatch.setattr(driver, '_cpu', lambda t=None: frozenset(('bf16',)))
    assert driver._key('int f(void) { return 1; }') != a


_X86 = r'''
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
float bf512(const uint16_t *a, const uint16_t *b, size_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_dpbf16_ps(s0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
        s1 = _mm512_dpbf16_ps(s1, (__m512bh)_mm512_loadu_si512(a + i + 32), (__m512bh)_mm512_loadu_si512(b + i + 32));
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}
void lanes256(const uint16_t *a, const uint16_t *b, float *acc) {
    _mm256_storeu_ps(acc, _mm256_dpbf16_ps(_mm256_loadu_ps(acc), (__m256bh)_mm256_loadu_si256(a),
                                           (__m256bh)_mm256_loadu_si256(b)));
}
void lanes128(const uint16_t *a, const uint16_t *b, float *acc) {
    _mm_storeu_ps(acc, _mm_dpbf16_ps(_mm_loadu_ps(acc), (__m128bh)_mm_loadu_si128(a), (__m128bh)_mm_loadu_si128(b)));
}
int us512(const uint8_t *a, const int8_t *b, size_t n) {
    __m512i s = _mm512_setzero_si512();
    for (size_t i = 0; i + 64 <= n; i += 64)
        s = _mm512_dpbusd_epi32(s, _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
    return _mm512_reduce_add_epi32(s);
}
void us256(const uint8_t *a, const int8_t *b, int *acc) {
    _mm256_storeu_si256(acc, _mm256_dpbusd_epi32(_mm256_loadu_si256(acc), _mm256_loadu_si256(a), _mm256_loadu_si256(b)));
}
'''
_X86_AVXVNNI = r'''
#include <immintrin.h>
void us256(const unsigned char *a, const signed char *b, int *acc) {
    _mm256_storeu_si256(acc, _mm256_dpbusd_avx_epi32(_mm256_loadu_si256(acc), _mm256_loadu_si256(a),
                                                     _mm256_loadu_si256(b)));
}
'''


def test_x86_code(monkeypatch):
    """vdpbf16ps on ymm, accumulating in place in the loop; vpdpbusd as EVEX, or VEX with AVX-VNNI."""
    text = _x86asm(_X86, _X86_ALL, monkeypatch)
    body = text[text.index('\nbf512:'):text.index('end function bf512')]
    loop = body[body.index('.p2align 6'):body.index('jmp')]
    assert loop.count('vdpbf16ps') == 4 and 'vmovaps' not in loop and '(%rsp)' not in loop, loop
    assert '\tvpdpbusd %ymm' in text  # 16-byte vectors too: in a ymm's lower half (AVX2's q)
    text = _x86asm(_X86_AVXVNNI, frozenset(('avx', 'avx2', 'fma', 'avxvnni')), monkeypatch)
    assert '{vex} vpdpbusd %ymm' in text


def test_x86_without_the_feature(monkeypatch):
    with pytest.raises(cimo.CompileError, match=r"'_mm256_dpbusd_avx_epi32' needs the CPU feature avxvnni"):
        _x86asm(_X86_AVXVNNI, _X86_ALL, monkeypatch)


@pytest.mark.skipif(not hosts.AMD64, reason='runs x86-64 code')
def test_x86_against_scalar():
    have = cpu.x86_features()
    if not {'avx512bf16', 'avx512vnni', 'avx512vl'} <= have:
        pytest.skip('a CPU without AVX512-BF16 and AVX512-VNNI')
    m = cimo.compile(_X86, cache=False)
    rnd = random.Random(3)
    for n in (0, 64, 192):
        xs = [float(rnd.randint(-64, 64)) for _ in range(n)]
        ys = [float(rnd.randint(-64, 64)) for _ in range(n)]
        a = (ctypes.c_uint16 * max(n, 1))(*[_bf16(x) for x in xs])
        b = (ctypes.c_uint16 * max(n, 1))(*[_bf16(y) for y in ys])
        assert m.bf512(a, b, n) == sum(x * y for x, y in zip(xs, ys))
        u8 = [rnd.randint(0, 255) for _ in range(n)]
        s8 = [rnd.randint(-128, 127) for _ in range(n)]
        assert m.us512((ctypes.c_uint8 * max(n, 1))(*u8), (ctypes.c_int8 * max(n, 1))(*s8), n) == \
            sum(x * y for x, y in zip(u8, s8))
    xs = [float(k % 7 - 3) for k in range(16)]
    ys = [float(k % 5 - 2) for k in range(16)]
    for f, lanes in ((m.lanes256, 8), (m.lanes128, 4)):
        acc = (ctypes.c_float * 8)(*[100.0 * k for k in range(8)])
        f((ctypes.c_uint16 * 16)(*map(_bf16, xs)), (ctypes.c_uint16 * 16)(*map(_bf16, ys)), acc)
        want = [100.0 * k + xs[2 * k] * ys[2 * k] + xs[2 * k + 1] * ys[2 * k + 1] if k < lanes else 100.0 * k
                for k in range(8)]
        assert list(acc) == want
    u8 = [rnd.randint(0, 255) for _ in range(32)]
    s8 = [rnd.randint(-128, 127) for _ in range(32)]
    acc = (ctypes.c_int * 8)(*range(8))
    m.us256((ctypes.c_uint8 * 32)(*u8), (ctypes.c_int8 * 32)(*s8), acc)
    assert list(acc) == [k + sum(u8[4 * k + j] * s8[4 * k + j] for j in range(4)) for k in range(8)]


_HALFVEC = r'''
typedef __bf16 bf8 __attribute__((vector_size(16)));
typedef _Float16 h8 __attribute__((vector_size(16)));
typedef short s8 __attribute__((vector_size(16)));
void add(const bf8 *a, const bf8 *b, bf8 *r) { *r = *a + *b; }
void mul(const h8 *a, const h8 *b, h8 *r) { *r = *a * *b; }
void neg(const bf8 *a, bf8 *r) { *r = -*a; }
void lt(const h8 *a, const h8 *b, s8 *r) { *r = *a < *b; }
static float lane(bf8 a, int i) { return a[3] + a[i]; }
float lanes(const bf8 *p, int i) { return lane(*p, i); }
'''


def test_half_vectors_compute_in_float():
    """Arithmetic on vectors of __bf16 and _Float16 lanes is each lane's in float, rounded back (it was integer or
    double arithmetic on the bits)."""
    if not hosts.RUNS:
        pytest.skip('compiles native code')
    m = cimo.compile(_HALFVEC, cache=False)
    xs = [1.0, -2.0, 3.5, 0.25, 8.0, -0.5, 100.0, 6.0]
    ys = [2.0, 2.0, -1.5, 4.0, 0.125, -0.5, 3.0, 6.0]
    a = (ctypes.c_uint16 * 8)(*map(_bf16, xs))
    b = (ctypes.c_uint16 * 8)(*map(_bf16, ys))
    r = (ctypes.c_uint16 * 8)()
    m.add(a, b, r)
    assert list(r) == [_bf16(x + y) for x, y in zip(xs, ys)]
    m.neg(a, r)
    assert list(r) == [_bf16(-x) for x in xs]
    assert m.lanes(a, 6) == 0.25 + 100.0
    h = (lambda vs: (ctypes.c_uint16 * 8)(*[struct.unpack('<H', struct.pack('<e', v))[0] for v in vs]))
    m.mul(h(xs), h(ys), r)
    assert list(r) == list(h([x * y for x, y in zip(xs, ys)]))
    s = (ctypes.c_int16 * 8)()
    m.lt(h(xs), h(ys), s)
    assert list(s) == [-1 if x < y else 0 for x, y in zip(xs, ys)]


_BCAST = r'''
typedef float v4 __attribute__((vector_size(16)));
typedef float v8 __attribute__((vector_size(32)));
typedef float v16 __attribute__((vector_size(64)));
void axpy(long n, const float *a, const v4 *x, v4 *y) {
    for (long i = 0; i < n; i++) {
        float s = a[i];
        y[i] += (v4){s, s, s, s} * x[i];
    }
}
void zero16(v16 *p) { *p = (v16){0}; }
void zero8(v8 *p) { v8 z = {0}; *p = z; }
'''


def test_broadcast_from_memory(monkeypatch):
    """A value loaded only to be broadcast is broadcast from memory (x86-64 vbroadcastss, arm64 ldr s and dup of the
    lane), and a vector of 2 q values set to {0} is one qdup in registers (sroa used to put its slots before the
    parameters, which then reached the emitter)."""
    if hosts.ARM64:
        m = cimo.compile(_BCAST, cache=False)
        a = (ctypes.c_float * 3)(2.0, -1.0, 0.5)
        x = (ctypes.c_float * 12)(*range(12))
        y = (ctypes.c_float * 12)(*[1.0] * 12)
        m.axpy(3, a, x, y)
        assert list(y) == [1.0 + a[k // 4] * k for k in range(12)]
        ssa, params = driver._ir(driver._preprocess(_BCAST, 'x.c'), 'x.c')
        asm = driver._qbe(ssa, params)
        b = asm[asm.index('axpy:\n'):asm.index('end function')]  # _axpy on macOS
        assert 'ldr\ts' in b and '.s[0]' in b, b
    text = _x86asm(_BCAST, _X86_ALL, monkeypatch)
    body = text[text.index('\naxpy:'):text.index('end function axpy')]
    assert 'vbroadcastss (' in body and 'vmovd' not in body, body
    z = text[text.index('\nzero16:'):text.index('end function zero16')]
    assert 'vpxor' in z and 'movl' not in z and 'movq' not in z, z
