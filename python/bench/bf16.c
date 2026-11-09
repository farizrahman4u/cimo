/* bf16 kernels written with the CPU's intrinsics (cimo's bundled arm_neon.h and immintrin.h, or gcc's), for
   python/bench/bf16.py: a dot product with several accumulators and a matrix product block, both over bf16
   inputs into float. B of a matrix product is packed in pairs along k ([k/2][n][2]), the layout bfdot,
   vdpbf16ps and AMX's tdpbf16ps read; each kernel keeps C's order of the float sums, so cimo's and gcc's
   results are the same bits. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>

/* four accumulators of 4 lanes, 32 bf16 pairs an iteration */
float bfdot(const uint16_t *a, const uint16_t *b, long n)
{
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
    long i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = vbfdotq_f32(s0, vld1q_bf16((const bfloat16_t *)a + i), vld1q_bf16((const bfloat16_t *)b + i));
        s1 = vbfdotq_f32(s1, vld1q_bf16((const bfloat16_t *)a + i + 8), vld1q_bf16((const bfloat16_t *)b + i + 8));
        s2 = vbfdotq_f32(s2, vld1q_bf16((const bfloat16_t *)a + i + 16), vld1q_bf16((const bfloat16_t *)b + i + 16));
        s3 = vbfdotq_f32(s3, vld1q_bf16((const bfloat16_t *)a + i + 24), vld1q_bf16((const bfloat16_t *)b + i + 24));
    }
    float32x4_t s = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    return (s[0] + s[1]) + (s[2] + s[3]);
}

/* C[m][n] = A[m][k] B[k][n]: blocks of 4 rows by 8 columns, 8 accumulators; m % 4, n % 8, k % 2 zero */
void gemm(long m, long n, long k, const uint16_t *A, const uint16_t *Bp, float *C)
{
    for (long i = 0; i < m; i += 4)
        for (long j = 0; j < n; j += 8) {
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            for (long p = 0; p < k / 2; p++) {
                bfloat16x8_t b0 = vld1q_bf16((const bfloat16_t *)Bp + (p * n + j) * 2),
                             b1 = vld1q_bf16((const bfloat16_t *)Bp + (p * n + j + 4) * 2);
                uint32_t w0, w1, w2, w3;
                memcpy(&w0, A + (i + 0) * k + 2 * p, 4);
                memcpy(&w1, A + (i + 1) * k + 2 * p, 4);
                memcpy(&w2, A + (i + 2) * k + 2 * p, 4);
                memcpy(&w3, A + (i + 3) * k + 2 * p, 4);
                bfloat16x8_t a0 = (bfloat16x8_t)vdupq_n_u32(w0), a1 = (bfloat16x8_t)vdupq_n_u32(w1);
                bfloat16x8_t a2 = (bfloat16x8_t)vdupq_n_u32(w2), a3 = (bfloat16x8_t)vdupq_n_u32(w3);
                c00 = vbfdotq_f32(c00, a0, b0); c01 = vbfdotq_f32(c01, a0, b1);
                c10 = vbfdotq_f32(c10, a1, b0); c11 = vbfdotq_f32(c11, a1, b1);
                c20 = vbfdotq_f32(c20, a2, b0); c21 = vbfdotq_f32(c21, a2, b1);
                c30 = vbfdotq_f32(c30, a3, b0); c31 = vbfdotq_f32(c31, a3, b1);
            }
            vst1q_f32(C + (i + 0) * n + j, c00); vst1q_f32(C + (i + 0) * n + j + 4, c01);
            vst1q_f32(C + (i + 1) * n + j, c10); vst1q_f32(C + (i + 1) * n + j + 4, c11);
            vst1q_f32(C + (i + 2) * n + j, c20); vst1q_f32(C + (i + 2) * n + j + 4, c21);
            vst1q_f32(C + (i + 3) * n + j, c30); vst1q_f32(C + (i + 3) * n + j + 4, c31);
        }
}

#elif defined(__x86_64__)
#include <immintrin.h>

static float sum8(__m256 s) { return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7])); }

/* 256 bits (AVX512-BF16 with AVX512-VL): four accumulators of 8 lanes */
float bfdot256(const uint16_t *a, const uint16_t *b, long n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    for (long i = 0; i + 64 <= n; i += 64) {
        s0 = _mm256_dpbf16_ps(s0, (__m256bh)_mm256_loadu_si256(a + i), (__m256bh)_mm256_loadu_si256(b + i));
        s1 = _mm256_dpbf16_ps(s1, (__m256bh)_mm256_loadu_si256(a + i + 16), (__m256bh)_mm256_loadu_si256(b + i + 16));
        s2 = _mm256_dpbf16_ps(s2, (__m256bh)_mm256_loadu_si256(a + i + 32), (__m256bh)_mm256_loadu_si256(b + i + 32));
        s3 = _mm256_dpbf16_ps(s3, (__m256bh)_mm256_loadu_si256(a + i + 48), (__m256bh)_mm256_loadu_si256(b + i + 48));
    }
    return sum8(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
}

/* 512 bits: four accumulators of 16 lanes */
float bfdot512(const uint16_t *a, const uint16_t *b, long n)
{
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps(), s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (long i = 0; i + 128 <= n; i += 128) {
        s0 = _mm512_dpbf16_ps(s0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
        s1 = _mm512_dpbf16_ps(s1, (__m512bh)_mm512_loadu_si512(a + i + 32), (__m512bh)_mm512_loadu_si512(b + i + 32));
        s2 = _mm512_dpbf16_ps(s2, (__m512bh)_mm512_loadu_si512(a + i + 64), (__m512bh)_mm512_loadu_si512(b + i + 64));
        s3 = _mm512_dpbf16_ps(s3, (__m512bh)_mm512_loadu_si512(a + i + 96), (__m512bh)_mm512_loadu_si512(b + i + 96));
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

#define GEMM(name, W, vt, bt, zero, ld, dp, st, set1) \
void name(long m, long n, long k, const uint16_t *A, const uint16_t *Bp, float *C) \
{ \
    for (long i = 0; i < m; i += 4) \
        for (long j = 0; j < n; j += 2 * W) { \
            vt c00 = zero, c01 = zero, c10 = zero, c11 = zero, c20 = zero, c21 = zero, c30 = zero, c31 = zero; \
            for (long p = 0; p < k / 2; p++) { \
                bt b0 = (bt)ld(Bp + (p * n + j) * 2), b1 = (bt)ld(Bp + (p * n + j + W) * 2); \
                int w0, w1, w2, w3; \
                memcpy(&w0, A + (i + 0) * k + 2 * p, 4); \
                memcpy(&w1, A + (i + 1) * k + 2 * p, 4); \
                memcpy(&w2, A + (i + 2) * k + 2 * p, 4); \
                memcpy(&w3, A + (i + 3) * k + 2 * p, 4); \
                bt a0 = (bt)set1(w0), a1 = (bt)set1(w1), a2 = (bt)set1(w2), a3 = (bt)set1(w3); \
                c00 = dp(c00, a0, b0); c01 = dp(c01, a0, b1); c10 = dp(c10, a1, b0); c11 = dp(c11, a1, b1); \
                c20 = dp(c20, a2, b0); c21 = dp(c21, a2, b1); c30 = dp(c30, a3, b0); c31 = dp(c31, a3, b1); \
            } \
            st(C + (i + 0) * n + j, c00); st(C + (i + 0) * n + j + W, c01); \
            st(C + (i + 1) * n + j, c10); st(C + (i + 1) * n + j + W, c11); \
            st(C + (i + 2) * n + j, c20); st(C + (i + 2) * n + j + W, c21); \
            st(C + (i + 3) * n + j, c30); st(C + (i + 3) * n + j + W, c31); \
        } \
}
/* C[m][n] = A[m][k] B[k][n] in blocks of 4 rows by 2 vectors of columns: m % 4, n % 16 (or 32), k % 2 zero */
GEMM(gemm256, 8, __m256, __m256bh, _mm256_setzero_ps(), _mm256_loadu_si256, _mm256_dpbf16_ps, _mm256_storeu_ps,
     _mm256_set1_epi32)
GEMM(gemm512, 16, __m512, __m512bh, _mm512_setzero_ps(), _mm512_loadu_si512, _mm512_dpbf16_ps, _mm512_storeu_ps,
     _mm512_set1_epi32)

/* AMX (amx-tile, amx-bf16): blocks of 16 rows by 32 columns in tiles 0 and 1, 16 rows by 32 of k of A in tile 2,
   16 pairs of k by 16 columns of B in tiles 3 and 4; m % 16, n % 32, k % 32 zero */
void gemm_amx(long m, long n, long k, const uint16_t *A, const uint16_t *Bp, float *C)
{
    unsigned char cfg[64] = {1};
    for (int t = 0; t < 5; t++) {
        cfg[16 + 2 * t] = 64;  /* bytes a row */
        cfg[48 + t] = 16;      /* rows */
    }
    _tile_loadconfig(cfg);
    for (long i = 0; i < m; i += 16)
        for (long j = 0; j < n; j += 32) {
            _tile_zero(0);
            _tile_zero(1);
            for (long p = 0; p < k; p += 32) {
                _tile_loadd(2, A + i * k + p, 2 * k);
                _tile_loadd(3, Bp + (p / 2 * n + j) * 2, 4 * n);
                _tile_loadd(4, Bp + (p / 2 * n + j + 16) * 2, 4 * n);
                _tile_dpbf16ps(0, 2, 3);
                _tile_dpbf16ps(1, 2, 4);
            }
            _tile_stored(0, C + i * n + j, 4 * n);
            _tile_stored(1, C + i * n + j + 16, 4 * n);
        }
    _tile_release();
}
#endif
