/* cimo's immintrin.h: only what cimo implements, so a kernel written against it builds under gcc and clang
   too. The dot products are instructions and need their CPU features (an error names them otherwise; see
   cimo.cpu_features()); the rest is C on GNU vector types, as gcc's own header computes it. */
#ifndef __CIMO_IMMINTRIN_H
#define __CIMO_IMMINTRIN_H

#ifndef __x86_64__
#error "immintrin.h is for x86-64"
#endif

typedef float __m128 __attribute__((vector_size(16)));
typedef double __m128d __attribute__((vector_size(16)));
typedef long long __m128i __attribute__((vector_size(16)));
typedef __bf16 __m128bh __attribute__((vector_size(16)));
typedef float __m256 __attribute__((vector_size(32)));
typedef double __m256d __attribute__((vector_size(32)));
typedef long long __m256i __attribute__((vector_size(32)));
typedef __bf16 __m256bh __attribute__((vector_size(32)));
typedef float __m512 __attribute__((vector_size(64)));
typedef double __m512d __attribute__((vector_size(64)));
typedef long long __m512i __attribute__((vector_size(64)));
typedef __bf16 __m512bh __attribute__((vector_size(64)));
typedef int __cimo_v4si __attribute__((vector_size(16)));
typedef int __cimo_v8si __attribute__((vector_size(32)));
typedef int __cimo_v16si __attribute__((vector_size(64)));

/* AVX512-BF16 (with AVX512-VL below 512 bits): each float lane + the products of its two bf16 pairs */
__m128 _mm_dpbf16_ps(__m128 __w, __m128bh __a, __m128bh __b) __asm__("__cimo_intrin__mm_dpbf16_ps");
__m256 _mm256_dpbf16_ps(__m256 __w, __m256bh __a, __m256bh __b) __asm__("__cimo_intrin__mm256_dpbf16_ps");
__m512 _mm512_dpbf16_ps(__m512 __w, __m512bh __a, __m512bh __b) __asm__("__cimo_intrin__mm512_dpbf16_ps");
/* AVX512-VNNI, or AVX-VNNI's _avx_ ones: each 32-bit lane + the products of its four unsigned bytes of a and
   signed bytes of b */
__m128i _mm_dpbusd_epi32(__m128i __w, __m128i __a, __m128i __b) __asm__("__cimo_intrin__mm_dpbusd_epi32");
__m256i _mm256_dpbusd_epi32(__m256i __w, __m256i __a, __m256i __b) __asm__("__cimo_intrin__mm256_dpbusd_epi32");
__m512i _mm512_dpbusd_epi32(__m512i __w, __m512i __a, __m512i __b) __asm__("__cimo_intrin__mm512_dpbusd_epi32");
__m128i _mm_dpbusd_avx_epi32(__m128i __w, __m128i __a, __m128i __b) __asm__("__cimo_intrin__mm_dpbusd_avx_epi32");
__m256i _mm256_dpbusd_avx_epi32(__m256i __w, __m256i __a, __m256i __b)
    __asm__("__cimo_intrin__mm256_dpbusd_avx_epi32");

/* AMX (amx-tile; amx-bf16 for _tile_dpbf16ps, amx-int8 for the byte ones): tiles by number, a constant from 0 to 7.
   A kernel configures them (64 bytes: palette 1, then each tile's bytes a row and its rows), works, and releases
   them; cimo asks the OS for the tile state when it loads a module that uses them. */
void _tile_loadconfig(const void *__config) __asm__("__cimo_intrin__tile_loadconfig");
void _tile_storeconfig(void *__config) __asm__("__cimo_intrin__tile_storeconfig");
void _tile_release(void) __asm__("__cimo_intrin__tile_release");
void _tile_zero(int __tile) __asm__("__cimo_intrin__tile_zero");
void _tile_loadd(int __dst, const void *__base, long __stride) __asm__("__cimo_intrin__tile_loadd");
void _tile_stream_loadd(int __dst, const void *__base, long __stride) __asm__("__cimo_intrin__tile_stream_loadd");
void _tile_stored(int __src, void *__base, long __stride) __asm__("__cimo_intrin__tile_stored");
void _tile_dpbf16ps(int __dst, int __a, int __b) __asm__("__cimo_intrin__tile_dpbf16ps");
void _tile_dpbssd(int __dst, int __a, int __b) __asm__("__cimo_intrin__tile_dpbssd");
void _tile_dpbsud(int __dst, int __a, int __b) __asm__("__cimo_intrin__tile_dpbsud");
void _tile_dpbusd(int __dst, int __a, int __b) __asm__("__cimo_intrin__tile_dpbusd");
void _tile_dpbuud(int __dst, int __a, int __b) __asm__("__cimo_intrin__tile_dpbuud");

/* macros, not functions: cimo copies functions with vector arguments or results into their callers only as calls */
#define __CIMO_LD(vt, p) ({ vt __v; __builtin_memcpy(&__v, (p), sizeof __v); __v; })
#define __CIMO_ST(vt, p, v) ({ vt __v = (v); __builtin_memcpy((p), &__v, sizeof __v); (void)0; })
#define _mm_loadu_ps(p) __CIMO_LD(__m128, (const float *)(p))
#define _mm256_loadu_ps(p) __CIMO_LD(__m256, (const float *)(p))
#define _mm512_loadu_ps(p) __CIMO_LD(__m512, (const float *)(p))
#define _mm_loadu_si128(p) __CIMO_LD(__m128i, (const void *)(p))
#define _mm256_loadu_si256(p) __CIMO_LD(__m256i, (const void *)(p))
#define _mm512_loadu_si512(p) __CIMO_LD(__m512i, (const void *)(p))
#define _mm_storeu_ps(p, v) __CIMO_ST(__m128, (float *)(p), (v))
#define _mm256_storeu_ps(p, v) __CIMO_ST(__m256, (float *)(p), (v))
#define _mm512_storeu_ps(p, v) __CIMO_ST(__m512, (float *)(p), (v))
#define _mm_storeu_si128(p, v) __CIMO_ST(__m128i, (void *)(p), (v))
#define _mm256_storeu_si256(p, v) __CIMO_ST(__m256i, (void *)(p), (v))
#define _mm512_storeu_si512(p, v) __CIMO_ST(__m512i, (void *)(p), (v))
#define _mm_setzero_ps() ((__m128){0})
#define _mm256_setzero_ps() ((__m256){0})
#define _mm512_setzero_ps() ((__m512){0})
#define _mm_setzero_si128() ((__m128i){0})
#define _mm256_setzero_si256() ((__m256i){0})
#define _mm512_setzero_si512() ((__m512i){0})
#define _mm_set1_ps(x) ({ float __x = (x); (__m128){__x, __x, __x, __x}; })
#define _mm256_set1_ps(x) ({ float __x = (x); (__m256){__x, __x, __x, __x, __x, __x, __x, __x}; })
#define _mm512_set1_ps(x) ({ float __x = (x); \
    (__m512){__x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x}; })
#define _mm_set1_epi32(x) ({ int __x = (x); (__m128i)(__cimo_v4si){__x, __x, __x, __x}; })
#define _mm256_set1_epi32(x) ({ int __x = (x); (__m256i)(__cimo_v8si){__x, __x, __x, __x, __x, __x, __x, __x}; })
#define _mm512_set1_epi32(x) ({ int __x = (x); \
    (__m512i)(__cimo_v16si){__x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x}; })
#define _mm_add_ps(a, b)((__m128)(a) + (__m128)(b))
#define _mm256_add_ps(a, b) ((__m256)(a) + (__m256)(b))
#define _mm512_add_ps(a, b) ((__m512)(a) + (__m512)(b))
#define _mm_mul_ps(a, b) ((__m128)(a) * (__m128)(b))
#define _mm256_mul_ps(a, b) ((__m256)(a) * (__m256)(b))
#define _mm512_mul_ps(a, b) ((__m512)(a) * (__m512)(b))
#define _mm_add_epi32(a, b) ((__m128i)((__cimo_v4si)(a) + (__cimo_v4si)(b)))
#define _mm256_add_epi32(a, b) ((__m256i)((__cimo_v8si)(a) + (__cimo_v8si)(b)))
#define _mm512_add_epi32(a, b) ((__m512i)((__cimo_v16si)(a) + (__cimo_v16si)(b)))
/* the sums of all lanes in gcc's order: the halves added, then their halves, down to one lane */
#define _mm512_reduce_add_ps(a) ({ __m512 __a = (a); \
    float __t0 = (__a[8] + __a[0]) + (__a[12] + __a[4]), __t1 = (__a[9] + __a[1]) + (__a[13] + __a[5]); \
    float __t2 = (__a[10] + __a[2]) + (__a[14] + __a[6]), __t3 = (__a[11] + __a[3]) + (__a[15] + __a[7]); \
    (__t0 + __t2) + (__t1 + __t3); })
#define _mm512_reduce_add_epi32(a) ({ __cimo_v16si __a = (__cimo_v16si)(a); unsigned __s = 0; \
    for (int __i = 0; __i < 16; __i++) __s += (unsigned)__a[__i]; (int)__s; })

#endif
