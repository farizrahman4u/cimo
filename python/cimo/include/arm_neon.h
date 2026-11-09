/* cimo's arm_neon.h: only what cimo implements, so a kernel written against it builds under gcc and clang
   too. The dot products are instructions and need their CPU feature (an error names it otherwise; see
   cimo.cpu_features()); the rest is C on GNU vector types, as gcc's own header computes it. */
#ifndef __CIMO_ARM_NEON_H
#define __CIMO_ARM_NEON_H

#ifndef __aarch64__
#error "arm_neon.h is for arm64"
#endif

#include <stdint.h>

typedef float float32_t;
typedef __bf16 bfloat16_t;
typedef float float32x4_t __attribute__((vector_size(16)));
typedef int32_t int32x4_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));
typedef int16_t int16x8_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef int8_t int8x16_t __attribute__((vector_size(16)));
typedef uint8_t uint8x16_t __attribute__((vector_size(16)));
typedef __bf16 bfloat16x8_t __attribute__((vector_size(16)));

/* FEAT_BF16: each float lane + the products of its two bf16 pairs */
float32x4_t vbfdotq_f32(float32x4_t __r, bfloat16x8_t __a, bfloat16x8_t __b) __asm__("__cimo_intrin_vbfdotq_f32");
/* FEAT_DotProd: each 32-bit lane + the products of its four bytes */
int32x4_t vdotq_s32(int32x4_t __r, int8x16_t __a, int8x16_t __b) __asm__("__cimo_intrin_vdotq_s32");
uint32x4_t vdotq_u32(uint32x4_t __r, uint8x16_t __a, uint8x16_t __b) __asm__("__cimo_intrin_vdotq_u32");
/* FEAT_I8MM: unsigned bytes of a times signed bytes of b */
int32x4_t vusdotq_s32(int32x4_t __r, uint8x16_t __a, int8x16_t __b) __asm__("__cimo_intrin_vusdotq_s32");

/* macros, not functions: cimo copies functions with vector arguments or results into their callers only as calls */
#define __CIMO_LD1(vt, p) ({ vt __v; __builtin_memcpy(&__v, (p), sizeof __v); __v; })
#define __CIMO_ST1(vt, p, v) ({ vt __v = (v); __builtin_memcpy((p), &__v, sizeof __v); (void)0; })
#define vld1q_f32(p) __CIMO_LD1(float32x4_t, (const float32_t *)(p))
#define vld1q_s32(p) __CIMO_LD1(int32x4_t, (const int32_t *)(p))
#define vld1q_u32(p) __CIMO_LD1(uint32x4_t, (const uint32_t *)(p))
#define vld1q_s16(p) __CIMO_LD1(int16x8_t, (const int16_t *)(p))
#define vld1q_u16(p) __CIMO_LD1(uint16x8_t, (const uint16_t *)(p))
#define vld1q_s8(p) __CIMO_LD1(int8x16_t, (const int8_t *)(p))
#define vld1q_u8(p) __CIMO_LD1(uint8x16_t, (const uint8_t *)(p))
#define vld1q_bf16(p) __CIMO_LD1(bfloat16x8_t, (const bfloat16_t *)(p))
#define vst1q_f32(p, v) __CIMO_ST1(float32x4_t, (float32_t *)(p), (v))
#define vst1q_s32(p, v) __CIMO_ST1(int32x4_t, (int32_t *)(p), (v))
#define vst1q_u32(p, v) __CIMO_ST1(uint32x4_t, (uint32_t *)(p), (v))
#define vst1q_s16(p, v) __CIMO_ST1(int16x8_t, (int16_t *)(p), (v))
#define vst1q_u16(p, v) __CIMO_ST1(uint16x8_t, (uint16_t *)(p), (v))
#define vst1q_s8(p, v) __CIMO_ST1(int8x16_t, (int8_t *)(p), (v))
#define vst1q_u8(p, v) __CIMO_ST1(uint8x16_t, (uint8_t *)(p), (v))
#define vst1q_bf16(p, v) __CIMO_ST1(bfloat16x8_t, (bfloat16_t *)(p), (v))
#define vdupq_n_f32(x) ({ float32_t __x = (x); (float32x4_t){__x, __x, __x, __x}; })
#define vdupq_n_s32(x) ({ int32_t __x = (x); (int32x4_t){__x, __x, __x, __x}; })
#define vdupq_n_u32(x) ({ uint32_t __x = (x); (uint32x4_t){__x, __x, __x, __x}; })
#define vaddq_f32(a, b) ((float32x4_t)(a) + (float32x4_t)(b))
#define vaddq_s32(a, b) ((int32x4_t)(a) + (int32x4_t)(b))
#define vaddq_u32(a, b) ((uint32x4_t)(a) + (uint32x4_t)(b))
#define vsubq_f32(a, b) ((float32x4_t)(a) - (float32x4_t)(b))
#define vmulq_f32(a, b) ((float32x4_t)(a) * (float32x4_t)(b))
/* the sums of all lanes, in faddp's order: (a0 + a1) + (a2 + a3) */
#define vaddvq_f32(a) ({ float32x4_t __a = (a); (__a[0] + __a[1]) + (__a[2] + __a[3]); })
#define vaddvq_s32(a) ({ int32x4_t __a = (a); (int32_t)((uint32_t)__a[0] + (uint32_t)__a[1] + (uint32_t)__a[2] + (uint32_t)__a[3]); })
#define vaddvq_u32(a) ({ uint32x4_t __a = (a); __a[0] + __a[1] + __a[2] + __a[3]; })
#define vgetq_lane_f32(a, i) ((float32_t)(a)[i])
#define vgetq_lane_s32(a, i) ((int32_t)(a)[i])
#define vgetq_lane_u32(a, i) ((uint32_t)(a)[i])

#endif
