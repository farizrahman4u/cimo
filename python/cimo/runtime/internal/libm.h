/* cimo runtime: musl's src/internal/libm.h, cut down for cimo (see ../COPYRIGHT).
   Differences from musl: no long double (cimo rejects binary128), no floating-point
   exception or errno side effects (the runtime's math is pure: fp_barrier and FORCE_EVAL
   are no-ops), helpers are macros so no unused static functions land in every object, and
   aliases/visibility attributes are dropped. */
#ifndef _LIBM_H
#define _LIBM_H

#include <stdint.h>
#include <float.h>
#include <math.h>
#include "musl.h"

/* Support non-nearest rounding mode.  */
#define WANT_ROUNDING 1
/* Support signaling NaNs.  */
#define WANT_SNAN 0

#define issignalingf_inline(x) 0
#define issignaling_inline(x) 0

#define TOINT_INTRINSICS 0

#define predict_true(x) (x)
#define predict_false(x) (x)

#define eval_as_float(x) ((float)(x))
#define eval_as_double(x) ((double)(x))
#define fp_barrierf(x) ((float)(x))
#define fp_barrier(x) ((double)(x))
#define fp_force_evalf(x) ((void)0)
#define fp_force_eval(x) ((void)0)
#define FORCE_EVAL(x) ((void)0)

#define asuint(f) ((union{float _f; uint32_t _i;}){f})._i
#define asfloat(i) ((union{uint32_t _i; float _f;}){i})._f
#define asuint64(f) ((union{double _f; uint64_t _i;}){f})._i
#define asdouble(i) ((union{uint64_t _i; double _f;}){i})._f

#define EXTRACT_WORDS(hi,lo,d)                    \
do {                                              \
  uint64_t __u = asuint64(d);                     \
  (hi) = __u >> 32;                               \
  (lo) = (uint32_t)__u;                           \
} while (0)

#define GET_HIGH_WORD(hi,d)                       \
do {                                              \
  (hi) = asuint64(d) >> 32;                       \
} while (0)

#define GET_LOW_WORD(lo,d)                        \
do {                                              \
  (lo) = (uint32_t)asuint64(d);                   \
} while (0)

#define INSERT_WORDS(d,hi,lo)                     \
do {                                              \
  (d) = asdouble(((uint64_t)(hi)<<32) | (uint32_t)(lo)); \
} while (0)

#define SET_HIGH_WORD(d,hi)                       \
  INSERT_WORDS(d, hi, (uint32_t)asuint64(d))

#define SET_LOW_WORD(d,lo)                        \
  INSERT_WORDS(d, asuint64(d)>>32, lo)

#define GET_FLOAT_WORD(w,d)                       \
do {                                              \
  (w) = asuint(d);                                \
} while (0)

#define SET_FLOAT_WORD(d,w)                       \
do {                                              \
  (d) = asfloat(w);                               \
} while (0)

hidden int    __rem_pio2_large(double*,double*,int,int,int);

hidden int    __rem_pio2(double,double*);
hidden double __sin(double,double,int);
hidden double __cos(double,double);
hidden double __tan(double,double,int);
hidden double __expo2(double,double);

hidden int    __rem_pio2f(float,double*);
hidden float  __sindf(double);
hidden float  __cosdf(double);
hidden float  __tandf(double,int);
hidden float  __expo2f(float,float);

extern int __signgam;
hidden double __lgamma_r(double, int *);
hidden float __lgammaf_r(float, int *);

/* error handling functions */
hidden float __math_xflowf(uint32_t, float);
hidden float __math_uflowf(uint32_t);
hidden float __math_oflowf(uint32_t);
hidden float __math_divzerof(uint32_t);
hidden float __math_invalidf(float);
hidden double __math_xflow(uint32_t, double);
hidden double __math_uflow(uint32_t);
hidden double __math_oflow(uint32_t);
hidden double __math_divzero(uint32_t);
hidden double __math_invalid(double);

#endif
