/* cimo: <math.h> (C11 7.12), double and float functions. The long double functions (and
   nexttoward) are left out: long double is binary128 on Linux aarch64, which cimo rejects.
   Classification and comparison macros are implemented inline (bit tests, no library calls). */
#ifndef __CIMO_MATH_H
#define __CIMO_MATH_H

#if __FLT_EVAL_METHOD__ == 1
typedef double float_t;
typedef double double_t;
#elif __FLT_EVAL_METHOD__ == 2
typedef long double float_t;
typedef long double double_t;
#else
typedef float float_t;
typedef double double_t;
#endif

#define HUGE_VAL ((double)__builtin_inff())
#define HUGE_VALF __builtin_inff()
#define HUGE_VALL ((long double)__builtin_inff())
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")

#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4

#define FP_ILOGB0 (-2147483647 - 1)
#define FP_ILOGBNAN (-2147483647 - 1)

#define MATH_ERRNO 1
#define MATH_ERREXCEPT 2
#define math_errhandling MATH_ERREXCEPT

static inline unsigned __cimo_fbits(float __f)
{
	union { float __f; unsigned __i; } __u;
	__u.__f = __f;
	return __u.__i;
}

static inline unsigned long long __cimo_dbits(double __d)
{
	union { double __d; unsigned long long __i; } __u;
	__u.__d = __d;
	return __u.__i;
}

static inline int __cimo_fpclassifyf(float __x)
{
	unsigned __i = __cimo_fbits(__x);
	unsigned __e = __i >> 23 & 0xff;
	if (!__e)
		return __i << 1 ? FP_SUBNORMAL : FP_ZERO;
	if (__e == 0xff)
		return __i << 9 ? FP_NAN : FP_INFINITE;
	return FP_NORMAL;
}

static inline int __cimo_fpclassify(double __x)
{
	unsigned long long __i = __cimo_dbits(__x);
	unsigned __e = __i >> 52 & 0x7ff;
	if (!__e)
		return __i << 1 ? FP_SUBNORMAL : FP_ZERO;
	if (__e == 0x7ff)
		return __i << 12 ? FP_NAN : FP_INFINITE;
	return FP_NORMAL;
}

/* Dispatch on the argument type without evaluating it twice (sizeof does not evaluate). */
#define __CIMO_TG(x, f, d) (sizeof(x) == sizeof(float) ? f(x) : d(x))

#define fpclassify(x) __CIMO_TG(x, __cimo_fpclassifyf, __cimo_fpclassify)
#define isinf(x) (sizeof(x) == sizeof(float) \
	? (__cimo_fbits(x) & 0x7fffffff) == 0x7f800000 \
	: (__cimo_dbits(x) & -1ULL >> 1) == 0x7ffULL << 52)
#define isnan(x) (sizeof(x) == sizeof(float) \
	? (__cimo_fbits(x) & 0x7fffffff) > 0x7f800000 \
	: (__cimo_dbits(x) & -1ULL >> 1) > 0x7ffULL << 52)
#define isfinite(x) (sizeof(x) == sizeof(float) \
	? (__cimo_fbits(x) & 0x7fffffff) < 0x7f800000 \
	: (__cimo_dbits(x) & -1ULL >> 1) < 0x7ffULL << 52)
#define isnormal(x) (fpclassify(x) == FP_NORMAL)
#define signbit(x) (sizeof(x) == sizeof(float) \
	? (int)(__cimo_fbits(x) >> 31) \
	: (int)(__cimo_dbits(x) >> 63))

static inline int __cimo_isunordered(double __x, double __y)
{
	return __x != __x || __y != __y;
}

static inline int __cimo_isunorderedf(float __x, float __y)
{
	return __x != __x || __y != __y;
}

static inline int __cimo_gt(double __x, double __y) { return !__cimo_isunordered(__x, __y) && __x > __y; }
static inline int __cimo_ge(double __x, double __y) { return !__cimo_isunordered(__x, __y) && __x >= __y; }
static inline int __cimo_lt(double __x, double __y) { return !__cimo_isunordered(__x, __y) && __x < __y; }
static inline int __cimo_le(double __x, double __y) { return !__cimo_isunordered(__x, __y) && __x <= __y; }
static inline int __cimo_lg(double __x, double __y) { return !__cimo_isunordered(__x, __y) && __x != __y; }
static inline int __cimo_gtf(float __x, float __y) { return !__cimo_isunorderedf(__x, __y) && __x > __y; }
static inline int __cimo_gef(float __x, float __y) { return !__cimo_isunorderedf(__x, __y) && __x >= __y; }
static inline int __cimo_ltf(float __x, float __y) { return !__cimo_isunorderedf(__x, __y) && __x < __y; }
static inline int __cimo_lef(float __x, float __y) { return !__cimo_isunorderedf(__x, __y) && __x <= __y; }
static inline int __cimo_lgf(float __x, float __y) { return !__cimo_isunorderedf(__x, __y) && __x != __y; }

#define __CIMO_TG2(x, y, f, d) (sizeof((x) + (y)) == sizeof(float) ? f(x, y) : d(x, y))
#define isgreater(x, y) __CIMO_TG2(x, y, __cimo_gtf, __cimo_gt)
#define isgreaterequal(x, y) __CIMO_TG2(x, y, __cimo_gef, __cimo_ge)
#define isless(x, y) __CIMO_TG2(x, y, __cimo_ltf, __cimo_lt)
#define islessequal(x, y) __CIMO_TG2(x, y, __cimo_lef, __cimo_le)
#define islessgreater(x, y) __CIMO_TG2(x, y, __cimo_lgf, __cimo_lg)
#define isunordered(x, y) __CIMO_TG2(x, y, __cimo_isunorderedf, __cimo_isunordered)

double acos(double x);
float acosf(float x);
double asin(double x);
float asinf(float x);
double atan(double x);
float atanf(float x);
double cos(double x);
float cosf(float x);
double sin(double x);
float sinf(float x);
double tan(double x);
float tanf(float x);
double acosh(double x);
float acoshf(float x);
double asinh(double x);
float asinhf(float x);
double atanh(double x);
float atanhf(float x);
double cosh(double x);
float coshf(float x);
double sinh(double x);
float sinhf(float x);
double tanh(double x);
float tanhf(float x);
double exp(double x);
float expf(float x);
double exp2(double x);
float exp2f(float x);
double expm1(double x);
float expm1f(float x);
double log(double x);
float logf(float x);
double log10(double x);
float log10f(float x);
double log1p(double x);
float log1pf(float x);
double log2(double x);
float log2f(float x);
double logb(double x);
float logbf(float x);
double cbrt(double x);
float cbrtf(float x);
double fabs(double x);
float fabsf(float x);
double sqrt(double x);
float sqrtf(float x);
double erf(double x);
float erff(float x);
double erfc(double x);
float erfcf(float x);
double lgamma(double x);
float lgammaf(float x);
double tgamma(double x);
float tgammaf(float x);
double ceil(double x);
float ceilf(float x);
double floor(double x);
float floorf(float x);
double nearbyint(double x);
float nearbyintf(float x);
double rint(double x);
float rintf(float x);
double round(double x);
float roundf(float x);
double trunc(double x);
float truncf(float x);
double atan2(double x, double y);
float atan2f(float x, float y);
double hypot(double x, double y);
float hypotf(float x, float y);
double pow(double x, double y);
float powf(float x, float y);
double fmod(double x, double y);
float fmodf(float x, float y);
double remainder(double x, double y);
float remainderf(float x, float y);
double copysign(double x, double y);
float copysignf(float x, float y);
double nextafter(double x, double y);
float nextafterf(float x, float y);
double fdim(double x, double y);
float fdimf(float x, float y);
double fmax(double x, double y);
float fmaxf(float x, float y);
double fmin(double x, double y);
float fminf(float x, float y);
double frexp(double value, int *exp);
float frexpf(float value, int *exp);
int ilogb(double x);
int ilogbf(float x);
double ldexp(double x, int exp);
float ldexpf(float x, int exp);
double modf(double value, double *iptr);
float modff(float value, float *iptr);
double scalbn(double x, int n);
float scalbnf(float x, int n);
double scalbln(double x, long n);
float scalblnf(float x, long n);
long lrint(double x);
long lrintf(float x);
long long llrint(double x);
long long llrintf(float x);
long lround(double x);
long lroundf(float x);
long long llround(double x);
long long llroundf(float x);
double remquo(double x, double y, int *quo);
float remquof(float x, float y, int *quo);
double nan(const char *tagp);
float nanf(const char *tagp);
double fma(double x, double y, double z);
float fmaf(float x, float y, float z);

/* Not C11, but commonly expected (POSIX/XSI). */
#define M_E 2.7182818284590452354
#define M_LOG2E 1.4426950408889634074
#define M_LOG10E 0.43429448190325182765
#define M_LN2 0.69314718055994530942
#define M_LN10 2.30258509299404568402
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.78539816339744830962
#define M_1_PI 0.31830988618379067154
#define M_2_PI 0.63661977236758134308
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2 1.41421356237309504880
#define M_SQRT1_2 0.70710678118654752440

#endif
