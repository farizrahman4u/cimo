/* Includes every bundled header and prints sizes and limits. Built by cimo (bundled headers,
   cimo.cpp, cproc, QBE) and by gcc (system headers); the outputs must match. */
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>
#include <stdalign.h>
#include <stdnoreturn.h>
#include <iso646.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

int printf(const char *, ...);

#define PI(e) printf("%s = %lld\n", #e, (long long)(e))
#define PU(e) printf("%s = %llu\n", #e, (unsigned long long)(e))
#define PD(e) printf("%s = %.17g\n", #e, (double)(e))
#define PS(t) printf("sizeof(%s) = %d, signed = %d\n", #t, (int)sizeof(t), (t)-1 < (t)0)

struct s { char c; double d; };

static int sum(int n, ...)
{
	va_list ap, aq;
	int i, t = 0;
	va_start(ap, n);
	va_copy(aq, ap);
	for (i = 0; i < n; i++)
		t += va_arg(aq, int);
	va_end(aq);
	va_end(ap);
	return t;
}

static int cmp(const void *a, const void *b)
{
	return *(const int *)a - *(const int *)b;
}

static noreturn void never(void) { abort(); }

int main(void)
{
	int arr[5] = {5, 3, 4, 1, 2};
	char buf[32];
	alignas(16) char aligned[16];
	volatile double zero = 0.0, one = 1.0;
	volatile float fzero = 0.0f;
	bool b = true;

	PS(size_t); PS(ptrdiff_t); PS(wchar_t); PS(char);
	PS(int8_t); PS(int16_t); PS(int32_t); PS(int64_t);
	PS(uint8_t); PS(uint16_t); PS(uint32_t); PS(uint64_t);
	PS(int_least8_t); PS(int_least16_t); PS(int_least32_t); PS(int_least64_t);
	PS(uint_least8_t); PS(uint_least16_t); PS(uint_least32_t); PS(uint_least64_t);
	PS(int_fast8_t); PS(int_fast16_t); PS(int_fast32_t); PS(int_fast64_t);
	PS(uint_fast8_t); PS(uint_fast16_t); PS(uint_fast32_t); PS(uint_fast64_t);
	PS(intptr_t); PS(uintptr_t); PS(intmax_t); PS(uintmax_t); PS(double_t);
	PS(float_t); /* cproc: FLT_EVAL_METHOD 1, so double; gcc: float */
	PI(sizeof(max_align_t)); PI(alignof(max_align_t)); PI(sizeof(va_list));
	PI(sizeof(bool)); PI(b); PI(true); PI(false); PI(__bool_true_false_are_defined);
	PI(offsetof(struct s, d)); PI(NULL == (void *)0); PI(alignof(double));
	PI(__alignas_is_defined); PI(__alignof_is_defined); PI((uintptr_t)aligned % 16);

	PI(CHAR_BIT); PI(SCHAR_MIN); PI(SCHAR_MAX); PI(UCHAR_MAX); PI(CHAR_MIN); PI(CHAR_MAX);
	PI(MB_LEN_MAX); PI(SHRT_MIN); PI(SHRT_MAX); PI(USHRT_MAX); PI(INT_MIN); PI(INT_MAX);
	PU(UINT_MAX); PI(LONG_MIN); PI(LONG_MAX); PU(ULONG_MAX); PI(LLONG_MIN); PI(LLONG_MAX);
	PU(ULLONG_MAX);
	PI(sizeof(UINT_MAX)); PI(sizeof(LONG_MAX)); PI(sizeof(INT_MIN)); PI(sizeof(LLONG_MIN));

	PI(INT8_MIN); PI(INT8_MAX); PU(UINT8_MAX); PI(INT16_MIN); PI(INT16_MAX); PU(UINT16_MAX);
	PI(INT32_MIN); PI(INT32_MAX); PU(UINT32_MAX); PI(INT64_MIN); PI(INT64_MAX); PU(UINT64_MAX);
	PI(INT_LEAST8_MIN); PI(INT_LEAST16_MAX); PU(UINT_LEAST32_MAX); PI(INT_LEAST64_MIN);
	PI(INT_FAST8_MIN); PI(INT_FAST16_MIN); PI(INT_FAST32_MAX); PU(UINT_FAST16_MAX);
	PU(UINT_FAST32_MAX); PI(INT_FAST64_MAX); PU(UINT_FAST64_MAX);
	PI(INTPTR_MIN); PI(INTPTR_MAX); PU(UINTPTR_MAX); PI(INTMAX_MIN); PI(INTMAX_MAX);
	PU(UINTMAX_MAX); PI(PTRDIFF_MIN); PI(PTRDIFF_MAX); PU(SIZE_MAX); PI(SIG_ATOMIC_MIN);
	PI(SIG_ATOMIC_MAX); PI(WCHAR_MIN); PU(WCHAR_MAX); PI(WINT_MIN); PU(WINT_MAX);
	PI(INT8_C(-5)); PU(UINT16_C(7)); PU(UINT32_C(4000000000)); PI(INT64_C(-9000000000));
	PU(UINT64_C(18000000000000000000)); PI(INTMAX_C(1)); PU(UINTMAX_C(2));
	PI(sizeof(INT64_C(1))); PI(sizeof(UINT32_C(1))); PI(sizeof(INT32_MAX));
	PI(sizeof(UINT32_MAX)); PI(sizeof(INT64_MAX)); PI(sizeof(SIZE_MAX));

	PI(FLT_ROUNDS); PI(FLT_RADIX); PI(FLT_MANT_DIG); PI(FLT_DIG); PI(FLT_DECIMAL_DIG);
	PI(FLT_MIN_EXP); PI(FLT_MIN_10_EXP); PI(FLT_MAX_EXP); PI(FLT_MAX_10_EXP);
	PD(FLT_MAX); PD(FLT_MIN); PD(FLT_EPSILON); PD(FLT_TRUE_MIN); PI(FLT_HAS_SUBNORM);
	PI(DBL_MANT_DIG); PI(DBL_DIG); PI(DBL_DECIMAL_DIG); PI(DBL_MIN_EXP); PI(DBL_MIN_10_EXP);
	PI(DBL_MAX_EXP); PI(DBL_MAX_10_EXP); PD(DBL_MAX); PD(DBL_MIN); PD(DBL_EPSILON);
	PD(DBL_TRUE_MIN); PI(DBL_HAS_SUBNORM); PI(LDBL_MANT_DIG); PI(LDBL_DIG); PI(DECIMAL_DIG);
	PI(LDBL_MAX_EXP); PI(LDBL_MIN_EXP); PI(sizeof(FLT_MAX)); PI(sizeof(DBL_MAX));

	PD(HUGE_VAL); PD(HUGE_VALF); PD(INFINITY); PI(sizeof(HUGE_VAL)); PI(sizeof(INFINITY));
	PI(isnan(NAN)); PI(isnan(zero / zero)); PI(isnan(one)); PI(isnan(fzero / fzero));
	PI(!!isinf(INFINITY)); PI(!!isinf(-HUGE_VAL)); PI(!!isinf(one)); PI(!!isinf(one / zero));
	PI(isfinite(one)); PI(isfinite(NAN)); PI(isfinite(1.0f)); PI(isnormal(one));
	PI(isnormal(DBL_TRUE_MIN)); PI(isnormal(FLT_MIN)); PI(!!signbit(-one)); PI(!!signbit(one));
	PI(!!signbit(-0.0f)); PI(!!signbit(-zero));
	PI(fpclassify(zero) == FP_ZERO); PI(fpclassify(one) == FP_NORMAL);
	PI(fpclassify(DBL_TRUE_MIN) == FP_SUBNORMAL); PI(fpclassify(NAN) == FP_NAN);
	PI(fpclassify(-INFINITY) == FP_INFINITE); PI(fpclassify(FLT_TRUE_MIN) == FP_SUBNORMAL);
	PI(isgreater(2.0, 1.0)); PI(isgreater(NAN, 1.0)); PI(isless(1.0f, 2.0f));
	PI(islessequal(1.0, 1.0)); PI(isgreaterequal(zero / zero, 1.0)); PI(islessgreater(1.0, 2.0));
	PI(isunordered(NAN, 1.0)); PI(isunordered(1.0, 2.0f));
	PD(sqrt(2.0)); PD(sqrtf(2.0f)); PD(sin(1.0)); PD(cosf(1.0f)); PD(pow(2.0, 10.0));
	PD(floor(-1.5)); PD(fabs(-3.0)); PD(fma(2.0, 3.0, 4.0)); PD(fmaxf(1.0f, 2.0f));
	PD(exp(1.0)); PD(log(M_E)); PD(atan2(1.0, 1.0) * 4); PD(M_PI); PI(lround(2.5));
	PI(ilogb(8.0)); PD(ldexp(1.0, 10)); PD(nan("") != nan("")); PD(copysign(1.0, -0.0));
	PD(hypotf(3.0f, 4.0f)); PD(cbrt(27.0)); PD(trunc(-2.7)); PD(remainder(5.0, 3.0));

	PI(strlen("hello")); PI(strcmp("a", "b") < 0); PI(memcmp("ab", "ab", 2));
	strcpy(buf, "abc"); strcat(buf, "def"); PI(strlen(buf)); PI(strchr(buf, 'd') - buf);
	memset(buf, 'x', 3); PI(buf[2]); memmove(buf + 1, buf, 4); PI(buf[4]);
	PI(strstr("haystack", "st") != NULL); PI(strspn("aab", "a")); PI(strcspn("abc", "c"));
	PI(isalpha('a')); PI(isdigit('5')); PI(isspace(' ')); PI(toupper('q')); PI(tolower('Q'));
	PI(isxdigit('f')); PI(ispunct('!')); PI(isupper('a')); PI(isblank('\t'));

	qsort(arr, 5, sizeof(int), cmp);
	PI(arr[0]); PI(arr[4]);
	{ int key = 4; int *p = bsearch(&key, arr, 5, sizeof(int), cmp); PI(p - arr); }
	PI(abs(-5)); PI(labs(-7L)); PI(div(7, 2).quot); PI(div(7, 2).rem); PI(atoi("123"));
	PI(strtol("ff", NULL, 16)); PD(strtod("2.5", NULL));
	{ void *p = malloc(10); void *q = calloc(2, 4); p = realloc(p, 20);
	  void *r = aligned_alloc(64, 64); PI((uintptr_t)r % 64); free(p); free(q); free(r); }
	PI(sum(3, 1, 2, 3));
	PI(1 and 0); PI(1 or 0); PI(not 0); PI(6 bitand 3); PI(6 bitor 3); PI(6 xor 3);
	PI(compl 0); PI(1 not_eq 2);
	if (arr[0] != 1)
		never();
	return 0;
}
