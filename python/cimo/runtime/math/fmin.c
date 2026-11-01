/* cimo: calls to fmin() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. Where it is not an instruction (x86-64:
   __CIMO_SOFT_MATH), it is musl's portable code. */
#include <math.h>
#include "musl.h"

#if __CIMO_SOFT_MATH
#include <float.h>
#include <stdint.h>

double fmin(double x, double y)
{
	union {double f; uint64_t i;} ux = {x}, uy = {y};

	if (ux.i << 1 > 0xffe0000000000000ULL) /* x is a NaN */
		return y;
	if (uy.i << 1 > 0xffe0000000000000ULL)
		return x;
	/* signed zeros, C99 Annex F.9.9.2 */
	if ((ux.i ^ uy.i) >> 63)
		return ux.i >> 63 ? x : y;
	return x < y ? x : y;
}
#else
double fmin(double x, double y)
{
	return __builtin_fmin(x, y);
}
#endif
