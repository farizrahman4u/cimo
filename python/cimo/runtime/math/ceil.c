/* cimo: calls to ceil() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. Where it is not an instruction (x86-64:
   __CIMO_SOFT_MATH), it is musl's portable code. */
#include <math.h>
#include "musl.h"

#if __CIMO_SOFT_MATH
#include <float.h>
#include <stdint.h>

static const double toint = 1 / DBL_EPSILON;

double ceil(double x)
{
	union {double f; uint64_t i;} u = {x};
	int e = u.i >> 52 & 0x7ff;
	double y;

	if (e >= 0x3ff + 52 || x == 0)
		return x;
	/* y = int(x) - x, where int(x) is an integer neighbor of x */
	if (u.i >> 63)
		y = x - toint + toint - x;
	else
		y = x + toint - toint - x;
	if (e <= 0x3ff - 1)
		return u.i >> 63 ? -0.0 : 1;
	if (y < 0)
		return x + y + 1;
	return x + y;
}
#else
double ceil(double x)
{
	return __builtin_ceil(x);
}
#endif
