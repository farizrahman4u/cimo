/* cimo: calls to trunc() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. Where it is not an instruction (x86-64:
   __CIMO_SOFT_MATH), it is musl's portable code. */
#include <math.h>
#include "musl.h"

#if __CIMO_SOFT_MATH
#include <float.h>
#include <stdint.h>

double trunc(double x)
{
	union {double f; uint64_t i;} u = {x};
	int e = (int)(u.i >> 52 & 0x7ff) - 0x3ff + 12;
	uint64_t m;

	if (e >= 52 + 12)
		return x;
	if (e < 12)
		e = 1;
	m = -1ULL >> e;
	if ((u.i & m) == 0)
		return x;
	u.i &= ~m;
	return u.f;
}
#else
double trunc(double x)
{
	return __builtin_trunc(x);
}
#endif
