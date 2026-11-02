/* cimo: calls to fmaxf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. Where it is not an instruction (x86-64:
   __CIMO_SOFT_MATH), it is musl's portable code. */
#include <math.h>
#include "musl.h"

#if __CIMO_SOFT_MATH
#include <float.h>
#include <stdint.h>

float fmaxf(float x, float y)
{
	union {float f; uint32_t i;} ux = {x}, uy = {y};

	if (ux.i << 1 > 0xff000000U) /* x is a NaN */
		return y;
	if (uy.i << 1 > 0xff000000U)
		return x;
	/* signed zeros, C99 Annex F.9.9.2 */
	if ((ux.i ^ uy.i) >> 31)
		return ux.i >> 31 ? y : x;
	return x < y ? y : x;
}
#else
float fmaxf(float x, float y)
{
	return __builtin_fmaxf(x, y);
}
#endif
