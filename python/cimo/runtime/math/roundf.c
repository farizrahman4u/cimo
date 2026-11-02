/* cimo: calls to roundf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. Where it is not an instruction (x86-64:
   __CIMO_SOFT_MATH), it is musl's portable code. */
#include <math.h>
#include "musl.h"

#if __CIMO_SOFT_MATH
#include <float.h>
#include <stdint.h>

static const float toint = 1 / FLT_EPSILON;

float roundf(float x)
{
	union {float f; uint32_t i;} u = {x};
	int e = u.i >> 23 & 0xff;
	float y;

	if (e >= 0x7f + 23)
		return x;
	if (u.i >> 31)
		x = -x;
	if (e < 0x7f - 1)
		return 0 * u.f;
	y = x + toint - toint - x;
	if (y > 0.5f)
		y = y + x - 1;
	else if (y <= -0.5f)
		y = y + x + 1;
	else
		y = y + x;
	if (u.i >> 31)
		y = -y;
	return y;
}
#else
float roundf(float x)
{
	return __builtin_roundf(x);
}
#endif
