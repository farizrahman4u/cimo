/* cimo: calls to floorf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. Where it is not an instruction (x86-64:
   __CIMO_SOFT_MATH), it is musl's portable code. */
#include <math.h>
#include "musl.h"

#if __CIMO_SOFT_MATH
#include <float.h>
#include <stdint.h>

float floorf(float x)
{
	union {float f; uint32_t i;} u = {x};
	int e = (int)(u.i >> 23 & 0xff) - 0x7f;
	uint32_t m;

	if (e >= 23)
		return x;
	if (e >= 0) {
		m = 0x007fffff >> e;
		if ((u.i & m) == 0)
			return x;
		if (u.i >> 31)
			u.i += m;
		u.i &= ~m;
	} else {
		if (u.i >> 31 == 0)
			u.i = 0;
		else if (u.i << 1)
			u.f = -1.0f;
	}
	return u.f;
}
#else
float floorf(float x)
{
	return __builtin_floorf(x);
}
#endif
