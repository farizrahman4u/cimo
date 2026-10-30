/* cimo: calls to fmaf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float fmaf(float x, float y, float z)
{
	return __builtin_fmaf(x, y, z);
}
