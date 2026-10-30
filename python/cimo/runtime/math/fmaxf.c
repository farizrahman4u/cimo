/* cimo: calls to fmaxf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float fmaxf(float x, float y)
{
	return __builtin_fmaxf(x, y);
}
