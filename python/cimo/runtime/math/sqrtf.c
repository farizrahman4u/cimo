/* cimo: calls to sqrtf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float sqrtf(float x)
{
	return __builtin_sqrtf(x);
}
