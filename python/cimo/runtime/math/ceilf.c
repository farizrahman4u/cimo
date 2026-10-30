/* cimo: calls to ceilf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float ceilf(float x)
{
	return __builtin_ceilf(x);
}
