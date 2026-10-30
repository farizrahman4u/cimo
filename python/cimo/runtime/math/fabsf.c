/* cimo: calls to fabsf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float fabsf(float x)
{
	return __builtin_fabsf(x);
}
