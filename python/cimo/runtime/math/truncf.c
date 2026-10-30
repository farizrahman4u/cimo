/* cimo: calls to truncf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float truncf(float x)
{
	return __builtin_truncf(x);
}
