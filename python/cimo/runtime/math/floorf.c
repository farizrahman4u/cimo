/* cimo: calls to floorf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float floorf(float x)
{
	return __builtin_floorf(x);
}
