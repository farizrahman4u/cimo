/* cimo: calls to roundf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float roundf(float x)
{
	return __builtin_roundf(x);
}
