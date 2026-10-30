/* cimo: calls to fminf() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

float fminf(float x, float y)
{
	return __builtin_fminf(x, y);
}
