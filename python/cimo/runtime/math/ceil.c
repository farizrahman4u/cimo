/* cimo: calls to ceil() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double ceil(double x)
{
	return __builtin_ceil(x);
}
