/* cimo: calls to floor() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double floor(double x)
{
	return __builtin_floor(x);
}
