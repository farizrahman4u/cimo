/* cimo: calls to round() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double round(double x)
{
	return __builtin_round(x);
}
