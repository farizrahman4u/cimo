/* cimo: calls to fmin() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double fmin(double x, double y)
{
	return __builtin_fmin(x, y);
}
