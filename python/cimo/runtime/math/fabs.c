/* cimo: calls to fabs() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double fabs(double x)
{
	return __builtin_fabs(x);
}
