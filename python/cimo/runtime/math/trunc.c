/* cimo: calls to trunc() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double trunc(double x)
{
	return __builtin_trunc(x);
}
