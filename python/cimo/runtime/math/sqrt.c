/* cimo: calls to sqrt() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double sqrt(double x)
{
	return __builtin_sqrt(x);
}
