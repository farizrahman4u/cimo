/* cimo: calls to fmax() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double fmax(double x, double y)
{
	return __builtin_fmax(x, y);
}
