/* cimo: calls to fma() compile to one instruction (cimo.frontend builtins); this definition
   serves indirect calls. */
#include <math.h>
#include "musl.h"

double fma(double x, double y, double z)
{
	return __builtin_fma(x, y, z);
}
