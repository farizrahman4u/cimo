/* cimo: no floating-point exception flags, so nearbyint is rint. */
#include <math.h>

double nearbyint(double x)
{
	return rint(x);
}
