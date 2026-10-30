/* cimo: no floating-point exception flags, so nearbyintf is rintf. */
#include <math.h>

float nearbyintf(float x)
{
	return rintf(x);
}
