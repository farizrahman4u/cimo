/* cimo wasm runtime: fma and fmaf, exact, in integer arithmetic (wasm has no scalar fused
   multiply-add; softfma.h). */

#include "softfma.h"

double
fma(double x, double y, double z)
{
	return __cimo_softfma(x, y, z);
}

float
fmaf(float x, float y, float z)
{
	return __cimo_softfmaf(x, y, z);
}
