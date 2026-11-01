/* cimo: where the target has a fused multiply-add (__FP_FAST_FMA: arm64), calls to fma()
   compile to one instruction (cimo.frontend builtins) and this definition serves indirect
   calls; elsewhere (x86-64 without FMA) it is exact integer arithmetic (softfma.h). */
#include <math.h>
#include "musl.h"

#if __FP_FAST_FMA
double fma(double x, double y, double z)
{
	return __builtin_fma(x, y, z);
}
#else
#include "softfma.h"

double fma(double x, double y, double z)
{
	return __cimo_softfma(x, y, z);
}
#endif
