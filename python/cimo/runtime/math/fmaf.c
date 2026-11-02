/* cimo: where the target has a fused multiply-add (__FP_FAST_FMAF: arm64), calls to fmaf()
   compile to one instruction (cimo.frontend builtins) and this definition serves indirect
   calls; elsewhere (x86-64 without FMA) it is exact integer arithmetic (softfma.h). */
#include <math.h>
#include "musl.h"

#if __FP_FAST_FMAF
float fmaf(float x, float y, float z)
{
	return __builtin_fmaf(x, y, z);
}
#else
#include "softfma.h"

float fmaf(float x, float y, float z)
{
	return __cimo_softfmaf(x, y, z);
}
#endif
