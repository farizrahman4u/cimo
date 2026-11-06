/* cimo: musl's src/internal/complex_impl.h, with CMPLX from __builtin_complex. */
#ifndef _COMPLEX_IMPL_H
#define _COMPLEX_IMPL_H

#include <complex.h>
#include "libm.h"

#undef CMPLX
#undef CMPLXF
#define CMPLX(x, y) __builtin_complex((double)(x), (double)(y))
#define CMPLXF(x, y) __builtin_complex((float)(x), (float)(y))

hidden double complex __ldexp_cexp(double complex,int);
hidden float complex __ldexp_cexpf(float complex,int);

#endif
