/* cimo: <complex.h> (C11 7.3), double and float functions (musl's, in the runtime). The long double
   functions are left out, as in <math.h>. creal and cimag are GNU's __real__ and __imag__; CMPLX is
   __builtin_complex. */
#ifndef __CIMO_COMPLEX_H
#define __CIMO_COMPLEX_H

#define complex _Complex
#define _Complex_I (0.0f + 1.0fi)
#define I _Complex_I

double complex cacos(double complex);
float complex cacosf(float complex);
double complex casin(double complex);
float complex casinf(float complex);
double complex catan(double complex);
float complex catanf(float complex);
double complex ccos(double complex);
float complex ccosf(float complex);
double complex csin(double complex);
float complex csinf(float complex);
double complex ctan(double complex);
float complex ctanf(float complex);
double complex cacosh(double complex);
float complex cacoshf(float complex);
double complex casinh(double complex);
float complex casinhf(float complex);
double complex catanh(double complex);
float complex catanhf(float complex);
double complex ccosh(double complex);
float complex ccoshf(float complex);
double complex csinh(double complex);
float complex csinhf(float complex);
double complex ctanh(double complex);
float complex ctanhf(float complex);
double complex cexp(double complex);
float complex cexpf(float complex);
double complex clog(double complex);
float complex clogf(float complex);
double cabs(double complex);
float cabsf(float complex);
double complex cpow(double complex, double complex);
float complex cpowf(float complex, float complex);
double complex csqrt(double complex);
float complex csqrtf(float complex);
double carg(double complex);
float cargf(float complex);
double cimag(double complex);
float cimagf(float complex);
double complex conj(double complex);
float complex conjf(float complex);
double complex cproj(double complex);
float complex cprojf(float complex);
double creal(double complex);
float crealf(float complex);

#define creal(x) ((double)__real__ (double complex)(x))
#define crealf(x) ((float)__real__ (float complex)(x))
#define cimag(x) ((double)__imag__ (double complex)(x))
#define cimagf(x) ((float)__imag__ (float complex)(x))

#define CMPLX(x, y) __builtin_complex((double)(x), (double)(y))
#define CMPLXF(x, y) __builtin_complex((float)(x), (float)(y))

#endif
