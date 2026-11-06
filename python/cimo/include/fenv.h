/* cimo: <fenv.h> (C11 7.6), the runtime's (fenv.c): the hardware's rounding modes and exception
   flags, arm64 FPCR/FPSR and x86-64 MXCSR. The compiler folds floating-point operations on constants
   in round-to-nearest without flags unless `#pragma STDC FENV_ACCESS ON` is in effect. */
#ifndef __CIMO_FENV_H
#define __CIMO_FENV_H

#ifdef __x86_64__
#define FE_INVALID 1
#define FE_DIVBYZERO 4
#define FE_OVERFLOW 8
#define FE_UNDERFLOW 16
#define FE_INEXACT 32
#define FE_ALL_EXCEPT 61
#define FE_TONEAREST 0
#define FE_DOWNWARD 0x400
#define FE_UPWARD 0x800
#define FE_TOWARDZERO 0xc00
#else
#define FE_INVALID 1
#define FE_DIVBYZERO 2
#define FE_OVERFLOW 4
#define FE_UNDERFLOW 8
#define FE_INEXACT 16
#define FE_ALL_EXCEPT 31
#define FE_TONEAREST 0
#define FE_UPWARD 0x400000
#define FE_DOWNWARD 0x800000
#define FE_TOWARDZERO 0xc00000
#endif

typedef unsigned int fexcept_t;
typedef struct {
	unsigned int __cr, __sr;
} fenv_t;

#define FE_DFL_ENV ((const fenv_t *)-1)

int feclearexcept(int);
int fegetexceptflag(fexcept_t *, int);
int feraiseexcept(int);
int fesetexceptflag(const fexcept_t *, int);
int fetestexcept(int);
int fegetround(void);
int fesetround(int);
int fegetenv(fenv_t *);
int feholdexcept(fenv_t *);
int fesetenv(const fenv_t *);
int feupdateenv(const fenv_t *);

#endif
