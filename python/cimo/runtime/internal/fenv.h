/* cimo runtime: no floating-point environment (no exception flags, round to nearest only).
   musl sources that test for FE_INEXACT etc. take their exception-free paths. */
#ifndef __CIMO_RT_FENV_H
#define __CIMO_RT_FENV_H
#endif
