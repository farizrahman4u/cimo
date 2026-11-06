/* cimo: <fenv.h> (C11 7.6) on the hardware's floating-point control and status registers, which
   __cimo_rdfpcr and the rest read and write (one instruction each: arm64 FPCR and FPSR, x86-64 MXCSR
   for both). On wasm they read 0 and drop writes: one rounding mode, no exception flags, which the
   functions report by reading back what they wrote. The exceptions are raised by setting their flags
   (traps stay disabled, as C starts). */

#include "../include/fenv.h" /* the bundled header (internal/fenv.h is the musl sources' empty one) */

unsigned __cimo_rdfpcr(void);
unsigned __cimo_rdfpsr(void);
void __cimo_wrfpcr(unsigned);
void __cimo_wrfpsr(unsigned);

#ifdef __x86_64__
#define RSHIFT 3 /* MXCSR's rounding control is bits 13-14; FE_ values are the x87's, bits 10-11 */
#define RMASK 0x6000u
#define DFLCR 0x1f80u /* all exceptions masked, round to nearest */
#else
#define RSHIFT 0 /* FPCR bits 22-23 */
#define RMASK 0xc00000u
#define DFLCR 0u
#endif

int feclearexcept(int excepts)
{
	excepts &= FE_ALL_EXCEPT;
	__cimo_wrfpsr(__cimo_rdfpsr() & ~(unsigned)excepts);
	return 0;
}

int feraiseexcept(int excepts)
{
	excepts &= FE_ALL_EXCEPT;
	__cimo_wrfpsr(__cimo_rdfpsr() | (unsigned)excepts);
	return (__cimo_rdfpsr() & excepts) != (unsigned)excepts;
}

int fetestexcept(int excepts)
{
	return __cimo_rdfpsr() & excepts & FE_ALL_EXCEPT;
}

int fegetexceptflag(fexcept_t *flagp, int excepts)
{
	*flagp = __cimo_rdfpsr() & excepts & FE_ALL_EXCEPT;
	return 0;
}

int fesetexceptflag(const fexcept_t *flagp, int excepts)
{
	excepts &= FE_ALL_EXCEPT;
	__cimo_wrfpsr((__cimo_rdfpsr() & ~(unsigned)excepts) | (*flagp & excepts));
	return 0;
}

int fegetround(void)
{
	return (__cimo_rdfpcr() & RMASK) >> RSHIFT;
}

int fesetround(int round)
{
	unsigned mode = (unsigned)round << RSHIFT;
	if (round & ~(FE_TONEAREST | FE_UPWARD | FE_DOWNWARD | FE_TOWARDZERO) || mode & ~RMASK)
		return 1;
	__cimo_wrfpcr((__cimo_rdfpcr() & ~RMASK) | mode);
	return fegetround() != round;
}

int fegetenv(fenv_t *envp)
{
	envp->__cr = __cimo_rdfpcr();
	envp->__sr = __cimo_rdfpsr();
	return 0;
}

int fesetenv(const fenv_t *envp)
{
	if (envp == FE_DFL_ENV) {
		__cimo_wrfpcr(DFLCR);
		__cimo_wrfpsr(DFLCR);
		return 0;
	}
	__cimo_wrfpcr(envp->__cr);
	__cimo_wrfpsr(envp->__sr);
	return 0;
}

int feholdexcept(fenv_t *envp)
{
	fegetenv(envp);
	feclearexcept(FE_ALL_EXCEPT);
	return 0;
}

int feupdateenv(const fenv_t *envp)
{
	int ex = fetestexcept(FE_ALL_EXCEPT);
	fesetenv(envp);
	feraiseexcept(ex);
	return 0;
}
