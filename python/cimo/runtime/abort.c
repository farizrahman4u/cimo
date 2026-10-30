/* cimo runtime: abort() and assert failures, which unwind back to Python (KernelError).

   Exports: abort, __assert_fail (called by the assert macro in <assert.h>)
   Imports: __cimo_fail(code, expr, file, line, func), assembly glue in cimo/execution.py:
            records the error in the calling thread's context and unwinds to the innermost
            entry trampoline. The faulting function is taken from the caller of the function
            that calls it, so these wrappers must keep their frames (QBE always emits one).
   Also in the glue: __builtin_trap (declared by the driver's prelude) and __cimo_trap
   (called where the backend would emit a trap instruction for unreachable code).

   Freestanding C; no other dependencies. */

_Noreturn void __cimo_fail(long code, const char *expr, const char *file, long line,
                           const char *func);

_Noreturn void
abort(void)
{
	__cimo_fail(1, 0, 0, 0, 0);
}

_Noreturn void
__assert_fail(const char *expr, const char *file, int line, const char *func)
{
	__cimo_fail(2, expr, file, line, func);
}
