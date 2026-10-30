/* cimo: <assert.h>. A failed assert unwinds the kernel and raises cimo.KernelError with the
   expression, file, line and function. Not include-guarded, as the standard requires. */
#undef assert
#ifdef NDEBUG
#define assert(e) ((void)0)
#else
_Noreturn void __assert_fail(const char *, const char *, int, const char *);
#define assert(e) ((void)((e) || (__assert_fail(#e, __FILE__, __LINE__, __func__), 0)))
#endif
#ifndef __cplusplus
#undef static_assert
#define static_assert _Static_assert
#endif
