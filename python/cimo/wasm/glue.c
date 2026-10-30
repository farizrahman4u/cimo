/* cimo wasm glue: what the native modules get from the assembly in cimo/execution.py, for a
   module loaded into Pyodide (single-threaded, so one context and no atomics).

   Exports: __cimo_ctx, __cimo_cas, __cimo_fail, __cimo_trap, __builtin_trap, and the functions
            the runtime's hook pointers are initialized with at load time (cimo/wasm/link.py):
            __cimo_wasm_write, __cimo_wasm_grow, __cimo_wasm_release
   Imports: the __cimo_js_* functions below, from Python (cimo/wasm/pyodide.py). Addresses
            are passed as 32-bit values, so no BigInt crosses the boundary.
   __cimo_js_fail records the error and throws, unwinding to the Python caller. */

typedef unsigned long size_t;

void __cimo_js_write(unsigned buf, unsigned len);
unsigned __cimo_js_grow(unsigned bytes, int kind);
void __cimo_js_release(unsigned p);
void __cimo_js_fail(int code, unsigned expr, unsigned file, int line, unsigned func);

static _Alignas(16) char ctx[8192];  /* layout: cimo/execution.py */

static unsigned
u32(const void *p)
{
	return (unsigned)(unsigned long)p;
}

void *
__cimo_ctx(void)
{
	return ctx;
}

void *
__cimo_cas(void **p, void *expect, void *desired)
{
	void *old = *p;
	if (old == expect)
		*p = desired;
	return old;
}

_Noreturn void
__cimo_fail(long code, const char *expr, const char *file, long line, const char *func)
{
	__cimo_js_fail(code, u32(expr), u32(file), line, u32(func));
	for (;;)
		;
}

_Noreturn void
__cimo_trap(void)
{
	__cimo_js_fail(3, 0, 0, 0, 0);
	for (;;)
		;
}

_Noreturn void
__builtin_trap(void)
{
	__cimo_js_fail(3, 0, 0, 0, 0);
	for (;;)
		;
}

void
__cimo_wasm_write(const char *buf, size_t len)
{
	__cimo_js_write(u32(buf), len);
}

void *
__cimo_wasm_grow(void *heap, unsigned long bytes, int kind)
{
	return (void *)(unsigned long)__cimo_js_grow(bytes, kind);
}

void
__cimo_wasm_release(void *p)
{
	__cimo_js_release(u32(p));
}
