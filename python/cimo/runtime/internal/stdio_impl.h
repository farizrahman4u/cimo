/* cimo runtime: formatted output without FILE. A sink receives the formatter's output. */
#ifndef __CIMO_RT_STDIO_IMPL_H
#define __CIMO_RT_STDIO_IMPL_H

#include <stddef.h>
#include <stdarg.h>

struct __cimo_sink {
	void (*write)(struct __cimo_sink *, const char *, size_t);
	char *buf;      /* string sinks: the destination */
	size_t len;     /* bytes stored so far */
	size_t cap;     /* bytes that fit, not counting the terminating NUL */
};

/* musl's vfprintf core: formats into f, returns the number of bytes the output has (also
   when the sink drops some), or -1 on an invalid format or overflow of int */
int __cimo_vformat(struct __cimo_sink *f, const char *restrict fmt, va_list ap);

/* stdout: buffered in the module, written to Python's sys.stdout through the
   __cimo_hook_write callback at a newline, when the buffer is full, and when the kernel returns (the Python side
   calls __cimo_flush when __cimo_stdout.len is nonzero) */
void __cimo_stdout_write(struct __cimo_sink *, const char *, size_t);
void __cimo_flush(void);

#endif
