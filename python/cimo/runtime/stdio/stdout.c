/* cimo: stdout for printf, puts and putchar.

   Output collects in a module-global buffer and goes to Python's sys.stdout through one
   hook, __cimo_hook_write(buf, len): a function pointer that Python sets after loading to a
   ctypes callback (so the module imports nothing).
   The buffer is written out when output contains a newline, when it is full, and when a
   kernel returns (Python calls __cimo_flush if __cimo_stdout.len is nonzero), so C and
   Python output interleave in program order, and sys.stdout redirection (Jupyter,
   contextlib.redirect_stdout) captures it.

   Concurrent calls from several threads share the buffer: the indices are clamped so a race
   cannot write out of bounds, but output of racing threads may interleave or be lost. */
#include <string.h>
#include "stdio_impl.h"

#define EOF (-1)

void (*__cimo_hook_write)(const char *buf, size_t len);

static void hook(const char *buf, size_t len)
{
	if (__cimo_hook_write) __cimo_hook_write(buf, len);
}

struct __cimo_stdout {
	size_t len;
	char buf[4096];
} __cimo_stdout;

void __cimo_flush(void)
{
	size_t n = __cimo_stdout.len;
	if (n > sizeof __cimo_stdout.buf) n = sizeof __cimo_stdout.buf;
	__cimo_stdout.len = 0;
	if (n) hook(__cimo_stdout.buf, n);
}

static void put(const char *s, size_t l)
{
	size_t n = __cimo_stdout.len;
	if (n > sizeof __cimo_stdout.buf) n = 0;
	if (l > sizeof __cimo_stdout.buf - n) {
		__cimo_flush();
		n = 0;
		if (l >= sizeof __cimo_stdout.buf) {
			hook(s, l);
			return;
		}
	}
	memcpy(__cimo_stdout.buf + n, s, l);
	__cimo_stdout.len = n + l;
}

void __cimo_stdout_write(struct __cimo_sink *f, const char *s, size_t l)
{
	put(s, l);
	if (!f->len && memchr(s, '\n', l)) f->len = 1;  /* flush when the call ends */
}

int putchar(int c)
{
	char ch = c;
	put(&ch, 1);
	if (ch == '\n') __cimo_flush();
	return (unsigned char)ch;
}

int puts(const char *s)
{
	put(s, strlen(s));
	put("\n", 1);
	__cimo_flush();
	return 0;
}

int vprintf(const char *restrict fmt, va_list ap)
{
	struct __cimo_sink f = { __cimo_stdout_write, 0, 0, 0 };
	int ret = __cimo_vformat(&f, fmt, ap);
	if (f.len) __cimo_flush();
	return ret;
}

int printf(const char *restrict fmt, ...)
{
	int ret;
	va_list ap;
	va_start(ap, fmt);
	ret = vprintf(fmt, ap);
	va_end(ap);
	return ret;
}
