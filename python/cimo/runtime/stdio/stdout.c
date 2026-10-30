/* cimo: stdout for printf, puts and putchar.

   Output collects in a per-thread buffer (in the calling thread's context, see
   cimo/execution.py) and goes to Python's sys.stdout through one hook,
   __cimo_hook_write(buf, len): a function pointer that Python sets after loading to a ctypes
   callback (so the module imports nothing).
   A buffer is written out when output contains a newline, when it is full, and when a kernel
   returns (Python calls __cimo_flush if the thread's buffer is nonempty), so C and Python
   output interleave in program order, sys.stdout redirection (Jupyter,
   contextlib.redirect_stdout) captures it, and lines printed by concurrent threads come out
   whole. A thread without a context (one the kernel started itself) writes unbuffered. */
#include <string.h>
#include "stdio_impl.h"

#define EOF (-1)
#define CTX_OUT 4096  /* cimo/execution.py */

void (*__cimo_hook_write)(const char *buf, size_t len);
void *__cimo_ctx(void);

static void hook(const char *buf, size_t len)
{
	if (__cimo_hook_write) __cimo_hook_write(buf, len);
}

struct out {
	size_t len;
	char buf[4096 - sizeof(size_t)];
};

static struct out *out(void)
{
	char *c = __cimo_ctx();
	return c ? (struct out *)(c + CTX_OUT) : 0;
}

static void flush(struct out *o)
{
	size_t n = o->len;
	o->len = 0;
	if (n) hook(o->buf, n);
}

void __cimo_flush(void)
{
	struct out *o = out();
	if (o) flush(o);
}

static void put(struct out *o, const char *s, size_t l)
{
	if (!o) {
		hook(s, l);
		return;
	}
	if (l > sizeof o->buf - o->len) {
		flush(o);
		if (l >= sizeof o->buf) {
			hook(s, l);
			return;
		}
	}
	memcpy(o->buf + o->len, s, l);
	o->len += l;
}

void __cimo_stdout_write(struct __cimo_sink *f, const char *s, size_t l)
{
	put(f->ctx, s, l);
	if (!f->len && memchr(s, '\n', l)) f->len = 1;  /* flush when the call ends */
}

int putchar(int c)
{
	struct out *o = out();
	char ch = c;
	put(o, &ch, 1);
	if (o && ch == '\n') flush(o);
	return (unsigned char)ch;
}

int puts(const char *s)
{
	struct out *o = out();
	put(o, s, strlen(s));
	put(o, "\n", 1);
	if (o) flush(o);
	return 0;
}

int vprintf(const char *restrict fmt, va_list ap)
{
	struct out *o = out();
	struct __cimo_sink f = { __cimo_stdout_write, 0, 0, 0, o };
	int ret = __cimo_vformat(&f, fmt, ap);
	if (o && f.len) flush(o);
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
