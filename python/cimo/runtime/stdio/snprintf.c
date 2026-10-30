/* cimo: snprintf and friends over musl's formatter (see ../COPYRIGHT). */
#include <limits.h>
#include <string.h>
#include "stdio_impl.h"

static void sn_write(struct __cimo_sink *f, const char *s, size_t l)
{
	size_t k = f->cap - f->len;
	if (k > l) k = l;
	memcpy(f->buf + f->len, s, k);
	f->len += k;
}

int vsnprintf(char *restrict s, size_t n, const char *restrict fmt, va_list ap)
{
	char dummy[1];
	struct __cimo_sink f = { sn_write, n ? s : dummy, 0, n ? n - 1 : 0 };
	int ret = __cimo_vformat(&f, fmt, ap);
	f.buf[f.len] = 0;
	return ret;
}

int snprintf(char *restrict s, size_t n, const char *restrict fmt, ...)
{
	int ret;
	va_list ap;
	va_start(ap, fmt);
	ret = vsnprintf(s, n, fmt, ap);
	va_end(ap);
	return ret;
}

int vsprintf(char *restrict s, const char *restrict fmt, va_list ap)
{
	return vsnprintf(s, INT_MAX, fmt, ap);
}

int sprintf(char *restrict s, const char *restrict fmt, ...)
{
	int ret;
	va_list ap;
	va_start(ap, fmt);
	ret = vsprintf(s, fmt, ap);
	va_end(ap);
	return ret;
}
