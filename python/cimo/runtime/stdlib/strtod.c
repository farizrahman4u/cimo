/* cimo: musl src/stdlib/strtod.c over the string pseudo-FILE, without strtold (see
   ../COPYRIGHT). */
#include <stdlib.h>
#include "floatscan.h"

static double strtox(const char *s, char **p, int prec)
{
	FILE f;
	sh_fromstring(&f, s);
	shlim(&f, 0);
	double y = __floatscan(&f, prec, 1);
	ptrdiff_t cnt = shcnt(&f);
	if (p) *p = cnt ? (char *)s + cnt : (char *)s;
	return y;
}

float strtof(const char *restrict s, char **restrict p)
{
	return strtox(s, p, 0);
}

double strtod(const char *restrict s, char **restrict p)
{
	return strtox(s, p, 1);
}
