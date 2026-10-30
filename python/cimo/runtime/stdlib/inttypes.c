/* cimo: <inttypes.h> functions. intmax_t is long, so they are the long versions. */
#include <inttypes.h>
#include <stdlib.h>

intmax_t imaxabs(intmax_t a)
{
	return a > 0 ? a : -a;
}

imaxdiv_t imaxdiv(intmax_t num, intmax_t den)
{
	return (imaxdiv_t){ num/den, num%den };
}

intmax_t strtoimax(const char *restrict s, char **restrict p, int base)
{
	return strtol(s, p, base);
}

uintmax_t strtoumax(const char *restrict s, char **restrict p, int base)
{
	return strtoul(s, p, base);
}
