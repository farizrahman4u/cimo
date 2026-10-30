/* cimo: "C" locale only, where collation is byte order. */
#include <string.h>

int strcoll(const char *l, const char *r)
{
	return strcmp(l, r);
}
