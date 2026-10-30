/* ctype.h: every function on EOF and every unsigned char value. */
#include <ctype.h>
#include "out.h"

long run(char *out, long cap)
{
	o_init(out, cap);
	for (int c = -1; c < 256; c++) {
		o_d(c);
		o_c(' ');
		o_d(!!isalnum(c)); o_d(!!isalpha(c)); o_d(!!isblank(c)); o_d(!!iscntrl(c));
		o_d(!!isdigit(c)); o_d(!!isgraph(c)); o_d(!!islower(c)); o_d(!!isprint(c));
		o_d(!!ispunct(c)); o_d(!!isspace(c)); o_d(!!isupper(c)); o_d(!!isxdigit(c));
		o_c(' ');
		o_d(tolower(c));
		o_c(' ');
		o_d(toupper(c));
		o_c('\n');
	}
	return o_end();
}
