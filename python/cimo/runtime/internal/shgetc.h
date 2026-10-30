/* cimo runtime: musl's string pseudo-FILE for the strto* scanners (src/internal/shgetc.h
   with sh_fromstring), without stdio: the input is always a NUL-terminated string. */
#ifndef __CIMO_RT_SHGETC_H
#define __CIMO_RT_SHGETC_H

#include <stddef.h>
#include "musl.h"

typedef struct __cimo_sfile {
	const unsigned char *buf, *rpos;
	ptrdiff_t shcnt;
	long shlim;
} FILE;

#define shcnt(f) ((f)->shcnt + ((f)->rpos - (f)->buf))
#define shlim(f, lim) ((void)((f)->shlim = (lim), (f)->shcnt = (f)->buf - (f)->rpos))
#define shgetc(f) (*(f)->rpos++)
#define shunget(f) ((f)->shlim>=0 ? (void)(f)->rpos-- : (void)0)

#define sh_fromstring(f, s) \
	((f)->buf = (f)->rpos = (const unsigned char *)(s))

#endif
