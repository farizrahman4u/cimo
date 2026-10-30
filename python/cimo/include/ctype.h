/* cimo: <ctype.h> (C11 7.4), "C" locale only. The functions are in cimo's runtime; the
   macros (from musl) inline the simple ones. */
#ifndef __CIMO_CTYPE_H
#define __CIMO_CTYPE_H

int isalnum(int c);
int isalpha(int c);
int isblank(int c);
int iscntrl(int c);
int isdigit(int c);
int isgraph(int c);
int islower(int c);
int isprint(int c);
int ispunct(int c);
int isspace(int c);
int isupper(int c);
int isxdigit(int c);
int tolower(int c);
int toupper(int c);

#define isalpha(a) (0 ? isalpha(a) : (((unsigned)(a)|32)-'a') < 26)
#define isdigit(a) (0 ? isdigit(a) : ((unsigned)(a)-'0') < 10)
#define islower(a) (0 ? islower(a) : ((unsigned)(a)-'a') < 26)
#define isupper(a) (0 ? isupper(a) : ((unsigned)(a)-'A') < 26)
#define isprint(a) (0 ? isprint(a) : ((unsigned)(a)-0x20) < 0x5f)
#define isgraph(a) (0 ? isgraph(a) : ((unsigned)(a)-0x21) < 0x5e)

#endif
