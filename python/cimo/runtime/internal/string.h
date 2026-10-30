/* cimo runtime: the public <string.h> plus musl's internal names. */
#include_next <string.h>
#ifndef __CIMO_RT_STRING_H
#define __CIMO_RT_STRING_H
char *__stpcpy(char *restrict, const char *restrict);
char *__stpncpy(char *restrict, const char *restrict, size_t);
char *__strchrnul(const char *, int);
void *__memrchr(const void *, int, size_t);
#endif
