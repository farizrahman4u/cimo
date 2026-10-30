/* cimo: <stdlib.h>, the OS-free subset (no stdio, exit, getenv, rand, locale). Memory comes
   from the module heap. abort() is the error-unwind path back to Python. */
#ifndef __CIMO_STDLIB_H
#define __CIMO_STDLIB_H

#ifndef __CIMO_SIZE_T
#define __CIMO_SIZE_T
typedef unsigned long size_t;
#endif
#ifndef __CIMO_WCHAR_T
#define __CIMO_WCHAR_T
#ifdef __APPLE__
typedef int wchar_t;
#else
typedef unsigned int wchar_t;
#endif
#endif
#undef NULL
#define NULL ((void *)0)

typedef struct {
	int quot;
	int rem;
} div_t;

void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);
void *aligned_alloc(size_t alignment, size_t size);

void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void *bsearch(const void *key, const void *base, size_t nmemb, size_t size,
              int (*compar)(const void *, const void *));

int abs(int j);
long labs(long j);
div_t div(int numer, int denom);

long strtol(const char *restrict nptr, char **restrict endptr, int base);
double strtod(const char *restrict nptr, char **restrict endptr);
int atoi(const char *nptr);

_Noreturn void abort(void);

#endif
