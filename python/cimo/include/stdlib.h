/* cimo: <stdlib.h>, the OS-free subset (no getenv, system, atexit, locale), provided by
   cimo's runtime. Memory comes from the module heap. abort() and exit() unwind back to
   Python (KernelError). */
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
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define RAND_MAX 0x7fffffff

typedef struct { int quot, rem; } div_t;
typedef struct { long quot, rem; } ldiv_t;
typedef struct { long long quot, rem; } lldiv_t;

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
long long llabs(long long j);
div_t div(int numer, int denom);
ldiv_t ldiv(long numer, long denom);
lldiv_t lldiv(long long numer, long long denom);

int atoi(const char *nptr);
long atol(const char *nptr);
long long atoll(const char *nptr);
double atof(const char *nptr);
long strtol(const char *restrict nptr, char **restrict endptr, int base);
unsigned long strtoul(const char *restrict nptr, char **restrict endptr, int base);
long long strtoll(const char *restrict nptr, char **restrict endptr, int base);
unsigned long long strtoull(const char *restrict nptr, char **restrict endptr, int base);
double strtod(const char *restrict nptr, char **restrict endptr);
float strtof(const char *restrict nptr, char **restrict endptr);

int rand(void);
void srand(unsigned seed);

_Noreturn void abort(void);
_Noreturn void exit(int status);
_Noreturn void _Exit(int status);

#endif
