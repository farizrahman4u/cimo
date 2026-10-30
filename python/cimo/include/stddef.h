/* cimo: <stddef.h> (C11 7.19), LP64 aarch64 (Linux and macOS). */
#ifndef __CIMO_STDDEF_H
#define __CIMO_STDDEF_H

typedef long ptrdiff_t;
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
#ifdef __APPLE__
/* Apple's is long double, which is binary64 there: size 8, alignment 8. */
typedef double max_align_t;
#else
/* Matches gcc's max_align_t on aarch64 Linux: size 32, alignment 16. */
typedef struct {
	_Alignas(16) long long __cimo_max_align_ll;
	_Alignas(16) unsigned char __cimo_max_align_ld[16];
} max_align_t;
#endif

#undef NULL
#define NULL ((void *)0)
#define offsetof(type, member) __builtin_offsetof(type, member)

#endif
