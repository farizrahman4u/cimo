/* cimo: <stdint.h> (C11 7.20), LP64 aarch64. 64-bit types are long on Linux and long long on
   Apple and Windows, as in the platform headers (this matters for C++-style mangling and for printf formats,
   not for code generation). */
#ifndef __CIMO_STDINT_H
#define __CIMO_STDINT_H

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__APPLE__) || defined(_WIN32)
typedef long long int64_t;
typedef unsigned long long uint64_t;
#define __CIMO_I64(c) c ## LL
#define __CIMO_U64(c) c ## ULL
#else
typedef long int64_t;
typedef unsigned long uint64_t;
#define __CIMO_I64(c) c ## L
#define __CIMO_U64(c) c ## UL
#endif

typedef int8_t int_least8_t;
typedef int16_t int_least16_t;
typedef int32_t int_least32_t;
typedef int64_t int_least64_t;
typedef uint8_t uint_least8_t;
typedef uint16_t uint_least16_t;
typedef uint32_t uint_least32_t;
typedef uint64_t uint_least64_t;

typedef signed char int_fast8_t;
typedef unsigned char uint_fast8_t;
#ifdef __APPLE__
typedef int16_t int_fast16_t;
typedef int32_t int_fast32_t;
typedef uint16_t uint_fast16_t;
typedef uint32_t uint_fast32_t;
#define INT_FAST16_MIN INT16_MIN
#define INT_FAST16_MAX INT16_MAX
#define UINT_FAST16_MAX UINT16_MAX
#define INT_FAST32_MIN INT32_MIN
#define INT_FAST32_MAX INT32_MAX
#define UINT_FAST32_MAX UINT32_MAX
#else
typedef long int_fast16_t;
typedef long int_fast32_t;
typedef unsigned long uint_fast16_t;
typedef unsigned long uint_fast32_t;
#define INT_FAST16_MIN INT64_MIN
#define INT_FAST16_MAX INT64_MAX
#define UINT_FAST16_MAX UINT64_MAX
#define INT_FAST32_MIN INT64_MIN
#define INT_FAST32_MAX INT64_MAX
#define UINT_FAST32_MAX UINT64_MAX
#endif
typedef int64_t int_fast64_t;
typedef uint64_t uint_fast64_t;

typedef long intptr_t;
typedef unsigned long uintptr_t;
typedef long intmax_t;
typedef unsigned long uintmax_t;

#define INT8_MIN (-128)
#define INT16_MIN (-32767 - 1)
#define INT32_MIN (-2147483647 - 1)
#define INT64_MIN (-INT64_MAX - 1)
#define INT8_MAX 127
#define INT16_MAX 32767
#define INT32_MAX 2147483647
#define INT64_MAX __CIMO_I64(9223372036854775807)
#define UINT8_MAX 255
#define UINT16_MAX 65535
#define UINT32_MAX 4294967295U
#define UINT64_MAX __CIMO_U64(18446744073709551615)

#define INT_LEAST8_MIN INT8_MIN
#define INT_LEAST16_MIN INT16_MIN
#define INT_LEAST32_MIN INT32_MIN
#define INT_LEAST64_MIN INT64_MIN
#define INT_LEAST8_MAX INT8_MAX
#define INT_LEAST16_MAX INT16_MAX
#define INT_LEAST32_MAX INT32_MAX
#define INT_LEAST64_MAX INT64_MAX
#define UINT_LEAST8_MAX UINT8_MAX
#define UINT_LEAST16_MAX UINT16_MAX
#define UINT_LEAST32_MAX UINT32_MAX
#define UINT_LEAST64_MAX UINT64_MAX

#define INT_FAST8_MIN INT8_MIN
#define INT_FAST8_MAX INT8_MAX
#define UINT_FAST8_MAX UINT8_MAX
#define INT_FAST64_MIN INT64_MIN
#define INT_FAST64_MAX INT64_MAX
#define UINT_FAST64_MAX UINT64_MAX

#define INTPTR_MIN (-INTPTR_MAX - 1)
#define INTPTR_MAX 9223372036854775807L
#define UINTPTR_MAX 18446744073709551615UL
#define INTMAX_MIN (-INTMAX_MAX - 1)
#define INTMAX_MAX 9223372036854775807L
#define UINTMAX_MAX 18446744073709551615UL

#define PTRDIFF_MIN (-PTRDIFF_MAX - 1)
#define PTRDIFF_MAX 9223372036854775807L
#define SIZE_MAX 18446744073709551615UL
#define SIG_ATOMIC_MIN (-2147483647 - 1)
#define SIG_ATOMIC_MAX 2147483647
#ifdef _WIN32
#define WCHAR_MIN 0
#define WCHAR_MAX 65535
#define WINT_MIN 0
#define WINT_MAX 65535
#elif defined(__APPLE__)
#define WCHAR_MIN (-2147483647 - 1)
#define WCHAR_MAX 2147483647
#define WINT_MIN (-2147483647 - 1)
#define WINT_MAX 2147483647
#else
#define WCHAR_MIN 0U
#define WCHAR_MAX 4294967295U
#define WINT_MIN 0U
#define WINT_MAX 4294967295U
#endif

#define INT8_C(c) c
#define INT16_C(c) c
#define INT32_C(c) c
#define INT64_C(c) __CIMO_I64(c)
#define UINT8_C(c) c
#define UINT16_C(c) c
#define UINT32_C(c) c ## U
#define UINT64_C(c) __CIMO_U64(c)
#define INTMAX_C(c) c ## L
#define UINTMAX_C(c) c ## UL

#endif
