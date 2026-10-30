#if 1 + 2 * 3 == 7 && (1 ? 2 : (1/0)) && !(0 && 1/0)
ok1
#endif
#if -1 > 0u
ok2_unsigned_promotion
#else
bad2
#endif
#if (-1 >> 63) == -1 && (0xffffffffffffffff >> 63) == 1
ok3
#endif
#if 'a' == 97 && '\377' == 255 && '\x41' == 65 && L'\xffffffff' > 0 && '\0' == 0
ok4
#endif
#if defined(FOO) || defined BAR
bad5
#elif !defined FOO
ok5
#else
bad5b
#endif
#define FOO 1
#ifdef FOO
ok6
#if 0
# garbage here ' unterminated
#error not reached
#elif FOO == 1
ok7
#else
bad7
#endif
#endif
#if 18446744073709551615u == -1
ok8
#endif
#if 0x7fffffffffffffff + 0 > 0 && (1 ? -1 : 0u) > 0
ok9
#endif
#if __STDC__ && __STDC_VERSION__ == 201112L && __STDC_HOSTED__ == 0
ok10
#endif
#if UNDEFINED_THING == 0 && (2 || 1/0) && (0 ? 1/0 : 3)
ok11
#endif
#define ZERO 0
#if ZERO
bad12
#elif 1/ (ZERO + 1)
ok12
#endif
#if (1 << 63) < 0 && (1u << 63) > 0 && 7 / -2 == -3 && 7 % -2 == 1 && -7 % 2 == -1
ok13
#endif
#if 0
#elif 1
ok14
#elif 1/0
#endif
#if 1, 2
ok15
#endif
#if ~0u == 0xffffffffffffffff && -0x8000000000000000 < 0 == 0
ok16
#endif
#define DEFD defined(FOO)
#ifndef __linux__
bad17
#endif
#if __aarch64__ && __LP64__ && __SIZEOF_POINTER__ == 8 && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
ok18
#endif
