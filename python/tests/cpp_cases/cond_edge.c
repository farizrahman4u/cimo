#define D defined(FOO)
#define FOO
#if D
ok1
#endif
#if defined FOO && defined(FOO) && !defined BAR
ok2
#endif
#if 0x8000000000000000 == -0x8000000000000000
ok3
#endif
#if (2 || 1/0) && (0 && 1/0 || 1) && (1 ? 1 : 1/0)
ok4
#endif
#if 0
#if garbage (( ))
#else
#endif
#elif 1
ok5
#endif
#   /* null directive */
# /* comment */ define SPACED 6
SPACED
#if -1 < 0 && -1 < 0u == 0 && (0 ? 1u : -1) > 0 && (1 ? -1 : 0) < 0
ok7
#endif
#if 1 << 2 == 4 && 256 >> 4 == 16 && (-16 >> 2) == -4 && 0b101 == 5
ok8
#endif
#if 'A' == 65 && '\n' == 10 && '\\' == 92 && '\'' == 39 && u'x' == 120 && U'é' == 0xe9
ok9
#endif
#if 10 / 3 == 3 && -10 / 3 == -3 && 10 % -3 == 1 && 0x10 == 16 && 010 == 8 && 1000000000000LL > 0
ok10
#endif
#if !0 == 1 && ~0 == -1 && -(-1) == 1 && +1 == 1 && (1, 0) == 0
ok11
#endif
#ifdef SPACED
#else
bad
#endif
#ifndef SPACED
bad
#elif defined SPACED
ok12
#endif
#if __INT_MAX__ == 2147483647 && __LONG_LONG_MAX__ > __INT_MAX__ && __SIZEOF_LONG__ == 8
ok13
#endif
#if __STDC_ENDIAN_NATIVE__ == 1234 && defined __STDC_NO_ATOMICS__ && defined __STDC_NO_COMPLEX__
ok14
#endif
#define FN(x) ((x) * 2)
#if FN(3) == 6 && FN(FN(1)) == 4
ok15
#endif
#if UNDEF_FN_LIKE + 0 == 0
ok16
#endif
#if 18446744073709551615 == -1
ok17
#endif
