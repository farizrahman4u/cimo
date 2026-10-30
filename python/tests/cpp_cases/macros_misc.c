#define EMPTY
#define LPAREN (
#define RPAREN )
#define F(x, y) x + y
#define ELLIP_FUNC(...) __VA_ARGS__
ELLIP_FUNC(F, LPAREN, 'a', 'b', RPAREN);
#define CAT(a, b) a ## b
#define XCAT(a, b) CAT(a, b)
CAT(1, 2) CAT(x, y) CAT(+, =) CAT(<, <=) XCAT(CAT(a,b), c) CAT(L, 'a') CAT(L, "s")
CAT(0x, 1p-3) CAT(1e, +) CAT(-, >) CAT(/, =) CAT(%:, %:) CAT(<, :)
#define STR(x) #x
#define XSTR(x) STR(x)
STR( a   b
  c ) STR("a\n" '\'' "\\" '"') STR() STR(   ) XSTR(__LINE__) XSTR(EMPTY) STR(@) STR(a/**/b)
#define EPRINTF(fmt, ...) eprintf(fmt, ## __VA_ARGS__)
EPRINTF("x") EPRINTF("x",) EPRINTF("x", 1, 2) EPRINTF("x", EMPTY)
#define NAMEDVA(fmt, args...) f(fmt, args)
NAMEDVA(1, 2, 3)
#define ZEROARG() zero
ZEROARG() ZEROARG( ) ZEROARG
#define ONEARG(x) [x]
ONEARG() ONEARG(()) ONEARG((a,b)) ONEARG(
 multi
 line
)
a EMPTY b EMPTY+EMPTY+ -EMPTY- c EMPTY ## d
#define PLUS +
+PLUS -PLUS- PLUS+ PLUS PLUS
#define f1(x) x
f1(f1)(1) f1(f1(f1))(2)
#define recur recur
#define self(x) self(x)
recur self(self(1))
#define AFTER(x) x AFTER
AFTER(1)(2)
#define q(x) x
#define r(x) q(x
r(1))
__FILE__ __LINE__ __COUNTER__ __COUNTER__ __INCLUDE_LEVEL__
#line 100 "renamed.c"
__LINE__ __FILE__
#line 200
__LINE__
# 300 "gnu_marker.c"
__LINE__ __FILE__
#undef F
#ifdef F
bad
#endif
#define hash_hash # ## #
#define mkstr(a) # a
#define in_between(a) mkstr(a)
#define join(c, d) in_between(c hash_hash d)
char p[] = join(x, y);
#define t2(x,y) x ## y
t2(L, 'x') t2(u8, "s") t2(., 5) t2(1, .2e+3)
#define OBJPASTE a ## b ## c
OBJPASTE
