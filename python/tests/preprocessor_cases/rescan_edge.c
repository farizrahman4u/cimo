#define foo(x) bar x
foo(foo) (2)
#define m() m
m()() m()
#define a a b
a
#define f(x) g(x)
#define g(x) x
#define h f(h)
h
#define ff(x) x
ff(1
#undef ff
#define ff 2
ff)
#define EMPTY
#define bracket(x) [x]
bracket(EMPTY) bracket( EMPTY ) bracket(EMPTY EMPTY)
#define CAT(a, b) a ## b
#define AB done
CAT(A, B) CAT(A,) CAT(,B)
#define V(x, ...) x __VA_ARGS__
V(1) V() V(1, 2, 3) V((a, b), c)
#define SL(...) #__VA_ARGS__
SL( a ,  b ) SL() SL(  "x"  ,  'y'  ) SL( /* c */ a /* d */ b /* e */ )
#define OBJF ff2
#define ff2(x) <x>
OBJF
#define UNUSED 1
(3) OBJF (4)
#define str(x) #x
#define xstr(x) str(x)
xstr(__LINE__) str(__LINE__) __LINE__ xstr(
__LINE__
)
#define LINE_AT __LINE__
LINE_AT bracket(
LINE_AT)
#define nest1(x) nest2(x) + nest2
#define nest2(x) nest1(x) * x
nest1(1)(2)(3)
#define id(x) x
id(id)(id)(5) id(id(id))(6)
#define lp (
#define rp )
id lp 7 rp id(lp) 8 rp
#define apply(m, x) m(x)
apply(id, 9) apply(bracket, apply(id, 10)) apply(apply, id)
#define CALL(f) f(CALL)
CALL(id) CALL(bracket)
#define PASTE3(a,b,c) a ## b ## c
PASTE3(x, , y) PASTE3(, , ) PASTE3(1, 2, 3) PASTE3(<, <, =)
#define glue(a, b) a ## b
#define xglue(a, b) glue(a, b)
#define LOW low
xglue(glue(a, b), LOW) glue(LOW, LOW) xglue(LOW, LOW)
#define obj2 + obj2
obj2 id(obj2)
#define SELF(x) x SELF(x)
SELF(1) SELF(SELF(2))
#define X1 X2
#define X2 X3
#define X3 X1 end
X1 X2 X3
