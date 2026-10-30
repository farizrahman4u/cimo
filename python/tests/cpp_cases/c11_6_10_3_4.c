/* C11 6.10.3.4 EXAMPLE and DR 017 style rescanning */
#define f(a) a*g
#define g(a) f(a)
f(2)(9)
#define AA BB
#define BB AA
AA BB
#define lparen (
#define F(x) [x]
F lparen 1)
#define NIL(x) x
#define G_0(arg) NIL(G_1)(arg)
#define G_1(arg) NIL(arg)
G_0(42)
#define obj obj + 1
obj
#define fn(x) fn(x) + x
fn(fn(1))
#define e1 e2(
#define e2(x) x e1
e1 1) 2) 3)
