/* declarations, enums, typedefs, attributes, generic, typeof, builtins */
#include <stddef.h>
enum E { A, B = 5, C, D = -1, F };
enum E2 { BIG = 0xffffffff };
enum E3 { NEG = -2147483649LL };
enum E4 : unsigned char { X1 = 0, X2 = 255 };
enum E5 : long { Y1 = -1, Y2 };
enum { ANON = 7 };
enum E6 { Z1 = 0x7fffffff, Z2 };
enum E7 { W1 = 0xffffffffffffffffull };
typedef enum E Et;
Et ev = C;
enum E2 e2 = BIG;
enum E3 e3 = NEG;
enum E4 e4 = X2;
long e5 = Y2 + sizeof(enum E5) + sizeof(enum E6) + Z2 + sizeof(enum E7);
int earr[F + 2];
static_assert(B == 5, "B");
_Static_assert(sizeof(int) == 4, "int");
static_assert(1);
typedef int T, *TP, TA[3], (*TF)(void);
T t0 = 1; TP tp0 = &t0; TA ta0 = {1, 2, 3}; TF tf0;
typedef struct L { struct L *next; int v; } L;
L l0 = {&l0, 1};
int g1 = _Generic(1.0f, float: 1, double: 2, default: 3);
int g2 = _Generic((char)1, int: 1, default: 3);
int g3 = _Generic("x", char *: 4, const char *: 5);
typeof(int *) ty0;
typeof(t0) ty1 = 4;
__typeof__(ta0) ty2;
typeof_unqual(const int) ty3;
int tcp = __builtin_types_compatible_p(int, signed) + __builtin_types_compatible_p(int, long);
int cst = __builtin_constant_p(1 + 2) + __builtin_constant_p(t0);
extern int asmname __asm__("real_name");
int asmname2 __asm("other" "name") = 3;
int useasm(void) { return asmname; }
static int sfn(void) { return 1; }
inline int ifn(void) { return 2; }
extern inline int eifn(void) { return 3; }
static inline int sifn(void) { return 4; }
struct __attribute__((packed)) { char a; long b; } pkd;
[[deprecated, maybe_unused]] int depr;
int callall(void) { return sfn() + ifn() + eifn() + sifn(); }
int nested_scopes(int x)
{
	int y = x;
	{ int x = 2; y += x; { extern int g1; y += g1; } }
	typedef long T;
	T z = sizeof(T);
	enum { Q = 3 };
	struct L l = {0, Q};
	static int counter = 5;
	static const char *msg = "msg";
	counter++;
	return y + z + l.v + counter + msg[0];
}
int stmtexpr(int a) { return (a, a + 1, a * 2); }
void *alloca_(int n) { return __builtin_alloca(n * 4); }
long expect_(long x) { return __builtin_expect(x, 1) ? 1 : 0; }
int kr();
int proto(int, char *, ...);
int arrparam(int a[static 3], int b[const], int n, int c[n], int d[*]);
int arrparam(int a[3], int b[], int n, int c[n], int d[]) { return a[0] + b[1] + c[n - 1]; }
const volatile int cv = 3;
int readcv(void) { return cv; }
int restrict_(int *restrict a, const int *restrict b) { *a = *b; return *a; }
static int fwd;
static int fwd = 4;
bool cbool = true;
