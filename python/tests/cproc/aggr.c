/* structs, unions, bitfields, initializers, compound literals, strings */
#include <stddef.h>
#include <stdalign.h>
struct P { int x, y; };
struct S { char c; short s; int i; long l; float f; double d; void *p; struct P pt; char arr[3]; };
union U { int i; float f; char c[7]; double d; };
struct BF { unsigned a : 3, b : 5; signed c : 4; int : 0; unsigned d : 31; long e : 40; _Bool f : 1; char g : 2; };
struct BF2 { int a : 1; int : 3; int b : 28; unsigned long long c : 63; };
struct F { int n; double v[]; };
struct N { struct { int a; union { int b; float c; }; }; int d; struct P p[2]; };
struct __attribute__((packed)) PK { char c; int i; };
struct AL { char c; alignas(16) int i; _Alignas(long) char d; };

struct S s0 = {1, 2, 3, 4, 5.5f, 6.5, &s0, {7, 8}, "ab"};
struct S s1 = {.d = 1.0, .pt.y = 3, .arr = {1, [2] = 3}, .c = 'x'};
struct S s2 = {.pt = {.x = 1}, 'z'};
union U u0 = {5}, u1 = {.f = 2.5f}, u2 = {.c = "hello"};
struct BF bf0 = {1, 2, -3, 4, -5, 1, 1}, bf1 = {.d = 0x7fffffff, .e = -1, .b = 31};
struct BF2 bf2 = {-1, 0xfffffff, 0x7fffffffffffffff};
struct N n0 = {{1, {.c = 2.0f}}, 3, {{4, 5}, [1].y = 6}};
struct N n1 = {.b = 7, .p[0].x = 8, .a = 9};
struct PK pk = {1, 2};
struct AL al = {1, 2, 3};
int arr[] = {[3] = 1, 2, [1] = 3};
int arr2[2][3] = {{1, 2, 3}, {4}};
int arr3[2][3] = {1, 2, 3, 4};
char str[] = "hello\n\tworld\"\\\x01\177\200\377";
char str2[10] = "abc";
char str3[3] = "abc";
char str4[2][4] = {"ab", "cde"};
unsigned char ustr[] = u8"héllo";
typeof(L'x') wstr[] = L"wide é 😀";
unsigned short u16[] = u"utf16 é 😀";
unsigned int u32[] = U"utf32 😀";
const char *strs[] = {"one", "two", "one", "two" "three"};
char *cat = "abc" "def" "\x41" "B";
int *pi = &arr[2];
int *pi2 = arr + 3;
void *nul = 0;
struct P *cl = &(struct P){1, 2};
int *clarr = (int[]){1, 2, 3};
char override[] = {"abcdef", [2] = 'X'};
struct { char s[4]; int i; } ov2 = {"xyz", .s[1] = 'Q', 5};
static int sarr[5];
int tentative;
int tentative;
extern int ext;
int *pext = &ext;
_Thread_local int tl = 5;
static _Thread_local int tl2;
int uninit[100];
double dz[4] = {0};
struct F fam = {2};
int fn(int);
int (*fp)(int) = fn;
int (*fps[])(int) = {fn, &fn, 0};

struct S getS(struct S *p) { return *p; }
union U getU(union U u) { u.i++; return u; }
struct P mkP(int x, int y) { struct P p = {x, y}; return p; }

int bitfields(struct BF *b, struct BF2 *c)
{
	b->a = 5;
	b->b += 3;
	b->c = -2;
	b->d++;
	--b->e;
	b->f = !b->f;
	b->g |= 1;
	c->c *= 3;
	c->a = c->b;
	return b->a + b->b + b->c + b->d + b->e + b->f + b->g + c->a + (int)c->c;
}

int locals(int n)
{
	struct S s = {1, 2, 3};
	struct S t = s;
	union U u = {.f = 1.0f};
	struct BF b = {1, 2, 3, 4, 5};
	int a[10] = {1, 2, [7] = 8};
	char str[] = "local string";
	char str2[20] = "short";
	typeof(L'x') wide[] = L"wide";
	struct N nn = {.p[1].y = n, .d = n + 1};
	struct P *pp = &(struct P){n, n};
	int *ip = (int[3]){n};
	alignas(32) char big[33];
	alignas(64) int big2;
	char vla[n];
	int vla2[n][n + 1];
	big[0] = 1;
	big2 = 2;
	vla[n - 1] = 1;
	vla2[1][2] = sizeof vla2 + sizeof vla2[0];
	t.pt = mkP(1, 2);
	s = getS(&t);
	u = getU(u);
	return s.i + t.pt.y + u.i + b.c + a[7] + str[3] + str2[4] + wide[1] + nn.d + pp->x + ip[0] + big[0] + big2 + vla[n - 1] + vla2[1][2] + sizeof vla;
}

int strfn(void) { return "abc"[1] + sizeof(__func__) + __func__[0]; }
size_t offs(void) { return offsetof(struct S, pt.y) + __builtin_offsetof(struct N, p[1].y) + offsetof(struct N, c); }
int flex(struct F *f) { return f->v[f->n] + sizeof(struct F); }
