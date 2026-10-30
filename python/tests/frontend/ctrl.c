/* control flow, functions, varargs, pointers */
#include <stdarg.h>
#include <stddef.h>

int sum(int n, ...)
{
	va_list ap, aq;
	int s = 0;
	va_start(ap, n);
	va_copy(aq, ap);
	for (int i = 0; i < n; i++)
		s += va_arg(ap, int);
	s += (int)va_arg(aq, double);
	s += *va_arg(aq, int *);
	va_end(aq);
	va_end(ap);
	return s;
}

int vsum(int n, va_list ap) { int s = 0; while (n--) s += va_arg(ap, long); return s; }

int sw(int x, long y, unsigned char c)
{
	int r = 0;
	switch (x) {
	case 1: r = 1;
	case 2: r += 2; break;
	case -1: r = -1; break;
	case 100: case 200: case 300: case 400: case 500: case 600: case 700: case 800: r = 9; break;
	case 0x7fffffff: r = 7; break;
	default: r = 42;
	}
	switch (y) {
	case 1: return 1;
	case -5: return 2;
	case 1L << 40: return 3;
	}
	switch (c) { default: r++; case 'a': r--; }
	switch (x) { case 5: { int z = 3; r += z; } }
	switch (x) {}
	return r;
}

int loops(int n, int *p)
{
	int s = 0, i, j;
	for (i = 0; i < n; ++i) {
		if (i == 3) continue;
		if (i > 10) break;
		s += p[i];
	}
	for (;;) { if (s > 100) break; s += 7; }
	for (int k = 0, m = 5; k < m; k++, m--) s ^= k * m;
	while (n-- > 0) { s--; if (!n) break; }
	do s += 2; while (s < 50 && n);
	do { if (s & 1) continue; s >>= 1; } while (s > 1000);
	i = 0;
again:
	if (++i < 3) goto again;
	j = 0;
	goto skip;
	j = 5;
skip:
	for (i = 0; i < 2; i++) for (j = 0; j < 2; j++) { if (j) goto out; }
out:
	return s + i + j;
}

int logic(int a, int b, double c, long d, char e, void *p)
{
	int r = a && b;
	r += a || b;
	r += !a;
	r += !c + !d + !e + !p;
	if (a && (b || c)) r++;
	if (!(a || d) && e) r--;
	if (p) r++;
	if (!p) r++;
	if (c) r++;
	if (d) r++;
	if (e) r++;
	if (a == 0) r++;
	if (0 != a) r++;
	if (a, b) r++;
	r += a ? b : c;
	r += p ? 1 : 0;
	r += (a ?: b);
	void *q = a ? p : 0;
	const void *q2 = a ? (const int *)p : (void *)p;
	a ? (void)0 : (void)1;
	while (a && b) a--;
	return r + (q == q2) + (p != NULL) + (q < q2);
}

typedef int (*binop)(int, int);
static int add(int a, int b) { return a + b; }
static int mul(int a, int b) { return a * b; }
binop pick(int i) { return i ? add : &mul; }
int callfp(binop f, int (*g[2])(int, int)) { return f(1, 2) + (*f)(3, 4) + g[1](5, 6) + pick(0)(7, 8); }

long ptrs(int *p, int *q, char *c, int (*a)[4], void *v)
{
	long r = q - p;
	r += *(p + 3) + p[-1] + 2[p] + *c++ + *--c;
	r += (*a)[2] + a[1][3] + sizeof *a + sizeof a[0][0];
	p += 2; p -= 1; p++; ++p; p--;
	r += (char *)v - c;
	r += p > q;
	r += (long)p & 0xff;
	int **pp = &p;
	**pp = 5;
	return r + !p + (p == 0) + (0 == q);
}

_Noreturn void die(void);
__attribute__((noreturn)) void die2(int);
[[noreturn]] void die3(void);
void unr(int x) { if (x) die(); if (x > 1) die2(x); if (x > 2) __builtin_unreachable(); die3(); }
_Noreturn void loop(void) { for (;;); }

int recur(int n) { return n <= 1 ? 1 : n * recur(n - 1); }
void voidret(int *p) { if (!p) return; *p = 1; }
int main(void) { int a[4] = {1, 2, 3, 4}; return sum(3, 1, 2, 3, 4.0, &a[0]) + loops(4, a) + sw(2, 3, 'a'); }
int compound(int x, int *p) { x += 3; x -= 1; x *= 2; x /= 3; x %= 5; x <<= 1; x >>= 1; x &= 7; x |= 8; x ^= 1; *p += x; p[1] *= *p; return x; }
float fcomp(float f, double *d) { f += 1; f *= 2.0; *d /= f; f -= *d; return f++ + ++f + f-- - --f; }
int incdec(int *p, long l, char c, float f, double d, int **pp)
{
	int r = (*p)++ + ++*p + l++ + --l + c++ + c-- + (int)(f++) + (int)(--d);
	(*pp)++;
	++*pp;
	return r;
}
