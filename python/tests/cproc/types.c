/* integer/float types, conversions, constants, constant folding */
#include <stdint.h>
#include <stddef.h>



typedef unsigned char u8;
typedef signed char s8;
_Bool b0 = 2, b1 = 0.5, b2 = (_Bool)256;
char c0 = 'a', c1 = '\xff', c2 = '\377', c3 = -1;
s8 sc = -128;
u8 uc = 255;
short sh = -32768;
unsigned short ush = 65535u;
int i0 = -2147483647 - 1, i1 = 2147483647, i2 = 0x7fffffff, i3 = 017, i4 = 0b101;
unsigned u0 = 4294967295u, u1 = -1;
long l0 = -9223372036854775807L - 1, l1 = 0x123456789abcdefL;
unsigned long long ull = 18446744073709551615ULL, ull2 = 0xffffffffffffffff;
long long ll = -9223372036854775807LL - 1;
float f0 = 1.5f, f1 = 1e-45f, f2 = 3.4028235e38f, f3 = 0x1.8p3f, f4 = 1e39f, f5 = .5e1f;
double d0 = 0.1, d1 = 1e308, d2 = 4.9e-324, d3 = 0x1p-1074, d4 = 1e400, d5 = -0.0, d6 = 123456789012345678901234567890.0;
double d7 = 0x.8p1, d8 = 1.e3, d9 = 0x1p+2, d10 = 1.7976931348623157e308;
float fi = __builtin_inff();
float fn = __builtin_nanf("");
double dinf = 1.0 / 0.0, dnan = 0.0 / 0.0, dninf = -1.0 / 0.0;
int folded[] = {
	1 + 2 * 3, 7 / 2, -7 / 2, 7 % -3, -7 % 3, 1 << 31, (int)(1u << 31) >> 31, -1 >> 1,
	(unsigned)-1 >> 1, 3 < 4, -1 < 0u, 5 == 5.0, 2.5 > 2, !0, ~0, 1 ? 2 : 3, 0 ? 2 : 3,
	(char)300, (signed char)200, (short)70000, (unsigned char)-1, 1 && 0, 0 || 2,
	sizeof(long double), _Alignof(double), (int)3.99, (int)-3.99, (unsigned)3e9, 10 / 0 == 0,
};
unsigned long long folded2[] = {
	(unsigned long long)-1 / 3, 1ull << 63, (long long)0x8000000000000000 >> 63,
	(unsigned long long)1e19, (long long)-9.2e18, (unsigned long long)(float)16777217,
	-9223372036854775807LL / -1, 18446744073709551615ULL % 7, 5 % 0,
};
double dfold[] = {1.0 / 3, 1.0f / 3, (float)0.1 + 0.2, 1e308 * 10, -1e308 * 10, 3 / 2.0, (double)(1ull << 63),
	(float)1e40, (double)-1, (double)18446744073709551615ULL, 0.1f * 3};
int sizes[] = {sizeof(char), sizeof(short), sizeof(int), sizeof(long), sizeof(long long), sizeof(void *),
	sizeof(float), sizeof(double), sizeof(_Bool), sizeof(int[10]), sizeof(struct { char c; double d; }),
	sizeof "abc", sizeof L"abc", sizeof u"abc", sizeof U"abc", sizeof u8"abc", sizeof(size_t), sizeof(ptrdiff_t)};
_BitInt(7) bi = 63;
unsigned _BitInt(33) ubi = 0x1ffffffffuwb;
int wbc = sizeof(1wb) + sizeof(-1wb) + sizeof(0xffffffffffffffffuwb);
int chars[] = {'\a', '\b', '\f', '\n', '\r', '\t', '\v', '\\', '\'', '\"', '\?', '\0', '\12', '\x7f', L'\x7fffffff', u'\xffff', U'\xffff', u8'a', L'é', u'é', U'😀'};
int nul = '\0';

long conv(long x, int i, unsigned u, short s, unsigned char c, float f, double d, _Bool b)
{
	long r = 0;
	r += i + u;
	r += s * c;
	r += (long)f + (long)d;
	r += b + (int)f;
	f = i; d = u; f = d; d = f; f = s; d = c; f = x; d = (unsigned long)x;
	i = f; u = d; c = f; s = d; b = f; b = d; b = x; b = i; b = s; b = c;
	r += (unsigned long)d + (unsigned)f;
	r += x / i + x % u + (unsigned long)x / 3 + (unsigned)i % 5u + (x >> 3) + ((unsigned long)x >> 2);
	r += i < u ? 1 : 2;
	r += f < d;
	r += (s == c) + (f != d) + (x <= i) + (u >= c);
	return r + f * d;
}

_BitInt(13) bitint(_BitInt(13) a, unsigned _BitInt(13) b, _BitInt(40) c)
{
	_BitInt(13) x = a * b + c;
	unsigned _BitInt(3) y = 7;
	y++;
	x >>= 2;
	return x + y + (_BitInt(13))c;
}

uint64_t fold_ptr(void) { return (uint64_t)(uintptr_t)&((struct { int a, b[4]; } *)0)->b[2]; }
int cast_ptr(void) { static int x; return (long)&x != 0; }
