/* cimo: GNU builtins that are library functions in cimo (declared by the driver's prelude):
   bit counting, byte swaps, floating-point classification. Portable C; arguments of 0 to clz
   and ctz are undefined, as in gcc. */
#include <stdint.h>

int __builtin_popcountll(unsigned long long x)
{
	x = x - ((x >> 1) & 0x5555555555555555ULL);
	x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
	x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
	return (x * 0x0101010101010101ULL) >> 56;
}

int __builtin_popcountl(unsigned long x) { return __builtin_popcountll(x); }
int __builtin_popcount(unsigned x) { return __builtin_popcountll(x); }

int __builtin_clzll(unsigned long long x)
{
	int n = 0;
	if (!(x >> 32)) n += 32, x <<= 32;
	if (!(x >> 48)) n += 16, x <<= 16;
	if (!(x >> 56)) n += 8, x <<= 8;
	if (!(x >> 60)) n += 4, x <<= 4;
	if (!(x >> 62)) n += 2, x <<= 2;
	if (!(x >> 63)) n += 1;
	return n;
}

int __builtin_clzl(unsigned long x) { return __builtin_clzll(x); }
int __builtin_clz(unsigned x) { return __builtin_clzll(x) - 32; }

int __builtin_ctzll(unsigned long long x) { return __builtin_popcountll((x & -x) - 1); }
int __builtin_ctzl(unsigned long x) { return __builtin_ctzll(x); }
int __builtin_ctz(unsigned x) { return __builtin_ctzll(x); }

int __builtin_ffsll(long long x) { return x ? __builtin_ctzll(x) + 1 : 0; }
int __builtin_ffsl(long x) { return __builtin_ffsll(x); }
int __builtin_ffs(int x) { return __builtin_ffsll((unsigned)x); }

int __builtin_parityll(unsigned long long x) { return __builtin_popcountll(x) & 1; }
int __builtin_parityl(unsigned long x) { return __builtin_popcountll(x) & 1; }
int __builtin_parity(unsigned x) { return __builtin_popcountll(x) & 1; }

int __builtin_clrsbll(long long x)
{
	return __builtin_clzll((unsigned long long)(x ^ (x >> 63)) << 1 | 1);
}
int __builtin_clrsbl(long x) { return __builtin_clrsbll(x); }
int __builtin_clrsb(int x) { return __builtin_clrsbll(x) - 32; }

unsigned short __builtin_bswap16(unsigned short x) { return x << 8 | x >> 8; }

unsigned __builtin_bswap32(unsigned x)
{
	return x >> 24 | (x >> 8 & 0xff00) | (x << 8 & 0xff0000) | x << 24;
}

unsigned long __builtin_bswap64(unsigned long x)
{
	return (unsigned long)__builtin_bswap32(x) << 32 | __builtin_bswap32(x >> 32);
}

static uint64_t bits(double x)
{
	union { double f; uint64_t i; } u = { x };
	return u.i;
}

int __builtin_isnan(double x) { return (bits(x) << 1) > 0xffe0000000000000ULL; }
int __builtin_isinf(double x) { return (bits(x) << 1) == 0xffe0000000000000ULL; }
int __builtin_isfinite(double x) { return (bits(x) << 1) < 0xffe0000000000000ULL; }
int __builtin_signbit(double x) { return bits(x) >> 63; }

static uint32_t fbits(float x)
{
	union { float f; uint32_t i; } u = { x };
	return u.i;
}

int __builtin_isinff(float x) { return (fbits(x) << 1) == 0xff000000u; }

/* __builtin_issignaling (the prelude dispatches on the type: a float converted to double
   would be quieted). A signaling NaN has the top mantissa bit clear. */
int __cimo_issignalingf(float x)
{
	return (fbits(x) & 0x7fffffffu) > 0x7f800000u && !(fbits(x) & 0x00400000u);
}

int __cimo_issignaling(double x)
{
	return (bits(x) & -1ULL >> 1) > 0x7ffULL << 52 && !(bits(x) & 1ULL << 51);
}

/* __builtin_isunordered, isgreater, isgreaterequal, isless, islessequal, islessgreater
   (op 0 to 5): quiet comparisons, false when either is a NaN. float arguments arrive as
   double, exactly. */
int __cimo_fcmp(int op, double x, double y)
{
	if (x != x || y != y)
		return op == 0;
	switch (op) {
	case 1: return x > y;
	case 2: return x >= y;
	case 3: return x < y;
	case 4: return x <= y;
	case 5: return x != y;
	}
	return 0;
}

/* __builtin_{add,sub,mul}_overflow (op 0, 1, 2) for any integer types: a and b are the
   operands' bits, as and bs whether their types are signed; the exact result is computed
   (sign and magnitude, the magnitude up to 128 bits), stored truncated to the result type
   (bits wide, a _BitInt's narrower than its storage; rsigned) at r unless r is null, and 1
   returned if it does not fit. */
int __cimo_overflow(int op, uint64_t a, int as, uint64_t b, int bs, void *r,
                    unsigned long bits, int rsigned)
{
	int an = as && (int64_t)a < 0, bn = bs && (int64_t)b < 0, neg;
	uint64_t am = an ? -a : a, bm = bn ? -b : b, hi = 0, lo, max, v;

	if (op == 2) {
		uint64_t a0 = am & 0xffffffff, a1 = am >> 32, b0 = bm & 0xffffffff, b1 = bm >> 32;
		uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
		uint64_t mid = (p00 >> 32) + (p01 & 0xffffffff) + (p10 & 0xffffffff);
		lo = mid << 32 | (p00 & 0xffffffff);
		hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
		neg = an != bn;
	} else {
		if (op == 1)
			bn = !bn;
		if (an == bn) {
			lo = am + bm;
			hi = lo < am;
			neg = an;
		} else if (am >= bm) {
			lo = am - bm;
			neg = an;
		} else {
			lo = bm - am;
			neg = bn;
		}
	}
	if (!hi && !lo)
		neg = 0;
	if (rsigned)
		max = neg ? 1ULL << (bits - 1) : (1ULL << (bits - 1)) - 1;
	else
		max = neg ? 0 : bits == 64 ? -1ULL : (1ULL << bits) - 1;
	v = neg ? -lo : lo;
	if (bits < 64) {  /* the truncated value, sign- or zero-extended from its width */
		v &= (1ULL << bits) - 1;
		if (rsigned && v >> (bits - 1))
			v |= -1ULL << bits;
	}
	if (r) {
		if (bits <= 8) *(uint8_t *)r = v;
		else if (bits <= 16) *(uint16_t *)r = v;
		else if (bits <= 32) *(uint32_t *)r = v;
		else *(uint64_t *)r = v;
	}
	return hi || lo > max;
}

/* signaling NaNs (the payload string is ignored) */
double __builtin_nans(const char *s)
{
	union { uint64_t i; double f; } u = { 0x7ff4000000000000ULL };
	return u.f;
}

float __builtin_nansf(const char *s)
{
	union { uint32_t i; float f; } u = { 0x7fa00000u };
	return u.f;
}
