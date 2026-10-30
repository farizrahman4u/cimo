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
