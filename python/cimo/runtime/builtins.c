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

/* cimo: __int128 operations the frontend lowers to calls (it lowers the rest to 64-bit halves): division,
   shifts by a variable amount, and conversions from and to double and float. A value is its low and high
   64 bits; results go to r (r[0] low, r[1] high). Plain C on the halves. */
static void u128divmod(uint64_t nl, uint64_t nh, uint64_t dl, uint64_t dh, uint64_t *q, uint64_t *rm)
{
	uint64_t ql = 0, qh = 0, rl = 0, rh = 0;
	int i;
	if (!dh && !nh) {  /* 64-bit operands */
		q[0] = dl ? nl / dl : 0; q[1] = 0;
		rm[0] = dl ? nl % dl : nl; rm[1] = 0;
		return;
	}
	for (i = 127; i >= 0; i--) {  /* restoring division, a bit at a time */
		uint64_t bit = i >= 64 ? (nh >> (i - 64)) & 1 : (nl >> i) & 1;
		rh = rh << 1 | rl >> 63;
		rl = rl << 1 | bit;
		if (rh > dh || (rh == dh && rl >= dl)) {
			uint64_t b = rl < dl;
			rl -= dl;
			rh -= dh + b;
			if (i >= 64) qh |= 1ULL << (i - 64); else ql |= 1ULL << i;
		}
	}
	q[0] = ql; q[1] = qh; rm[0] = rl; rm[1] = rh;
}

static void neg128(uint64_t *v)
{
	v[0] = -v[0];
	v[1] = ~v[1] + !v[0];
}

/* kind: bit 0 remainder (else quotient), bit 1 signed */
void __cimo_i128div(int kind, uint64_t al, uint64_t ah, uint64_t bl, uint64_t bh, uint64_t *r)
{
	uint64_t a[2] = { al, ah }, b[2] = { bl, bh }, q[2], m[2];
	int an = 0, bn = 0;
	if (kind & 2) {
		if (ah >> 63) { neg128(a); an = 1; }
		if (bh >> 63) { neg128(b); bn = 1; }
	}
	u128divmod(a[0], a[1], b[0], b[1], q, m);
	if (kind & 1) {
		if (an) neg128(m);
		r[0] = m[0]; r[1] = m[1];
	} else {
		if (an != bn) neg128(q);
		r[0] = q[0]; r[1] = q[1];
	}
}

/* kind: 0 shift left, 1 logical right, 2 arithmetic right; n from 0 to 127 */
void __cimo_i128shift(int kind, uint64_t lo, uint64_t hi, int n, uint64_t *r)
{
	n &= 127;
	if (!n) { r[0] = lo; r[1] = hi; return; }
	if (kind == 0) {
		if (n >= 64) { r[1] = lo << (n - 64); r[0] = 0; }
		else { r[1] = hi << n | lo >> (64 - n); r[0] = lo << n; }
		return;
	}
	uint64_t fill = kind == 2 && hi >> 63 ? ~0ULL : 0;
	if (n >= 64) {
		r[0] = n == 64 ? hi : (kind == 2 ? (uint64_t)((int64_t)hi >> (n - 64)) : hi >> (n - 64));
		r[1] = fill;
	} else {
		r[0] = lo >> n | hi << (64 - n);
		r[1] = kind == 2 ? (uint64_t)((int64_t)hi >> n) : hi >> n;
	}
}

/* |v| rounded once to the target precision: the bits below 64 become one sticky bit, so that converting the
   top 64 bits rounds as the whole value does; then scaled by the dropped bits' count */
static double u128tod(uint64_t lo, uint64_t hi, int isfloat)
{
	int s = 0;
	if (hi) {
		int lz = __builtin_clzll(hi);
		s = 64 - lz;  /* bits to drop so that the value fits in 64 */
		uint64_t top = hi << lz | (lz ? lo >> (64 - lz) : 0);
		uint64_t dropped = lz ? lo << lz : lo;
		lo = top | (dropped != 0);
	}
	if (isfloat) {
		float x = (float)lo;
		return (double)x * (double)(1ULL << (s > 62 ? 62 : s)) * (s > 62 ? (double)(1ULL << (s - 62)) : 1.0);
	}
	double x = (double)lo;
	return x * (double)(1ULL << (s > 62 ? 62 : s)) * (s > 62 ? (double)(1ULL << (s - 62)) : 1.0);
}

void __cimo_i128tod(int sign, uint64_t lo, uint64_t hi, double *r)
{
	uint64_t v[2] = { lo, hi };
	int neg = sign && hi >> 63;
	if (neg) neg128(v);
	double x = u128tod(v[0], v[1], 0);
	*r = neg ? -x : x;
}

void __cimo_i128tof(int sign, uint64_t lo, uint64_t hi, float *r)
{
	uint64_t v[2] = { lo, hi };
	int neg = sign && hi >> 63;
	if (neg) neg128(v);
	float x = (float)u128tod(v[0], v[1], 1);
	*r = neg ? -x : x;
}

/* truncation toward zero; out of range is undefined in C (here: wraps) */
void __cimo_i128fromd(int sign, double d, uint64_t *r)
{
	int neg = d < 0;
	double a = neg ? -d : d;
	uint64_t v[2];
	if (a < 18446744073709551616.0) {
		v[0] = (uint64_t)a; v[1] = 0;
	} else {
		/* a = m * 2^e with m a 53-bit integer, e >= 11 */
		union { double f; uint64_t i; } u = { a };
		int e = (int)((u.i >> 52) & 0x7ff) - 1075;
		uint64_t m = (u.i & ((1ULL << 52) - 1)) | (1ULL << 52);
		if (e >= 128 - 53) e = 128 - 53;  /* out of range */
		if (e >= 64) { v[1] = m << (e - 64); v[0] = 0; }
		else { v[1] = m >> (64 - e); v[0] = m << e; }
	}
	if (neg) neg128(v);
	r[0] = v[0]; r[1] = v[1];
}

/* __cimo_overflow with an __int128 operand or result: the magnitude exact in four 64-bit limbs */
int __cimo_overflow128(int op, unsigned __int128 a, int as, unsigned __int128 b, int bs, void *r,
                       unsigned long bits, int rsigned)
{
	int an = as && (__int128)a < 0, bn = bs && (__int128)b < 0, neg;
	unsigned __int128 am = an ? -a : a, bm = bn ? -b : b, lo, hi = 0, max, v;
	if (op == 2) {
		uint64_t x[2] = { (uint64_t)am, (uint64_t)(am >> 64) }, y[2] = { (uint64_t)bm, (uint64_t)(bm >> 64) };
		uint64_t p[4] = { 0, 0, 0, 0 };
		for (int i = 0; i < 2; i++) {
			uint64_t carry = 0;
			for (int j = 0; j < 2; j++) {
				unsigned __int128 t = (unsigned __int128)x[i] * y[j] + p[i + j] + carry;
				p[i + j] = (uint64_t)t;
				carry = (uint64_t)(t >> 64);
			}
			p[i + 2] = carry;
		}
		lo = (unsigned __int128)p[1] << 64 | p[0];
		hi = (unsigned __int128)p[3] << 64 | p[2];
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
		max = neg ? (unsigned __int128)1 << (bits - 1) : ((unsigned __int128)1 << (bits - 1)) - 1;
	else
		max = neg ? 0 : bits == 128 ? ~(unsigned __int128)0 : ((unsigned __int128)1 << bits) - 1;
	v = neg ? -lo : lo;
	if (bits < 128) {
		v &= ((unsigned __int128)1 << bits) - 1;
		if (rsigned && (v >> (bits - 1)) & 1)
			v |= ~(unsigned __int128)0 << bits;
	}
	if (r) {
		if (bits <= 8) *(uint8_t *)r = (uint8_t)v;
		else if (bits <= 16) *(uint16_t *)r = (uint16_t)v;
		else if (bits <= 32) *(uint32_t *)r = (uint32_t)v;
		else if (bits <= 64) *(uint64_t *)r = (uint64_t)v;
		else *(unsigned __int128 *)r = v;
	}
	return hi || lo > max;
}

/* cimo: _Float16 and __bf16 from and to float (the frontend computes them in float): round to nearest,
   ties to even, with subnormals, infinities and NaNs. */
float __cimo_h2f(unsigned h)
{
	unsigned s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
	union { uint32_t i; float f; } u;
	if (e == 31)
		u.i = s | 0x7f800000 | m << 13;
	else if (e)
		u.i = s | (e + 112) << 23 | m << 13;
	else if (m) {  /* subnormal: normalize */
		e = 113;
		while (!(m & 0x400)) { m <<= 1; e--; }
		u.i = s | e << 23 | (m & 0x3ff) << 13;
	} else
		u.i = s;
	return u.f;
}

unsigned __cimo_f2h(float x)
{
	union { float f; uint32_t i; } u = { x };
	uint32_t s = (u.i >> 16) & 0x8000, a = u.i & 0x7fffffff;
	if (a >= 0x7f800000)  /* infinity, NaN (kept quiet) */
		return s | 0x7c00 | (a > 0x7f800000 ? 0x200 | (a >> 13 & 0x3ff) : 0);
	if (a >= 0x477ff000)  /* rounds to 65520 or more: infinity */
		return s | 0x7c00;
	if (a >= 0x38800000) {  /* normal half */
		uint32_t r = a - (112 << 23);
		r += 0xfff + ((r >> 13) & 1);
		return s | r >> 13;
	}
	if (a < 0x33000000)  /* below half the smallest subnormal: zero */
		return s;
	/* subnormal: shift the significand (with its hidden bit) into place, rounding to even */
	uint32_t e = a >> 23, m = (a & 0x7fffff) | 0x800000;
	int sh = 126 - e;  /* 14 to 24 */
	uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
	if (rem > half || (rem == half && (r & 1)))
		r++;
	return s | r;
}

float __cimo_bf2f(unsigned h)
{
	union { uint32_t i; float f; } u = { (uint32_t)(h & 0xffff) << 16 };
	return u.f;
}

unsigned __cimo_f2bf(float x)
{
	union { float f; uint32_t i; } u = { x };
	if ((u.i & 0x7fffffff) > 0x7f800000)
		return (u.i >> 16) | 0x40;
	return (u.i + 0x7fff + ((u.i >> 16) & 1)) >> 16;
}

int __cimo_issignalingh(_Float16 x)
{
	union { _Float16 h; unsigned short u; } v = { x };
	unsigned short u = v.u;
	return (u & 0x7c00) == 0x7c00 && (u & 0x3ff) && !(u & 0x200);
}

int __cimo_issignalingbf(__bf16 x)
{
	union { __bf16 h; unsigned short u; } v = { x };
	unsigned short u = v.u;
	return (u & 0x7f80) == 0x7f80 && (u & 0x7f) && !(u & 0x40);
}
