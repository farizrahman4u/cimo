/* cimo wasm runtime: fma and fmaf, exact (rounded once), in integer arithmetic. The native
   runtime's fma.c is one instruction; wasm has no scalar fused multiply-add.

   x*y is formed exactly as a 106-bit integer, z is aligned to it in a 128-bit window (bits
   shifted out are kept as a sticky bit), they are added, and the sum is rounded to nearest
   even once at the target precision, subnormals included. Infinities, NaNs and zero operands
   take the plain x*y + z path, which is exact for them. */

typedef unsigned long u64;

struct u128 {
	u64 hi, lo;
};

union dbl {
	double f;
	u64 i;
};

union flt {
	float f;
	unsigned i;
};

static int
clz64(u64 x)
{
	int n = 0;
	if (!x)
		return 64;
	while (!(x >> 63)) {
		x <<= 1;
		n++;
	}
	return n;
}

static int
clz128(struct u128 a)
{
	return a.hi ? clz64(a.hi) : 64 + clz64(a.lo);
}

static struct u128
shl(struct u128 a, int s)
{
	struct u128 r;
	if (s == 0)
		return a;
	if (s >= 128) {
		r.hi = r.lo = 0;
	} else if (s >= 64) {
		r.hi = a.lo << (s - 64);
		r.lo = 0;
	} else {
		r.hi = a.hi << s | a.lo >> (64 - s);
		r.lo = a.lo << s;
	}
	return r;
}

/* right shift with the bits shifted out ORed into bit 0 */
static struct u128
shr_jam(struct u128 a, int s)
{
	struct u128 r;
	u64 lost;
	if (s == 0)
		return a;
	if (s >= 128) {
		r.hi = 0;
		r.lo = (a.hi | a.lo) != 0;
		return r;
	}
	if (s >= 64) {
		lost = a.lo | (s > 64 ? a.hi << (128 - s) : 0);
		r.lo = s == 64 ? a.hi : a.hi >> (s - 64);
		r.hi = 0;
	} else {
		lost = a.lo << (64 - s);
		r.lo = a.lo >> s | a.hi << (64 - s);
		r.hi = a.hi >> s;
	}
	r.lo |= lost != 0;
	return r;
}

static struct u128
mul64(u64 a, u64 b)
{
	u64 a0 = a & 0xffffffff, a1 = a >> 32, b0 = b & 0xffffffff, b1 = b >> 32;
	u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
	u64 mid = (p00 >> 32) + (p01 & 0xffffffff) + (p10 & 0xffffffff);
	struct u128 r;
	r.lo = mid << 32 | (p00 & 0xffffffff);
	r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
	return r;
}

static int
lt(struct u128 a, struct u128 b)
{
	return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

static struct u128
add(struct u128 a, struct u128 b)
{
	struct u128 r;
	r.lo = a.lo + b.lo;
	r.hi = a.hi + b.hi + (r.lo < a.lo);
	return r;
}

static struct u128
sub(struct u128 a, struct u128 b)
{
	struct u128 r;
	r.lo = a.lo - b.lo;
	r.hi = a.hi - b.hi - (a.lo < b.lo);
	return r;
}

/* finite nonzero double: value = m * 2^e */
static void
decode(double d, u64 *m, int *e, int *s)
{
	union dbl u;
	int be;
	u.f = d;
	*s = u.i >> 63;
	be = u.i >> 52 & 0x7ff;
	*m = u.i & ((1UL << 52) - 1);
	if (be) {
		*m |= 1UL << 52;
		*e = be - 1075;
	} else {
		*e = -1074;
	}
}

/* x*y + z for finite nonzero x, y, z, rounded to p bits with exponent range [emin, emax].
   Returns 0 and sets *sign, *keep (the significand, < 2^p, or 2^p after a carry) and *exp
   (the unbiased exponent of its leading bit position p-1); 1 for an exact zero; 2 for
   overflow. */
static int
core(double x, double y, double z, int p, int emin, int emax, int *sign, u64 *keep, int *exp)
{
	u64 mx, my, mz;
	int ex, ey, ez, sx, sy, sz, sp, ep, eb, sb, lz, e, shift, prec;
	struct u128 P, Z, big, small, R;
	u64 rem_hi_bit, kept;
	int roundbit, sticky;

	decode(x, &mx, &ex, &sx);
	decode(y, &my, &ey, &sy);
	decode(z, &mz, &ez, &sz);
	P = mul64(mx, my);
	sp = sx ^ sy;
	lz = clz128(P);
	P = shl(P, lz - 2);
	ep = ex + ey - (lz - 2);
	Z.hi = 0;
	Z.lo = mz;
	lz = clz128(Z);
	Z = shl(Z, lz - 2);
	ez -= lz - 2;
	if (ep >= ez) {
		big = P;
		sb = sp;
		eb = ep;
		small = shr_jam(Z, ep - ez > 200 ? 200 : ep - ez);
		if (sp == sz) {
			R = add(big, small);
		} else if (!lt(big, small)) {
			R = sub(big, small);
		} else {
			R = sub(small, big);
			sb = sz;
		}
	} else {
		big = Z;
		sb = sz;
		eb = ez;
		small = shr_jam(P, ez - ep > 200 ? 200 : ez - ep);
		if (sp == sz) {
			R = add(big, small);
		} else if (!lt(big, small)) {
			R = sub(big, small);
		} else {
			R = sub(small, big);
			sb = sp;
		}
	}
	if (!R.hi && !R.lo)
		return 1;
	lz = clz128(R);
	R = shl(R, lz);
	e = eb - lz + 127; /* R's leading bit (bit 127) has weight 2^e */
	prec = p;
	if (e < emin)
		prec = p - (emin - e);
	*sign = sb;
	if (prec <= 0) {
		/* below half the smallest subnormal unless prec == 0 and above the halfway point */
		kept = 0;
		if (prec == 0) {
			roundbit = 1;
			sticky = (R.hi << 1) != 0 || R.lo != 0;
			kept = roundbit && sticky;
		}
		*keep = kept;
		*exp = emin;
		return 0;
	}
	shift = 128 - prec;
	if (shift >= 64) {
		kept = R.hi >> (shift - 64);
		rem_hi_bit = shift - 64 ? (R.hi >> (shift - 65)) & 1 : R.lo >> 63;
		roundbit = rem_hi_bit;
		if (shift - 64 >= 1)
			sticky = (R.hi & ((1UL << (shift - 65)) - 1)) != 0 || R.lo != 0;
		else
			sticky = (R.lo << 1) != 0;
	} else {
		kept = R.hi << (64 - shift) | R.lo >> shift;
		roundbit = R.lo >> (shift - 1) & 1;
		sticky = shift > 1 && (R.lo & ((1UL << (shift - 1)) - 1)) != 0;
	}
	if (roundbit && (sticky || (kept & 1)))
		kept++;
	if (prec == p && kept >> p) { /* carry into a new leading bit */
		kept >>= 1;
		e++;
	}
	if (e > emax)
		return 2;
	*keep = kept;
	*exp = e < emin ? emin : e;
	return 0;
}

double
fma(double x, double y, double z)
{
	union dbl u;
	int sign, exp, r;
	u64 keep;

	u.f = x;
	if ((u.i >> 52 & 0x7ff) == 0x7ff || x == 0)
		return x * y + z;
	u.f = y;
	if ((u.i >> 52 & 0x7ff) == 0x7ff || y == 0)
		return x * y + z;
	u.f = z;
	if ((u.i >> 52 & 0x7ff) == 0x7ff)
		return z + z;
	if (z == 0)
		return x * y + z;
	r = core(x, y, z, 53, -1022, 1023, &sign, &keep, &exp);
	if (r == 1)
		return 0.0;
	if (r == 2)
		u.i = 0x7ffUL << 52;
	else if (keep >> 52) /* normal (a subnormal that rounded up to the smallest normal too) */
		u.i = (u64)(exp + 1023) << 52 | (keep & ((1UL << 52) - 1));
	else
		u.i = keep;
	u.i |= (u64)sign << 63;
	return u.f;
}

float
fmaf(float x, float y, float z)
{
	union flt u;
	int sign, exp, r;
	u64 keep;

	if (x != x || y != y || z != z || x == 0 || y == 0 || x - x != 0 || y - y != 0 ||
	    z - z != 0)
		return (float)((double)x * y + z); /* NaN, infinity or a zero factor: exact */
	if (z == 0)
		return x * y + z;
	r = core(x, y, z, 24, -126, 127, &sign, &keep, &exp);
	if (r == 1)
		return 0.0f;
	if (r == 2)
		u.i = 0xffU << 23;
	else if (keep >> 23)
		u.i = (unsigned)(exp + 127) << 23 | (unsigned)(keep & ((1U << 23) - 1));
	else
		u.i = (unsigned)keep;
	u.i |= (unsigned)sign << 31;
	return u.f;
}
