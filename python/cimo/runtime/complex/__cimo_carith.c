/* cimo: complex multiplication's and division's special cases (C11 Annex G, G.5.1), called by code the
   frontend generates for float and double _Complex. Multiplication is computed inline and calls
   __cimo_cmul only when both parts come out NaN (an infinity may be hiding there); division always
   calls __cimo_cdiv. Written for cimo (the special cases follow the standard's example code). Each writes the real and imaginary parts to r[0] and r[1]. */
#include <math.h>

#define CMUL(NAME, T, SUF)                                                              \
void NAME(T a, T b, T c, T d, T *r)                                                     \
{                                                                                       \
	T ac = a * c, bd = b * d, ad = a * d, bc = b * c;                               \
	T x = ac - bd, y = ad + bc;                                                     \
	if (isnan(x) && isnan(y)) {                                                     \
		int recalc = 0;                                                         \
		if (isinf(a) || isinf(b)) {                                             \
			a = copysign##SUF(isinf(a) ? 1 : 0, a);                         \
			b = copysign##SUF(isinf(b) ? 1 : 0, b);                         \
			if (isnan(c)) c = copysign##SUF(0, c);                          \
			if (isnan(d)) d = copysign##SUF(0, d);                          \
			recalc = 1;                                                     \
		}                                                                       \
		if (isinf(c) || isinf(d)) {                                             \
			c = copysign##SUF(isinf(c) ? 1 : 0, c);                         \
			d = copysign##SUF(isinf(d) ? 1 : 0, d);                         \
			if (isnan(a)) a = copysign##SUF(0, a);                          \
			if (isnan(b)) b = copysign##SUF(0, b);                          \
			recalc = 1;                                                     \
		}                                                                       \
		if (!recalc && (isinf(ac) || isinf(bd) || isinf(ad) || isinf(bc))) {    \
			if (isnan(a)) a = copysign##SUF(0, a);                          \
			if (isnan(b)) b = copysign##SUF(0, b);                          \
			if (isnan(c)) c = copysign##SUF(0, c);                          \
			if (isnan(d)) d = copysign##SUF(0, d);                          \
			recalc = 1;                                                     \
		}                                                                       \
		if (recalc) {                                                           \
			x = (T)INFINITY * (a * c - b * d);                              \
			y = (T)INFINITY * (a * d + b * c);                              \
		}                                                                       \
	}                                                                               \
	r[0] = x;                                                                       \
	r[1] = y;                                                                       \
}

/* Finite operands, a nonzero divisor: (ac + bd) / (cc + dd) and (bc - ad) / (cc + dd) with every product
   carried as a mantissa and an exponent (frexp), so that nothing overflows or underflows on the way, and
   each step rounds once; ldexp makes the result. gcc's checks of extreme quotients (subnormal parts,
   exponents 1500 apart: ieee/cdivchkd.c) need it; scaling the divisor first (G.5.1) flushes a tiny part
   whose product with a huge numerator matters. */
#define XDIV(NAME, T, SUF)                                                              \
static void NAME(T a, T b, T c, T d, T *r)                                              \
{                                                                                       \
	int ea, eb, ec, ed;                                                             \
	T ma = frexp##SUF(a, &ea), mb = frexp##SUF(b, &eb);                             \
	T mc = frexp##SUF(c, &ec), md = frexp##SUF(d, &ed);                             \
	T p[2], q[2];                                                                   \
	int pe[2], qe[2];                                                               \
	/* den = cc + dd, xn = ac + bd, yn = bc - ad: each the sum of two products */  \
	T dm, xm, ym;                                                                   \
	int de, xe, ye, e;                                                              \
	p[0] = mc * mc; pe[0] = 2 * ec; p[1] = md * md; pe[1] = 2 * ed;                 \
	if (!c) p[0] = 0; if (!d) p[1] = 0;                                             \
	SUM(dm, de, p, pe, T, SUF);                                                     \
	p[0] = ma * mc; pe[0] = ea + ec; p[1] = mb * md; pe[1] = eb + ed;               \
	if (!a || !c) p[0] = 0; if (!b || !d) p[1] = 0;                                 \
	SUM(xm, xe, p, pe, T, SUF);                                                     \
	q[0] = mb * mc; qe[0] = eb + ec; q[1] = -(ma * md); qe[1] = ea + ed;            \
	if (!b || !c) q[0] = 0; if (!a || !d) q[1] = 0;                                 \
	SUM(ym, ye, q, qe, T, SUF);                                                     \
	e = xe - de;                                                                    \
	r[0] = xm ? ldexp##SUF(xm / dm, e) : copysign##SUF(0, xm / dm);                \
	e = ye - de;                                                                    \
	r[1] = ym ? ldexp##SUF(ym / dm, e) : copysign##SUF(0, ym / dm);                \
}

/* m * 2^e = v[0] * 2^ve[0] + v[1] * 2^ve[1] (a zero v[i] has no exponent) */
#define SUM(m, e, v, ve, T, SUF)                                                        \
	do {                                                                            \
		if (!v[0]) { m = v[1]; e = ve[1]; }                                     \
		else if (!v[1]) { m = v[0]; e = ve[0]; }                                \
		else {                                                                  \
			e = ve[0] > ve[1] ? ve[0] : ve[1];                              \
			m = ldexp##SUF(v[0], ve[0] - e) + ldexp##SUF(v[1], ve[1] - e);  \
		}                                                                       \
	} while (0)

#define CDIV(NAME, XNAME, T, SUF)                                                       \
void NAME(T a, T b, T c, T d, T *r)                                                     \
{                                                                                       \
	int ilogbw = 0;                                                                 \
	T logbw, denom, x, y;                                                           \
	if (isfinite(a) && isfinite(b) && isfinite(c) && isfinite(d) && (c || d)) {     \
		XNAME(a, b, c, d, r);                                                   \
		return;                                                                 \
	}                                                                               \
	logbw = logb##SUF(fmax##SUF(fabs##SUF(c), fabs##SUF(d)));                       \
	if (isfinite(logbw)) {                                                          \
		ilogbw = (int)logbw;                                                    \
		c = scalbn##SUF(c, -ilogbw);                                            \
		d = scalbn##SUF(d, -ilogbw);                                            \
	}                                                                               \
	denom = c * c + d * d;                                                          \
	x = scalbn##SUF((a * c + b * d) / denom, -ilogbw);                              \
	y = scalbn##SUF((b * c - a * d) / denom, -ilogbw);                              \
	if (isnan(x) && isnan(y)) {                                                     \
		if (denom == 0 && (!isnan(a) || !isnan(b))) {                           \
			x = copysign##SUF((T)INFINITY, c) * a;                          \
			y = copysign##SUF((T)INFINITY, c) * b;                          \
		} else if ((isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)) {      \
			a = copysign##SUF(isinf(a) ? 1 : 0, a);                         \
			b = copysign##SUF(isinf(b) ? 1 : 0, b);                         \
			x = (T)INFINITY * (a * c + b * d);                              \
			y = (T)INFINITY * (b * c - a * d);                              \
		} else if (isinf(logbw) && logbw > 0 && isfinite(a) && isfinite(b)) {   \
			c = copysign##SUF(isinf(c) ? 1 : 0, c);                         \
			d = copysign##SUF(isinf(d) ? 1 : 0, d);                         \
			x = 0 * (a * c + b * d);                                        \
			y = 0 * (b * c - a * d);                                        \
		}                                                                       \
	}                                                                               \
	r[0] = x;                                                                       \
	r[1] = y;                                                                       \
}

CMUL(__cimo_cmul, double, )
CMUL(__cimo_cmulf, float, f)
XDIV(xdiv, double, )
XDIV(xdivf, float, f)
CDIV(__cimo_cdiv, xdiv, double, )
CDIV(__cimo_cdivf, xdivf, float, f)
