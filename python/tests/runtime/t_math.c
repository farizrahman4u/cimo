/* math.h: every function on special values and random arguments, results as bit patterns.
   One line per call: "<function> <argument index> <result bits> [<second result>]". */
#include <math.h>
#include "out.h"

#define N 600

static double xs[N], ys[N], zs[N];
static float fx[N], fy[N], fz[N];

static const double special[] = { 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 2.0,
	3.0, 0.49999999999999994, -0.49999999999999994, 4503599627370495.5, 4503599627370497.0,
	1e-310, -1e-310, 2.2250738585072014e-308, 1.7976931348623157e308, -1.7976931348623157e308,
	1e300, -1e300, 710.0, -745.2, 0.1, 3.141592653589793, 1.5707963267948966, 6.283185307179586,
	1e22, 1e-22, 100.0, -100.0, 0.7, 1e-8, 22.0, -22.0, 1.0000000000000002, 0.9999999999999999,
	INFINITY, -INFINITY, NAN };
#define NS (int)(sizeof special / sizeof *special)

static double rnd(int kind)
{
	union { unsigned long long u; double d; } x;
	switch (kind % 4) {
	case 0: /* uniform bits: every exponent */
		x.u = o_rand();
		if ((x.u >> 52 & 0x7ff) == 0x7ff) x.u ^= 1ULL << 62;
		return x.d;
	case 1: return ((double)(o_rand() >> 11) / 9007199254740992.0) * 20 - 10;
	case 2: return ((double)(o_rand() >> 11) / 9007199254740992.0) * 2 - 1;
	default: return ((double)(o_rand() >> 11) / 9007199254740992.0) * 1400 - 700;
	}
}

#define D1(f) for (int i = 0; i < N; i++) { o_s(#f " "); o_d(i); o_c(' '); o_f(f(xs[i])); o_c('\n'); }
#define F1(f) for (int i = 0; i < N; i++) { o_s(#f " "); o_d(i); o_c(' '); o_ff(f(fx[i])); o_c('\n'); }
#define D2(f) for (int i = 0; i < N; i++) { o_s(#f " "); o_d(i); o_c(' '); o_f(f(xs[i], ys[i])); o_c('\n'); }
#define F2(f) for (int i = 0; i < N; i++) { o_s(#f " "); o_d(i); o_c(' '); o_ff(f(fx[i], fy[i])); o_c('\n'); }

long run(char *out, long cap)
{
	o_init(out, cap);
	for (int i = 0; i < N; i++) {
		if (i < NS * NS && i < 300) {
			xs[i] = special[i % NS];
			ys[i] = special[i / NS % NS];
			zs[i] = special[(i * 7) % NS];
		} else {
			xs[i] = rnd(i);
			ys[i] = rnd(i / 4 + 1);
			zs[i] = rnd(i / 16 + 2);
		}
		fx[i] = (float)xs[i];
		fy[i] = (float)ys[i];
		fz[i] = (float)zs[i];
	}
	D1(acos) D1(asin) D1(atan) D1(cos) D1(sin) D1(tan) D1(acosh) D1(asinh) D1(atanh)
	D1(cosh) D1(sinh) D1(tanh) D1(exp) D1(exp2) D1(expm1) D1(log) D1(log10) D1(log1p)
	D1(log2) D1(logb) D1(cbrt) D1(fabs) D1(sqrt) D1(erf) D1(erfc) D1(lgamma) D1(tgamma)
	D1(ceil) D1(floor) D1(nearbyint) D1(rint) D1(round) D1(trunc)
	F1(acosf) F1(asinf) F1(atanf) F1(cosf) F1(sinf) F1(tanf) F1(acoshf) F1(asinhf) F1(atanhf)
	F1(coshf) F1(sinhf) F1(tanhf) F1(expf) F1(exp2f) F1(expm1f) F1(logf) F1(log10f) F1(log1pf)
	F1(log2f) F1(logbf) F1(cbrtf) F1(fabsf) F1(sqrtf) F1(erff) F1(erfcf) F1(lgammaf) F1(tgammaf)
	F1(ceilf) F1(floorf) F1(nearbyintf) F1(rintf) F1(roundf) F1(truncf)
	D2(atan2) D2(hypot) D2(pow) D2(fmod) D2(remainder) D2(copysign) D2(nextafter) D2(fdim)
	D2(fmax) D2(fmin)
	F2(atan2f) F2(hypotf) F2(powf) F2(fmodf) F2(remainderf) F2(copysignf) F2(nextafterf)
	F2(fdimf) F2(fmaxf) F2(fminf)
	for (int i = 0; i < N; i++) {
		int e = 0, q = 0;
		double ip = 0;
		float fip = 0;
		int n = (int)(ys[i] * 100) % 2200;
		o_s("fma "); o_d(i); o_c(' '); o_f(fma(xs[i], ys[i], zs[i])); o_c('\n');
		o_s("fmaf "); o_d(i); o_c(' '); o_ff(fmaf(fx[i], fy[i], fz[i])); o_c('\n');
		o_s("frexp "); o_d(i); o_c(' '); o_f(frexp(xs[i], &e)); o_c(' ');
		o_d(isfinite(xs[i]) ? e : 0); o_c('\n');
		o_s("frexpf "); o_d(i); o_c(' '); o_ff(frexpf(fx[i], &e)); o_c(' ');
		o_d(isfinite(fx[i]) ? e : 0); o_c('\n');
		o_s("modf "); o_d(i); o_c(' '); o_f(modf(xs[i], &ip)); o_c(' '); o_f(ip); o_c('\n');
		o_s("modff "); o_d(i); o_c(' '); o_ff(modff(fx[i], &fip)); o_c(' '); o_ff(fip); o_c('\n');
		o_s("ldexp "); o_d(i); o_c(' '); o_f(ldexp(xs[i], n)); o_c('\n');
		o_s("ldexpf "); o_d(i); o_c(' '); o_ff(ldexpf(fx[i], n)); o_c('\n');
		o_s("scalbn "); o_d(i); o_c(' '); o_f(scalbn(xs[i], -n)); o_c('\n');
		o_s("scalblnf "); o_d(i); o_c(' '); o_ff(scalblnf(fx[i], n)); o_c('\n');
		o_s("remquo "); o_d(i); o_c(' '); o_f(remquo(xs[i], ys[i], &q)); o_c(' ');
		o_d(isfinite(xs[i]) && isfinite(ys[i]) && ys[i] != 0 ? q & 7 : 0); o_c('\n');
		o_s("remquof "); o_d(i); o_c(' '); o_ff(remquof(fx[i], fy[i], &q)); o_c(' ');
		o_d(isfinite(fx[i]) && isfinite(fy[i]) && fy[i] != 0 ? q & 7 : 0); o_c('\n');
		/* integer results: only where defined */
		if (isfinite(xs[i]) && fabs(xs[i]) < 1e18) {
			o_s("lrint "); o_d(i); o_c(' '); o_d(lrint(xs[i])); o_c(' '); o_d(llrint(xs[i]));
			o_c(' '); o_d(lround(xs[i])); o_c(' '); o_d(llround(xs[i])); o_c('\n');
		}
		if (isfinite(fx[i]) && fabsf(fx[i]) < 1e18f) {
			o_s("lrintf "); o_d(i); o_c(' '); o_d(lrintf(fx[i])); o_c(' '); o_d(llrintf(fx[i]));
			o_c(' '); o_d(lroundf(fx[i])); o_c(' '); o_d(llroundf(fx[i])); o_c('\n');
		}
		if (xs[i] != 0 && isfinite(xs[i])) {
			/* ilogbf(0) (x underflows as a float) is FP_ILOGB0: implementation-defined */
			o_s("ilogb "); o_d(i); o_c(' '); o_d(ilogb(xs[i])); o_c(' ');
			o_d(fx[i] != 0 ? ilogbf(fx[i]) : 0);
			o_c('\n');
		}
	}
	o_s("nan "); o_d(isnan(nan(""))); o_d(isnan(nanf(""))); o_c('\n');
	return o_end();
}
