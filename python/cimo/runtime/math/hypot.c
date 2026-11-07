/* cimo: glibc's algorithm (Borges, "An Improved Algorithm for hypot(a,b)", 2019): with a fused multiply-add,
   one sqrt of an fma, written so that the sum of squares rounds once; without, a sqrt and one correction step.
   Error under 1 ulp either way. Inputs are scaled only when a square could overflow or lose bits to underflow. */
#include <math.h>
#include <stdint.h>

#define SCALE 0x1p-600
#define LARGE 0x1p511
#define TINY 0x1p-511
#define EPS 0x1p-54

static inline double kernel(double ax, double ay)
{
#if __FP_FAST_FMA
	double t1 = ay + ay, t2 = ax - ay;
	/* ax^2 + ay^2 = 2ay*ax + (ax - ay)^2: (ax - ay)^2 is exact enough when 2ay >= ax */
	if (t1 >= ax)
		return sqrt(fma(t1, ax, t2 * t2));
	return sqrt(fma(ax, ax, ay * ay));
#else
	double h = sqrt(ax * ax + ay * ay), t1, t2;
	if (h <= 2.0 * ay) {
		double d = h - ay;
		t1 = ax * (2.0 * d - ax);
		t2 = (d - 2.0 * (ax - ay)) * d;
	} else {
		double d = h - ax;
		t1 = 2.0 * d * (ax - 2.0 * ay);
		t2 = (4.0 * d - ay) * ay + d * d;
	}
	return h - (t1 + t2) / (2.0 * h);
#endif
}

double hypot(double x, double y)
{
	union {double f; uint64_t i;} ux = {x}, uy = {y};
	ux.i &= -1ULL >> 1;
	uy.i &= -1ULL >> 1;
	/* note: hypot(inf, nan) == inf */
	if (ux.i >= 0x7ffULL << 52 || uy.i >= 0x7ffULL << 52) {
		if (ux.i == 0x7ffULL << 52 || uy.i == 0x7ffULL << 52)
			return INFINITY;
		return x + y;
	}
	double ax = ux.i < uy.i ? uy.f : ux.f;
	double ay = ux.i < uy.i ? ux.f : uy.f;
	if (ax > LARGE) {
		if (ay <= ax * EPS)
			return ax + ay;
		return kernel(ax * SCALE, ay * SCALE) / SCALE;
	}
	if (ay < TINY) {
		if (ax >= ay / EPS)
			return ax + ay;
		return kernel(ax / SCALE, ay / SCALE) * SCALE;
	}
	if (ay <= ax * EPS)
		return ax + ay;
	return kernel(ax, ay);
}
