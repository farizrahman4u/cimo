/* expf for GPU code: in float only (double is slow on most GPUs). x = k ln2 + r, |r| <= ln2/2, with ln2 in two parts
   (Cody and Waite); e^r by its Taylor polynomial of degree 7; 2^k put into the exponent in two steps, so that results
   near overflow and in the subnormal range are right. Within 1 ulp (checked against the host's exp in
   tests/test_gpu.py). */
#include <math.h>

static float __cimo_scale(float y, int k)
{
	union { float f; unsigned u; } a, b;
	int k1 = k / 2;
	a.u = (unsigned)(k1 + 127) << 23;
	b.u = (unsigned)(k - k1 + 127) << 23;
	return y * a.f * b.f;
}

float expf(float x)
{
	if (!(x == x))
		return x + x;
	if (x > 88.72284f)
		return x * 0x1p127f * 0x1p127f;  /* inf */
	if (x < -103.97208f)
		return 0x1p-127f * 0x1p-127f;  /* 0 */
	float k = rintf(x * 1.44269504f);
	float r = fmaf(k, -0.693145752f, x);
	r = fmaf(k, -1.42860677e-6f, r);
	float p = 1.98412698e-4f;
	p = fmaf(p, r, 1.38888889e-3f);
	p = fmaf(p, r, 8.33333333e-3f);
	p = fmaf(p, r, 4.16666667e-2f);
	p = fmaf(p, r, 1.66666667e-1f);
	p = fmaf(p, r, 0.5f);
	p = fmaf(p, r * r, r);
	return __cimo_scale(1.0f + p, (int)k);
}
