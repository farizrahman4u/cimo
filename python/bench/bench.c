/* Timing loops for python/bench/run.py, compiled once with gcc -O2 into a shared library.
 *
 * Mirrors scratch/qbe_eval/harness.c (same sizes, reps and data), but every kernel is called
 * through a function pointer, so cimo's in-process code and gcc's shared libraries go through
 * the same call path. Each kernel has:
 *   init_<k>()             allocate inputs and fill them from a per-kernel seed (once)
 *   run_<k>(fp, &cs)       untimed setup, timed body, checksum; returns milliseconds
 * Inputs never depend on which configuration ran before. */
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long rng;
static unsigned next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static float frand(void) { return (next() & 0xffffff) / 16777216.0f; }
static void seed(unsigned long long s) { rng = 88172645463325252ULL ^ (s * 0x9e3779b97f4a7c15ULL); next(); }

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static double fsum(const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }
static double dsum(const double *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }
static double isum_h(const int *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }

#define TIMED(BODY) ({ double t0_ = now(); BODY; now() - t0_; })

/* saxpy: cache-resident, many reps */
static float *sx_x, *sx_y; enum { SX_N = 4096, SX_REPS = 50000 };
void init_saxpy(void) { seed(1); sx_x = malloc(SX_N * 4); sx_y = malloc(SX_N * 4); for (int i = 0; i < SX_N; i++) sx_x[i] = frand(); }
double run_saxpy(void (*f)(int, float, const float *, float *), double *cs)
{
	for (int i = 0; i < SX_N; i++) sx_y[i] = 0;
	double ms = TIMED(for (int r = 0; r < SX_REPS; r++) f(SX_N, 0.5f, sx_x, sx_y));
	*cs = fsum(sx_y, SX_N);
	return ms;
}

static float *gm_a, *gm_b, *gm_c; enum { GM_N = 384 };
void init_gemm(void) { seed(2); gm_a = malloc(GM_N * GM_N * 4); gm_b = malloc(GM_N * GM_N * 4); gm_c = malloc(GM_N * GM_N * 4);
	for (int i = 0; i < GM_N * GM_N; i++) { gm_a[i] = frand(); gm_b[i] = frand(); } }
double run_gemm(void (*f)(int, const float *, const float *, float *), double *cs)
{
	memset(gm_c, 0, GM_N * GM_N * 4);
	double ms = TIMED(f(GM_N, gm_a, gm_b, gm_c));
	*cs = fsum(gm_c, GM_N * GM_N);
	return ms;
}

static float *jc_s, *jc_d; enum { JC_N = 1024, JC_STEPS = 100 };
void init_jacobi2d(void) { jc_s = malloc(JC_N * JC_N * 4); jc_d = malloc(JC_N * JC_N * 4); }
double run_jacobi2d(void (*f)(int, const float *, float *), double *cs)
{
	float *s = jc_s, *d = jc_d;
	for (int i = 0; i < JC_N * JC_N; i++) s[i] = d[i] = (i % 97) * 0.01f;
	double ms = TIMED(for (int t = 0; t < JC_STEPS; t++) { f(JC_N, s, d); float *tmp = s; s = d; d = tmp; });
	*cs = fsum(s, JC_N * JC_N);
	return ms;
}

static float *cv_img, *cv_out, cv_k[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1}; enum { CV_H = 1024, CV_W = 1024, CV_REPS = 20 };
void init_conv3x3(void) { seed(4); cv_img = malloc(CV_H * CV_W * 4); cv_out = malloc(CV_H * CV_W * 4);
	for (int i = 0; i < CV_H * CV_W; i++) cv_img[i] = frand(); }
double run_conv3x3(void (*f)(int, int, const float *, const float *, float *), double *cs)
{
	memset(cv_out, 0, CV_H * CV_W * 4);
	double ms = TIMED(for (int r = 0; r < CV_REPS; r++) f(CV_H, CV_W, cv_img, cv_k, cv_out));
	*cs = fsum(cv_out, CV_H * CV_W);
	return ms;
}

static unsigned char *rg_rgb, *rg_g; enum { RG_N = 1 << 20, RG_REPS = 50 };
void init_rgb2gray(void) { seed(5); rg_rgb = malloc(3 * RG_N); rg_g = malloc(RG_N); for (int i = 0; i < 3 * RG_N; i++) rg_rgb[i] = next() & 255; }
double run_rgb2gray(void (*f)(int, const unsigned char *, unsigned char *), double *cs)
{
	memset(rg_g, 0, RG_N);
	double ms = TIMED(for (int r = 0; r < RG_REPS; r++) f(RG_N, rg_rgb, rg_g));
	double s = 0; for (int i = 0; i < RG_N; i++) s += rg_g[i];
	*cs = s;
	return ms;
}

static double *dt_x, *dt_y; enum { DT_N = 4096, DT_REPS = 50000 };
void init_dot(void) { seed(6); dt_x = malloc(DT_N * 8); dt_y = malloc(DT_N * 8); for (int i = 0; i < DT_N; i++) { dt_x[i] = frand(); dt_y[i] = frand(); } }
double run_dot(double (*f)(int, const double *, const double *), double *cs)
{
	double acc = 0;
	double ms = TIMED(for (int r = 0; r < DT_REPS; r++) acc += f(DT_N, dt_x, dt_y));
	*cs = acc;
	return ms;
}

static int *is_x; enum { IS_N = 4096, IS_REPS = 50000 };
void init_isum(void) { seed(7); is_x = malloc(IS_N * 4); for (int i = 0; i < IS_N; i++) is_x[i] = next() & 0xffff; }
double run_isum(long (*f)(int, const int *), double *cs)
{
	long acc = 0;
	double ms = TIMED(for (int r = 0; r < IS_REPS; r++) acc += f(IS_N, is_x));
	*cs = (double)acc;
	return ms;
}

static float *ps_x, *ps_y; enum { PS_N = 4096, PS_REPS = 50000 };
void init_prefix_sum(void) { seed(8); ps_x = malloc(PS_N * 4); ps_y = malloc(PS_N * 4); for (int i = 0; i < PS_N; i++) ps_x[i] = frand(); }
double run_prefix_sum(void (*f)(int, const float *, float *), double *cs)
{
	memset(ps_y, 0, PS_N * 4);
	double ms = TIMED(for (int r = 0; r < PS_REPS; r++) f(PS_N, ps_x, ps_y));
	*cs = fsum(ps_y, PS_N);
	return ms;
}

static float *ir_x, *ir_y; enum { IR_N = 4096, IR_REPS = 50000 };
void init_iir(void) { seed(9); ir_x = malloc(IR_N * 4); ir_y = malloc(IR_N * 4); for (int i = 0; i < IR_N; i++) ir_x[i] = frand() - 0.5f; }
double run_iir(void (*f)(int, float, float, float, const float *, float *), double *cs)
{
	memset(ir_y, 0, IR_N * 4);
	double ms = TIMED(for (int r = 0; r < IR_REPS; r++) f(IR_N, 0.3f, -0.5f, 0.2f, ir_x, ir_y));
	*cs = fsum(ir_y, IR_N);
	return ms;
}

static unsigned char *ed_a, *ed_b; static int *ed_p, *ed_c; enum { ED_N = 6000, ED_M = 6000 };
void init_edit_distance(void) { seed(10); ed_a = malloc(ED_N); ed_b = malloc(ED_M); ed_p = malloc((ED_M + 1) * 4); ed_c = malloc((ED_M + 1) * 4);
	for (int i = 0; i < ED_N; i++) ed_a[i] = 'a' + next() % 4;
	for (int i = 0; i < ED_M; i++) ed_b[i] = 'a' + next() % 4; }
double run_edit_distance(int (*f)(int, int, const unsigned char *, const unsigned char *, int *, int *), double *cs)
{
	int d = 0;
	double ms = TIMED(d = f(ED_N, ED_M, ed_a, ed_b, ed_p, ed_c));
	*cs = d;
	return ms;
}

static unsigned char *hg_x; static int hg_hist[256]; enum { HG_N = 1 << 22, HG_REPS = 20 };
void init_histogram(void) { seed(11); hg_x = malloc(HG_N); for (int i = 0; i < HG_N; i++) hg_x[i] = (next() & 255) & (next() & 255); }
double run_histogram(void (*f)(int, const unsigned char *, int *), double *cs)
{
	double ms = TIMED(for (int r = 0; r < HG_REPS; r++) f(HG_N, hg_x, hg_hist));
	*cs = isum_h(hg_hist, 256);
	return ms;
}

static int *mb_out; enum { MB_W = 800, MB_H = 600 };
void init_mandelbrot(void) { mb_out = malloc(MB_W * MB_H * 4); }
double run_mandelbrot(void (*f)(int, int, int, int *), double *cs)
{
	double ms = TIMED(f(MB_W, MB_H, 256, mb_out));
	*cs = isum_h(mb_out, MB_W * MB_H);
	return ms;
}

static double *nb[6]; enum { NB_N = 2048 };
void init_nbody_acc(void) { seed(13); for (int k = 0; k < 6; k++) nb[k] = malloc(NB_N * 8);
	for (int i = 0; i < NB_N; i++) { nb[0][i] = frand(); nb[1][i] = frand(); nb[2][i] = frand(); } }
double run_nbody_acc(void (*f)(int, const double *, const double *, const double *, double *, double *, double *), double *cs)
{
	double ms = TIMED(f(NB_N, nb[0], nb[1], nb[2], nb[3], nb[4], nb[5]));
	*cs = dsum(nb[3], NB_N) + dsum(nb[4], NB_N) + dsum(nb[5], NB_N);
	return ms;
}

static unsigned char *ff_g; static int *ff_st; enum { FF_W = 2048, FF_H = 2048 };
void init_flood_fill(void) { ff_g = malloc(FF_W * FF_H); ff_st = malloc(FF_W * FF_H * 4 * sizeof(int)); }
double run_flood_fill(int (*f)(int, int, int, int, unsigned char *, int *), double *cs)
{
	rng = 12345;
	for (int i = 0; i < FF_W * FF_H; i++) ff_g[i] = (next() % 100) < 30;
	ff_g[(FF_H / 2) * FF_W + FF_W / 2] = 0;
	int cnt = 0;
	double ms = TIMED(cnt = f(FF_W, FF_H, FF_W / 2, FF_H / 2, ff_g, ff_st));
	*cs = cnt;
	return ms;
}

static unsigned char *sv_comp; enum { SV_N = 30000000 };
void init_sieve(void) { sv_comp = malloc(SV_N + 1); }
double run_sieve(int (*f)(int, unsigned char *), double *cs)
{
	int c = 0;
	double ms = TIMED(c = f(SV_N, sv_comp));
	*cs = c;
	return ms;
}
