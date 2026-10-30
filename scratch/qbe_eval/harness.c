/* Timing harness: always compiled with gcc -O2. Prints "name ms checksum" per kernel. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void saxpy(int, float, const float *, float *);
void gemm(int, const float *, const float *, float *);
void jacobi2d(int, const float *, float *);
void conv3x3(int, int, const float *, const float *, float *);
void rgb2gray(int, const unsigned char *, unsigned char *);
double dot(int, const double *, const double *);
long isum(int, const int *);
void prefix_sum(int, const float *, float *);
void iir(int, float, float, float, const float *, float *);
int edit_distance(int, int, const unsigned char *, const unsigned char *, int *, int *);
void histogram(int, const unsigned char *, int *);
void mandelbrot(int, int, int, int *);
void nbody_acc(int, const double *, const double *, const double *, double *, double *, double *);
int flood_fill(int, int, int, int, unsigned char *, int *);
int sieve(int, unsigned char *);

static unsigned long long rng = 88172645463325252ULL;
static unsigned next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static float frand(void) { return (next() & 0xffffff) / 16777216.0f; }

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define TRIALS 5
/* SETUP runs untimed before each trial; BODY is timed; best trial reported */
#define BENCH(name, SETUP, BODY, CHECK) do { \
	double best = 1e30, cs = 0; \
	for (int t_ = 0; t_ < TRIALS; t_++) { \
		SETUP; \
		double t0_ = now(); \
		BODY; \
		double dt_ = now() - t0_; \
		if (dt_ < best) best = dt_; \
		cs = (CHECK); \
	} \
	printf("%-14s %10.3f %.10g\n", name, best, cs); \
	fflush(stdout); \
} while (0)

static double fsum(const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }
static double dsum(const double *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }
static double isum_h(const int *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }

int main(void)
{
	/* saxpy: cache-resident, many reps */
	{
		int n = 4096, reps = 50000;
		float *x = malloc(n * sizeof *x), *y = malloc(n * sizeof *y);
		for (int i = 0; i < n; i++) x[i] = frand();
		BENCH("saxpy", for (int i = 0; i < n; i++) y[i] = 0,
		      for (int r = 0; r < reps; r++) saxpy(n, 0.5f, x, y), fsum(y, n));
		free(x); free(y);
	}
	{
		int n = 384;
		float *a = malloc(n * n * 4), *b = malloc(n * n * 4), *c = malloc(n * n * 4);
		for (int i = 0; i < n * n; i++) { a[i] = frand(); b[i] = frand(); }
		BENCH("gemm", memset(c, 0, n * n * 4), gemm(n, a, b, c), fsum(c, n * n));
		free(a); free(b); free(c);
	}
	{
		int n = 1024, steps = 100;
		float *s = malloc(n * n * 4), *d = malloc(n * n * 4);
		BENCH("jacobi2d", for (int i = 0; i < n * n; i++) s[i] = d[i] = (i % 97) * 0.01f,
		      for (int t = 0; t < steps; t++) { jacobi2d(n, s, d); float *tmp = s; s = d; d = tmp; },
		      fsum(s, n * n));
		free(s); free(d);
	}
	{
		int h = 1024, w = 1024, reps = 20;
		float *img = malloc(h * w * 4), *out = calloc(h * w, 4), k[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
		for (int i = 0; i < h * w; i++) img[i] = frand();
		BENCH("conv3x3", , for (int r = 0; r < reps; r++) conv3x3(h, w, img, k, out), fsum(out, h * w));
		free(img); free(out);
	}
	{
		int n = 1 << 20, reps = 50;
		unsigned char *rgb = malloc(3 * n), *g = malloc(n);
		for (int i = 0; i < 3 * n; i++) rgb[i] = next() & 255;
		BENCH("rgb2gray", , for (int r = 0; r < reps; r++) rgb2gray(n, rgb, g),
		      ({ double s = 0; for (int i = 0; i < n; i++) s += g[i]; s; }));
		free(rgb); free(g);
	}
	{
		int n = 4096, reps = 50000;
		double *x = malloc(n * 8), *y = malloc(n * 8), acc = 0;
		for (int i = 0; i < n; i++) { x[i] = frand(); y[i] = frand(); }
		BENCH("dot", acc = 0, for (int r = 0; r < reps; r++) acc += dot(n, x, y), acc);
		free(x); free(y);
	}
	{
		int n = 4096, reps = 50000;
		int *x = malloc(n * 4);
		long acc = 0;
		for (int i = 0; i < n; i++) x[i] = next() & 0xffff;
		BENCH("isum", acc = 0, for (int r = 0; r < reps; r++) acc += isum(n, x), (double)acc);
		free(x);
	}
	{
		int n = 4096, reps = 50000;
		float *x = malloc(n * 4), *y = malloc(n * 4);
		for (int i = 0; i < n; i++) x[i] = frand();
		BENCH("prefix_sum", , for (int r = 0; r < reps; r++) prefix_sum(n, x, y), fsum(y, n));
		free(x); free(y);
	}
	{
		int n = 4096, reps = 50000;
		float *x = malloc(n * 4), *y = malloc(n * 4);
		for (int i = 0; i < n; i++) x[i] = frand() - 0.5f;
		BENCH("iir", , for (int r = 0; r < reps; r++) iir(n, 0.3f, -0.5f, 0.2f, x, y), fsum(y, n));
		free(x); free(y);
	}
	{
		int n = 6000, m = 6000;
		unsigned char *a = malloc(n), *b = malloc(m);
		int *p = malloc((m + 1) * 4), *c = malloc((m + 1) * 4), d = 0;
		for (int i = 0; i < n; i++) a[i] = 'a' + next() % 4;
		for (int i = 0; i < m; i++) b[i] = 'a' + next() % 4;
		BENCH("edit_distance", , d = edit_distance(n, m, a, b, p, c), (double)d);
		free(a); free(b); free(p); free(c);
	}
	{
		int n = 1 << 22, reps = 20;
		unsigned char *x = malloc(n);
		int hist[256];
		for (int i = 0; i < n; i++) x[i] = (next() & 255) & (next() & 255);
		BENCH("histogram", , for (int r = 0; r < reps; r++) histogram(n, x, hist), isum_h(hist, 256));
		free(x);
	}
	{
		int w = 800, h = 600;
		int *out = malloc(w * h * 4);
		BENCH("mandelbrot", , mandelbrot(w, h, 256, out), isum_h(out, w * h));
		free(out);
	}
	{
		int n = 2048;
		double *p[6];
		for (int k = 0; k < 6; k++) p[k] = malloc(n * 8);
		for (int i = 0; i < n; i++) { p[0][i] = frand(); p[1][i] = frand(); p[2][i] = frand(); }
		BENCH("nbody_acc", , nbody_acc(n, p[0], p[1], p[2], p[3], p[4], p[5]),
		      dsum(p[3], n) + dsum(p[4], n) + dsum(p[5], n));
		for (int k = 0; k < 6; k++) free(p[k]);
	}
	{
		int w = 2048, h = 2048, cnt = 0;
		unsigned char *g = malloc(w * h);
		int *st = malloc(w * h * 4 * sizeof(int));
		BENCH("flood_fill",
		      { rng = 12345; for (int i = 0; i < w * h; i++) g[i] = (next() % 100) < 30; g[(h / 2) * w + w / 2] = 0; },
		      cnt = flood_fill(w, h, w / 2, h / 2, g, st), (double)cnt);
		free(g); free(st);
	}
	{
		int n = 30000000, c = 0;
		unsigned char *comp = malloc(n + 1);
		BENCH("sieve", , c = sieve(n, comp), (double)c);
		free(comp);
	}
	return 0;
}
