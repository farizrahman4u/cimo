/* Benchmark kernels. Plain C11, restrict on pointer params, no libc except math.
 * Compiled separately by each compiler under test; harness.c calls them. */
#include <math.h>

/* --- vectorizable --- */

void saxpy(int n, float a, const float *restrict x, float *restrict y)
{
	for (int i = 0; i < n; i++)
		y[i] = a * x[i] + y[i];
}

void gemm(int n, const float *restrict a, const float *restrict b, float *restrict c)
{
	for (int i = 0; i < n; i++)
		for (int k = 0; k < n; k++) {
			float aik = a[i * n + k];
			for (int j = 0; j < n; j++)
				c[i * n + j] += aik * b[k * n + j];
		}
}

void jacobi2d(int n, const float *restrict src, float *restrict dst)
{
	for (int i = 1; i < n - 1; i++)
		for (int j = 1; j < n - 1; j++)
			dst[i * n + j] = 0.2f * (src[i * n + j] + src[i * n + j - 1] + src[i * n + j + 1]
			                         + src[(i - 1) * n + j] + src[(i + 1) * n + j]);
}

void conv3x3(int h, int w, const float *restrict img, const float *restrict k, float *restrict out)
{
	for (int y = 1; y < h - 1; y++)
		for (int x = 1; x < w - 1; x++) {
			float s = 0.0f;
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++)
					s += img[(y + dy) * w + x + dx] * k[(dy + 1) * 3 + dx + 1];
			out[y * w + x] = s;
		}
}

void rgb2gray(int n, const unsigned char *restrict rgb, unsigned char *restrict gray)
{
	for (int i = 0; i < n; i++)
		gray[i] = (unsigned char)((77 * rgb[3 * i] + 150 * rgb[3 * i + 1] + 29 * rgb[3 * i + 2]) >> 8);
}

/* --- reductions (gcc won't vectorize fp reductions without -ffast-math) --- */

double dot(int n, const double *restrict x, const double *restrict y)
{
	double s = 0.0;
	for (int i = 0; i < n; i++)
		s += x[i] * y[i];
	return s;
}

long isum(int n, const int *restrict x)
{
	long s = 0;
	for (int i = 0; i < n; i++)
		s += x[i];
	return s;
}

/* --- sequential / branchy --- */

void prefix_sum(int n, const float *restrict x, float *restrict y)
{
	float s = 0.0f;
	for (int i = 0; i < n; i++) {
		s += x[i];
		y[i] = s;
	}
}

void iir(int n, float b0, float a1, float a2, const float *restrict x, float *restrict y)
{
	float y1 = 0.0f, y2 = 0.0f;
	for (int i = 0; i < n; i++) {
		float v = b0 * x[i] - a1 * y1 - a2 * y2;
		y[i] = v;
		y2 = y1;
		y1 = v;
	}
}

int edit_distance(int n, int m, const unsigned char *restrict a, const unsigned char *restrict b,
                  int *restrict prev, int *restrict cur)
{
	for (int j = 0; j <= m; j++)
		prev[j] = j;
	for (int i = 1; i <= n; i++) {
		cur[0] = i;
		for (int j = 1; j <= m; j++) {
			int sub = prev[j - 1] + (a[i - 1] != b[j - 1]);
			int del = prev[j] + 1;
			int ins = cur[j - 1] + 1;
			int v = sub < del ? sub : del;
			cur[j] = v < ins ? v : ins;
		}
		int *t = prev; prev = cur; cur = t;
	}
	return prev[m];
}

void histogram(int n, const unsigned char *restrict x, int *restrict hist)
{
	for (int i = 0; i < 256; i++)
		hist[i] = 0;
	for (int i = 0; i < n; i++)
		hist[x[i]]++;
}

void mandelbrot(int w, int h, int maxit, int *restrict out)
{
	for (int py = 0; py < h; py++)
		for (int px = 0; px < w; px++) {
			double cr = -2.0 + 3.0 * px / w, ci = -1.5 + 3.0 * py / h;
			double zr = 0.0, zi = 0.0;
			int it = 0;
			while (it < maxit && zr * zr + zi * zi < 4.0) {
				double t = zr * zr - zi * zi + cr;
				zi = 2.0 * zr * zi + ci;
				zr = t;
				it++;
			}
			out[py * w + px] = it;
		}
}

void nbody_acc(int n, const double *restrict px, const double *restrict py, const double *restrict pz,
               double *restrict ax, double *restrict ay, double *restrict az)
{
	for (int i = 0; i < n; i++) {
		double sx = 0.0, sy = 0.0, sz = 0.0;
		for (int j = 0; j < n; j++) {
			double dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
			double r2 = dx * dx + dy * dy + dz * dz + 1e-9;
			double inv = 1.0 / (r2 * sqrt(r2));
			sx += dx * inv;
			sy += dy * inv;
			sz += dz * inv;
		}
		ax[i] = sx;
		ay[i] = sy;
		az[i] = sz;
	}
}

/* 4-connected flood fill with an explicit stack; returns number of filled cells */
int flood_fill(int w, int h, int sx, int sy, unsigned char *restrict grid, int *restrict stack)
{
	int top = 0, count = 0;
	stack[top++] = sy * w + sx;
	grid[sy * w + sx] = 2;
	while (top > 0) {
		int p = stack[--top];
		int x = p % w, y = p / w;
		count++;
		if (x > 0 && grid[p - 1] == 0) { grid[p - 1] = 2; stack[top++] = p - 1; }
		if (x < w - 1 && grid[p + 1] == 0) { grid[p + 1] = 2; stack[top++] = p + 1; }
		if (y > 0 && grid[p - w] == 0) { grid[p - w] = 2; stack[top++] = p - w; }
		if (y < h - 1 && grid[p + w] == 0) { grid[p + w] = 2; stack[top++] = p + w; }
	}
	return count;
}

/* sieve of Eratosthenes; returns prime count */
int sieve(int n, unsigned char *restrict comp)
{
	for (int i = 0; i <= n; i++)
		comp[i] = 0;
	int count = 0;
	for (int i = 2; i <= n; i++) {
		if (comp[i])
			continue;
		count++;
		for (long j = (long)i * i; j <= n; j += i)
			comp[j] = 1;
	}
	return count;
}
