"""Without cimo vs with cimo: the same loops in pure Python, in numpy where numpy can express
them, and in C compiled by cimo. Every version must give the same answer; times are the best
of a few runs. Compiled with fma=False so cimo's floating-point results are bit-identical to
Python's (fused multiply-add rounds once instead of twice)."""

import time

import numpy as np

import cimo

SRC = r"""
/* exponential moving average: each output depends on the previous one */
void ema(long n, const double *x, double a, double *y) {
    double s = x[0];
    for (long i = 0; i < n; i++) {
        s = a * x[i] + (1 - a) * s;
        y[i] = s;
    }
}

/* Levenshtein distance, the classic dynamic program */
int edit_distance(const char *a, int n, const char *b, int m) {
    int prev[m + 1], cur[m + 1];
    for (int j = 0; j <= m; j++) prev[j] = j;
    for (int i = 1; i <= n; i++) {
        cur[0] = i;
        for (int j = 1; j <= m; j++) {
            int best = prev[j - 1] + (a[i - 1] != b[j - 1]);
            if (prev[j] + 1 < best) best = prev[j] + 1;
            if (cur[j - 1] + 1 < best) best = cur[j - 1] + 1;
            cur[j] = best;
        }
        for (int j = 0; j <= m; j++) prev[j] = cur[j];
    }
    return prev[m];
}

/* Mandelbrot set: iterations until escape, per pixel */
void mandelbrot(int h, int w, double x0, double y0, double step, int maxit, int out[h][w]) {
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            double cr = x0 + c * step, ci = y0 + r * step, zr = 0, zi = 0;
            int k = 0;
            while (k < maxit && zr * zr + zi * zi <= 4.0) {
                double t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
                k++;
            }
            out[r][c] = k;
        }
}

/* plain sum: numpy is already native code here */
double total(long n, const double *x) {
    double s = 0;
    for (long i = 0; i < n; i++) s += x[i];
    return s;
}
"""


def best(f, runs=3):
    """(seconds, result) of the fastest of `runs` calls."""
    t, r = float('inf'), None
    for _ in range(runs):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


# --- pure Python -------------------------------------------------------------------------

def ema_py(x, a):
    y, s = [0.0] * len(x), x[0]
    for i, v in enumerate(x):
        s = a * v + (1 - a) * s
        y[i] = s
    return y


def edit_distance_py(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def mandelbrot_py(h, w, x0, y0, step, maxit):
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            cr, ci, zr, zi, k = x0 + c * step, y0 + r * step, 0.0, 0.0, 0
            while k < maxit and zr * zr + zi * zi <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                k += 1
            row.append(k)
        out.append(row)
    return out


# --- numpy (where it can express the loop) ------------------------------------------------

def mandelbrot_np(h, w, x0, y0, step, maxit):
    cr = x0 + np.arange(w) * step
    ci = y0 + np.arange(h)[:, None] * step
    cr, ci = np.broadcast_to(cr, (h, w)).copy(), np.broadcast_to(ci, (h, w)).copy()
    zr, zi = np.zeros((h, w)), np.zeros((h, w))
    out = np.zeros((h, w), dtype=np.int32)
    live = np.ones((h, w), dtype=bool)
    for _ in range(maxit):
        live &= zr * zr + zi * zi <= 4.0
        if not live.any():
            break
        t = zr[live] * zr[live] - zi[live] * zi[live] + cr[live]
        zi[live] = 2.0 * zr[live] * zi[live] + ci[live]
        zr[live] = t
        out[live] += 1
    return out


def main():
    t0 = time.perf_counter()
    k = cimo.compile(SRC, fma=False, cache=False)
    cold = time.perf_counter() - t0
    cimo.compile(SRC, fma=False)  # fills the cache
    t0 = time.perf_counter()
    cimo.compile(SRC, fma=False)
    cached = time.perf_counter() - t0
    print('compile: %.0f ms the first time, %.1f ms from the cache\n' % (cold * 1e3, cached * 1e3))

    rows = []
    rng = np.random.default_rng(0)

    # moving average over a million samples
    x = rng.random(1_000_000)
    xl = x.tolist()
    tp, yp = best(lambda: ema_py(xl, 0.01), 1)
    y = np.empty_like(x)
    tc, _ = best(lambda: k.ema(len(x), x, 0.01, y))
    assert y.tolist() == yp
    rows.append(('moving average, 1M samples', tp, None, tc))

    # edit distance between two 2000-character strings
    a = ''.join(rng.choice(list('ACGT'), 2000))
    b = ''.join(rng.choice(list('ACGT'), 2000))
    tp, dp = best(lambda: edit_distance_py(a, b), 1)
    tc, dc = best(lambda: k.edit_distance(a.encode(), len(a), b.encode(), len(b)))
    assert dc == dp
    rows.append(('edit distance, 2000 x 2000', tp, None, tc))

    # Mandelbrot set, 320 x 240, up to 200 iterations
    args = (240, 320, -2.2, -1.2, 0.01, 200)
    tp, mp = best(lambda: mandelbrot_py(*args), 1)
    tn, mn = best(lambda: mandelbrot_np(*args))
    out = np.empty(args[:2], dtype=np.int32)
    tc, _ = best(lambda: k.mandelbrot(*args, out))
    assert out.tolist() == mp == mn.tolist()
    rows.append(('Mandelbrot, 320 x 240', tp, tn, tc))

    # sum of 10 million numbers (numpy's home turf)
    x = rng.random(10_000_000)
    xl = x.tolist()
    tp, sp = best(lambda: sum(xl), 1)
    tn, sn = best(lambda: x.sum())
    tc, sc = best(lambda: k.total(len(x), x))
    assert abs(sc - sp) < 1e-6 * sp and abs(sn - sp) < 1e-6 * sp
    rows.append(('sum, 10M numbers', tp, tn, tc))

    def ms(t):
        return '-' if t is None else '%.1f ms' % (t * 1e3)
    print('%-28s %12s %12s %12s %16s' % ('', 'pure Python', 'numpy', 'cimo', 'cimo vs Python'))
    for name, tp, tn, tc in rows:
        print('%-28s %12s %12s %12s %15.0fx' % (name, ms(tp), ms(tn), ms(tc), tp / tc))
    print("\n'-': no simple numpy version (each step depends on the previous one)")


if __name__ == '__main__':
    main()
