"""The same functions in pure Python and in C compiled by cimo: run each, time them, print the
speedups. fma=False keeps cimo's floating-point results identical to Python's."""

import array
import random
import time

import cimo

k = cimo.compile(r"""
double ema(const double *x, long n, double a) {         /* exponential moving average */
    double s = x[0];
    for (long i = 0; i < n; i++) s = a * x[i] + (1 - a) * s;
    return s;
}

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

long collatz(long n) {                     /* the start below n with the longest chain */
    long best = 1, best_len = 1;
    for (long i = 1; i < n; i++) {
        long x = i, len = 1;
        while (x != 1) { x = x % 2 ? 3 * x + 1 : x / 2; len++; }
        if (len > best_len) { best_len = len; best = i; }
    }
    return best;
}

long mandelbrot(int h, int w, int maxit) {          /* total iterations over the image */
    long total = 0;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            double cr = -2.2 + c * 3.2 / w, ci = -1.2 + r * 2.4 / h, zr = 0, zi = 0;
            int k = 0;
            while (k < maxit && zr * zr + zi * zi <= 4.0) {
                double t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
                k++;
            }
            total += k;
        }
    return total;
}

long primes(char *composite, long n) {             /* sieve of Eratosthenes: count below n */
    long count = 0;
    for (long i = 2; i < n; i++) {
        if (composite[i]) continue;
        count++;
        for (long j = i * i; j < n; j += i) composite[j] = 1;
    }
    return count;
}
""", fma=False)


def ema(x, n, a):
    s = x[0]
    for i in range(n):
        s = a * x[i] + (1 - a) * s
    return s


def edit_distance(a, n, b, m):
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[m]


def collatz(n):
    best, best_len = 1, 1
    for i in range(1, n):
        x, length = i, 1
        while x != 1:
            x = 3 * x + 1 if x % 2 else x // 2
            length += 1
        if length > best_len:
            best, best_len = i, length
    return best


def mandelbrot(h, w, maxit):
    total = 0
    for r in range(h):
        for c in range(w):
            cr, ci, zr, zi, k = -2.2 + c * 3.2 / w, -1.2 + r * 2.4 / h, 0.0, 0.0, 0
            while k < maxit and zr * zr + zi * zi <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                k += 1
            total += k
    return total


def primes(composite, n):
    count = 0
    for i in range(2, n):
        if composite[i]:
            continue
        count += 1
        for j in range(i * i, n, i):
            composite[j] = 1
    return count


def timed(f, *args):
    t = time.perf_counter()
    r = f(*args)
    return r, time.perf_counter() - t


random.seed(0)
xs = array.array('d', (random.random() for _ in range(1_000_000)))
a = bytes(random.choice(b'ACGT') for _ in range(1500))
b = bytes(random.choice(b'ACGT') for _ in range(1500))
N = 2_000_000

tests = [  # name, function name, arguments (a fresh copy for each run where it writes)
    ('moving average, 1M samples', 'ema', lambda: (xs, len(xs), 0.01)),
    ('edit distance, 1500 x 1500', 'edit_distance', lambda: (a, len(a), b, len(b))),
    ('longest Collatz chain < 100k', 'collatz', lambda: (100_000,)),
    ('Mandelbrot, 320 x 240', 'mandelbrot', lambda: (240, 320, 200)),
    ('primes below 2M', 'primes', lambda: (bytearray(N), N)),
]

print('%-30s %12s %12s %9s' % ('', 'Python', 'cimo', 'speedup'))
for name, fn, args in tests:
    want, tp = timed(globals()[fn], *args())
    got, tc = min((timed(getattr(k, fn), *args()) for _ in range(3)), key=lambda r: r[1])
    assert got == want, (name, got, want)
    print('%-30s %9.1f ms %9.1f ms %8.0fx' % (name, tp * 1e3, tc * 1e3, tp / tc))
