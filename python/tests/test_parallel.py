"""Parallel loops: `#pragma omp parallel for` on the runtime's thread pool (frontend/omp.py, runtime/omp.c,
cimo/parallel.py). Results against gcc -fopenmp's on the same file where gcc is here, errors and faults in the
threads, Ctrl-C, each thread's heap and output, the thread count, and the code for the other targets."""

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

import hosts
import cimo
from cimo import driver, linker, parallel

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='cimo does not run here')

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSIX = sys.platform != 'win32'


@contextlib.contextmanager
def threads(n):
    old = parallel._default
    parallel.set_num_threads(n)
    try:
        yield
    finally:
        parallel.set_num_threads(old)


def output(m, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(*args)
    return buf.getvalue()


# Every result here is the same whatever order the threads run in (integer sums, sums of small integers in double),
# or depends only on the static schedule's split, which libgomp's and cimo's share.
GCC_PROGRAM = r'''
#include <stdio.h>
#include <omp.h>

#define N 1000
static int a[N], b[N], tid[64];
static double d[N];
int glob;

static void reductions(void) {
    long s = 0, p = 1, x = 0, o = 0, an = -1, mx = -1000000, mn = 1000000, sub = 0;
    int land = 1, lor = 0;
    double ds = 0, dmax = -1e300;
    unsigned char ub = 0;
    #pragma omp parallel for reduction(+:s) reduction(*:p) reduction(^:x) reduction(|:o) reduction(&:an) \
        reduction(max:mx) reduction(min:mn) reduction(-:sub) reduction(&&:land) reduction(||:lor) \
        reduction(+:ds) reduction(max:dmax) reduction(+:ub)
    for (int i = 0; i < N; i++) {
        s += a[i]; p *= (i % 97 == 5) ? 3 : 1; x ^= a[i] * 7; o |= 1L << (i % 40); an &= ~(1L << (i % 30));
        if (a[i] * 3 % 1001 > mx) mx = a[i] * 3 % 1001;
        if (a[i] * 5 % 997 < mn) mn = a[i] * 5 % 997 + 1;
        sub -= i; land = land && i < 2000; lor = lor || i == 777; ds += i * 0.5; if (d[i] > dmax) dmax = d[i];
        ub += i;
    }
    glob = 0;
    #pragma omp parallel for reduction(+:glob)
    for (int i = 0; i < N; i++) glob += i & 7;
    printf("red %ld %ld %ld %ld %ld %ld %ld %ld %d %d %.1f %.1f %d %d\n", s, p, x, o, an, mx, mn, sub, land, lor, ds,
           dmax, ub, glob);
}

static void forms(void) {
    long t = 0;
    int i;
    unsigned u;
    long l;
    short sh;
    unsigned char c;
    i = 77;
    #pragma omp parallel for reduction(+:t)
    for (i = 0; i <= 100; i += 3) t += i;
    printf("le %ld %d\n", t, i);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (i = 100; i > -7; i--) t += i * i;
    printf("gt %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (i = 99; i >= 5; i -= 4) t += i;
    printf("ge %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (u = 10; u != 1000; u++) t += u;
    printf("ne %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (u = 4000000000u; u > 3999999000u; u = u - 7) t += u & 15;
    printf("unsigned %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (l = -5000000000L; l < 5000000000L; l = 1000000007 + l) t += l % 1000;
    printf("long %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (sh = -300; sh < 300; sh += 7) t += sh;
    printf("short %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (c = 250; c >= 3; c -= 2) t += c;
    printf("uchar %ld\n", t);
    t = 0;
    int lo = 5, hi = 3;
    #pragma omp parallel for reduction(+:t)
    for (int k = lo; k < hi; k++) t += 1;
    printf("empty %ld\n", t);
    t = 0;
    int st = 5;
    #pragma omp parallel for reduction(+:t)
    for (int k = 1; 200 > k; k += st) t += k;
    printf("swapped %ld\n", t);
}

static void sharing(void) {
    int x = 10, y = 20, z = 30, w = 40;
    int arr[4] = {1, 2, 3, 4};
    #pragma omp parallel for private(x) firstprivate(y, arr) lastprivate(z) shared(w)
    for (int i = 0; i < N; i++) {
        x = i;
        y += 1;
        arr[i % 4] += i;
        z = i * 2 + x - i + arr[0] * 0;
        b[i] = y > 20 && w == 40;
    }
    long s = 0;
    for (int i = 0; i < N; i++) s += b[i];
    printf("sharing %d %d %d %d %ld %d\n", x, y, z, w, s, arr[1]);
    int k = -1, j = -1;
    #pragma omp parallel for lastprivate(k)
    for (k = 0; k < 17; k += 4) j = k;
    printf("lastvar %d\n", k);
    int q = 5;
    #pragma omp parallel for firstprivate(q) lastprivate(q)
    for (int i = 0; i < 10; i++) q = q * 0 + i + 100;
    printf("firstlast %d\n", q);
    (void)j;
}

static void schedules(void) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < 30; i++) tid[i] = omp_get_thread_num() * 10 + omp_get_num_threads();
    for (i = 0; i < 30; i++) printf("%d ", tid[i]);
    printf("\n");
    #pragma omp parallel for schedule(static, 4)
    for (i = 29; i >= 0; i--) tid[i] = omp_get_thread_num();
    for (i = 0; i < 30; i++) printf("%d ", tid[i]);
    printf("\n");
    #pragma omp parallel for num_threads(3)
    for (i = 0; i < 10; i++) tid[i] = omp_get_thread_num() * 10 + omp_get_num_threads();
    for (i = 0; i < 10; i++) printf("%d ", tid[i]);
    printf("\n");
    int on = 0;
    #pragma omp parallel for if(on)
    for (i = 0; i < 10; i++) tid[i] = omp_get_thread_num() * 10 + omp_get_num_threads();
    for (i = 0; i < 10; i++) printf("%d ", tid[i]);
    printf("\n");
    long t = 0;
    #pragma omp parallel for schedule(dynamic, 3) reduction(+:t)
    for (i = 0; i < N; i++) t += a[i];
    long g = 0;
    #pragma omp parallel for schedule(guided) reduction(+:g)
    for (i = 0; i < N; i++) g += a[i];
    long r = 0;
    #pragma omp parallel for schedule(runtime) reduction(+:r)
    for (i = 0; i < N; i++) r += a[i];
    printf("sched %ld %ld %ld\n", t, g, r);
}

static void nested_and_sync(void) {
    long cnt = 0, cnt2 = 0;
    int hist[8] = {0};
    #pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        int inner = 0;
        #pragma omp parallel for reduction(+:inner)
        for (int j = 0; j < 100; j++) inner += omp_get_num_threads();
        tid[i] = inner;
    }
    for (int i = 0; i < 8; i++) printf("%d ", tid[i]);
    printf("\n");
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        hist[a[i] % 8]++;
        #pragma omp critical
        cnt += i;
        #pragma omp critical (other)
        {
            cnt2 += 1;
        }
    }
    printf("sync %ld %ld %d %d %d\n", cnt, cnt2, hist[0], hist[3], hist[7]);
    omp_lock_t lk;
    omp_init_lock(&lk);
    long t = 0;
    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        omp_set_lock(&lk);
        t += i;
        omp_unset_lock(&lk);
    }
    omp_destroy_lock(&lk);
    printf("lock %ld %d %d\n", t, omp_in_parallel(), omp_get_num_threads());
    omp_set_num_threads(2);
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) tid[i] = omp_get_num_threads() * 10 + omp_get_max_threads();
    printf("set %d %d %d %d\n", tid[0], tid[1], tid[2], tid[3]);
    omp_set_num_threads(4);
    #pragma omp for
    for (int i = 0; i < 4; i++) tid[i] = omp_get_num_threads();
    printf("orphan %d %d\n", tid[0], tid[3]);
}

int main(void) {
    for (int i = 0; i < N; i++) { a[i] = (i * 37) % 101; d[i] = (i * 13) % 977; }
    reductions();
    forms();
    sharing();
    schedules();
    nested_and_sync();
    return 0;
}
'''


# what the loop's body may use of the function around it
GCC_PROGRAM2 = r"""
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <omp.h>

struct big { long v[6]; int tag; };
static long vla(int n, int m, int a[n][m]) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) {
        int row[m];
        for (int j = 0; j < m; j++) row[j] = a[i][j] * 2;
        for (int j = 0; j < m; j++) s += row[j] + sizeof row / sizeof row[0];
    }
    return s;
}
static long by_struct(struct big b, int n) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += b.v[i % 6] + b.tag;
    return s;
}
static long escaped(int n) {
    int k = 3;
    int *p = &k;
    long out[64];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) out[i] = *p + k + i;
    long s = 0;
    for (int i = 0; i < n; i++) s += out[i];
    return s;
}
static long control(int n) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) {
        if (i % 3 == 0) continue;
        switch (i % 4) { case 0: s += 1; break; case 1: s += 10; break; default: s += 100; }
        int k = 0;
    again:
        if (++k < 3) goto again;
        s += k;
        s += ({ int t = i * 2; t + 1; });
        s += ((int[]){i, 2 * i, 3 * i})[i % 3];
        _Alignas(64) char buf[64];
        memset(buf, i & 0x7f, sizeof buf);
        s += buf[63] + ((unsigned long)buf % 64);
        for (int j = 0; j < 4; j++) { if (j == 2) break; s += j; }
    }
    return s;
}
static long callee(int n) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += omp_get_num_threads();
    return s;
}
static long dynamic_nesting(int n) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += callee(10);
    return s;
}
static long types(int n) {
    _Bool flag = 1; char c = 7; short h = -3; float f = 1.5f; double d = 2.25; unsigned long ul = 1UL << 40;
    long long ld = 4; double _Complex z = 1.0 + 2.0 * 1.0i;
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += flag + c + h + (long)(f * 2) + (long)(d * 4) + (long)(ul >> 38) + (long)ld
                                   + (long)__imag__ z;
    float fs = 0;
    #pragma omp parallel for simd reduction(+:fs)
    for (int i = 0; i < n; i++) fs += 0.5f;
    return s * 1000 + (long)fs;
}
static long privates(int n) {
    int arr[8] = {0};
    struct big b = {{1, 2, 3, 4, 5, 6}, 7};
    long s = 0;
    #pragma omp parallel for private(arr) firstprivate(b) reduction(+:s)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 8; k++) arr[k] = i + k;
        b.tag += 1;
        s += arr[7] + b.v[5];
    }
    return s * 100 + arr[0] + b.tag;
}
static long jumps(int n) {
    jmp_buf env;
    volatile long s = 0;
    if (setjmp(env) == 0) {
        long t = 0;
        #pragma omp parallel for reduction(+:t)
        for (int i = 0; i < n; i++) t += i;
        s = t;
        longjmp(env, 1);
    }
    return s;
}
int main(void) {
    int a[5][7];
    for (int i = 0; i < 5; i++) for (int j = 0; j < 7; j++) a[i][j] = i * 7 + j;
    struct big b = {{1, 2, 3, 4, 5, 6}, 100};
    printf("%ld %ld %ld %ld\n", vla(5, 7, a), by_struct(b, 50), escaped(40), control(200));
    printf("%ld %ld %ld %ld\n", dynamic_nesting(16), types(64), privates(32), jumps(1000));
    return 0;
}
"""


def test_what_the_loop_uses_as_gcc():
    m = cimo.compile(GCC_PROGRAM2)
    with threads(4):
        ours = output(m)
    assert ours == '1435 5171 1020 74396\n160 1728032 91207 499500\n'  # gcc -fopenmp's
    if _gcc_openmp():
        assert ours == gcc_run(GCC_PROGRAM2)


# clause expressions, and loops at the ends of their types' ranges
GCC_PROGRAM3 = r"""
#include <stdio.h>
#include <limits.h>
#include <omp.h>

static int tid[64];

int main(void) {
    long t = 0;
    int *p = &tid[0];
    int k = 2, n = 40, st = 3;
    #pragma omp parallel for if(p) num_threads(k + 1) schedule(static, n / 8)
    for (int i = 0; i < n; i++) tid[i] = omp_get_thread_num() * 10 + omp_get_num_threads();
    for (int i = 0; i < n; i++) printf("%d ", tid[i]);
    printf("\n");
    t = 0;
    #pragma omp parallel for reduction(+:t) schedule(guided, 2)
    for (unsigned long u = ULONG_MAX - 10; u < ULONG_MAX; u++) t += (long)(u & 0xff);
    printf("ulong %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (int i = n; i > 0; i -= st) t += i;
    printf("down %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (int i = n; i >= -5; i = i - st) t += i;
    printf("down2 %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (long i = LONG_MIN + 5; i < LONG_MIN + 50; i += 7) t += i - LONG_MIN;
    printf("lmin %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (int i = INT_MAX - 20; i < INT_MAX; i++) t += INT_MAX - i;
    printf("imax %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t)
    for (unsigned i = 0; i < 3000000000u; i += 1000000000u) t += i / 1000;
    printf("ustep %ld\n", t);
    t = 0;
    #pragma omp parallel for reduction(+:t) schedule(dynamic)
    for (long i = 0; i < 100000; i++) t += i;
    printf("dyn1 %ld\n", t);
    t = 0;
    int big = 1000003;
    #pragma omp parallel for reduction(+:t) schedule(static, 7)
    for (int i = 0; i < big; i++) t += i % 13;
    printf("chunked %ld\n", t);
    return 0;
}
"""
GCC_OUTPUT3 = (
    '3 3 3 3 3 13 13 13 13 13 23 23 23 23 23 3 3 3 3 3 13 13 13 13 13 23 23 23 23 23 3 3 3 3 3 13 13 13 13 13 \n'
    'ulong 2495\ndown 287\ndown2 280\nlmin 182\nimax 210\nustep 3000000\ndyn1 4999950000\nchunked 6000000\n')


def test_clause_expressions_and_ranges_as_gcc():
    m = cimo.compile(GCC_PROGRAM3)
    with threads(4):
        ours = output(m)
    assert ours == GCC_OUTPUT3
    if _gcc_openmp():
        assert ours == gcc_run(GCC_PROGRAM3)


def test_narrow_loop_variables_run_as_written():
    """A char or short loop variable whose step doesn't land on the bound: the iterations the loop has run as
    written (gcc 13 -fopenmp runs none of these two; gcc without -fopenmp, and cimo, run them all)."""
    m = cimo.compile(r"""
long sc(void) { long t = 0;
    #pragma omp parallel for reduction(+:t)
    for (signed char c = -100; c < 100; c += 9) t += c;
    return t; }
long us(void) { long t = 0;
    #pragma omp parallel for reduction(+:t)
    for (unsigned short h = 65000; h > 100; h -= 999) t += h;
    return t; }
""")
    for n in (1, 3, 8):
        with threads(n):
            assert m.sc() == sum(range(-100, 100, 9)) and m.us() == sum(range(65000, 100, -999))


def _gcc_openmp():
    gcc = shutil.which('gcc')
    if not gcc or sys.platform == 'win32':
        return None
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, 't.c')
        with open(c, 'w') as f:
            f.write('int main(void) { int s = 0;\n#pragma omp parallel for reduction(+:s)\n'
                    'for (int i = 0; i < 4; i++) s += i; return s != 6; }\n')
        r = subprocess.run([gcc, '-fopenmp', c, '-o', os.path.join(d, 't')], capture_output=True)
        return gcc if r.returncode == 0 else None


def gcc_run(src, nthreads=4):
    gcc = _gcc_openmp()
    if gcc is None:
        pytest.skip('no gcc with -fopenmp')
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, 't.c')
        with open(c, 'w') as f:
            f.write(src)
        exe = os.path.join(d, 't')
        subprocess.run([gcc, '-O3', '-fopenmp', c, '-o', exe], check=True, capture_output=True)
        r = subprocess.run([exe], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=str(nthreads)),
                           check=True, timeout=60)
        return r.stdout


def test_results_as_gcc_openmp():
    m = cimo.compile(GCC_PROGRAM)
    with threads(4):
        ours = output(m)
    assert ours == gcc_run(GCC_PROGRAM), ours


# what gcc -O3 -fopenmp's build of it prints with OMP_NUM_THREADS=4
GCC_OUTPUT = (
    'red 50010 177147 32 1099511627775 -1073741824 300 1 -499500 1 1 249750.0 976.0 44 3500\nle 1683 77\n'
    'gt 338441\nge 1272\nne 499455\nunsigned 1073\nlong -3685\nshort -215\nuchar 15748\nempty 0\nswapped 3940\n'
    'sharing 10 20 1998 40 1000 2\nlastvar 20\nfirstlast 109\n'
    '4 4 4 4 4 4 4 4 14 14 14 14 14 14 14 14 24 24 24 24 24 24 24 34 34 34 34 34 34 34 \n'
    '3 3 2 2 2 2 1 1 1 1 0 0 0 0 3 3 3 3 2 2 2 2 1 1 1 1 0 0 0 0 \n3 3 3 3 13 13 13 23 23 23 \n1 1 1 1 1 1 1 1 1 1 \n'
    'sched 50010 50010 50010\n100 100 100 100 100 100 100 100 \nsync 499500 1000 129 128 118\nlock 4950 0 1\n'
    'set 22 22 22 22\norphan 1 1\n')


def test_results_without_gcc():
    """The program above's results, as gcc's build printed them: so they are checked where gcc isn't."""
    m = cimo.compile(GCC_PROGRAM)
    with threads(4):
        assert output(m) == GCC_OUTPUT


@pytest.mark.parametrize('n', [1, 2, 3, 8, 16])
def test_team_sizes(n):
    m = cimo.compile(r'''
#include <omp.h>
void ids(int *t, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) t[i] = omp_get_thread_num() * 1000 + omp_get_num_threads();
}
long sum(const long *x, long n) {
    long s = 0;
    #pragma omp parallel for reduction(+:s) schedule(dynamic, 7)
    for (long i = 0; i < n; i++) s += x[i];
    return s;
}
''')
    import array
    t = array.array('i', [0] * 100)
    with threads(n):
        m.ids(t, 100)
        x = array.array('q', range(100000))
        assert m.sum(x, 100000) == sum(range(100000))
    q, r = divmod(100, n)
    want = []
    for k in range(n):
        want += [k * 1000 + n] * (q + (k < r))
    assert list(t) == want


SRC = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int rd(int *p, int n, int bad) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += i == bad ? *(int *)8 : p[i];
    return s;
}
int ab(int n, int bad) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) if (i == bad) abort();
    return 1;
}
int asr(int n, int bad) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) assert(i != bad);
    return 1;
}
int dv(int n, int z) {
    int s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += i / (i == n - 1 ? z : 1);
    return s;
}
static int deep(int n) { volatile char b[4096]; b[0] = n; return n == -1 ? b[0] : deep(n + 1) + 1; }
int so(int n) {
    int s = 0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += i == n - 1 ? deep(0) : 0;
    return s;
}
long heap(int n) {
    long *ps[64];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) { ps[i] = malloc(1000 * sizeof(long)); for (int k = 0; k < 1000; k++) ps[i][k] = i; }
    long s = 0;
    for (int i = 0; i < n; i++) { s += ps[i][999]; free(ps[i]); }
    return s;
}
long *give(int n) {
    long *p = 0;
    #pragma omp parallel for lastprivate(p)
    for (int i = 0; i < n; i++) { p = malloc(64); p[0] = i; }
    return p;
}
void say(int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) printf("line %d of %d\n", i, n);
}
'''


def test_errors_in_the_threads_raise():
    m = cimo.compile(SRC)
    import array
    p = array.array('i', range(64))
    with threads(4):
        for bad in (0, 30, 63):  # in the calling thread's part, and in workers'
            with pytest.raises(cimo.KernelError) as e:
                m.rd(p, 64, bad)
            assert e.value.kind == 'memory fault' and e.value.function == 'rd' and e.value.address == 8
            assert m.rd(p, 64, -1) == sum(range(64))
            with pytest.raises(cimo.KernelError) as e:
                m.ab(64, bad)
            assert e.value.kind == 'abort' and e.value.function == 'ab'
            with pytest.raises(cimo.KernelError) as e:
                m.asr(64, bad)
            assert e.value.kind == 'assert' and e.value.expr == 'i != bad'
        from cimo import loader
        if loader._X86:  # arm64's division by 0 gives 0
            with pytest.raises(cimo.KernelError) as e:
                m.dv(64, 0)
            assert e.value.kind == 'division fault' and e.value.function == 'dv'
        assert m.dv(64, 1) == sum(range(64))
        with pytest.raises(cimo.KernelError) as e:
            m.so(64)
        assert e.value.kind == 'stack overflow'
        assert m.rd(p, 64, -1) == sum(range(64))


def test_each_thread_has_its_heap():
    m = cimo.compile(SRC)
    with threads(8):
        for _ in range(20):
            assert m.heap(64) == sum(range(64))
        p = m.give(16)
        assert p
        m.free(p)
        m.reset_heap()
        assert m.heap(64) == sum(range(64))


def test_output_of_the_threads():
    m = cimo.compile(SRC)
    buf = io.StringIO()
    with threads(4), contextlib.redirect_stdout(buf):
        m.say(40)
    assert sorted(buf.getvalue().splitlines()) == sorted('line %d of 40' % i for i in range(40))


def test_thread_count():
    m = cimo.compile(r'''
#include <omp.h>
int n(void) { int k = 0;
    #pragma omp parallel for reduction(max:k)
    for (int i = 0; i < 1000; i++) if (omp_get_num_threads() > k) k = omp_get_num_threads();
    return k; }
int maxt(void) { return omp_get_max_threads(); }
int procs(void) { return omp_get_num_procs(); }
''')
    with threads(3):
        assert m.n() == 3 and m.maxt() == 3 and parallel.get_num_threads() == 3
    with threads(1):
        assert m.n() == 1
    assert m.procs() >= 1
    with pytest.raises(ValueError):
        parallel.set_num_threads(0)
    r = subprocess.run([sys.executable, '-c', 'import cimo\nm = cimo.compile("""%s""")\nprint(m.n(), m.maxt(), '
                        'cimo.get_num_threads())' % m.source.replace('\\', '\\\\')],
                       env=dict(os.environ, PYTHONPATH=PY, OMP_NUM_THREADS='5'), cwd=PY, capture_output=True,
                       text=True, timeout=300)
    assert r.stdout.split() == ['5', '5', '5'], r.stderr


def test_many_python_threads_run_parallel_loops_at_once():
    """Each calling thread gets a pool of its own while another's is busy (as libgomp)."""
    from concurrent.futures import ThreadPoolExecutor
    m = cimo.compile(r'''
long sum(const long *x, long n, long add) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) s += x[i] + add;
    return s;
}''')
    import array
    x = array.array('q', range(10000))
    with threads(3), ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda k: m.sum(x, 10000, k), range(200)))
    assert got == [sum(range(10000)) + 10000 * k for k in range(200)]


CTRL_C = r'''
import cimo, os, signal, threading, time
cimo.set_num_threads(4)
m = cimo.compile(r"""
#include <math.h>
long spin(long n) {
    long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) { volatile long k = 0; for (;;) { k++; if (i < 0) break; } s += k; }
    return s;
}
double heavy(long n) {
    double s = 0;
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) { volatile double t = 0; for (;;) t += sqrt(t + i); }
    return s;
}
long ok(long n) { long s = 0;
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) s += i;
    return s; }
""", cache=False)
kill = lambda: os.kill(os.getpid(), signal.SIGINT)
for f in (m.spin, m.heavy):
    threading.Timer(0.3, kill).start()
    t0 = time.time()
    try:
        f(8)
        print('returned')
    except KeyboardInterrupt:
        print('stopped', time.time() - t0 < 10)
print(m.ok(1000))
for _ in range(20):  # SIGURG (which stops a stopping region's threads) does nothing to a region that isn't stopping
    threading.Timer(0.01, lambda: os.kill(os.getpid(), signal.SIGURG)).start()
    assert m.ok(3000000) == 3000000 * 2999999 // 2
print('urg ok')
'''


@pytest.mark.skipif(not POSIX, reason='POSIX signals')
def test_ctrl_c_stops_every_thread():
    if os.environ.get('CIMO_FAULTS') == '0':
        pytest.skip('CIMO_FAULTS=0')
    r = subprocess.run([sys.executable, '-c', CTRL_C], capture_output=True, text=True, timeout=120, cwd=PY,
                       env=dict(os.environ, PYTHONPATH=PY))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split('\n')[:4] == ['stopped True', 'stopped True', '499500', 'urg ok'], r.stdout + r.stderr


@pytest.mark.parametrize('src, msg', [
    ('void f(int *a) {\n#pragma omp parallel\n{ a[0] = 1; } }', "'parallel for' (a loop"),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) { if (a[i]) break; } }', "'break' can't"),
    ('int f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) { if (a[i]) return 1; } return 0; }',
     "'return' in a parallel loop"),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) { if (a[i]) goto out; } out:; }',
     'jump into or out of'),
    ('void f(int *a) { goto in;\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) { in: a[i] = 1; } }',
     'jump into or out of'),
    ('void f(int *a, int k) { switch (k) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) { case 1: a[i] = 1; '
     '} } }', "'case' label"),
    ('void f(int *a) {\n#pragma omp parallel for collapse(2)\nfor (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++)'
     ' a[i] = j; }', "'collapse(2)' is not supported"),
    ('void f(int *a) {\n#pragma omp parallel for ordered\nfor (int i = 0; i < 9; i++) a[i] = 1; }',
     "'ordered' clause"),
    ('void f(int *a) {\n#pragma omp parallel for linear(a)\nfor (int i = 0; i < 9; i++) a[i] = 1; }', "'linear'"),
    ('void f(int *a) {\n#pragma omp parallel for reduction(+:a[0:4])\nfor (int i = 0; i < 9; i++) a[0] += 1; }',
     'not array sections'),
    ('void f(int *a) {\n#pragma omp parallel for reduction(foo:a)\nfor (int i = 0; i < 9; i++) a[0] += 1; }',
     "reduction operator 'foo'"),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i * 2 < 9; i++) a[i] = 1; }', 'a parallel loop'),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i *= 2) a[i] = 1; }', 'a parallel loop'),
    ('void f(float *a) {\n#pragma omp parallel for\nfor (float x = 0; x < 9; x++) a[0] = x; }', 'integer variable'),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i--) a[i] = 1; }', 'wrong way'),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i != 9; i += 2) a[i] = 1; }', "with '!='"),
    ('void f(int *a) {\n#pragma omp parallel for\nint x; }', "followed by a 'for'"),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) {\n#pragma omp barrier\n} }',
     "barrier' inside"),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) {\n#pragma omp single\na[0] = 1; } }',
     "single' inside"),
    ('void f(int *a) {\n#pragma omp parallel for\nfor (int i = 0; i < 9; i++) {\n#pragma omp for\n'
     'for (int j = 0; j < 9; j++) a[j] = 1; } }', "'#pragma omp for' inside"),
    ('void f(int *a) {\n#pragma omp task\na[0] = 1; }', "'#pragma omp task' is not supported"),
    ('int x;\n#pragma omp threadprivate(x)\n', "threadprivate' is not supported"),
    ('void f(int *a) {\n#pragma omp single private(a)\na[0] = 1; }', "'private' clause on '#pragma omp single'"),
    ('void f(int *a) {\n#pragma omp frobnicate\na[0] = 1; }', "unknown directive"),
    ('void f(int *a) {\n#pragma omp parallel for private(nosuch)\nfor (int i = 0; i < 9; i++) a[i] = 1; }',
     "'nosuch' in a 'private' clause"),
    ('void f(int *a) { int s;\n#pragma omp parallel for private(s) reduction(+:s)\nfor (int i = 0; i < 9; i++) s += '
     'a[i]; }', "in both a 'private' and a 'reduction'"),
    ('void f(int *a) {\n#pragma omp error message("nope")\n}', 'nope'),
])
def test_unsupported_is_an_error(src, msg):
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile(src)
    assert msg in str(e.value), str(e.value)


def test_ignored_directives_still_compile():
    """simd (its reductions), declare simd, the ones a team of one runs as written."""
    m = cimo.compile(r'''
#pragma omp declare simd
static int sq(int x) { return x * x; }
long f(int n) {
    long s = 0;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < n; i++) s += sq(i);
    #pragma omp single
    s += 1;
    #pragma omp master
    { s += 1; }
    #pragma omp barrier
    #pragma omp flush
    #pragma omp taskwait
    return s;
}''')
    assert m.f(10) == 285 + 2


def test_the_loop_vectorizes_and_reads_its_variables_by_value():
    ssa, _ = driver._frontend(driver._preprocess(r'''
void axpy(float *restrict y, const float *restrict x, float a, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}''', 't.c', (), False, ()), 't.c')
    body = ssa[ssa.index('function $axpy._omp_fn.0'):]
    body = body[:body.index('\n}\n')]
    assert 'loads %' in body  # a, by value from the data block
    asm = driver._qbe(ssa, _)
    fn = asm[asm.index('axpy._omp_fn.0:'):]
    if driver.TARGET.startswith('arm64'):
        assert '.4s' in fn[:4000]
    else:
        assert re.search(r'(mul|add|fmadd\w*)ps\s', fn[:4000])


@pytest.mark.parametrize('target', ['arm64', 'arm64_apple', 'arm64_win', 'amd64_sysv', 'amd64_windows'])
def test_every_target_builds_parallel_loops(target):
    """The frontend's code, the runtime, the glue and the pool and fault handlers' C, assembled for each target."""
    from cimo import execution, fault
    pp = driver._preprocess(GCC_PROGRAM, 't.c', (), False, (), target=target)
    ssa, params = driver._frontend(pp, 't.c', target=target, builtins=True, prune=True)
    asm = driver._qbe(ssa, params, stkchk=execution.stack_checked(ssa), target=target)
    apple, windows = target == 'arm64_apple', target.endswith('win') or target.endswith('windows')
    linker.assemble(asm + execution.glue(['main'], apple, windows, target), target)
    rt = linker.compile_runtime_source(os.path.join(linker.RUNTIME_DIR, 'omp.c'), target)
    linker.assemble(rt, target)
    for mod, platform in ((parallel, None), (fault, None)):
        old = (mod._WIN, mod._APPLE)
        mod._WIN, mod._APPLE = windows, apple
        try:
            from cimo import loader
            oldarm = loader._ARM64
            loader._ARM64 = target.startswith('arm64')
            defs = mod._defines()
        finally:
            mod._WIN, mod._APPLE = old
            loader._ARM64 = oldarm
        pp = driver._preprocess(mod._SOURCE, 'pool.c', (), False, tuple(defs.items()), target=target)
        ssa, params = driver._frontend(pp, 'pool.c', target=target, builtins=True, prune=True)
        linker.assemble(driver._qbe(ssa, params, stkchk=execution.stack_checked(ssa), target=target), target)


def test_wasm_runs_the_loop_on_one_thread():
    from cimo import wasm
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasm'))
    from host import Host, available
    if not available():
        pytest.skip('needs node')
    h = Host()
    try:
        mod, sigs = wasm.build(r'''
#include <omp.h>
long f(long n) {
    long s = 0, last = 0;
    #pragma omp parallel for reduction(+:s) lastprivate(last) schedule(dynamic, 3)
    for (long i = 0; i < n; i++) { s += i * omp_get_num_threads(); last = i; }
    #pragma omp critical
    s += omp_get_thread_num();
    return s * 1000 + last;
}''')
        h.load(mod)
        assert h.call('f', 'l', 100)[0] == 4950 * 1000 + 99
    finally:
        h.close()
