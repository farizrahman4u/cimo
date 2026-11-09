"""Parallel loops (`#pragma omp parallel for`), cimo against gcc -O3 -march=native -fopenmp on the same C file:
the cost of an empty region, and the scaling of a memory-bound matrix-vector product (float and bf16 weights).

    python scratch/spikes/omp_bench.py [--threads 1,2,4,8,16] [--rows 8192] [--cols 8192]

gcc's build is a shared library called through ctypes, as cimo's kernels are called; both set their team size with
omp_set_num_threads in the same kernel. Interleaved, best of several runs (requirements.md, "Engineering rules").
EC2 only (requirements.md: benchmarks are not run on the laptop).
"""

import argparse
import array
import ctypes
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import cimo  # noqa: E402

SRC = r'''
#include <omp.h>
#include <stddef.h>

void threads(int n) { omp_set_num_threads(n); }

static volatile int sink;
void empty(long reps, int n) {
    for (long r = 0; r < reps; r++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) sink = i;
    }
}

void matvec(int rows, int cols, const float *w, const float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        const float *r = w + (size_t)i * cols;
        float s = 0;
        #pragma omp simd reduction(+:s)
        for (int k = 0; k < cols; k++) s += r[k] * x[k];
        y[i] = s;
    }
}

void matvec_bf16(int rows, int cols, const __bf16 *w, const float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        const __bf16 *r = w + (size_t)i * cols;
        float s = 0;
        #pragma omp simd reduction(+:s)
        for (int k = 0; k < cols; k++) s += (float)r[k] * x[k];
        y[i] = s;
    }
}
'''


def gcc_lib(d):
    c = os.path.join(d, 'k.c')
    with open(c, 'w') as f:
        f.write(SRC)
    so = os.path.join(d, 'k.so')
    subprocess.check_call(['gcc', '-O3', '-march=native', '-fopenmp', '-shared', '-fPIC', c, '-o', so])
    lib = ctypes.CDLL(so)
    lib.empty.argtypes = [ctypes.c_long, ctypes.c_int]
    lib.threads.argtypes = [ctypes.c_int]
    for f in (lib.matvec, lib.matvec_bf16):
        f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def addr(a):
    return a.buffer_info()[0]


def best(f, runs):
    t = float('inf')
    for _ in range(runs):
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', default='1,2,4,8,16')
    ap.add_argument('--rows', type=int, default=8192)
    ap.add_argument('--cols', type=int, default=8192)
    ap.add_argument('--runs', type=int, default=7)
    a = ap.parse_args()
    ns = [int(n) for n in a.threads.split(',')]
    m = cimo.compile(SRC)
    with tempfile.TemporaryDirectory() as d:
        g = gcc_lib(d)
        R, C = a.rows, a.cols
        w = array.array('f', [((i * 7) % 13 - 6) * 0.01 for i in range(R * C)])
        wb = array.array('H', [(((i * 7) % 13) + 0x3c00) & 0xffff for i in range(R * C)])
        x = array.array('f', [((i * 3) % 11 - 5) * 0.1 for i in range(C)])
        y1, y2 = array.array('f', [0] * R), array.array('f', [0] * R)
        print('cpus %d, %d x %d' % (os.cpu_count(), R, C))
        print('threads  empty us (cimo gcc)   float GB/s (cimo gcc)   bf16 GB/s (cimo gcc)')
        for n in ns:
            m.threads(n)
            g.threads(n)
            reps = 20000
            m.empty(1000, n)
            g.empty(1000, n)
            te = [float('inf')] * 2
            tf = [float('inf')] * 2
            tb = [float('inf')] * 2
            for _ in range(a.runs):  # interleaved, with a pause between the two: each one's threads spin for a
                # while after a loop (libgomp's 300000 pauses, about 10 ms on Sapphire Rapids), which would take
                # cores and hyperthread siblings from the other's run
                te[0] = min(te[0], best(lambda: m.empty(reps, n), 1) / reps)
                tf[0] = min(tf[0], best(lambda: m.matvec(R, C, w, x, y1), 1))
                tb[0] = min(tb[0], best(lambda: m.matvec_bf16(R, C, wb, x, y1), 1))
                time.sleep(0.2)
                te[1] = min(te[1], best(lambda: g.empty(reps, n), 1) / reps)
                tf[1] = min(tf[1], best(lambda: g.matvec(R, C, addr(w), addr(x), addr(y2)), 1))
                tb[1] = min(tb[1], best(lambda: g.matvec_bf16(R, C, addr(wb), addr(x), addr(y2)), 1))
                time.sleep(0.2)
            m.matvec(R, C, w, x, y1)
            g.matvec(R, C, addr(w), addr(x), addr(y2))
            err = max(abs(p - q) for p, q in zip(y1, y2))
            gb, gbb = R * C * 4 / 1e9, R * C * 2 / 1e9
            print('%7d  %6.2f %6.2f        %6.1f %6.1f            %6.1f %6.1f     (|dy| %.1e)'
                  % (n, te[0] * 1e6, te[1] * 1e6, gb / tf[0], gb / tf[1], gbb / tb[0], gbb / tb[1], err), flush=True)


if __name__ == '__main__':
    main()
