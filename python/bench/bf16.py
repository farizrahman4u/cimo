"""cimo against gcc on bench/bf16.c: bf16 kernels written with the CPU's intrinsics.

    python3 python/bench/bf16.py [-n TRIALS]

Both builds of the same file (cimo.compile with an empty cache; gcc -O3 -mcpu=native on arm64, -march=native on x86-64,
or $BENCH_GCC_FLAGS, as a shared library) are
called through the same function pointers from a timing loop built with gcc -O2, interleaved, best trial
reported. Each kernel's result must be gcc's, bit for bit. The kernels each build only for a CPU with their
features (cimo.cpu_features()); the others are listed as skipped.
"""

import argparse
import ctypes
import os
import random
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cimo  # noqa: E402

TIMER = r'''
#define _POSIX_C_SOURCE 199309L
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
double time_dot(float (*f)(const void *, const void *, long), const void *a, const void *b, long n, long reps, float *out)
{
    double t0 = now();
    float s = 0;
    for (long r = 0; r < reps; r++)
        s += f(a, b, n);
    *out = s;
    return now() - t0;
}
double time_gemm(void (*f)(long, long, long, const void *, const void *, float *), long m, long n, long k,
                 const void *a, const void *b, float *c, long reps)
{
    double t0 = now();
    for (long r = 0; r < reps; r++)
        f(m, n, k, a, b, c);
    return now() - t0;
}
'''

# kernel: (features, kind)
KERNELS = {
    'bfdot': (('bf16',), 'dot'), 'gemm': (('bf16',), 'gemm'),
    'bfdot256': (('avx512bf16', 'avx512vl'), 'dot'), 'gemm256': (('avx512bf16', 'avx512vl'), 'gemm'),
    'bfdot512': (('avx512bf16',), 'dot'), 'gemm512': (('avx512bf16',), 'gemm'),
    'gemm_amx': (('amx-tile', 'amx-bf16'), 'gemm'),
}
DOT_N, DOT_REPS = 1 << 14, 2000  # 32 KB a vector: L1/L2
GEMM_M, GEMM_N, GEMM_K, GEMM_REPS = 64, 64, 512, 50


def bf16s(n, rnd):
    return (ctypes.c_uint16 * n)(*[struct.unpack('<I', struct.pack('<f', rnd.uniform(-1, 1)))[0] >> 16
                                   for _ in range(n)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-n', type=int, default=7, help='trials')
    ap.add_argument('kernels', nargs='*')
    a = ap.parse_args()
    feats = cimo.cpu_features()
    want = [k for k in (a.kernels or KERNELS) if k in KERNELS]
    run = [k for k in want if set(KERNELS[k][0]) <= feats]
    for k in want:
        if k not in run:
            print('%-10s skipped: needs %s' % (k, ', '.join(KERNELS[k][0])))
    src = os.path.join(HERE, 'bf16.c')
    tmp = tempfile.mkdtemp(prefix='cimo-bf16-')
    native = os.environ.get('BENCH_GCC_FLAGS', '-O3 ' + ('-mcpu=native' if 'bf16' in feats else '-march=native')).split()
    for name, text, flags in (('timer', TIMER, ['-O2']), ('gcc', None, native)):
        c = os.path.join(tmp, name + '.c')
        if text:
            with open(c, 'w') as f:
                f.write(text)
        subprocess.check_call([os.environ.get('CC', 'gcc')] + flags + ['-shared', '-fPIC', '-o',
                              os.path.join(tmp, name + '.so'), c if text else src])
    timer = ctypes.CDLL(os.path.join(tmp, 'timer.so'))
    timer.time_dot.restype = timer.time_gemm.restype = ctypes.c_double
    timer.time_dot.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long] * 2 + [ctypes.c_void_p]
    timer.time_gemm.argtypes = [ctypes.c_void_p] + [ctypes.c_long] * 3 + [ctypes.c_void_p] * 3 + [ctypes.c_long]
    gcc = ctypes.CDLL(os.path.join(tmp, 'gcc.so'))
    os.environ['CIMO_CACHE_DIR'] = tmp
    with open(src) as f:
        mod = cimo.compile(f.read(), cache=False)
    rnd = random.Random(1)
    da, db = bf16s(DOT_N, rnd), bf16s(DOT_N, rnd)
    ga, gb = bf16s(GEMM_M * GEMM_K, rnd), bf16s(GEMM_K * GEMM_N, rnd)
    print('%-10s %10s %10s %7s  %s' % ('kernel', 'cimo ms', 'gcc ms', 'ratio', 'result'))
    for k in run:
        kind = KERNELS[k][1]
        addrs = {'cimo': getattr(mod, k).address, 'gcc': ctypes.cast(getattr(gcc, k), ctypes.c_void_p).value}
        best, res = {}, {}
        for t in range(a.n):
            for who in (('cimo', 'gcc') if t % 2 == 0 else ('gcc', 'cimo')):
                if kind == 'dot':
                    out = ctypes.c_float()
                    ms = timer.time_dot(addrs[who], da, db, DOT_N, DOT_REPS, ctypes.byref(out))
                    r = struct.pack('<f', out.value)
                else:
                    c = (ctypes.c_float * (GEMM_M * GEMM_N))()
                    ms = timer.time_gemm(addrs[who], GEMM_M, GEMM_N, GEMM_K, ga, gb, c, GEMM_REPS)
                    r = bytes(c)
                best[who] = min(best.get(who, ms), ms)
                if res.setdefault(who, r) != r:
                    sys.exit('%s: %s changed between runs' % (k, who))
        same = 'same bits' if res['cimo'] == res['gcc'] else 'DIFFERENT'
        print('%-10s %10.3f %10.3f %7.2f  %s' % (k, best['cimo'], best['gcc'], best['cimo'] / best['gcc'], same))
        if res['cimo'] != res['gcc']:
            sys.exit(1)


if __name__ == '__main__':
    main()
