"""cimo's PTX against nvcc's, on this machine's NVIDIA GPU: each kernel in tests/gpu/ compiled by cimo (PTX, compiled by
the driver), by nvcc to PTX (compiled by the driver) and by nvcc to a cubin (NVIDIA's bundled compiler); results
compared, and the GPU time of each, the source-to-PTX time and the driver's load time printed.

    CUDA_CACHE_DISABLE=1 python tests/gpu_bench.py [--nvcc /usr/local/cuda/bin/nvcc] [--reps 50]

nvcc is the oracle here only; cimo never uses it. The numbers go to requirements.md ("PTX: measured").
"""

import argparse
import array
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cimo  # noqa: E402
import cimo.gpu as G  # noqa: E402

KDIR = os.path.join(HERE, 'gpu')


def bf(v):
    b = struct.unpack('<I', struct.pack('<f', v))[0]
    return (b + 0x7fff + ((b >> 16) & 1)) >> 16


def fb(b):
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]


def cases(gpu):
    """(file, kernel, inputs maker, launch, output index, reference check)"""
    rnd = random.Random(5)
    out = []
    n = 1 << 24
    x = array.array('f', (rnd.random() for _ in range(n)))
    y = array.array('f', (rnd.random() for _ in range(n)))

    def saxpy_args():
        return [2.5, gpu.array(x), gpu.array(y), n]
    out.append(('saxpy.c', 'saxpy', saxpy_args, dict(grid=(n // 256,), block=(256,)), 2,
                lambda r: all(abs(r[i] - (2.5 * x[i] + y[i])) <= 1e-6 * abs(r[i]) + 1e-7 for i in range(0, n, 9973))))
    N = 2048
    A = array.array('f', (rnd.uniform(-1, 1) for _ in range(N * N)))
    B = array.array('f', (rnd.uniform(-1, 1) for _ in range(N * N)))

    def mm_args():
        return [gpu.array(A), gpu.array(B), gpu.zeros(4 * N * N), N]

    def mm_check(r):
        for i in range(0, N, 331):
            for j in range(0, N, 457):
                s = sum(A[i * N + t] * B[t * N + j] for t in range(N))
                if abs(s - r[i * N + j]) > 1e-3:
                    return False
        return True
    out.append(('matmul_tiled.c', 'matmul', mm_args, dict(grid=(N // 64, N // 64), block=(16, 16)), 2, mm_check))
    M = K = Nn = 2048
    Ab = array.array('H', (bf(rnd.uniform(-1, 1)) for _ in range(M * K)))
    Bt = array.array('H', (bf(rnd.uniform(-1, 1)) for _ in range(Nn * K)))

    def mma_args():
        return [gpu.array(Ab), gpu.array(Bt), gpu.zeros(4 * M * Nn), M, Nn, K]

    def mma_check(r):
        for i in range(0, M, 263):
            for j in range(0, Nn, 389):
                s = sum(fb(Ab[i * K + t]) * fb(Bt[j * K + t]) for t in range(K))
                if abs(s - r[i * Nn + j]) > 1e-3:
                    return False
        return True
    out.append(('gemm_mma.c', 'gemm_mma', mma_args, dict(grid=(Nn // 32, M // 16), block=(128,)), 2, mma_check))
    Mg = Ng = Kg = 2048
    Ag = array.array('H', (bf(rnd.uniform(-1, 1)) for _ in range(Mg * Kg)))
    Bg = array.array('H', (bf(rnd.uniform(-1, 1)) for _ in range(Ng * Kg)))
    bias = array.array('f', (rnd.uniform(-1, 1) for _ in range(Ng)))

    def gelu(x):
        import math
        return 0.5 * x * (1 + math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def gg_args():
        return [gpu.array(Ag), gpu.array(Bg), gpu.array(bias), gpu.zeros(4 * Mg * Ng), Mg, Ng, Kg]

    def gg_check(r):
        for i in range(0, Mg, 271):
            for j in range(0, Ng, 397):
                s = sum(fb(Ag[i * Kg + t]) * fb(Bg[j * Kg + t]) for t in range(Kg)) + bias[j]
                if abs(gelu(s) - r[i * Ng + j]) > 1e-3:
                    return False
        return True
    out.append(('gemm_gelu.c', 'gemm_gelu', gg_args, dict(grid=(Ng // 64, Mg // 64), block=(128,)), 3, gg_check))
    rows, cols = 4096, 4096
    X = array.array('H', (bf(rnd.uniform(-8, 8)) for _ in range(rows * cols)))

    def sm_args():
        return [gpu.array(X), gpu.zeros(4 * rows * cols), cols]

    def sm_check(r):
        import math
        for i in range(0, rows, 977):
            row = [fb(X[i * cols + j]) for j in range(cols)]
            m = max(row)
            s = sum(math.exp(v - m) for v in row)
            for j in range(0, cols, 101):
                w = math.exp(row[j] - m) / s
                if abs(r[i * cols + j] - w) > 1e-6 * w + 1e-12:
                    return False
        return True
    out.append(('softmax.c', 'softmax', sm_args, dict(grid=(rows,), block=(256,)), 1, sm_check))
    return out


def nvcc_build(nvcc, path, sm, kind):
    d = tempfile.mkdtemp(prefix='fariz-nvcc-')
    o = os.path.join(d, 'k.' + kind)
    subprocess.check_call([nvcc, '-arch=sm_%d' % sm, '-O3', '-' + kind, '-I', os.path.join(KDIR, 'nvcc'), '-x', 'cu',
                           path, '-o', o])
    with open(o, 'rb') as f:
        data = f.read()
    shutil.rmtree(d)
    return data.decode() if kind == 'ptx' else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nvcc', default=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--only')
    a = ap.parse_args()
    gpu = G.device()
    print('%s, sm_%d, CUDA driver %d.%d; CUDA_CACHE_DISABLE=%s' % ((gpu.name, gpu.sm) + G.driver_version()
                                                                   + (os.environ.get('CUDA_CACHE_DISABLE'),)))
    print('%-14s %9s %9s %9s %8s %8s %8s %6s' % ('kernel', 'cimo ms', 'nvccPTX', 'cubin', 'ptx s', 'load ms',
                                                'nvload', 'same'))
    for fname, kname, mk, launch, oi, check in cases(gpu):
        if a.only and a.only not in fname:
            continue
        path = os.path.join(KDIR, fname)
        src = open(path).read()
        t0 = time.perf_counter()
        text = G._ptx1(src, gpu.sm, (), (), "<string>", None, None)[0]
        tptx = time.perf_counter() - t0
        mod = gpu.compile(src, cache=False)
        sigs = mod.signatures
        mods = {'cimo': mod}
        try:
            mods['nvptx'] = gpu.load(nvcc_build(a.nvcc, path, gpu.sm, 'ptx'), sigs)
            mods['cubin'] = gpu.load(nvcc_build(a.nvcc, path, gpu.sm, 'cubin'), sigs)
        except (OSError, subprocess.CalledProcessError) as e:
            print('  nvcc: %s' % e)
        times, results, argss = {}, {}, {}
        for name, m in mods.items():
            args = argss[name] = mk()
            k = getattr(m, kname)
            k(*args, **launch)
            gpu.sync()
            results[name] = array.array('f', args[oi].to_host())
        for _ in range(a.rounds):  # interleaved, the best of each: clocks change as the GPU warms
            for name, m in mods.items():
                k, args = getattr(m, kname), argss[name]
                t = gpu.time(lambda: k(*args, **launch), a.reps)
                times[name] = min(times.get(name, t), t)
        ok = check(results['cimo'])
        same = all(results[n] == results['cimo'] for n in results)
        close = all(max(abs(p - q) for p, q in zip(results[n][::97], results['cimo'][::97])) < 1e-3 for n in results)
        print('%-14s %9.3f %9s %9s %8.3f %8.1f %8s %6s%s' % (
            kname, times['cimo'], '%.3f' % times['nvptx'] if 'nvptx' in times else '-',
            '%.3f' % times['cubin'] if 'cubin' in times else '-', tptx, mod.load_time,
            '%.1f' % mods['nvptx'].load_time if 'nvptx' in mods else '-', 'bits' if same else 'close' if close else 'NO',
            '' if ok else '  WRONG vs reference'))
        print('    %d lines of C, %d lines of PTX; registers: %s' % (
            len(src.splitlines()), len(text.splitlines()),
            ', '.join('%s %d' % (n, getattr(m, kname).registers) for n, m in mods.items())))


if __name__ == '__main__':
    main()
