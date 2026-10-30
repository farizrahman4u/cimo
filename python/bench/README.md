# Benchmarks

cimo against gcc on 15 kernels: vectorizable loops, reductions, and sequential or branchy code.
The kernels are plain C11 in [`scratch/qbe_eval/kernels.c`](../../scratch/qbe_eval/kernels.c).

```console
$ python3 python/bench/run.py            # all 15 kernels, best of 5
$ python3 python/bench/run.py gemm iir   # a subset
$ python3 python/bench/run.py -n 9       # more trials
```

Needs Linux arm64 and gcc (`CC` picks another compiler). `run.py --help` lists the options,
including cimo builds with chosen backend extensions (`--cimo NAME=FLAGS`).

## Method

- Every configuration is called in-process through the same function pointers and timing loops
  (`bench.c`, built once with `gcc -O2`). The gcc builds are shared libraries loaded with ctypes;
  cimo's are `cimo.compile(kernels.c)` with an empty cache.
- Configurations run interleaved, in rotating order, and the best trial counts. Single runs drift
  by about 10% on WSL, so compare ratios, not one table.
- Every result is checksummed against gcc. `=` matches `gcc -O3 -fno-tree-vectorize
  -ffp-contract=off` exactly; `=f` matches the same build with FMA contraction.

| Column | Build |
|---|---|
| `nofma` | `gcc -O3 -mcpu=native -fno-tree-vectorize -ffp-contract=off` (scalar, like for like) |
| `novec` | `gcc -O3 -mcpu=native -fno-tree-vectorize` (scalar, with FMA contraction) |
| `O3` | `gcc -O3 -mcpu=native` (vectorized) |

## Results

aarch64 WSL2 on mains power, 8 cores, gcc 13.3, CPython 3.13, measured 2026-10-30. Times in ms,
best of 7. Ratio = cimo time / gcc time; lower is better. cimo as `cimo.compile` builds it by
default (FMA contraction on).

| Kernel | cimo | nofma | novec | O3 | /nofma | /novec | /O3 | check |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| saxpy | 45.9 | 71.4 | 71.2 | 18.2 | 0.64x | 0.64x | 2.52x | = |
| gemm | 11.0 | 13.9 | 12.6 | 4.1 | 0.79x | 0.87x | 2.66x | =f |
| jacobi2d | 56.4 | 52.4 | 52.7 | 30.4 | 1.08x | 1.07x | 1.85x | = |
| conv3x3 | 19.6 | 32.4 | 17.0 | 13.6 | 0.60x | 1.15x | 1.44x | = |
| rgb2gray | 34.0 | 27.0 | 27.0 | 3.4 | 1.26x | 1.26x | 10.05x | = |
| dot | 274.0 | 207.8 | 274.3 | 274.7 | 1.32x | 1.00x | 1.00x | = |
| isum | 70.3 | 83.2 | 71.8 | 63.9 | 0.84x | 0.98x | 1.10x | = |
| prefix_sum | 209.0 | 209.3 | 208.2 | 208.2 | 1.00x | 1.00x | 1.00x | = |
| iir | 836.3 | 702.1 | 837.0 | 837.8 | 1.19x | 1.00x | 1.00x | =f |
| edit_distance | 93.4 | 37.1 | 37.1 | 36.9 | 2.52x | 2.52x | 2.53x | = |
| histogram | 36.1 | 37.7 | 37.7 | 37.7 | 0.96x | 0.96x | 0.96x | = |
| mandelbrot | 70.3 | 77.8 | 73.9 | 74.3 | 0.90x | 0.95x | 0.95x | = |
| nbody_acc | 7.3 | 9.9 | 7.4 | 4.4 | 0.74x | 0.99x | 1.66x | =f |
| flood_fill | 28.2 | 23.5 | 23.8 | 23.7 | 1.20x | 1.18x | 1.19x | = |
| sieve | 131.8 | 120.1 | 126.8 | 119.9 | 1.10x | 1.04x | 1.10x | = |
| **geomean** | | | | | **1.01x** | **1.06x** | **1.58x** | |

With FMA contraction off on both sides (`CIMO_FMA=0` against `nofma`), the loop kernels the
project README quotes: saxpy 0.76x, gemm 0.85x, jacobi2d 1.10x, conv3x3 1.06x (geomean 0.93x).

- Scalar code is at parity with gcc overall. The gap to `gcc -O3` is vectorization (saxpy, gemm,
  rgb2gray), which cimo does not do yet.
- `edit_distance` needs a conditional select in the backend; `rgb2gray` needs better constant
  materialization. Both are in [`TODO.md`](../../TODO.md).
