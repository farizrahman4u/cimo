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

aarch64 WSL2, 8 cores, gcc 13.3, CPython 3.13, measured 2026-10-30. Times in ms, best of 5.
Ratio = cimo time / gcc time; lower is better.

| Kernel | cimo | nofma | novec | O3 | /nofma | /novec | /O3 | check |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|
| saxpy | 45.3 | 71.0 | 71.6 | 18.5 | 0.64x | 0.63x | 2.46x | = |
| gemm | 11.0 | 13.9 | 12.6 | 4.1 | 0.79x | 0.88x | 2.67x | =f |
| jacobi2d | 57.3 | 53.2 | 54.0 | 31.9 | 1.08x | 1.06x | 1.80x | = |
| conv3x3 | 19.8 | 32.8 | 17.5 | 13.6 | 0.60x | 1.14x | 1.46x | = |
| rgb2gray | 34.4 | 27.0 | 27.5 | 3.5 | 1.27x | 1.25x | 9.89x | = |
| dot | 278.3 | 209.9 | 276.1 | 275.8 | 1.33x | 1.01x | 1.01x | = |
| isum | 70.3 | 83.2 | 71.9 | 64.1 | 0.84x | 0.98x | 1.10x | = |
| prefix_sum | 210.4 | 209.8 | 209.8 | 208.9 | 1.00x | 1.00x | 1.01x | = |
| iir | 837.0 | 701.0 | 838.0 | 839.1 | 1.19x | 1.00x | 1.00x | =f |
| edit_distance | 93.6 | 37.2 | 37.3 | 36.9 | 2.52x | 2.51x | 2.54x | = |
| histogram | 36.1 | 38.0 | 38.0 | 38.1 | 0.95x | 0.95x | 0.95x | = |
| mandelbrot | 70.4 | 78.1 | 74.4 | 74.8 | 0.90x | 0.95x | 0.94x | = |
| nbody_acc | 7.3 | 9.9 | 7.5 | 4.4 | 0.74x | 0.98x | 1.66x | =f |
| flood_fill | 28.2 | 23.6 | 23.8 | 24.1 | 1.20x | 1.18x | 1.17x | = |
| sieve | 130.1 | 119.8 | 120.0 | 117.3 | 1.09x | 1.08x | 1.11x | = |
| **geomean** | | | | | **1.01x** | **1.06x** | **1.58x** | |

- Scalar code is at parity with gcc overall. The gap to `gcc -O3` is vectorization (saxpy, gemm,
  rgb2gray), which cimo does not do yet.
- `edit_distance` needs a conditional select in the backend; `rgb2gray` needs better constant
  materialization. Both are in [`TODO.md`](../../TODO.md).
