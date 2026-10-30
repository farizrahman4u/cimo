# QBE baseline vs gcc

**Current state:** see "Architecture check" in `../pure_python_cc.md` for the summary. This
directory reproduces it:

- `gen_opt.py`: hand-optimized QBE IR for saxpy, gemm, jacobi2d, conv3x3 (the `qbe-opt` config).
- `qbe-arm64-isel.patch`: arm64 backend fixes. The QBE clone at `~/code/qbe` (branch
  `cimo-arm64-isel`) already has it applied in its working tree. On a fresh clone:
  `git apply <path>/qbe-arm64-isel.patch && make`. Both the `qbe` and `qbe-opt` configs use
  whatever `~/code/qbe/qbe` is built (override with `CIMO_QBE_DIR` / `CIMO_CPROC_DIR`).
- `edge.c`: bit-exact check of the 4 optimized kernels against plain C, sizes 0-21.
  `gcc -O0 -ffp-contract=off edge.c build/kernels-qbe-opt.o -lm -o build/edge && build/edge`
- Timings drift by about 10% between runs on WSL. Compare configs by interleaving runs and
  taking the best of several, not from a single `run.py` table.

The table below is the original **stock QBE** baseline.

Measures how far cproc + QBE (the candidate backend to port to Python) is from gcc on kernel-style C.

```
git clone git://c9x.me/qbe.git ~/code/qbe && make -C ~/code/qbe
git clone https://git.sr.ht/~mcf/cproc ~/code/cproc && (cd ~/code/cproc && ./configure && make)
python3 run.py
```

`kernels.c` is compiled by each config; `harness.c` (always gcc -O2) times best-of-5 and checksums against gcc -O3.

## Results (2026-10-30, aarch64 WSL2, gcc 13.3, QBE e786f06, cproc d1c53dd)

| kernel | gcc-O2 | gcc-O3 novec | gcc-O3 | qbe | qbe / O3-novec | qbe / O3 |
|---|---|---|---|---|---|---|
| saxpy | 71.2 | 70.9 | 18.1 | 132.9 | 1.87x | 7.33x |
| gemm | 21.1 | 12.7 | 4.1 | 55.4 | 4.36x | 13.37x |
| jacobi2d | 74.2 | 53.5 | 32.3 | 230.1 | 4.30x | 7.12x |
| conv3x3 | 109.0 | 19.1 | 17.4 | 350.4 | 18.38x | 20.15x |
| rgb2gray | 27.6 | 27.0 | 3.4 | 93.0 | 3.45x | 27.29x |
| dot | 274.4 | 275.0 | 275.0 | 209.5 | 0.76x | 0.76x |
| isum | 70.9 | 76.4 | 63.9 | 153.0 | 2.00x | 2.40x |
| prefix_sum | 208.2 | 208.1 | 208.1 | 209.1 | 1.01x | 1.00x |
| iir | 832.1 | 830.4 | 833.8 | 698.6 | 0.84x | 0.84x |
| edit_distance | 37.1 | 36.9 | 37.0 | 150.9 | 4.09x | 4.08x |
| histogram | 37.6 | 37.4 | 37.5 | 65.8 | 1.76x | 1.76x |
| mandelbrot | 73.6 | 73.7 | 73.7 | 80.9 | 1.10x | 1.10x |
| nbody_acc | 7.4 | 7.5 | 4.4 | 18.7 | 2.50x | 4.25x |
| flood_fill | 23.7 | 23.5 | 23.9 | 30.2 | 1.28x | 1.27x |
| sieve | 120.2 | 122.8 | 128.0 | 157.7 | 1.28x | 1.23x |

All times in ms; all checksums match. Compile time for the whole file: cproc 3 ms + QBE 9 ms (native C).

## Observations

- QBE wins on `dot` and `iir`: gcc contracts `s += x*y` into `fmadd`, which lengthens the loop-carried
  dependency chain (FMA latency > FADD latency). Confirmed: with `-ffp-contract=off` gcc -O3 gives
  dot 207.5 ms and iir 695.9 ms, i.e. parity with QBE. So QBE ties gcc there, it doesn't beat it.
- QBE's arm64 inner loops lack: immediate operands (`mov x5,#4; mul`), reg+reg<<shift addressing,
  induction-variable strength reduction, loop rotation, FMA selection. See `saxpy` asm: 15 instrs/iter vs gcc's 6.
- `conv3x3` (18x) and `gemm`/`jacobi2d` (~4x) need loop unrolling, LICM of loads, and address
  strength reduction: exactly the structured mid-end planned in `../pure_python_cc.md`.
- Latency-bound kernels (`prefix_sum`, `mandelbrot`, `flood_fill`, `sieve`) are already within ~1.3x.
- Only the arm64 backend was measured; the amd64 isel is larger (3.5k vs 2k lines) and may fare better.
