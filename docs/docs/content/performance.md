# Performance

cimo is an optimizing compiler. On loops it can analyze it runs close to gcc and clang at `-O3` with vectorization off: within a few percent overall on the [benchmarks](https://github.com/farizrahman4u/cimo/tree/main/python/bench), sometimes faster. Against pure Python, loops like the ones below run about 100x faster.

## What the optimizer does

- promotes locals and parameters to registers
- unrolls loops and unroll-and-jams loop nests
- strength-reduces array indexing into pointers that step forward
- replaces repeated loads with registers (scalar replacement)
- adds runtime checks where two pointers might overlap, with a fast path when they don't
- contracts `a * b + c` into fused multiply-adds (arm64)
- selects arm64 or x86-64 instructions with their addressing modes; on arm64 it also pairs adjacent loads and stores

## Write loops it can see through

- **Count with a simple index.** `for (long i = 0; i < n; i++)` over arrays indexed by `i` is what it optimizes best.
- **Mark pointers that don't overlap with `restrict`.** It saves the runtime overlap check:

  ```c
  void saxpy(long n, float a, const float *restrict x, float *restrict y) {
      for (long i = 0; i < n; i++)
          y[i] = a * x[i] + y[i];
  }
  ```

- **Write small helpers as functions.** Small functions in the same source are inlined, so they cost the same as a macro. A call to a large function, a recursive one, or one taking a struct by value stays a call.
- **Prefer `float` when the data is `float32`.** Converting each element to `double` and back costs time.

## Look at the generated code

```console
$ cimo compile -S kernels.c          # assembly for this machine (arm64 or x86-64)
$ cimo compile --ir kernels.c        # the intermediate representation (QBE IR)
```

> [!PLANNED]
> `cimo report kernels.c`: which loops were optimized, and why the others weren't.

To find out where the time goes before changing anything, see [Measuring and profiling](profiling.md).

## Compile time

Compiling a kernel file takes well under a second. Results are cached by a hash of the preprocessed source, so the second run of the same code starts immediately; see [Files, includes and the cache](files-and-cache.md).

## Switches

| Setting | Effect |
|---|---|
| `CIMO_MIDDLE=0` | turns off the loop optimizer (to compare, or to rule it out) |
| `CIMO_FMA=0` | no fused multiply-add contraction (for one compile: `cimo.compile(src, fma=False)`) |
| `CIMO_INLINE=<n>` | largest function inlined, in IR instructions (default 40, `0` turns it off) |

> [!PLANNED]
> SIMD: auto-vectorization of the loops the optimizer analyzes. This is the remaining gap to gcc and clang with vectorization on.
