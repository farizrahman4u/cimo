# Performance

cimo is an optimizing compiler. On loops it can analyze it runs close to gcc and clang at `-O3` with vectorization off: within a few percent overall on the [benchmarks](https://github.com/farizrahman4u/cimo/tree/main/python/bench), sometimes faster. Against pure Python, loops like the ones below run about 100x faster.

## What the optimizer does

- promotes locals and parameters to registers
- unrolls loops and unroll-and-jams loop nests
- strength-reduces array indexing into pointers that step forward
- replaces repeated loads with registers (scalar replacement)
- adds runtime checks where two pointers might overlap, with a fast path when they don't
- contracts `a * b + c` into fused multiply-adds (arm64, and x86-64 CPUs with FMA)
- on x86-64, compiles for the CPU it runs on: FMA, POPCNT, LZCNT and TZCNT where present (`CIMO_X86=baseline` for SSE2 only)
- auto-vectorizes straight counted loops, 128 bits at a time (arm64, x86-64), 256 on x86-64 CPUs with AVX2
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
$ cimo compile -S kernels.c          # assembly for the machine you run it on (arm64 or x86-64)
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

On arm64 and x86-64, straight counted loops over 4- and 8-byte elements are auto-vectorized, 128 bits at a time (256 bits on x86-64 CPUs with AVX2, `CIMO_X86=baseline` for 128), including integer sums, and arrays that might overlap are compared when the loop runs. So are conditions that choose a value, like `x > 0 ? x : 0` or a clamp, with the same result as the scalar loop for NaNs and signed zeros; on x86-64 only integer ones for now. A running integer maximum or minimum (`if (a[i] > m) m = a[i];`) vectorizes too, and integer reductions keep four vectors of accumulators, so they aren't held up waiting on one. A loop that steps by a stride argument (`a[i * st]`) gets a vectorized copy that runs when the stride is 1. Loops that leave as soon as they find something (`any`, `all`, the index of the first match, `a[i] < lo || a[i] != a[i]`) check a chunk of elements at a time; the scalar loop then finds the exact element, so the result is the same, and reading ahead never crosses into memory the loop wouldn't have touched. Floating-point sums keep C's order, adding the lanes one by one, so they vectorize only where the rest of the loop is worth it (a square root or a division on x86-64, a polynomial), not in a plain sum or dot product. Other code is scalar.

With vectorization on both sides, the geomean over the 15 benchmark kernels is 1.11 to 1.18x gcc `-O3`'s time on Neoverse N1, V1 and V2, and 1.05 to 1.08x on Ice Lake, Sapphire Rapids and Zen 4.

> [!PLANNED]
> SIMD: auto-vectorization of more loops (float `max`/`min`, `argmax`, float conditions on x86-64, 8- and 16-bit elements, outer loops), and on WebAssembly. This is the remaining gap to gcc and clang with vectorization on.
