# Measuring and profiling

Before changing a kernel to make it faster, measure where the time goes.

## Time a kernel

Compile once, outside what you time, then take the best of several runs:

```python
import timeit

k = cimo.compile(src)   # not part of the measurement

per_call = min(timeit.repeat(lambda: k.blur(*src.shape, src, dst),
                             number=10, repeat=5)) / 10
```

- **Best of several:** the minimum is the most stable number; single runs vary by several percent, more on laptops on battery and in virtual machines.
- **No warm-up needed for the code:** cimo compiles before the first call, so the first call runs the same machine code as the hundredth. The first call can still be slower because the arrays' memory is touched for the first time.
- **Large enough work:** a call costs a few microseconds of Python and ctypes overhead. Time calls that do at least a millisecond of work, or pass more data per call.

## Compile time and run time

`cimo run --time` reports both, on stderr:

```console
$ cimo run --time program.c
compile 0.059 s, run 0.277 s
```

The first compile after installing also builds the parts of the C library the program uses, so time a second run for the real compile time. Unchanged code isn't compiled at all on later runs; see [the cache](files-and-cache.md#the-cache).

## Compare builds

Changing one thing at a time tells you what it's worth:

| Compare | How |
|---|---|
| with and without the loop optimizer | run with `CIMO_MIDDLE=0` |
| with and without fused multiply-add | run with `CIMO_FMA=0` |
| against numpy or pure Python | time the same work written both ways |
| against gcc or clang | the [benchmark harness](https://github.com/farizrahman4u/cimo/tree/main/python/bench) calls cimo and gcc builds of the same kernels through the same timing loops |

## Find the slow part

There is no per-line profile of a kernel yet. Two ways to narrow it down:

- **Split the kernel** into one function per phase, call them one at a time from Python, and time each call.
- **Shrink the input** to see how time grows with size: linear growth where you expect it, or a jump where the data stops fitting in cache.

## Read the generated code

```console
$ cimo compile -S kernels.c     # arm64 assembly
$ cimo compile --ir kernels.c   # the intermediate representation, before the backend
```

In the assembly of a hot loop, look for:

- **the loop body repeated** four times or so: it was unrolled
- **the same value loaded on every iteration:** the compiler couldn't prove two pointers don't overlap; mark them `restrict`
- **calls inside the loop:** each costs time, and blocks the optimizations around it

## System profilers

Whole-process tools such as `perf stat` work as for any Python program and report counters (instructions, cache misses, branch misses) for everything that ran, kernels included.

Sampling profilers such as `perf record` see the time spent in cimo's code, but can't name the functions: the code lives in memory cimo maps itself, without symbols.

> [!PLANNED]
> Symbols for profilers (a perf map and jitdump), so `perf` and other profilers name kernel functions. `cimo report`: which loops were optimized, and why the others weren't.
