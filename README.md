<div align="center">

# cimo

A C compiler written in pure Python.

[Live demo](https://www.cimo.dev/) · [Docs](https://www.cimo.dev/docs/)

</div>

## Usage

```python
import numpy as np
import cimo

k = cimo.compile(r"""
void blur(int h, int w, const float src[h][w], float dst[h][w]) {
    for (int i = 1; i < h - 1; i++)
        for (int j = 1; j < w - 1; j++)
            dst[i][j] = 0.2f * (src[i][j] + src[i-1][j] + src[i+1][j]
                                          + src[i][j-1] + src[i][j+1]);
}
""")

src = np.random.rand(4096, 4096).astype(np.float32)
dst = np.zeros_like(src)
k.blur(*src.shape, src, dst)   # native code, on the arrays in place
```

cimo parses the C, optimizes it, encodes arm64, x86-64 or WebAssembly itself and maps the result into
executable memory in-process. It needs no C compiler, SDK, LLVM or binary wheels. Arrays are passed
without copying, and compiled code releases the GIL.

## Install

```
pip install cimo
```

or download one file to put on `sys.path`:

```
wget https://github.com/farizrahman4u/cimo/releases/download/onefile/cimo.py
```

Requires CPython 3.9 or newer, or Pyodide. There are no dependencies outside the standard library;
numpy is only needed for your arrays.

## Motivation

Compute-graph engines, PDE solvers and symbolic code generators emit C at runtime and call the system
compiler on the user's machine. Where there is none, they fail or fall back to slow paths. Numba avoids
the compiler by bundling LLVM. cimo runs wherever Python does:

- **Libraries** can ship C kernels as source, with no wheel matrix and no compiler needed for a source
  install.
- **Desktop applications** can use native kernels without bundling LLVM or installing a C compiler on
  every build machine.
- **Browsers and restricted machines**: Pyodide, JupyterLite and marimo, plugin repositories that accept
  no binaries, and machines where nothing can be installed.

## Performance

cimo has a loop optimizer (unrolling, unroll-and-jam, tiling, scalar replacement, struct fields kept in registers,
strength reduction, runtime alias checks) and an instruction-selecting backend. On scalar code it runs close to gcc and clang at
`-O3`:

| Kernel   | vs `gcc -O3` (Linux arm64) | vs `clang -O3` (Apple M4) |
|:---------|---------------------------:|--------------------------:|
| saxpy    | 0.79x                      | 1.43x                     |
| gemm     | 0.89x                      | 1.11x                     |
| jacobi2d | 1.04x                      | 0.96x                     |
| conv3x3  | 1.04x                      | 1.07x                     |

<sub>cimo time ÷ reference time; lower is better. Scalar code on both sides, with FMA contraction off.
Results are bit-identical to the reference compiler.</sub>

On arm64 and x86-64, cimo also auto-vectorizes straight counted loops over 4- and 8-byte elements, 128 bits at a
time (256 with AVX2), conditions included (`x > 0 ? x : 0`; integer ones only on x86-64). Floating-point sums vectorize only where
the loop does enough else, since C fixes their order. Integer reductions use four vectors of accumulators, and a
running integer `max` or `min` vectorizes, with a fast path when a stride argument is 1 (`a[i * st]`), and so do
loops that leave early (`any`, `all`, finding the first match). Loops over bytes vectorize in 16-bit lanes when their
values fit (pixels: `rgb2gray`), and an outer loop vectorizes when its inner loop's sums must stay in order (`nbody_acc`).
Float `max`/`min`, `argmax`, 16-bit elements and WebAssembly stay scalar, so the
reference compilers are ahead where they vectorize those. Geomean over 15 kernels, against vectorized
`gcc -O3`: 0.83 to 0.86x on Neoverse N1, V1 and V2. On x86-64 cimo compiles for the CPU it runs on (FMA, POPCNT,
LZCNT, TZCNT, 256-bit AVX2): against `gcc -O3 -march=native`, 0.87x on Sapphire Rapids, 0.88x on Ice Lake and 0.95x on
Zen 4, from 1.16 to 1.41x with 128-bit vectors. See [`requirements.md`](requirements.md) for the full measurements.

Compiling a kernel file takes under a second, and compiled modules are cached by content hash.

## Language

cimo compiles C11 and bundles its own C library, so a kernel behaves the same on every platform,
including the browser.

- Structs, unions, bitfields, pointers of any depth, function pointers, recursion, varargs, VLAs, `goto`,
  `switch`, globals and `const` tables
- `stdio.h`, `stdlib.h`, `string.h`, `math.h`, `ctype.h`, `inttypes.h` and the freestanding headers.
  `printf` writes to `sys.stdout`, so output appears in notebooks
- `restrict`, used by the loop optimizer

To use the operating system, link the host libc or any other shared library (zlib, BLAS, the Python C
API):

```python
k = cimo.compile(src, link=["libc"], host_headers=True)
k = cimo.compile(src, link=["libz.so.1"])
```

Kernels run with the full privileges of the Python process, like ctypes. There is no sandbox.

Inline assembly, `setjmp`/`longjmp`, `long double` and GNU vector types are not supported. Each is a
compile error that says so. An empty `asm` (`asm volatile ("" ::: "memory")`, the usual optimization
barrier) compiles to nothing.

## Calling kernels

Every non-`static` function becomes a Python callable, typed from its C signature:

| C parameter                        | Pass from Python                        |
|:-----------------------------------|:----------------------------------------|
| `float *`, `const uint8_t *`, VLAs | numpy arrays, `bytearray`, `memoryview` |
| `int`, `double`, `size_t`          | Python numbers                          |
| `struct point *`                   | a ctypes structure                      |

Kernels can also live in a `.c` file:

```python
k = cimo.load("kernels.c")
```

### Threads

Kernels are safe to call concurrently. There is no launch API; use ordinary Python threads and give each
one a part of the work:

```c
void blur(int block, int nblocks, int h, int w, const float src[h][w], float dst[h][w]) {
    int rows = (h - 2 + nblocks - 1) / nblocks;
    int lo = 1 + block * rows;
    int hi = lo + rows < h - 1 ? lo + rows : h - 1;
    for (int i = lo; i < hi; i++)
        for (int j = 1; j < w - 1; j++)
            dst[i][j] = 0.2f * (src[i][j] + src[i-1][j] + src[i+1][j]
                                          + src[i][j-1] + src[i][j+1]);
}
```

```python
from concurrent.futures import ThreadPoolExecutor

n = 16
with ThreadPoolExecutor() as pool:
    list(pool.map(lambda b: k.blur(b, n, h, w, src, dst), range(n)))
```

Each thread gets its own part of the module's heap, so `malloc` doesn't take a lock.

### Errors

Errors in the C raise `cimo.CompileError`. `assert`, `abort()`, `exit()`, traps, stack overflow and
hardware faults raise `cimo.KernelError`, and the module stays usable:

```python
k = cimo.compile(r"""
#include <assert.h>

void check(int x) { assert(x > 0); }
""")

k.check(-1)
# cimo.KernelError: assertion failed: x > 0, at <string>:4 in check() (calling check)
```

```python
k = cimo.compile("int read(int *p) { return *p; }")

k.read(8)
# cimo.KernelError: memory fault at 0x8 in read() (calling read)
```

A write that lands on memory the process owns isn't a fault: it corrupts that memory, as in any C. So does
a kernel stopped halfway through a write.

## Command line

```console
$ cimo run life.c 64 64       # compile and run main(), with arguments
$ cimo compile -S kernel.c    # print the generated assembly
$ cimo onefile -o cimo.py     # write cimo as one file to vendor
```

## Browser

In Pyodide, JupyterLite and marimo, `cimo.compile` emits WebAssembly in the page and links it into
Pyodide's memory, with no compile server. Kernels work on numpy arrays in place. The
[live demo](https://www.cimo.dev/) runs this way, and recompiles as you edit the C.

## Platforms

| Platform             | Backend      | Loader                            |
|:---------------------|:-------------|:----------------------------------|
| Linux arm64 / x86-64 | arm64, amd64 | `mmap` + `mprotect` (W^X)         |
| macOS, Apple silicon | arm64        | `MAP_JIT`, W^X fallback           |
| Windows arm64 / x64  | arm64, amd64 | `VirtualAlloc` + `VirtualProtect` |
| Pyodide / browser    | WebAssembly  | shared-memory side module         |

## How it works

```
 C source ──▶ front end ──▶ loop optimizer ──▶ backend ──▶ encoder ──▶ executable memory
             (from cproc)     (cimo's own)     (from QBE)  (arm64, amd64,
                                                            WebAssembly)
```

The front end and backend are [cproc](https://sr.ht/~mcf/cproc/) and [QBE](https://c9x.me/compile/),
ported to Python and checked against the originals by diffing their IR pass by pass. The loop optimizer
and machine-code encoder are new, so no external assembler or linker is used. The compiler is
differential-tested against gcc on randomly generated programs.

## Status

Pre-release. Runs on Linux, macOS and Windows (arm64 and x86-64) and in the browser, and auto-vectorizes on
arm64 and x86-64. Differential-tested against gcc on random programs (Csmith, YARPGen) on both architectures.
Planned:

- Auto-vectorization of more loops (float `max`/`min`, `argmax`, 8- and 16-bit elements, outer loops), and on WebAssembly
- GNU vector types (`__attribute__((vector_size(16)))`)
- `debug=True`: bounds checks on every access, and detection of use-after-free, double free and leaks
- `cimo report`: which loops were optimized, and why the others weren't
- Checking at call time that an array matches a VLA parameter's shape
- Struct arguments by value, from a ctypes structure or a `dict`

## Repository

- [`python/cimo/`](python/cimo/): the compiler and Python runtime
- [`js/`](js/): browser glue
- [`docs/`](docs/): the website, [cimo.dev](https://www.cimo.dev/), and the documentation in
  [`docs/docs/`](docs/docs/)
- [`examples/`](examples/): kernels and demos
- [`requirements.md`](requirements.md): the design spec, measurements and roadmap

## License

[MIT](LICENSE). Bundled code from QBE (MIT), cproc (ISC) and musl (MIT) keeps its license.
