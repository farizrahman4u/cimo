<div align="center">

# cimo

**A C compiler written in pure Python.** Write a C kernel as a string, call it on your numpy arrays
a few hundred milliseconds later, and it runs at `-O3` speed.<br>
No compiler, no SDK, no wheels, no build step. In the browser too.

```
pip install cimo
```

</div>

---

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
k.blur(*src.shape, src, dst)   # native code, running on your arrays in place
```

That's it. No `setup.py`, no `gcc`, no Xcode, no MSVC, no LLVM wheel. cimo parses the C, optimizes
it, encodes arm64 / x86-64 / WebAssembly instructions itself, maps them into executable memory, and
hands you a Python callable. Your numpy data is never copied.

## Why

Shipping native code with Python is still a mess. Cython, cffi and C extensions need a toolchain
on the user's machine. Numba needs LLVM wheels. And some places allow only `.py` files:

- **Pyodide, JupyterLite and marimo** in the browser have no compiler at all.
- **The QGIS plugin repository** bans compiled binaries.
- **Locked-down corporate machines** and hosted Python sandboxes don't let you install anything.

cimo is ordinary Python, standard library only. If Python runs, your kernels run.

## Fast where it matters

cimo isn't a toy interpreter or a TCC-style one-pass compiler. Its loop optimizer handles
unrolling, unroll-and-jam, scalar replacement, strength reduction and runtime alias checks, and
its backend does real instruction selection. On analyzable loops it matches gcc and clang at
`-O3`:

| Kernel   | vs `gcc -O3` (Linux arm64) | vs `clang -O3` (Apple M4) |
|----------|:--------------------------:|:-------------------------:|
| saxpy    | **0.79x**                  | 1.43x                     |
| gemm     | **0.89x**                  | 1.11x                     |
| jacobi2d | **1.04x**                  | **0.96x**                 |
| conv3x3  | **1.04x**                  | **1.07x**                 |

<sub>cimo time ÷ reference time; lower is better. Scalar code with FMA contraction off on both sides.
Results are bit-identical to the reference compiler.</sub>

Compiling a whole kernel file takes well under a second, and results are cached by content
hash, so the second run starts instantly.

About **100x faster than pure Python** on the loops numpy can't express: stencils, flood fills,
dynamic programming.

## Real C, not a subset

cimo compiles **freestanding C11**: all of the language, none of the operating system.

- structs, unions, bitfields, pointers of any depth, function pointers, recursion, varargs,
  VLAs, `goto`, `switch`, globals and `const` tables
- the freestanding headers plus `math.h`, `string.h`, `ctype.h` and an OS-free `stdlib.h` with
  `malloc`, `free`, `qsort`, `bsearch` and `strtod`, all bundled
- GNU vector types (`__attribute__((vector_size(32)))`) for hand-written NEON / AVX2 / AVX-512
- `#pragma omp simd` and `restrict`, honored as written

Out of scope: stdio, files, threads and anything else that needs an operating system.

## Pythonic bindings

Every non-`static` function becomes a Python callable, typed from its C signature:

| C parameter                         | Pass from Python                                 |
|-------------------------------------|--------------------------------------------------|
| `float *`, `const uint8_t *`, VLAs  | numpy arrays, `bytearray`, `memoryview`          |
| `int`, `double`, `size_t`           | Python numbers                                   |
| `struct point`                      | a ctypes structure or a `dict`                   |

Keep kernels in a `.c` file next to your module if you prefer:

```python
k = cimo.load("kernels.c")
```

### Multicore with plain Python threads

Kernels release the GIL and are safe to call concurrently. No launch API; bring your own threads:

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

`malloc` is lock-free: each thread gets its own heap.

### Errors are Python exceptions

`assert`, `abort()` and traps unwind cleanly back to Python. A crashing kernel doesn't take
your interpreter down with it.

```python
k = cimo.compile(src, debug=True)
k.blur(h, w, src, dst[:10])
# cimo.BoundsError: blur: write to dst[10][1] is out of bounds (buffer is 10 x 4096)
#   at kernels.c:5
```

Debug builds bounds-check every access, against both Python buffers and heap allocations, and catch
use-after-free, double free and leaks.

### Know why a loop is slow

```console
$ python -m cimo report kernels.c
kernels.c:12  loop over j: optimized (unrolled x4, alias check on src/dst)
kernels.c:31  loop over i: not optimized: base of `p` depends on a runtime branch
```

## Runs in the browser

The same compiler emits WebAssembly. In Pyodide, JupyterLite or marimo:

```python
import micropip; await micropip.install("cimo")
import cimo
k = cimo.compile(src)   # compiled in the page, linked into Pyodide's own memory
```

Kernels share Pyodide's memory and work on numpy arrays **in place**. No server, no 50 MB
clang-in-WASM download.

## Platforms

| Platform                    | Backend        | Loader                          |
|-----------------------------|----------------|---------------------------------|
| Linux arm64 / x86-64        | arm64, amd64   | `mmap` + `mprotect` (W^X)       |
| macOS (Apple silicon)       | arm64          | `MAP_JIT`, W^X fallback         |
| Windows x64                 | amd64          | `VirtualAlloc` + `VirtualProtect` |
| Pyodide / browser           | WebAssembly    | shared-memory side module       |

CPython 3.9+ and Pyodide. Zero runtime dependencies.

## How it works

```
 C source ──▶ front end ──▶ loop optimizer ──▶ backend ──▶ encoder ──▶ executable memory
             (from cproc)     (cimo's own)     (from QBE)  (arm64, amd64,   (JIT, or ELF /
                                                            WASM)           Mach-O / PE)
```

cimo stands on two excellent small compilers: [cproc](https://sr.ht/~mcf/cproc/) for the C front
end and [QBE](https://c9x.me/compile/) for the backend, ported to Python and checked against the
originals by diffing their IR output pass by pass. On top of that sits cimo's own loop optimizer,
which is what closes the gap to gcc, plus a machine-code encoder, so no external assembler or
linker is ever needed. Every release is differential-tested against gcc and clang on generated
random programs.

## Repository

- [`python/cimo/`](python/cimo/): the compiler and Python runtime
- [`js/`](js/): browser glue
- [`docs/`](docs/): documentation
- [`examples/`](examples/): kernels and demos
- [`requirements.md`](requirements.md): the design spec, measurements and roadmap

## License

[MIT](LICENSE). Bundled code from QBE (MIT), cproc (ISC) and musl (MIT) keeps its original license.
