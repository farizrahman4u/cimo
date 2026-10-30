# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Big

1. ~~**Single-file distribution.**~~ **Done:** `cimo bundle [-o cimo.py]`
   (`cimo/bundle.py`, `cimo/resources.py`; `.c`/`.h` stay real files). See "Single-file
   distribution" in `requirements.md`.
2. **Python interop API (M6).** Struct/union arguments and returns (ctypes structures, dicts),
   VLA shape checks at call time (README promises them). The CLI is done (`cimo run`,
   `compile`, `bundle`, `cache`, `completion`) except `cimo report kernels.c`: per-loop
   optimization remarks from the middle end (README promises it). zsh/fish completion
   scripts are untested.
3. **amd64 backend.** Linux x86-64 and Windows x64 (`VirtualAlloc` loader, Win64 ABI).
4. ~~**WASM backend + Pyodide loading.**~~ **Done:** `cimo.backend.wasm`, `cimo.wasm` (see "WASM
   backend" in `requirements.md`). Open: host JS libraries via `link=`, async loading for large
   modules on a browser main thread, wasm SIMD.
5. **SIMD.** GNU vector types (`vector_size`, currently a clear error), then auto-vectorization
   of the loops the middle end already analyzes. Closes the 1.58x gap to gcc `-O3`.
6. **Debug mode.** `debug=True`: bounds checks against Python buffers and heap blocks
   (`cimo.BoundsError`, as in the README), use-after-free, double free, leak report.
7. **macOS (Apple silicon).** Never run: `MAP_JIT` loader, `tpidrro_el0` contexts, Mach-O
   assembly, Apple ABI variant. Needs `python/tests` on an M-series Mac.
8. **SIGSEGV/SIGBUS in a kernel kills the process.** Turn faults inside module code into
   `KernelError` (signal handler that unwinds to the entry trampoline when the pc is in a
   module; chain to Python's handler otherwise).
9. **`_Complex` / `complex.h`.** Not in cproc or QBE. Needs: frontend lowering (Annex G
   `__muldc3`/`__divdc3` semantics), arm64 ABI (two FP registers), musl `src/complex` in the
   runtime, numpy `complex128` binding.

## Performance

- GNU bit builtins (`popcount`, `clz`, `ctz`, `bswap`, ...) are portable C calls; lower them to
  `cnt`/`clz`/`rbit`/`rev` like the math builtins.
- `edit_distance` 2.5x of gcc: needs `csel` (select) in the backend.
- `rgb2gray` 1.3x of gcc: constant materialization / `madd`.
- Runtime: per-module copy of what it uses (printf ~20 KB); fine for now.

## Small

- Demos (see "Demos (planned)" in `requirements.md`).
