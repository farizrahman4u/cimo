# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Big

1. **Single-file distribution.** `python -m cimo bundle [-o cimo.py]` writes one `cimo.py` with
   every module's source and the bundled `.h`/`.c` text embedded as strings, plus an import
   hook so `import cimo` (and `cimo.frontend` etc.) loads from it. Standard library only, no
   resource files. All header/runtime reads go through one resource function (files in the
   repo, the embedded table in the bundle). Open question: keep `.c`/`.h` as real files in the
   repo (preferred: editor tooling, musl diffs, the gcc reference builds in
   `tests/test_runtime.py`) or store them as `.py`.
   - `--prebuilt`: also embed the compiled runtime (assembly text, a few MB) so a fresh process
     skips the 8 s first build and needs no writable `~/.cache`. Without a writable cache dir,
     fall back to an in-memory cache.
2. **Python interop API (M6).** Struct/union arguments and returns (ctypes structures, dicts),
   VLA shape checks at call time (README promises them), a CLI (`python -m cimo prog.c`,
   `python -m cimo report kernels.c`).
3. **amd64 backend.** Linux x86-64 and Windows x64 (`VirtualAlloc` loader, Win64 ABI).
4. **WASM backend + Pyodide loading.** Side module linked into Pyodide's memory; numpy arrays
   in place.
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

## Bugs

- `glob.h` (with `system_headers=True`) needs `__SIZE_TYPE__`, which `cimo.preprocessor` does
  not predefine; check the other gcc predefined type macros too.
- `nullptr` is `(void *)0` in the driver (cproc's `nullptr_t` is unfinished); `_Generic` and
  `typeof` can't tell it apart.
- "long double is not supported" points at the end of the statement, not the
  sub-expression.

## Small

- Rename `system_headers=` to `host_headers=` (docs and errors already say "host").
- README: "Out of scope: stdio, files, threads" is stale since `link=`; update the freestanding
  section.
- Frontend: drop unreferenced `static` functions and data.
- `float_t`/`FLT_EVAL_METHOD` consistency (cimo evaluates float in float; the runtime is built
  with `__FLT_EVAL_METHOD__` 0).
- Demos (see "Demos (planned)" in `requirements.md`).
