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
   module; chain to Python's handler otherwise). On Windows an access violation ends the
   process silently (exit 5); a vectored exception handler that resumes at the unwind glue
   would do the same job there.
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

- **Large struct copies** are unrolled (8 bytes per load/store pair, as QBE does): a 64 KB
  argument takes 4.4 s to compile and grows faster than linearly (gcc `pr28982b.c`, 256 KB,
  takes minutes). Emit a loop or a `memcpy` call above a size threshold.
- **`volatile` stores** are rejected (cproc: "not yet supported"; QBE has no volatile
  operations): 34 torture programs. Emit them as ordinary stores the optimizer may not remove.
- **Backend speed on huge functions:** a Csmith function with a 125 KB initializer compiles in
  18 s (spill, liveness bitsets, `filluse`); C QBE takes well under a second.
- `fn.tmp[rval(r)]` is evaluated before checking `rtype(r) == RTmp` in places (C computes the
  pointer without reading it); one caused an `IndexError` (`parse.typecheck`). Audit the rest.
- GNU overflow builtins (`__builtin_add_overflow`, ...) and `__builtin_isunordered` & co.

- Profiler symbols: write a perf map (`/tmp/perf-<pid>.map`) and/or jitdump for loaded modules, so
  `perf record` names kernel functions (the docs' profiling page marks it Planned).

- Demos (see "Demos (planned)" in `requirements.md`).
- Website demo (`docs/demo.js`): C vector field → particles, recompiled in Pyodide on every edit.
  `docs/cimo.py` is a `cimo bundle` build: regenerate it when the compiler changes.
- Website training demo (`docs/nn.js`, `docs/nn/codegen.py`): a drawn network → JSON → C
  (forward, backward, Adam) → cimo → WASM, trained in the page. `docs/runtime.js` shares Pyodide.
- Website Architecture section (`docs/arch.js`): real output of each compiler stage for one kernel,
  from `docs/arch/gen.py`; rerun it when the compiler's output changes.
