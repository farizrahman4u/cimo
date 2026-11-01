# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Big

1. ~~**Single-file distribution.**~~ **Done:** `cimo onefile [-o cimo.py]`
   (`cimo/onefile.py`, `cimo/resources.py`; `.c`/`.h` stay real files). See "Single-file
   distribution" in `requirements.md`.
2. **Python interop API (M6).** Struct/union arguments and returns (ctypes structures, dicts),
   VLA shape checks at call time (README promises them). The CLI is done (`cimo run`,
   `compile`, `onefile`, `cache`, `completion`) except `cimo report kernels.c`: per-loop
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

- **A public way to get a module's WebAssembly** (requested by scramjit): its site demo shows the
  compiled kernel's disassembly and calls `cimo.wasm.build` for the bytes, as cimo's own demo does,
  which isn't documented API. For example `module.wasm` (bytes, in Pyodide) and making
  `docs/nn/wat.py` a supported `cimo.wat`. **When done, note it in scramjit's `TODO.md`.**

- **Slower than gcc -O3 on scramjit's kernels** (requested by scramjit, 2026-11-01; arm64 WSL,
  gcc 13.3 `-O3 -ffp-contract=off`, cimo with `CIMO_FMA=0`; reproduce with scramjit's
  `python3 bench/gcc.py`, which compiles the same C both ways). On the same generated C:
  gray_scott 0.22 vs 0.17 ms (1.3x), flood_fill 3.60 vs 1.93 ms (1.9x), viewshed 7.84 vs 3.43 ms
  (2.3x); on the hand-written C of scramjit's `kernels/c/`: 1.2x, 1.3x and 2.5x. Pinned down so far:
  - ~~`rint` is a library call~~ **Done:** `rint`/`nearbyint` are `frintx`/`frinti` (wasm
    `nearest`): 2.5 to 0.44 ns per element, the same as `floor`.
  - The bundled `hypot` is 1.5x glibc's (10.3 vs 6.8 ns).
  - No common subexpression elimination: an expression repeated in the source is computed each
    time (scramjit's `x % n` macro computed `a % b` three times). scramjit now writes each
    division once, but flood_fill (branchy, many small blocks, bounds checks) is still 1.9x.
  **When fixed, add a note to scramjit's `TODO.md`** so it can drop the workarounds listed there.

- ~~**Inline small functions**~~ **Done** (`backend/inline.py`): helpers now run as fast as macros
  (`ema` 5.12 to 3.33 ns, a wrapping-add sum 2.05 to 0.34 ns). Costs 13-38% compile time and about
  25-30% more code on `t_printf` and `t_math`. Open: drop a static function once every call is
  inlined (its definition is still compiled and emitted); honor `noinline`.
- **FMA contraction per compile** (requested by scramjit): `cimo.compile(..., fma=False)`, like the
  CLI's `--no-fma`; today only the process-wide `CIMO_FMA`, read at import. **When done, note it
  in scramjit's `TODO.md`** so it can drop setting `CIMO_FMA` and its warning.
- **Linking the host libm for math calls** (requested by scramjit; see the scramjit paragraph in
  `requirements.md`): needed for results bit-identical to CPython's on native targets, on Linux,
  macOS and Windows. **When done, note it in scramjit's `TODO.md`.**

- `edit_distance` 2.5x of gcc: needs `csel` (select) in the backend.
- `rgb2gray` 1.3x of gcc: constant materialization / `madd`.
- Runtime: per-module copy of what it uses (printf ~20 KB); fine for now.

## Small

- **Backend speed on huge functions:** a Csmith function with a 125 KB initializer compiles in
  18 s (spill, liveness bitsets, `filluse`); C QBE takes well under a second.
- **Middle end on functions with hundreds of loops** (YARPGen, `tests/fuzz_yarpgen.py`): a
  474 KB program took 141 s, 77 s since `memdep.Roots` is computed once per round instead of
  once per loop tried. Still quadratic: after each transformed loop, `_phase` refreshes uses
  (`filluse`), re-finds loops and recomputes `Roots` over the whole function (~480 rounds).
  Fix: keep `Roots` across transforms (temporaries it has not seen must count as unknown, and
  rewritten uses must keep their roots) and continue with the loops still valid.

- Profiler symbols: `CIMO_PERFMAP=1` writes `/tmp/perf-<pid>.map` (done; not yet tried with
  `perf` itself). Open: jitdump (with code bytes, for `perf annotate`), and the docs' profiling
  page still says Planned.

- Demos (see "Demos (planned)" in `requirements.md`).
- Website demo (`docs/demo.js`): C vector field → particles, recompiled in Pyodide on every edit.
  `docs/cimo.py` is built by the Pages workflow on each deploy, after the wasm tests pass in
  Pyodide; for a local preview, `cimo onefile -o docs/cimo.py`.
- Website training demo (`docs/nn.js`, `docs/nn/codegen.py`): a drawn network → JSON → C
  (forward, backward, Adam) → cimo → WASM, trained in the page. `docs/runtime.js` shares Pyodide.
- Website Architecture section (`docs/arch.js`): real output of each compiler stage for one kernel,
  from `docs/arch/gen.py`; rerun it when the compiler's output changes.
