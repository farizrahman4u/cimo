# cimo: requirements

**C in, machine code out.** A pure-Python C compiler for CPU kernels. PyPI name `cimo` is reserved
(placeholder 0.0.1, published 2026-10-30).

This is the living spec: what to build, why, what has been measured, and in what order. Research
code behind the measurements is in `scratch/` (`scratch/qbe_eval/`, `scratch/spikes/`).
Third-party clones live outside the repo: QBE at `~/code/qbe` (branch `cimo-arm64-isel`, patch
applied) and cproc at `~/code/cproc`.

## Repository layout

- `python/cimo/`: the compiler and the Python runtime (the PyPI package).
- `js/`: reserved for a later JS port of the compiler and browser glue. Empty for now.
- `docs/`: user-facing documentation.
- `examples/`: example kernels and the demos (see "Demos").
- `scratch/`: research code and spikes (benchmarks, hand-optimized IR, loading tests). Not part of
  the package.
- `requirements.md`: this spec. `AGENTS.md`: rules for coding agents.

## Engineering rules

- **Pure Python, standard library only** at runtime. No native extensions, no third-party
  dependencies. Supports CPython 3.9+ (macOS Command Line Tools still ship 3.9) and Pyodide.
- **Port faithfully first, refactor later.** Ported code (QBE, cproc) must reproduce the C
  original's output exactly, checked by diffing IR text, before any redesign.
- **Differential testing from day one:** against the C originals for IR, and against gcc/clang for
  program results, including random programs.
- **Benchmarks:** interleave configurations and take the best of several runs. Single runs drift
  by about 10% on WSL. Compare against both scalar and default `-O3`, with and without FMA, on
  Linux arm64 and Apple M-series (results don't carry over between cores).
- **Record findings here, concisely.** This file is a spec, not a log.

## Pitch

"CUDA kernels for the CPU, with no toolchain." A C compiler written in pure Python that compiles
**freestanding C** kernels into optimized machine code and loads them into the running Python
process. No compiler, SDK, wheel or native dependency on the target machine. The same compiler
emits WASM for Python in the browser.

## Problem

- Shipping native code with Python is still a mess. Cython, cffi (API mode) and C extensions need a
  toolchain on the target (MSVC on Windows, Xcode CLT on macOS).
- Some environments allow `.py` files but no binaries or installs:
  - The **QGIS** official plugin repo has banned compiled binaries since it opened.
  - **Pyodide / JupyterLite / marimo WASM** have no compiler at all, and Numba doesn't run there.
  - Locked-down corporate machines and hosted Python sandboxes.
- Going from "this loop is slow" to "I have a native kernel" changes the build system and workflow.

## Competition

- `pip install ziglang` gives a clang-based compiler as a wheel. Fine where wheels are allowed;
  useless where they aren't, and not in the browser.
- Numba gives near-C speed for numeric Python with no system toolchain, but needs LLVM wheels and
  doesn't run in Pyodide.
- Cython + cibuildwheel solves distribution for library authors, not for users writing kernels.

## Demos (planned)

The wow isn't "fast in the browser": WebAssembly, JS engines and WebGPU already do that. It's
**C compiled at runtime, from inside Python, with no server, a small download, and no copying of
numpy data.** Demos should show the edit → run loop and the absence of infrastructure.

Why that isn't available today:
- Emscripten compiles ahead of time with a full toolchain; nothing compiles in the user's browser.
- Numba and Cython don't run in Pyodide; pure Python in WASM is slower than normal CPython.
- In-browser C compilers are either clang compiled to WASM (tens of MB, slow start) or TCC
  ports (unoptimized code). Neither shares memory with Pyodide's numpy arrays.
- numpy can't express stencils, flood fills, DP or other sequential loops, which is where
  Python is slowest.

1. **"Numba in the browser" notebook:** a JupyterLite/marimo simulation (reaction-diffusion,
   fluid) driven by sliders, with an editable C kernel cell. Edit, recompile in under a second,
   and it keeps running on the same numpy arrays. A toggle shows pure Python, roughly 100x slower.
2. **AI-written kernel runs locally:** prompt → LLM writes C → cimo compiles it in the page → it
   runs on local data. WASM is the sandbox, so no server sandbox is needed. Needs an LLM API key,
   so keep it optional.
3. **Private data crunching:** drop a large local file onto a static page and process it with
   custom C kernels; the data never leaves the machine. DuckDB-WASM covers SQL, not custom logic.
4. **Runtime specialization:** compile user formulas, regexes or filters into C and then into
   machine code on the fly (spreadsheet engine, per-pattern grep). Pitch: "a JIT target for
   Python developers", without writing a code generator per target.
5. **Same notebook, two worlds:** one notebook running natively on desktop Python and in the
   browser via WASM, with identical C kernels and no build step on either side.
6. **QGIS plugin:** viewshed or flood fill shipped as C source in the official repo (which bans
   binaries), about 100x faster than pure Python.

**Flagship:** one static page with tabs for (1), (2) and (3), plus (5) shown side by side.
(1) and (3) must stand alone without an API key.

## Language: freestanding C11 + the parts of libc that never touch the OS

Accept (almost) all of C syntax. The execution environment is a kernel, like CUDA device code:
no I/O and no OS. The only libc is the part that needs no OS.

- **Headers:** the C11 freestanding set (`stddef.h`, `stdint.h`, `stdbool.h`, `limits.h`,
  `float.h`, `stdarg.h`, `stdalign.h`, `stdnoreturn.h`, `iso646.h`) plus `math.h`,
  `string.h`, `ctype.h` and an OS-free `stdlib.h`. All bundled.
- **Library:** compiled from bundled C source (largely from musl, MIT) by our own compiler, then cached:
  - `stdlib.h`: `malloc`, `calloc`, `realloc`, `free`, `aligned_alloc`, `qsort`, `bsearch`,
    `abs`/`labs`/`div`, `strtol`/`strtod`, `atoi`
  - all of `string.h`, `ctype.h`, `math.h`
  - still out: `stdio`, `time`, `signal`, `locale`, files, `exit`. `abort()` is the error-unwind path.
- **Builtins:** `memcpy`, `memset`, `memmove` (needed for struct copies anyway) and common GNU
  builtins (`__builtin_expect`, etc.).
- **Allowed:** structs, unions, bitfields, pointers of any depth, pointer arithmetic, recursion,
  function pointers, varargs, VLAs, `goto`, `switch`, mutable globals, `const` tables.
  Everything compiles correctly; not everything gets the fast path (see "Optimization model").
- **Not supported:**
  - stdio, files, OS calls. Memory comes from caller buffers or the module heap.
  - inline asm
  - `_Atomic` / threads (v2, if parallel reductions need them)
  - `setjmp`/`longjmp`
  - `long double` where it is 128-bit quad (arm64 Linux); rejected with a clear error
  - most GNU extensions, except the common ones (vector types, statement expressions, simple attributes)
- **Semantics:** standard C. A program with defined behavior in our compiler behaves the same
  under gcc. `restrict` is honored when written; it is never implied.

## Execution model

- **Entry points:** non-`static` functions. Python binds them by parsing their signatures:
  pointers take buffers (numpy arrays, `bytearray`, `memoryview`), scalars take numbers,
  structs take ctypes structures or dicts.
- **Self-contained code:** the only external symbol is the heap's `grow` hook, so loading is just
  copying code into executable memory (JIT), or writing a minimal ELF / Mach-O (ad-hoc signed) / PE.
  WASM modules import nothing beyond their memory.
- **Heap:** `malloc` and friends come from our own size-class allocator (a few hundred lines of
  freestanding C), never from the host libc. That avoids external symbols and mixed C runtimes on Windows.
  - The Python runtime gives the module large regions (anonymous `mmap`). When they run out, the
    allocator calls the `grow(bytes)` hook back into Python. That's slow (it takes the GIL) but
    rare. In WASM the hook is `memory.grow`.
  - The heap belongs to the module and persists across calls. Kernels may return `malloc`'d
    pointers for Python to free later, and Python can reset the whole heap at once, for per-call
    arena use.
  - Under parallel launch, each worker has its own heap, selected by a hidden worker id. Freeing
    memory from another worker's heap goes through a deferred free list built on one internal
    compare-and-swap; users never see atomics.
  - For the optimizer, each `malloc` result is a new base that aliases nothing (C guarantees this).
- **Errors:** `assert`, `abort()` and traps unwind to the entry function's saved stack pointer and
  return an error code; Python raises an exception. Recursion depth is guarded with a stack limit check.
- **Debug mode:** Python knows every buffer's size, the allocator knows every heap allocation's
  size, and the compiler tracks each pointer's base. So debug builds bounds-check every access
  ("compute-sanitizer for CPU kernels"), and detect use-after-free, double free, and leaks
  (reported when the module unloads).
  Optional `printf`-style debug output through a callback into Python.
- **Parallel launch (v1):** ctypes releases the GIL during foreign calls, so a Python thread
  pool can call the kernel on chunks in parallel, with the block index passed in. Real multicore
  scaling with no threading runtime inside the kernel.

## Python interop API (to be designed)

Ergonomic Python-side sugar is what makes this feel native rather than "ctypes with extra steps".
The exact design is still open. Areas to cover:

- how kernels are declared and loaded: inline source strings, `.c` files next to the `.py`, decorators
- type mapping: numpy arrays / buffers, scalars, structs, return values, heap pointers handed back to Python
- launch syntax and parallelism
- error and debug surfaces
- compile cache control and ahead-of-time packaging (e.g. shipping prebuilt cache entries)

## Optimization model

The compiler accepts all of the language; it reaches `-O3` quality on **analyzable** code:

- canonical counted loops (`for (i = a; i < b; i += c)`, `i` not modified in the body)
- affine indexing (`a[i*n + j]`, C99 VLA parameters `float a[n][m]`)
- pointers whose base buffer can be tracked: `p = a + k`, `p++`, ping-pong swaps of two buffers
  (handled by unrolling the outer loop by 2), and read-only row tables (`float **rows`) treated like 2D arrays
- no aliasing between written and read buffers, known from either:
  - `restrict` written by the user, or
  - **runtime alias checks:** compute each buffer's address range for the loop, compare once
    before the loop, run the fast or the safe version (what gcc does)

Everything else compiles at "QBE quality" (measured 1-4x behind gcc on scalar code, see below).
`--report` explains why a loop wasn't optimized ("base of `p` depends on a runtime branch"), like
gcc's `-fopt-info`.

**Floating point:** strict C, no reassociation. `#pragma omp simd reduction(+:s)` (valid C) opts a
loop into vectorized reductions. Benchmark against gcc both with default FMA contraction and with
`-ffp-contract=off`.

**SIMD:**
- **v1:** GNU vector types (`__attribute__((vector_size(32)))`). They map 1:1 onto instructions; no vectorizer needed.
- **v2:** auto-vectorize innermost analyzable loops and `#pragma omp simd` loops, with a scalar remainder loop.

## Architecture

Start from proven small C compilers instead of writing one from scratch:

1. **Front end:** port **cproc** (~9k lines of C, ISC license, C11) to Python, cut down to
   freestanding C. cproc emits QBE IR as text, so the port is verified by diffing its output
   against C cproc, file by file.
2. **Middle stage (ours):** a structured IR before lowering to QBE IR:
   - loop passes: unrolling, LICM of loads, strength reduction of address arithmetic,
     interchange and unroll-and-jam for affine nests
   - runtime alias checks
   - the vectorizer (v2)
   This is the product's differentiator.
3. **Backend:** port **QBE** (~9k lines core + 2k arm64 + 3.5k amd64, MIT), verified pass by pass
   by diffing IR dumps against C QBE. Then extend it:
   - arm64 instruction selection: immediate operands, reg+reg<<shift addressing, FMA
   - vector types (NEON, AVX2, AVX-512)
   - our own machine-code encoder (QBE prints assembly and relies on an external assembler)
   - a WASM backend
4. **Emitters:** in-process JIT (W^X handling, `MAP_JIT` on macOS), ELF / Mach-O / PE, WASM.
5. **Runtime library:** the heap allocator plus OS-free libc (`string.h`, `ctype.h`, OS-free
   `stdlib.h`, `math.h`), written in or ported to freestanding C and compiled by our own compiler.
6. **Python runtime:** signature-based bindings, a content-hashed compile cache, heap region
   management and the `grow` hook, error unwinding, debug checks, the parallel launcher.

Estimated size: 25-35k lines of Python. It's large but mechanically verifiable, which suits
agent-driven porting.

## Baseline: cproc + QBE vs gcc (measured 2026-10-30)

Setup: `scratch/qbe_eval/` (15 kernels, best of 5, aarch64 WSL2, gcc 13.3, QBE e786f06). All checksums match gcc.

| Kernel group | QBE vs `gcc -O3 -fno-tree-vectorize` | QBE vs `gcc -O3` |
|---|---|---|
| Latency-bound: prefix_sum, mandelbrot, flood_fill, sieve | 1.0-1.3x | same |
| dot, iir | parity (once gcc's FMA contraction is turned off) | parity |
| Simple loops: saxpy, isum, histogram, rgb2gray | 1.8-3.5x | up to 27x (gcc vectorizes) |
| Loop nests: gemm, jacobi2d, edit_distance, nbody | 2.5-4.4x | 4-13x |
| conv3x3 | 18x | 20x |

- Compile time for the whole kernel file: cproc 3 ms + QBE 9 ms natively. Even 50-100x slower
  in Python, that's under a second, and the output is cached.
- The `saxpy` inner loop is 15 instructions vs gcc's 6. The losses come from missing arm64
  instruction-selection features (immediates, addressing modes, FMA) and missing loop
  optimizations. Both are planned work, not architectural dead ends.

## Architecture check: good IR + patched QBE reaches gcc -O3 (scalar)

Done 2026-10-30, arm64. Setup: `scratch/qbe_eval/gen_opt.py` emits the IR our loop passes should produce
for 4 kernels; `scratch/qbe_eval/qbe-arm64-isel.patch` (+200 lines on QBE `e786f06`) fixes the arm64
backend. Interleaved best-of-3; checked bit-exact against plain C for sizes 0-21 (`scratch/qbe_eval/edge.c`).

| Kernel | gcc -O3 scalar, no FMA | gcc -O3 scalar | IR transforms used |
|---|---|---|---|
| saxpy | **0.79x** | 0.79x | pointer IVs, 4x unroll + remainder loop |
| gemm | **0.89x** | 0.98x | + unroll-and-jam of `k` by 2 (keeps C evaluation order) |
| jacobi2d | **1.04x** | 1.03x | + scalar replacement of `src[j-1]`, `src[j]` across iterations |
| conv3x3 | **1.04x** | 1.79x | + 3x3 fully unrolled, 9 weights hoisted, 3-row sliding window |

(ratio = cimo time / gcc time; lower is better; "scalar" = `-fno-tree-vectorize`)

**The backend patch** (`arm64/isel.c`, `arm64/emit.c`):
- **Immediates:** add/sub (12-bit, add↔sub flip for negatives); logical immediates for 64-bit
  `and`/`or`/`xor`; `mul` by a power of 2 becomes `lsl`.
- **`[reg, #imm]` addressing:** a load/store address defined as `add base, const` in the same
  block is folded into an `RMem` (scaled uimm12 offsets); the `add` is removed once all its uses
  are folded. `RMem` was already handled generically by liveness, rega and spill; only isel and
  emit changed.
- **FP constants** are loaded once in the start block and reused. Stock QBE reloads them
  (`adrp`/`add`/`ldr`) at every use, inside loops too. This alone took jacobi2d from 1.31x to 1.04x.
  Hoisting in the IR doesn't help: QBE's copy propagation folds the constant back into each use.
- QBE's own test suite (58 tests) passes with the patch.

**Lessons for the build:**
- The middle stage is what closes the gap. Patched QBE on cproc's naive IR (addresses recomputed
  with `mul`/`add` at every access) barely improves; the same backend with transformed IR reaches parity.
- **FMA is the remaining gap** (conv3x3: 1.79x vs gcc with contraction). QBE IR has no fused
  multiply-add and loses C expression boundaries, so the middle stage must emit an explicit `fma`
  op where C allows contraction (within one expression, like clang's default
  `-ffp-contract=on`), and the backend must select `fmadd`. Both are needed for parity with gcc's
  default build.
- Loop back edges come out as `bcs exit; b loop`, an extra branch per iteration. A block-layout
  fix in emit is still open.

**macOS (Apple M4, macOS 26.7) vs Apple clang 21:** the patch works on QBE's `arm64_apple`
target, and all 4 kernels are bit-exact. Same IR as above; `scratch/qbe_eval/mac_bench.sh` reproduces it.

| Kernel | clang -O3 scalar, no FMA | clang -O3 scalar |
|---|---|---|
| saxpy | **1.43x** | 1.34x |
| gemm | 1.11x | 0.88x |
| jacobi2d | 0.96x | 0.95x |
| conv3x3 | 1.07x | 1.78x |

(ratio = cimo / clang; "scalar" = `-fno-vectorize -fno-slp-vectorize`)

- saxpy loses on M4 although it wins on Linux: clang pairs adjacent accesses into `ldp`/`stp`
  (half the memory instructions) and unrolls by 6; ours issues one access per value plus two
  branches per iteration. M4 is bound by memory-instruction throughput here.
- **Backend items this adds:** an `ldp`/`stp` pairing pass (adjacent `[base, #off]` accesses
  are easy to spot after address folding) and the loop-layout fix above.
- Benchmark on both microarchitectures. A result on one arm64 core doesn't carry over.
- Not yet tested: the transforms on the other 11 kernels (edit_distance 3.7x, rgb2gray 3.1x,
  isum 2.2x, nbody 1.9x, histogram 1.8x as compiled by cproc → patched QBE), amd64, SIMD.

## Platform loading findings

Spikes: `scratch/spikes/load_test/loadtest.py` (native) and `scratch/spikes/pyodide/` (WASM). Each puts a
hand-encoded `sum(const long *p, long n)` into executable memory from pure Python and calls it on
a Python-owned buffer.

| Platform | Status | How |
|---|---|---|
| Linux arm64 (CPython 3.12, 3.13) | works | anonymous `mmap` RW → `mprotect` RX via ctypes on libc (W^X compliant). Plain `mmap.mmap(prot=RWX)` also works. |
| Linux x86-64 (Ubuntu 24.04, CPython 3.12) | works | same two strategies; no cache flush needed on x86 |
| Windows Server 2022 x64 (embeddable CPython 3.12.10) | works, with Defender real-time protection on | `VirtualAlloc` RW → `VirtualProtect` RX via ctypes on kernel32 |
| Pyodide 0.27.7 (Node 22) | works, zero-copy | see below |
| macOS 26.7 arm64 (M4): Apple CLT Python 3.9, Homebrew 3.14, python.org 3.13 | works | `mmap(MAP_JIT)` + `pthread_jit_write_protect_np` + `sys_icache_invalidate`. W^X `mmap` RW → `mprotect` RX also works. Plain RWX `mmap` is **refused** (`EACCES`); never use it on macOS. |
| Locked-down Windows (AppLocker/WDAC, third-party EDR) | not yet tested | default Defender didn't object; enterprise policies may |

To retest a platform without opening any network access: put `loadtest.py` inline in EC2 user
data, run it at boot, write results to the serial console (Linux `/dev/ttyS0`; Windows a
`SerialPort` on COM1), and read them with `aws ec2 get-console-output`. No key pair, IAM role or
inbound rules needed.

- **arm64 needs an instruction-cache flush** after writing code.
  - macOS: `sys_icache_invalidate`; Windows: `FlushInstructionCache`.
  - Linux: `__clear_cache` is exported by `libgcc_s.so.1`, but that library isn't guaranteed
    (musl, minimal images). Plan to emit our own `dc cvau` / `ic ivau` flush routine; read the
    cache line size from `CTR_EL0`.
- **macOS code signing:** python.org Python runs under the hardened runtime with `allow-jit`,
  `disable-executable-page-protection` and `disable-library-validation`. An ad-hoc-signed
  hardened copy with only `disable-library-validation` (needed by anything that embeds Python)
  still loaded code by both strategies (SIP on). Unverified: apps signed with a Developer ID under
  the hardened runtime but without `allow-jit`, e.g. a notarized app that embeds Python. Prefer
  `MAP_JIT` first and fall back to W^X; fail with a clear error naming the missing entitlement.
- **Pyodide does not expose its `WebAssembly.Memory`.** `_module.wasmMemory` is undefined; the
  memory is a local inside `pyodide.asm.js`. What works: emit the module as an **Emscripten side
  module** (a `dylink.0` custom section with zero memory/table sizes, importing `env.memory`)
  and load it with `pyodide._module.loadWebAssemblyModule(bytes, {loadAsync: false, nodelete: true,
  allowUndefined: true}, name, {})`. Pyodide's own linear memory is then shared, so kernels read
  and write Python buffers in place (pointer from `ctypes.addressof`).
  - That's a private API: pin Pyodide versions and test each upgrade.
  - Unverified: synchronous compilation of larger modules on a browser's main thread. Chrome
    limits sync `WebAssembly.Module` size on the main thread; Pyodide usually runs in a worker.

## Plan

1. ~~Validate the architecture~~ **Done: GO.** Good IR + patched QBE is within 1.04x of gcc's
   scalar `-O3` on all 4 kernels tried on Linux arm64 (threshold was 1.2x). On Apple M4 vs clang,
   3 of 4 are within 1.11x; saxpy is at 1.43x until `ldp`/`stp` pairing lands. See "Architecture check".
2. ~~Check loading in each environment~~ **Done** for Linux arm64/x86-64, Windows x64, macOS
   arm64 and Pyodide. Still open: locked-down Windows (AppLocker/WDAC/EDR), notarized macOS apps
   that embed Python without `allow-jit`.
3. **Port QBE (arm64 first) with `qbe-arm64-isel.patch` applied, then cproc**, verified by IR
   diffs against the C originals, plus differential testing against gcc using generated random programs.
4. **Middle stage loop passes:** implement the transforms that `gen_opt.py` applies by hand, plus
   explicit `fma` emission and FMA selection in the backend. Target: about 1.2x of gcc's scalar
   `-O3` on the whole suite.
5. **SIMD** (vector types, then auto-vectorization), **amd64**, **WASM**, **parallel launch**.

## Milestones and acceptance criteria

**M1: QBE in Python (arm64).** Port QBE `e786f06` with `scratch/qbe_eval/qbe-arm64-isel.patch` applied.
- Every pass's IR dump matches C QBE (with the patch) on QBE's own tests and on `scratch/qbe_eval/kernels.c`.
- QBE's 58 tests pass using the Python port plus an external assembler.
- Record compile time for `kernels.c` (C QBE takes about 9 ms; the budget is under 1 s in Python).

**M2: encoder + loader.** arm64 machine-code encoder (no external assembler) and the JIT loader
for Linux and macOS (strategies under "Platform loading findings").
- All 15 kernels compile and run in-process from Python, with results matching gcc's.
- `scratch/qbe_eval/edge.c` cases are bit-exact.

**M3: cproc in Python.** Port cproc `d1c53dd`, restricted to freestanding C.
- QBE IR text identical to C cproc on cproc's tests and on `kernels.c`.
- Clear errors for unsupported features (see "Language").

**M4: middle stage.** Automate the transforms `scratch/qbe_eval/gen_opt.py` applies by hand, plus
explicit `fma` emission, and add backend `fmadd` selection, `ldp`/`stp` pairing and the loop-layout fix.
- On the 15-kernel suite: within 1.2x of scalar `-O3` (gcc on Linux arm64, clang on Apple M4), with matching FMA settings.

**M5 and later:** Python interop API, heap and runtime library, amd64 backend, WASM backend +
Pyodide loading, SIMD (vector types, then auto-vectorization), parallel launch, debug mode,
then the demos.

## Risks

- **Executable memory may be blocked** where it matters most: EDR tools, AppLocker/WDAC, hosted sandboxes.
- **Register allocation** under heavy unrolling. QBE's allocator may spill more than gcc's.
- **Vectorizer quality** is the long tail; v1 vector types limit the damage.
- **Miscompilations** destroy trust. Needs heavy differential fuzzing from day one.
- **Narrow commercial path:** open-source first. Possible paid niches: GIS and plugin
  ecosystems, browser/edtech notebook platforms, enterprises with "no installs" policies.
