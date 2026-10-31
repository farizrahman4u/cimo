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
- `docs/`: the website (GitHub Pages serves `main:/docs` at www.cimo.dev). The user documentation
  lives in `docs/docs/` (www.cimo.dev/docs/): Markdown pages in `docs/docs/content/`, ordered and
  grouped by `docs/docs/nav.json`, rendered in the browser by `docs/docs/docs.js` (no build step).
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

## Language: C11 with a bundled runtime; the host OS on request

Accept (almost) all of C syntax. Kernels link against cimo's **bundled runtime library** by
default; the host OS and other native libraries are available only when linked explicitly.
cimo is **explicitly unsafe**: native kernels run with the full privileges of the Python process
(like ctypes or a C extension), and there is no sandbox mode. A symbol allowlist would not be a
security boundary (C can write through any pointer), so offering one would be a false sense of
security. Isolation comes from the platform (WebAssembly in the browser) or from the user.

- **Headers:** the C11 freestanding set (`stddef.h`, `stdint.h`, `stdbool.h`, `limits.h`,
  `float.h`, `stdarg.h`, `stdalign.h`, `stdnoreturn.h`, `iso646.h`) plus `math.h`,
  `string.h`, `ctype.h`, `assert.h`, `inttypes.h`, an OS-free `stdlib.h` and a minimal
  `stdio.h` (formatted output only). All bundled.
- **Runtime library:** compiled from bundled C source (largely from musl, MIT) by our own compiler,
  then cached (or shipped precompiled):
  - `stdlib.h`: `malloc`, `calloc`, `realloc`, `free`, `aligned_alloc` (cimo's heap, see
    "Execution model"), `qsort`, `bsearch`, `abs`/`labs`/`div`, `strtol`/`strtod`, `atoi`,
    `rand`/`srand` (musl's), `abort`, `exit` (raises `KernelError` with the status)
  - all of `string.h`, `ctype.h`, `math.h` (no `errno`: math functions are pure, so the optimizer
    may hoist and CSE them)
  - `stdio.h`: `printf`, `vprintf`, `puts`, `putchar`, `snprintf`, `vsnprintf`. Formatting is done
    in C; output goes through one callback, `__cimo_write(buf, len)`, into Python's `sys.stdout`
    (buffered, flushed at newline or when the kernel returns), so it shows up in Jupyter cells,
    honors `contextlib.redirect_stdout`, and works in Pyodide.
- **Why a bundled runtime, not the host libc** (decided 2026-10-30): identical results on every
  platform and in Pyodide (no host libc there); `abort`/`assert` raise Python exceptions instead of
  killing the interpreter; `printf` reaches notebooks; heap features need an allocator we own
  (per-thread heaps, reset, debug checks, leak reports); and speed: the compiler knows the library's
  semantics (`malloc` results alias nothing, math is pure, small `memcpy`s inline) and maps
  `sqrt`, `fabs`, `fmin`/`fmax`, `floor`/`ceil`/`trunc`/`round` to single instructions. Large
  `memcpy`/`memset` may call the host's SIMD routines internally where available, until cimo
  vectorizes.
- **Host libraries on request:** `cimo.compile(src, link=["libc"])` (or `"libz.so.1"`, any path or
  soname) makes those libraries' symbols resolvable, which gives kernels the whole OS API (files,
  time, threads, sockets) and existing native code (BLAS, zlib, the Python C API). Only named
  libraries resolve; a symbol that is neither in the runtime nor in a linked library is a compile
  error naming the missing function. Native platforms only (Pyodide: maybe Emscripten's exported
  libc, unverified). Declarations come from the user, or from the system headers via an option.
  Notes: blocking calls are fine (kernels run without the GIL); `fork`, `exit`, signal handlers and
  closing Python's file descriptors fight the interpreter, as with ctypes. Memory from a host
  allocator must be freed by that allocator, not by cimo's `free`.
- **Builtins:** `memcpy`, `memset`, `memmove` (needed for struct copies anyway) and common GNU
  builtins: cproc's own (`__builtin_expect`, `__builtin_unreachable`, `__builtin_offsetof`,
  ...), `__builtin_trap`, `__builtin_huge_val`/`inf`/`nan` (macros over cproc's `inff`/`nanf`),
  and as runtime functions declared by the driver's prelude (`runtime/builtins.c`, portable C;
  lowering them to `clz`/`rbit`/`cnt`/`rev` is future work): `popcount`, `clz`, `ctz`, `ffs`,
  `parity`, `clrsb` (and `l`/`ll` versions), `bswap16/32/64`, `isnan`, `isinf`, `isfinite`,
  `signbit`.
- **Allowed:** structs, unions, bitfields, pointers of any depth, pointer arithmetic, recursion,
  function pointers, varargs, VLAs, `goto`, `switch`, mutable globals, `const` tables.
  Everything compiles correctly; not everything gets the fast path (see "Optimization model").
- **Not supported:**
  - inline asm (raw system calls go through a linked libc instead)
  - `_Atomic` / threads in kernels without a linked libc (v2, if parallel reductions need them)
  - `setjmp`/`longjmp`
  - `long double` where it is 128-bit quad (arm64 Linux); rejected with a clear error
  - most GNU extensions, except the common ones (vector types, statement expressions, simple attributes)
- **Semantics:** standard C. A program with defined behavior in our compiler behaves the same
  under gcc. `restrict` is honored when written; it is never implied.

## Execution model

- **Entry points:** non-`static` functions. Python binds them by parsing their signatures:
  pointers take buffers (numpy arrays, `bytearray`, `memoryview`), scalars take numbers,
  structs take ctypes structures or dicts.
- **Self-contained code:** by default the only external symbols are the runtime's hooks into
  Python (`grow` for the heap, `__cimo_write` for output), so loading is just copying code into
  executable memory (JIT), or writing a minimal ELF / Mach-O (ad-hoc signed) / PE, with no stubs or
  GOT. WASM modules import nothing beyond their memory and those hooks. Explicitly linked host
  libraries (`link=`) are reached through loader-generated stubs and GOT slots.
- **Heap:** `malloc` and friends come from our own size-class allocator (a few hundred lines of
  freestanding C), never from the host libc. That avoids external symbols and mixed C runtimes on Windows.
  - The Python runtime gives the module large regions (anonymous `mmap`). When they run out, the
    allocator calls the `grow(bytes)` hook back into Python. That's slow (it takes the GIL) but
    rare. In WASM the hook is `memory.grow`.
  - The heap belongs to the module and persists across calls. Kernels may return `malloc`'d
    pointers for Python to free later, and Python can reset the whole heap at once, for per-call
    arena use.
  - Each calling thread has its own heap, inside its per-call context (see "Errors"), so `malloc`
    takes no lock. Freeing memory from another thread's heap goes through a deferred free list built on one
    internal compare-and-swap; users never see atomics. A heap outlives its thread (its blocks may
    still be referenced; a later thread with the same thread pointer reuses it) and is released
    with the module.
  - **As built (M5, `runtime/heap.c`, Python side in `cimo/execution.py`):** 16-byte block
    header (owner heap, capacity | kind), so pointers stay 16-aligned and `free` needs no
    lookup. 52 size classes up to 256 KB (16..128 by 16, then 4 per power of two), carved by
    bump pointer from per-thread chunks (`grow` hook: 1 MB doubling to 64 MB), freed blocks on
    per-class lists, reused only for their class (no coalescing). Above 256 KB a block is its
    own mapping, unmapped on `free`. A cross-thread `free` pushes onto the owner's `deferred`
    list with one CAS; the owner takes the whole list (CAS to 0) when a class list runs dry, so
    push + take-all has no ABA problem. The CAS is an assembly helper (`__cimo_cas`, ldaxr/stlxr:
    ARMv8.0, no LSE detection needed) rather than a compiler builtin: it is only on the
    cross-thread path, and the ported frontend/backend stay untouched. malloc+free pair: 16 ns.
  - Python API: `m.free(ptr)` frees a pointer a kernel returned (any thread);
    `m.reset_heap()` frees every block of every thread's heap at once (chunks stay mapped for
    reuse, large blocks are unmapped; refused while a kernel of the module runs);
    `m.heap_bytes()` is the memory mapped for the heap. Kernels called through a raw `address`
    (no context) get NULL from `malloc`.
  - For the optimizer, each `malloc` result is a new base that aliases nothing (C guarantees this).
- **Errors:** `assert`, `abort()` and traps unwind to the entry function's saved stack pointer and
  return an error code; Python raises an exception. Recursion depth is guarded with a stack limit check.
  - **As built (M5, `cimo/execution.py`):** Python calls every kernel through a generated entry
    trampoline (`__cimo_entry_<f>`, assembled into the module with the other glue routines). It
    looks up the thread's context, saves x19-x29, lr, sp and d8-d15 into the context's next
    frame (up to 16 nested entries: kernel -> Python callback -> kernel), and calls the kernel
    with the arguments and sp untouched. `abort`, a failed `assert` (`__assert_fail`),
    `__builtin_trap`, unreachable code (QBE `hlt`, backend ext `trap`) and the stack guard
    record the error in the context and restore the innermost frame, returning 0; Python checks
    the context after each call and raises `cimo.KernelError` (`kind`, `function` from the
    faulting pc, `expr`/`file`/`line` for assert). Adds ~0.1 us to a ~0.9 us call.
  - **How kernel code finds its context:** a per-module hash table keyed by the thread-pointer
    register (`mrs tpidr_el0` on Linux; `tpidrro_el0` & ~7 on macOS, where `tpidr_el0` holds the
    CPU number), filled by Python on a thread's first call. ~20 instructions per lookup (once
    per call, `malloc`, `free`). Rejected: x18 (reserved on macOS/Windows, clobbered by host
    code; Windows would use x18 = TEB as the key instead), TLS (loader support, a TLS block per
    module), a hidden argument (changes every function's ABI and the ported frontend). The table
    only grows; Python inserts under a lock and publishes a rebuilt table after a `dmb ish`.
  - **Stack guard:** functions that can recurse (in a call-graph cycle; an indirect call counts
    as an edge to every address-taken function) call `__cimo_stkchk` in their prologue (backend
    ext `stkchk`): sp below the thread's stack bottom + 128 KB (from `pthread_getattr_np` /
    `pthread_get_stackaddr_np`) unwinds with "stack overflow". Non-recursive code pays nothing.
    Not covered: VLAs/`alloca`, single frames over 128 KB.
  - Hardware faults (SIGSEGV, SIGBUS) still kill the process: catching them needs signal
    handlers, which fight the interpreter. Untested on macOS (key register, stack bounds).
- **Debug mode:** Python knows every buffer's size, the allocator knows every heap allocation's
  size, and the compiler tracks each pointer's base. So debug builds bounds-check every access
  ("compute-sanitizer for CPU kernels"), and detect use-after-free, double free, and leaks
  (reported when the module unloads).
  In every build, arguments are checked at call time against VLA parameter shapes
  (`float a[n][m]` must hold `n*m` floats).
  Debug mode also checks the arguments of runtime functions (`memcpy` ranges, `free` of a
  non-heap pointer).
- **Multicore (v1): bring your own threads.** Kernel calls release the GIL (ctypes does this for
  foreign calls) and are safe to make concurrently from several Python threads. cimo has no launch
  API, no hidden block index and no `cimo.h`. The user splits the work with ordinary parameters
  (`block`, `nblocks`) and plain `threading` / `concurrent.futures`. Reasons: kernels stay
  ordinary C that gcc compiles too; LLMs already write this Python fluently but would not know a
  custom API; it composes with uneven splits, queues, asyncio and free-threaded CPython; and
  Pyodide has no threads anyway. No threading runtime inside the kernel.

## Python interop API (to be designed)

Ergonomic Python-side sugar is what makes this feel native rather than "ctypes with extra steps".
The exact design is still open. Areas to cover:

- how kernels are declared and loaded: inline source strings, `.c` files next to the `.py`, decorators
- type mapping: numpy arrays / buffers, scalars, structs, return values, heap pointers handed back to Python
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
   management and the `grow` hook, error unwinding, debug checks, per-thread heaps.

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
- Loop back edges come out as `bcs exit; b loop`, an extra branch per iteration. Fixed by
  `ext.layout` (see M4).

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
    (musl, minimal images). `cimo.loader` uses its own 25-instruction `dc cvau` / `ic ivau` routine
    (line sizes from `CTR_EL0`, loops skipped when its IDC/DIC bits say so).
  - Bootstrapping that routine: it gets its own page, written RW and then made executable once
    with `mprotect`. Linux arm64 makes a user page I-cache coherent when it first becomes
    executable (`set_pte_at` → `__sync_icache_dcache`), and the page is never rewritten or
    unmapped, so it is safe to call before anything else is flushed. If `__clear_cache` is
    already loaded in the process it is also run over that page.
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

## WASM backend (2026-10-30)

`cimo.backend.wasm` (codegen), `cimo.wasm` (link, build, cache), `cimo.wasm.pyodide` (loading).
`cimo.compile` takes this path automatically when `sys.platform == 'emscripten'`.

- **Pipeline:** QBE's target-independent passes and the middle end, then wasm codegen instead of
  isel/regalloc (the engine allocates registers). SSA temps are wasm locals; phis are copies on
  edges through the operand stack (a free parallel copy). Control flow is structured with
  Ramsey's "Beyond Relooper" algorithm over the dominator tree; irreducible flow (a goto into a
  loop) is a compile error.
- **Data model: LP64 on wasm32.** The frontend still targets aarch64, so `long` and pointers are
  64-bit in the IR and wrapped to i32 at each memory access. `long` is 64-bit in Pyodide too,
  unlike Emscripten: pass `int64`/numpy buffers, not `ctypes.c_long` (32-bit there).
- **ABI (cimo's own):** wasm params in order; struct returns through a hidden first pointer;
  struct args by pointer to a caller copy; varargs in 8-byte slots of a caller buffer passed as
  a last i64 (`va_list` holds the next slot's address). C stack below Emscripten's
  `__stack_pointer`; Python restores it after a kernel error.
- **Module:** one Emscripten side module per `compile`: user code + the runtime members it needs
  + `cimo/wasm/glue.c` (context, CAS, fail/trap, hook targets). Imports only Pyodide's memory,
  table, stack pointer, bases and four `__cimo_js_*` functions (Python callbacks, 32-bit
  arguments, passed through `loadWebAssemblyModule`'s `localScope`). Pointers inside data and
  the runtime's hook pointers are filled in by the module's `__wasm_apply_data_relocs`. A call
  that ignores a callee's result goes through a linker-generated thunk.
- **i64 parameters need BigInt** and Pyodide converts BigInt results back to int, so each kernel
  is called through a small JS wrapper that converts its `l` arguments.
- **No scalar fma in wasm:** `fma`/`fmaf` are exact software versions (`cimo/wasm/runtime/fma.c`),
  and the runtime is built without `__FP_FAST_FMA`, so musl's `pow` takes its non-FMA path: 3 of
  600 `pow` results in `tests/runtime/t_math.c` differ from arm64 by one ulp. Everything else in
  `tests/runtime` matches the native build byte for byte.
- **Measured** (Pyodide 0.27.7 on Node 22, aarch64 WSL2): a small kernel compiles in 57 ms warm
  (first `compile` 1.2 s: imports plus runtime members); a 1024x1024 5-point blur runs in
  1.1 ms and matches numpy bit for bit; the single-file bundle is 1.1 MB.
- **Tests:** `tests/test_wasm.py` runs modules under Node in a stand-in loader
  (`tests/wasm/host.mjs`); with `PYODIDE_JS` set, also in real Pyodide.

## Preprocessor and bundled headers (2026-10-30)

cproc doesn't preprocess: its driver runs `cpp -undef` with the flags in cproc's `config.h`, and
cproc-qbe only understands `#define`/`#undef`, gcc line markers and (ignored) `#pragma`.
`python/cimo/preprocessor.py` replaces that step: `preprocess(src, filename, include_dirs, defines,
target)` returns text with gcc line markers for cproc-qbe. The predefined macros are cproc's
`config.h` list plus `__STDC_VERSION__` 201112L and `__STDC_HOSTED__` 0, plus gcc's type,
size and limit macros (`__SIZE_TYPE__`, `__INT64_TYPE__`, `__INT64_C`, `__WCHAR_MAX__`, ...,
and `__CHAR_UNSIGNED__` on Linux; system headers such as glibc's `glob.h` need them; values from
`cpp -dM`). `target='arm64_apple'` swaps `__linux__`/`__unix__` for `__APPLE__`/`__MACH__` and
uses Apple's type macros (signed char and `wchar_t`, `int64_t` = `long long`). Search order: the includer's directory
(quotes only), then `include_dirs`, then `cimo/include/`.

- **Verified against GNU cpp 13.3:** the token streams match on cproc's tests, `scratch/qbe_eval`,
  `python/tests/preprocessor_cases/` (including the C11 6.10.3.5 examples), glibc headers and `Python.h`
  (with gcc's full predefined set), and 400 random mutually recursive macro programs. cproc-qbe's
  IR, and its diagnostics including line:col, are byte-identical to the `cproc -E` pipeline on all
  of them. Run `python3 python/tests/preprocessor_compare.py` or `pytest python/tests/test_preprocessor.py`.
- **gcc behaviours copied on purpose** (the standard leaves them open, and cproc's output was
  produced by gcc):
  - a directive between a function-like macro's name and `(` means no invocation;
  - an included file's end also ends a macro invocation;
  - `, ## __VA_ARGS__` drops the comma only when the variable arguments are omitted, not empty;
  - the first token of each output line keeps its source column, so cproc's columns match.
- **Deliberate deviations from gcc:**
  - character constants in `#if` have type `int`. gcc makes them unsigned when plain char is
    unsigned, so `#if ~'a' > 7` is true under gcc on aarch64.
  - digraphs are printed canonically, because cproc's scanner has no digraphs;
  - `#pragma` is dropped, except `#pragma omp`, which is kept and macro-expanded (OpenMP rule);
  - no trigraphs (also gcc's default) and no `__VA_OPT__`.
- **Headers** (`python/cimo/include/`) are declarations only; the functions are in the runtime
  library (M5). `stdlib.h` has the listed functions plus `abort` and the obvious siblings
  (`llabs`, `ldiv`, `lldiv`, `atol`, `atoll`, `atof`, `strtoul`, `strtoll`, `strtoull`,
  `strtof`); `string.h` adds `strnlen`, `strdup`, `strndup`; `stdio.h` has `printf`, `vprintf`,
  `puts`, `putchar`, `snprintf`, `vsnprintf`, `sprintf`, `vsprintf` and no `FILE`. `math.h`
  has the double and float functions, not long double or `nexttoward`. `ctype.h` inlines the
  simple tests as musl's macros do.
  - `NAN`/`INFINITY` are `__builtin_nanf("")`/`__builtin_inff()`, cproc's only float builtins.
    `HUGE_VAL` is a cast of the latter.
  - `stdarg.h` uses `__builtin_va_*` and `offsetof` uses `__builtin_offsetof`.
  - Apple differences are under `#ifdef __APPLE__`: signed char, `wchar_t` is `int`, `int64_t` is
    `long long`, 64-bit `long double` limits.
  - `python/tests/preprocessor_headers/all_headers.c` gives the same output when built by gcc with system
    headers and by cimo.preprocessor + cproc + QBE.
- **For the cproc port (M3):**
  - cproc emits unused `static` functions. math.h's 16 tiny classification and comparison
    helpers (e.g. `isnan`, no double evaluation) landed in every unit that includes it.
    **Done:** `frontend.compile(prune=True)` (driver and runtime build) drops static functions
    and data no exported definition reaches (`frontend/prune.py`, on the IR text); the
    runtime's first build went from 8.8 s to 5.5 s. QBE has no inliner, so `isnan(x)` is a call
    until the middle stage inlines.
  - cproc's config.h defines `__FLT_EVAL_METHOD__` as 1, but cproc + QBE compute floats in
    single precision (at run time and when folding constants). `cimo.preprocessor` defines 0,
    like gcc on arm64, so `float_t` is `float` (with 1 it was `double`: slower, and results
    differing from gcc).
  - cproc has no Apple target. Apple arm64 has signed `char` and an 8-byte `long double`, while
    cproc's `aarch64` has unsigned `char` and a 16-byte `long double`. The port adds target
    `aarch64-apple` (signed `char`, `wchar_t` = `int`, 8-byte `long double`, and `va_list` =
    `char *` as Apple's ABI and QBE's `arm64_apple` vastart expect); not diffable against C.

## Single-file distribution (2026-10-30)

`cimo bundle [-o cimo.py]` (`cimo.bundle.build`) writes one `cimo.py` that works
like the package in any CPython 3.9+ with only the standard library: put it on `sys.path`,
`import cimo`.

- **Resource layer:** every read of a package file goes through `cimo.resources` (`path`,
  `isfile`, `read`, `files`): bundled headers (preprocessor search path), runtime sources,
  and the compiler sources hashed into cache keys. In the repo or a pip install they are files
  next to the package. In the bundle `resources.ROOT` is `<cimo.py>/cimo`, a name that exists
  only in the embedded table; any other path (user and host headers, the cache) is read from
  the filesystem. The `.c`/`.h` files stay real files in the repo.
- **Format:** a license header (cimo MIT; cproc's and QBE's upstream `LICENSE` files, shipped
  as `frontend/LICENSE` and `backend/LICENSE`, and musl's `runtime/COPYRIGHT` reproduced), then `_FILES = {path relative to the package: text}` (every `.py`, `.h`, `.c`
  and `runtime/COPYRIGHT`, as readable triple-quoted literals, sorted), then a bootstrap: a
  meta-path finder/loader serving `cimo.*` from the table. Imported as `cimo`, the file
  becomes the package (`__path__ = []`) and runs `__init__.py`; run as a script it runs
  `cimo.__main__` (the CLI). Module filenames are `<cimo.py>/cimo/driver.py`, registered
  with `linecache`, so tracebacks and `inspect` show source. Deterministic (no timestamps):
  rebuilding, also from inside the bundle, gives the same bytes.
- **Stripped:** comments, docstrings and blank lines are left out of the embedded `.py`
  (checked: same AST minus docstrings) and `.c`/`.h` (comments become a space; checked: same
  pp-tokens); C comments carrying a copyright or license notice stay (musl's math files
  have Sun/FreeBSD/Arm notices that must be preserved). 1.26 MB -> 1.05 MB.
  `--keep-comments` turns it off.
- **Cache keys are content hashes over package-relative names**, computed from the original
  sources at bundle time and embedded (`bundle-keys.json`), so a bundle has the same
  `compiler_key()` and runtime key as the tree it was built from, stripped or not.
- **Sources only.** A `--prebuilt` option (embedding the compiled runtime) was tried and
  removed: it only matched the target it was built on and put generated, target-specific
  code in a file meant to be source. The first-use cost is handled by compiling runtime
  members on demand instead (M5, link model).
- **No writable cache** (`CIMO_CACHE_DIR` / `CIMO_RUNTIME_CACHE` / `~/.cache/cimo` cannot be
  created or written): the module cache and the runtime are kept in memory for the process.
- **Measured** (Linux arm64, CPython 3.13): bundle 1.05 MB; building it 1 s with stripping,
  0.05 s without; `import cimo` from it 0.05 s; first compile of a printf/sin/malloc kernel
  1.5 s with nothing cached (15 runtime members), then 0.1 s. `tests/test_bundle.py` runs it
  under `python -I -S` with no writable home.

## Command line (2026-10-30)

`cimo` (console script) = `python -m cimo` = `python cimo.py` for a bundle. Standard library
only: `cimo/cli/parse.py` is a small declarative framework (nested commands, options with
short/long/`=`/attached forms, choices, repeatable options, a `rest` argument that passes
everything after it through), `cimo/cli/term.py` does color (`--color`, `NO_COLOR`,
`FORCE_COLOR`, `TERM=dumb`, tty), unicode symbols with ASCII fallback, wrapping to the
terminal width and a spinner.

- Commands: `run FILE [ARGS...]` (calls `main(argc, argv[, envp])` or `--entry NAME`; exit
  status = return value or `exit()` status, 134 for abort/assert/trap; `-` reads stdin),
  `compile FILE` (lists the exported functions with their Python-side types) with `-E`,
  `--ir`, `-S` / `--emit pp|ir|asm` and `-o`, `bundle`, `cache [info|clear|path]`,
  `completion bash|zsh|fish`, `help [COMMAND...]`. Compile options shared by `run` and
  `compile`: `-I`, `-D`, `-l/--link`, `--host-headers`, `-O0/--no-opt`, `--no-fma`.
- Help at every level (`-h`, `--help`, `cimo help ...`, a group without a subcommand), with
  examples; "did you mean" (difflib) for commands, options, choice values, missing files and
  entry points; a usage line and hint on every usage error (exit status 2).
- Compile errors are shown with the source line and a caret; driver hints are rewritten from
  API spelling (`host_headers=True`) to flags (`--host-headers`). Failed asserts show the
  assert's source line.
- Completion: the scripts call the hidden `cimo __complete WORDS...`, so they follow the
  command tree; `__file__`/`__dir__` hand path completion back to the shell. bash is tested
  (`tests/test_cli.py`); zsh and fish scripts are untested (not installed here).
- Compiling runtime members reports through `linker.progress` (the CLI shows a spinner);
  without it the library prints one line on stderr per link that compiled any.

## Plan

1. ~~Validate the architecture~~ **Done: GO.** Good IR + patched QBE is within 1.04x of gcc's
   scalar `-O3` on all 4 kernels tried on Linux arm64 (threshold was 1.2x). On Apple M4 vs clang,
   3 of 4 are within 1.11x; saxpy is at 1.43x until `ldp`/`stp` pairing lands. See "Architecture check".
2. ~~Check loading in each environment~~ **Done** for Linux arm64/x86-64, Windows x64, macOS
   arm64 and Pyodide. Still open: locked-down Windows (AppLocker/WDAC/EDR), notarized macOS apps
   that embed Python without `allow-jit`.
3. **Port QBE (arm64 first) with `qbe-arm64-isel.patch` applied** (done, see M1), **then cproc**, verified by IR
   diffs against the C originals, plus differential testing against gcc using generated random programs.
4. **Middle stage loop passes:** implement the transforms that `gen_opt.py` applies by hand, plus
   explicit `fma` emission and FMA selection in the backend. Target: about 1.2x of gcc's scalar
   `-O3` on the whole suite. **Mostly done** (M4): geomean 1.11x; 12 of 15 kernels within
   1.2x; edit_distance, nbody_acc and rgb2gray wait on backend items (csel, sqrt, int
   constants/madd). Next for the middle stage: runtime alias checks (loop versioning on
   address ranges) for non-restrict pointers, multi-block loop bodies, `--report`.
5. **SIMD** (vector types, then auto-vectorization), **amd64**, **WASM**, **concurrent calls** (per-thread heaps).

## Milestones and acceptance criteria

**M1: QBE in Python (arm64).** Port QBE `e786f06` with `scratch/qbe_eval/qbe-arm64-isel.patch` applied.
- Every pass's IR dump matches C QBE (with the patch) on QBE's own tests and on `scratch/qbe_eval/kernels.c`.
- QBE's 58 tests pass using the Python port plus an external assembler.
- Record compile time for `kernels.c` (C QBE takes about 9 ms; the budget is under 1 s in Python).
- **Done 2026-10-30** (`python/cimo/backend/`, 9.5k lines). Every `-d` dump and the final assembly are
  byte-identical to C QBE on all 77 inputs (QBE's tests + `kernels.ssa`), for `arm64` and
  `arm64_apple` (`python/tests/backend_diff.py`). QBE's `tools/test.sh all` passes all 59 tests with
  the port (`bin=python/tests/pyqbe`). `kernels.ssa` compiles in 0.10 s (C QBE: 6 ms).
- Matching depends on glibc 2.39's stable `qsort`: the port uses Python's stable sort where QBE
  calls `qsort` with comparators that have ties (rega, mem, load). Another libc may order ties
  differently; that's a C-side difference, not a port bug. Conventions: `python/cimo/backend/PORTING.md`.

**`cimo.compile` (2026-10-30): pure Python.** `python/cimo/driver.py`: `cimo.preprocessor` → `cimo.frontend` →
`cimo.backend` → `cimo.assembler` → `cimo.loader`, no external tools. The assembly is cached in `~/.cache/cimo`,
keyed by the preprocessed source, the flags and a hash of the compiler's own sources (so an edited
or upgraded compiler never reuses old assembly). Signatures come from the QBE IR (`l` params accept buffers).
Modules link statically with the bundled runtime library (M5); nothing resolves from the
process unless named in `link=`. `examples/` uses it.
Defaults: middle end on (`CIMO_MIDDLE=0` off), FMA contraction on (`CIMO_FMA=0` for results
identical to `-ffp-contract=off`), ldp/stp pairing and loop layout on. Geomean over the 15
kernels with these defaults: 1.07x of gcc nofma, 1.13x of novec, 1.69x of `-O3` (vectorized).

**M2: encoder + loader.** arm64 machine-code encoder (no external assembler) and the JIT loader
for Linux and macOS (strategies under "Platform loading findings").
- All 15 kernels compile and run in-process from Python, with results matching gcc's.
- `scratch/qbe_eval/edge.c` cases are bit-exact.
- **Done 2026-10-30 on Linux arm64; macOS path written but untested.** `python/cimo/assembler.py`
  assembles QBE's GNU-syntax output (both `arm64` and `arm64_apple` syntax) into an in-memory
  object; `python/cimo/loader.py` links it into the process. No `cc`, no `dlopen` of generated code.
  - Byte-identical to GNU as (sections, relocations, symbols) on all 77 corpus inputs for both
    targets (Apple syntax via a textual translation to ELF syntax), a hand-written file with every
    emitter form, and sweeps of all logical immediates and random immediates/offsets
    (`python/tests/test_assembler.py`). `kernels.s` assembles in 4 ms; load + link takes a few ms.
  - All 15 kernels are bit-identical to `gcc -O2 -ffp-contract=off` on the same inputs, and
    `edge.c` (unchanged, its kernels pointed at the in-process code) is bit-exact
    (`python/tests/test_m2.py`). CPython 3.9 and 3.12 both pass.
- Assembler semantics follow GNU as where QBE relies on them: `mov #imm` picks movz, then movn,
  then orr; `ldr`/`str` offsets that don't fit the scaled form become `ldur`/`stur`; `add`/`sub`
  with `sp` and a register use the extended (UXTX) form; branches to non-global labels in the
  same section are resolved, everything else becomes a relocation; `.comm` without alignment
  gets the next power of 2 ≥ size, at most 16. Mach-O `.comm`'s third operand is log2 (QBE
  passes the byte alignment, so Apple over-aligns commons; harmless).
- Relocation types QBE needs: `CALL26`, `JUMP26`, `CONDBR19`, `ADR_PREL_PG_HI21`,
  `ADD_ABS_LO12_NC`, `ADR_GOT_PAGE`, `LD64_GOT_LO12_NC`, `ABS64`, `ABS32` (plus `LDST*_LO12_NC`
  and `ABS16`, supported but not emitted). Thread-local storage (`.tdata`, `mrs tpidr_el0`, `@tlvp`)
  is rejected with a clear error.
- Loader layout: code sections, then 16-byte stubs (`ldr x16, 8; br x16; .quad addr`) for calls
  to external functions; then a RW region with GOT slots, data, bss and commons. External
  data is always reached through a GOT slot (see M5), so library distance doesn't matter.
- macOS: code goes in a `MAP_JIT` mapping, data in a separate RW mapping hinted right after it
  (a `MAP_JIT` page is RX for the thread while kernels run, so writable data can't share it);
  if that fails or lands beyond `adrp` range, fall back to one W^X mapping. Untested: needs a
  run of `python/tests` on an M-series Mac.

**M3: cproc in Python.** Port cproc `d1c53dd`, restricted to freestanding C.
- QBE IR text identical to C cproc on cproc's tests and on `kernels.c`.
- Clear errors for unsupported features (see "Language"). **Done 2026-10-30**, see below.
- **Port done 2026-10-30** (`python/cimo/frontend/`, 7k lines; conventions in its `PORTING.md`):
  `cimo.frontend.compile(text, filename, target='aarch64')` and `python -m cimo.frontend` (= `cproc-qbe`:
  `-E`, `-t`, `-o`). All of cproc-qbe is ported literally, including its own pp (`#define`,
  `#undef`, line markers); `#include`/`#if` come from `cimo.preprocessor`. Targets `aarch64` (default),
  `x86_64-sysv`, `riscv64`, plus `aarch64-apple`.
- `python/tests/test_frontend.py`: stdout, stderr and exit status byte-identical to C `cproc-qbe`
  on all 1704 cases: cproc's 196 tests (as `runtests` runs them, including expected errors and
  `-E`), `scratch/qbe_eval/*.c` and `python/tests/frontend/*.c` through `cproc -E` on all 3 targets,
  and 300+ raw snippets (errors, pp corners, folding) on all targets and `-E`. Also identical on
  cproc's and QBE's own sources (52 files x 3 targets). Line coverage of the port: 94%.
- `kernels.c` (1262 preprocessed lines) compiles in 27 ms on CPython 3.13 and 51 ms on 3.9
  (C: about 2 ms). cproc + QBE sources (~25k lines with headers): 3.6 s.
- Matching relies on: constant folding of `x / 0` etc. following aarch64 (quotient 0, remainder =
  dividend), where the reference C build runs; C `assert`s, which the CLI mirrors with `abort()`;
  places where C cproc segfaults (e.g. `extern int x;` redeclared in an inner block), where the
  port raises an internal error (CLI status 70). cproc's quirks are kept on purpose, e.g.
  `enum E : unsigned char { A }` is rejected, `_Bool b = 2` emits `b 2`, and initializing two
  union members asserts.
- **Clear errors done 2026-10-30** (`frontend/diag.py`, behind `compile(diagnostics=True)`, which
  `cimo.compile` passes; the CLI stays cproc's; `python/tests/test_diagnostics.py`). Every error
  has `file:line:col` and names the feature:
  - Attributes: cproc's own (`aligned`, `packed`, `section`, `constructor`, `destructor`,
    `noreturn`) work; an allowlist of hints that don't change meaning for cimo (`noinline`,
    `unused`, `const`, `pure`, `format`, `nonnull`, `may_alias`, `visibility`, `weak`, ...) is
    ignored; any other is "attribute 'X' is not supported" (`vector_size`: "not supported yet").
    `aligned` after a declarator applies to the object/member, and on a struct typedef makes an
    over-aligned variant (glibc's `pthread.h`). The preprocessor ignores glibc's
    `#define __attribute__(x)` (made for non-GNU compilers), so attributes are never dropped.
  - Declaring `long double` (16-byte), `__int128` (an opaque 16-byte type) or thread-locals is
    fine (glibc's headers do); defining or using them is an error. `_Float16`, `__fp16`,
    `__bf16`, `_Float128` etc. are rejected outright; an undeclared identifier followed by a
    declarator is "unknown type name".
  - Rejected GNU extensions: statement expressions, `&&label`/`goto *`, nested functions, case
    ranges, range designators, `__label__`, `__auto_type`, `__builtin_choose_expr` and other
    undeclared `__builtin_`/`__atomic_`/`__sync_` names, imaginary constants, empty structs,
    inline assembly (`asm` is a keyword, as in gcc's default mode). `a ?: b` works (cproc
    evaluates `a` once).
  - Accepted: `__restrict(__)`, `__const(__)`, `__volatile`, `__alignof`, `asm` labels,
    `__PRETTY_FUNCTION__`/`__FUNCTION__`, C23 `nullptr` and `nullptr_t` (cproc d1c53dd parses
    them but asserts in code generation and rejects non-constant `nullptr_t` operands; cimo
    generates a pointer-sized zero and allows `nullptr_t` expressions in assignment, `==`/`!=`
    and `?:` as C23 does).
  - Unsupported types are reported where the expression is parsed (a hook on expression
    construction), so the column points at the `long double` sub-expression. A C99 inline definition with no external definition in the
    unit becomes a local function when referenced (cproc emits nothing, so the call failed to link).
  - Driver hints: a missing header suggests `host_headers=True` (`setjmp.h`: unsupported);
    undeclared hosted-libc names (`fopen`, `FILE`, `time`, `pthread_*`, ...) suggest
    `host_headers=True, link=['libc']`.
  - glibc 2.39 headers (stdio, stdlib, unistd, pthread, signal, math, sys/*, ...; 70+ tried) compile
    except `complex.h` (`_Complex`) and `glob.h` (`__SIZE_TYPE__` is not predefined).

**M4: middle stage.** Automate the transforms `scratch/qbe_eval/gen_opt.py` applies by hand, plus
explicit `fma` emission, and add backend `fmadd` selection, `ldp`/`stp` pairing and the loop-layout fix.
- On the 15-kernel suite: within 1.2x of scalar `-O3` (gcc on Linux arm64, clang on Apple M4), with matching FMA settings.
- **Harness:** `python3 python/bench/run.py [kernels...] [-n N] [--cimo NAME=fma,ldp,layout|none]
  [--gen-opt]`. Kernels are called in-process through ctypes from the same C timing loops
  (`python/bench/bench.c`, harness.c's sizes, per-kernel seeds), gcc builds as shared libraries
  (`-O3 -mcpu=native` with `-fno-tree-vectorize -ffp-contract=off` / `-fno-tree-vectorize` /
  nothing), configurations interleaved, best of N, checksums compared. `cimo` = `cimo.compile`;
  `--cimo` builds use explicit extension flags; `--gen-opt` swaps in gen_opt.py's IR for 4 kernels.
- **Baseline 2026-10-30** (cimo = cproc port + QBE port with the isel patch, no middle stage;
  Snapdragon X / WSL2, gcc 13.3, best of 5), geomean over 15 kernels: **1.98x** vs gcc nofma,
  2.08x vs novec (FMA), 3.10x vs `-O3`. Worst: conv3x3 10.2x, edit_distance 3.8x, gemm 3.5x,
  jacobi2d 3.1x, rgb2gray 3.1x, isum 2.2x; dot, iir, prefix_sum, mandelbrot at 1.0x.
- **Backend extensions (done 2026-10-30)**, each off by default (stock QBE output, so
  `backend_diff.py`, QBE's suite and `test_frontend.py` are unchanged), switched on with
  `cimo.backend.ext.options(fma=, ldp=, layout=)` and `frontend.compile(..., fma=True)`:
  - **FMA.** IR: `%r =s fma %a, %b, %c` (a*b + c, rounded once), parsed only with `ext.fma`.
    Stored as an adjacent pinned pair `fma0 %c; %r = fma1 %a, %b`, like QBE's `sel0/sel1`, so
    every pass sees all three uses unchanged; `igroup` keeps the pair together, pinning keeps gvn
    from merging and gcm from moving it (no LICM of FMAs). isel makes it one op whose 2nd and
    3rd operands ride in a `Mem`'s base/index: liveness, spilling and rega already handle those,
    so a reload can never land between the halves. Same-block `neg` of operands folds into
    `fmsub`/`fnmsub`/`fnmadd`. Frontend: contracts only an add/sub whose operand is directly a
    product of the same type (clang's `-ffp-contract=on`; left product preferred); never across
    statements, assignments, commas or casts (clang does fuse across a same-type cast; we don't).
    Bit-exact against cc with explicit `fma()` calls, incl. specials and a spill-heavy function
    (`test_fma.py`). Measured: conv3x3 1.31x faster, mandelbrot 1.17x; dot and iir 1.35-1.40x
    *slower* (fma latency on the loop-carried chain; gcc's default build pays the same on dot).
    iir is 1.18x behind gcc-fast: clang's rule fuses `b0*x - a1*y1` as `b0*x + -(a1*y1)`,
    keeping a multiply on the recurrence; gcc's contraction picks the other product.
  - **Loop layout.** In emit, a block ending in `jmp H`, H earlier in the layout with ≤ 4
    instructions and a conditional jump (cproc's top-tested loop header), gets a copy of H's
    code and branch: one conditional branch per iteration instead of `b H` + test. When neither
    successor of a conditional jump follows, the conditional branch goes to the backward
    (loop) target. Measured on cproc IR: conv3x3 1.62x, isum 1.12x, saxpy 1.08x faster;
    on gen_opt IR (already bottom-tested) saxpy 1.05x.
  - **ldp/stp.** After isel (SSA), the second access of a pair (same base temp, offsets o and
    o+size, plain 32/64-bit int/fp, lower offset in imm7 range) moves next to the first: loads
    up, stores down, only past instructions that can't touch the memory (loads past loads; past
    same-base accesses with disjoint ranges; never past calls or other-base accesses). emit then
    pairs any adjacent fit, including spill slots and callee-save saves/restores. Hook for alias
    knowledge: `ext.disjoint(fn, base1, base2)` (restrict) — without it, saxpy's x loads can't
    cross y stores. No effect on cproc IR (addresses recomputed per access); on gen_opt IR
    gemm 1.05x, others within noise on this core (M4 is where pairing should matter).
  - Geomean on cproc IR, vs gcc nofma / novec: layout alone 1.89x / 1.98x; all three
    1.96x / 2.06x (FMA's dot/iir losses offset its conv3x3 gain). The middle stage is what
    closes the gap.
- **Middle stage (done 2026-10-30, `python/cimo/middle/`, ~2.5k lines).** Runs on the backend's
  `Fn` after GCM (clean SSA, GCM has already hoisted pure invariants, the ABI/isel passes haven't
  run), then gvn/simplcfg/gcm run again. Off by default (`cimo.middle.enabled`); `cimo.driver`
  switches it on (`CIMO_MIDDLE=0` turns it off). Restrict info comes from the frontend through
  `frontend.qbe.paraminfo` (per function: param temporary, restrict flag; off by default), which
  `driver._frontend` collects and `driver._qbe(ssa, params)` passes on. Phases, per function:
  1. full unrolling of innermost loops with a constant trip count ≤ 16 (conv3x3's 3x3);
  2. unroll-and-jam by 2 of a loop around an innermost loop (`jam.py`), when an inner store has
     an address that doesn't depend on the outer IV (gemm's `c[i*n+j]`);
  3. arithmetic if-conversion of small integer diamonds in innermost loops
     (`x = b ^ ((a ^ b) & -c)`: the arm64 backend has no `sel`);
  4. innermost loops (`inner.py`): versioned against the untouched original loop; guard computes
     the exact trip count T and run-time checks; then strength reduction (one IV per
     (base, stride) group), LICM incl. loads no store may alias, unroll by 4 + remainder loop
     (bottom-tested, `iv != end`), scalar replacement of loads at contiguous offsets of one
     array, and store-to-load forwarding / redundant load and dead store elimination within the
     unrolled body (`LocalMem`).
- **Overflow semantics.** QBE IR doesn't say whether a `w` add came from signed or unsigned C,
  so nothing relies on "signed overflow is UB". Affine forms are exact mod 2^32/2^64 (add, sub,
  mul, shl are ring homomorphisms); the only non-homomorphic step, `extsw`/`extuw` of a 32-bit
  index, is checked at run time: the exact value at t = 0 and t = T-1 must fit (affine, so
  that covers every t). The trip count formulas also check that the IV can't wrap past the
  end of its type (`i += 3` near INT_MAX, unsigned wrap) and fall back to the original loop.
  Cost: a few instructions per loop entry (checks of one array are merged into min/max).
- **Alias analysis** (`memdep.py`): roots of an address through add/sub/copy/phi (64-bit
  params, stack slots, globals), with must-roots (intersection over phis) so a pointer that is
  `prev` or `cur` (edit_distance's swap) isn't treated as based on either one alone. A restrict
  param in must(A) and not in may(B) proves no alias; without restrict, distinct pointer params
  may alias and nothing moves across them (no runtime alias checks yet).
- **Results** (driver default = middle stage only; Snapdragon X/WSL2, gcc 13.3, best of 3,
  all checksums equal to gcc nofma): geomean **1.11x vs gcc nofma** (was 1.98x), 1.17x vs
  novec, 1.75x vs `-O3`. With fma, ldp and layout on as well: 1.07x / 1.13x / 1.68x.

  | kernel | before | middle | middle+fma,ldp,layout vs novec |
  |---|---|---|---|
  | saxpy | 1.78x | 0.77x | 0.63x |
  | gemm | 3.54x | 0.89x | 0.87x |
  | jacobi2d | 3.18x | 1.07x | 1.10x |
  | conv3x3 | 10.5x | 1.04x | 1.14x |
  | rgb2gray | 3.13x | 1.26x | 1.25x |
  | isum | 2.21x | 0.85x | 0.98x |
  | histogram | 1.83x | 0.96x | 0.96x |
  | sieve | 1.25x | 1.08x | 1.08x |
  | edit_distance | 3.69x | 2.48x | 2.57x |
  | nbody_acc | 1.91x | 1.88x | 2.54x |
  | flood_fill | 1.18x | 1.18x | 1.17x |
  | dot, prefix_sum, iir, mandelbrot | 1.0x | 1.0x | 0.9-1.0x |

  (ratio vs gcc `-O3 -fno-tree-vectorize -ffp-contract=off` unless noted.)
- **Left for M4's 1.2x, all backend items:** edit_distance needs `csel` (arm64 `sel` support,
  then QBE's ifopt instead of the 6-instruction arithmetic select on the critical path: gcc
  37 ms, us 93 ms, 137 ms with branches). nbody_acc needed `sqrt` as an instruction
  (done in M5: 0.74x of gcc nofma). rgb2gray: integer constants of `mul`
  are rematerialized (`mov w6, #77`) at every use inside the loop, shifts take no immediate,
  no `madd`. jacobi2d/flood_fill are within 1.2x.
- **Backend bug found (pre-existing, also in C QBE + isel patch):** gvn folds 32-bit constant
  chains in 64-bit without truncating, e.g. `(x + -1) + -30` (C: `int y = x + -1; return y +
  -30;`), and the arm64 isel emits the unnormalized value as an immediate: assembler error or an
  assert in emit. Fix belongs in isel (truncate w immediates). The middle stage works around it
  (`ir.normalize_wcons`: w add/sub constants as positive values), so it only bites with the
  middle stage off.
- **Tests:** `tests/test_middle.py`: 72 loop shapes in `tests/middle/loops.c` (trip counts
  0..100, steps ±1..4, `<=`/`!=`/down/unsigned/long/pointer IVs, INT_MAX/INT_MIN bounds,
  unsigned wrap, symbolic strides, aliasing calls without restrict, early exits, nests) and the
  15 kernels at sizes 0..21, 37, 64: middle on, middle off and gcc -O0 bit-identical.
  `tests/test_middle_random.py`: random loop programs (`tests/middle_gen.py`, 40 functions per
  seed, 6 seeds by default, `CIMO_RANDOM_SEEDS=a:b` for more; 120 seeds pass).
- Compile time of kernels.c: 0.17 s with the middle stage (0.09 s without).

**M5: runtime library.** The bundled runtime described under "Language" and "Execution model",
replacing host-libc resolution.
- `printf` family through `__cimo_write` into `sys.stdout`: output appears in Jupyter and under
  `redirect_stdout`, in order with Python's own output.
  **Done 2026-10-30.** `runtime/stdio/`: musl's `vfprintf` core writing to a sink instead of a
  `FILE` (long double = double, `%lc`/`%ls` encode UTF-8); stdout is a 4 KB module-global
  buffer written through `__cimo_hook_write` (a function pointer Python sets to a ctypes
  callback, so modules import no symbol at all) at a newline, when full, and after every
  kernel call (Python reads the buffer length and calls `__cimo_flush` only if nonzero). The
  callback decodes UTF-8 incrementally (per thread) and writes to the current `sys.stdout`.
  The buffer is **per thread**, at offset 4096 of the thread's context (contexts are two
  pages now), so concurrent threads' lines come out whole and in order per thread (checked:
  8 threads x 2000 lines); a thread without a context (started by the kernel through a
  linked libc) writes unbuffered.
- Heap: size-class allocator in freestanding C, per-thread heaps with a lock-free fast path,
  cross-thread `free` via a deferred list (one internal compare-and-swap), `grow` hook into Python,
  heap reset from Python; stress-tested from many Python threads.
  **Done 2026-10-30** on Linux arm64 (design under "Execution model"; `tests/test_execution.py`:
  8 threads churning malloc/realloc/free with fill patterns, rounds where every free is
  cross-thread and races the owner's mallocs, grow, reset reuse, large map/unmap,
  realloc/calloc/aligned_alloc semantics, buffers returned to Python and freed from another
  thread; also passes on CPython 3.9).
- `abort`, `assert` and traps raise a Python exception (`cimo.KernelError`) and leave the
  interpreter and the module usable. **Done 2026-10-30** on Linux arm64, plus the stack guard:
  unwinding from depth 0-1000, float returns and stack-passed arguments, re-entry through
  callbacks, the nesting limit, errors from 8 threads at once, stack overflow in main and
  secondary threads, each followed by successful calls.
- `string.h`, `ctype.h`, `math.h`, OS-free `stdlib.h` from musl, compiled by cimo; results match
  musl (and glibc where the standard pins them; within documented ulps for math). First-use build
  time recorded, then cached.
  **Done 2026-10-30.** Sources: musl 1.2.5 (`runtime/COPYRIGHT` lists what was taken and how
  it was changed): `math/` generic C unchanged (122 files), `string/` with musl's
  word-at-a-time paths enabled (they hide behind `__GNUC__`; cimo has no type-based alias
  analysis), `stdlib/` with `intscan`/`floatscan` over a string-only pseudo-`FILE` and
  `long double` → `double` in floatscan (what musl does where they are the same type),
  `ctype/ctype.c`; `internal/` holds cut-down musl headers (`libm.h` with macros instead of
  static inlines and no-op `FORCE_EVAL`/`fp_barrier`: no errno, no FP exception flags).
  Built with the driver's optimizations but **no FMA contraction** (musl builds with
  `-std=c99`, i.e. `-ffp-contract=off`, and its algorithms assume it), with
  `__FLT_EVAL_METHOD__` 0 like all code (with 1, `float_t` would be double and the float
  functions would differ from musl's), and `__FP_FAST_FMA`/`__FP_FAST_FMAF` as
  gcc defines them on arm64 (musl's `pow`, `log`, `log2` then use `fma()` for exact products;
  without them `pow` differs from gcc-built musl by an ulp).
  Fix over musl: floatscan's hex-float digit window is one digit wider (musl's is sized for its
  real long double; with long double = double, `0x1.00000000000008000001p0` rounded down).
  - **Compiled on demand:** all 197 files take 5.5 s (8.8 s before unreferenced statics were
    pruned; CPython 3.13, Snapdragon X/WSL2; vfprintf.c 0.6 s, `__rem_pio2_large.c` 0.4 s, the
    rest < 0.15 s each), but a link compiles only the members it pulls in: printf + sin +
    malloc is 15 members, 1.4 s. Each member is cached; a module whose members are cached
    links and loads in ~0.1 s the first time in a process and ~10 ms after.
  - `tests/test_runtime.py`: five C programs (`tests/runtime/t_*.c`: string edges of lengths
    and alignments, ctype on all of -1..255, strto* edge cases and 2000 random decimal
    strings, qsort/bsearch, ~30k printf formats, every math function on special and random
    arguments) are built by cimo, by gcc against the same runtime sources, and by gcc
    against glibc. cimo's output is **identical to gcc-built musl** for all five; identical to
    glibc for string/ctype/stdlib (NaN payloads, invalid bases and `FP_ILOGB0` left out) and
    printf, except `%a` precision rounding, `%5%` and wide characters (C leaves them open)
    and a glibc bug (2.39): `%#.3g` of 999.9995 prints `1.e+03` (C and musl: `1.00e+03`); math within
    1-2 ulps of glibc (lgamma/tgamma more), exact where Annex F pins it. The reference build
    needs `-fno-math-errno`, else gcc's `__builtin_sqrt` calls `sqrt` for negative inputs,
    which is the runtime's own `sqrt` (an endless tail call).
- `sqrt`, `fabs`, `fmin`/`fmax`, `floor`/`ceil`/`trunc`/`round` compile to single instructions;
  `nbody_acc` within 1.2x of gcc nofma.
  **Done 2026-10-30.** Frontend (`frontend.compile(builtins=True)`): a call to the external
  function of that name with the standard prototype (the names are reserved, C11 7.1.3; a
  `static` function or another prototype is left alone) becomes an IR instruction `fsqrt`,
  `fabs`, `fmin`, `fmax`, `ffloor`, `fceil`, `ftrunc`, `fround` (and `fma`/`fmaf` the `fma`
  instruction). Backend ext `math`: pure ops (gvn merges, gcm hoists, the middle end copies
  them), arm64 `fsqrt`, `fabs`, `fminnm`, `fmaxnm`, `frintm`, `frintp`, `frintz`, `frinta`;
  assembler encodings checked against GNU as. Semantics are C's exactly: `fminnm`/`fmaxnm`
  return the number when one operand is a quiet NaN and order -0 < +0 (as glibc/gcc on
  aarch64; musl's C version differs only for signaling NaNs), `frinta` rounds halfway away
  from zero, `fsqrt` is correctly rounded. The runtime's own `sqrt` etc. are one-line
  wrappers (for `&sqrt`). The driver always has backend `fma` on now: an explicit `fma()` is
  fused even with `CIMO_FMA=0`, which only controls contraction.
  **nbody_acc: 0.74x of gcc nofma** (7.3 ms vs 9.9 ms; 0.98x of gcc with FMA, whose result
  it matches bit for bit), was 1.88x. Suite geomean with the driver defaults now 1.01x of gcc
  nofma, 1.05x of novec, 1.58x of `-O3` (was 1.07x / 1.13x / 1.68x).
- `link=[...]` resolves host libraries; unknown symbols are compile errors naming the function.
  The default build has no host-libc dependency (checked by a test that resolves nothing from
  the process).
  **Done 2026-10-30** (`cimo/linker.py`). **Link model: static, per module.** The runtime is
  an archive (one member per source file, compiled when a link first needs it and cached as
  assembly in `~/.cache/cimo/runtime-<key>/`; `runtime/symbols.txt`, generated by `cimo
  __update-runtime-symbols` and checked by a test, maps symbols to members so nothing else
  is compiled; what a member uses comes from its object; key = runtime sources and headers +
  bundled headers + compiler sources (frontend, middle, backend, driver, linker, execution); `CIMO_RUNTIME_CACHE` overrides the location, a fresh
  `CIMO_CACHE_DIR` does not force a rebuild). Linking pulls only the members that resolve the
  module's undefined symbols, transitively, and merges them with the module into one object
  (sections concatenated, locals renamed apart). Chosen over one shared runtime image per
  process because the heap and stdout buffer belong to the module ("Execution model"),
  calls stay direct (no stubs), modules stay self-contained (the same shape as a future
  ELF/Mach-O/WASM module that imports nothing), and the middle end can later inline runtime
  functions. Cost: each module carries its copy of what it uses (printf ~ 20 KB).
  `cimo.compile(src, link=['libc'])` (or `'libm'`, `'python'` = the C API, a soname, a short
  name like `'z'`, a path): what neither the module nor the runtime defines resolves by dlsym
  in those libraries only, through the loader's stubs/GOT; runtime functions still win
  (`malloc` stays the module heap's, `printf` still goes to `sys.stdout`).
  `host_headers=True` adds the host's header directories after `include_dirs` (glibc's
  `stdio.h`, `unistd.h`, `time.h`, `errno.h`, ... preprocess and compile; the bundled
  `stdarg.h` defines gcc's `__gnuc_va_list` for them; cimo's freestanding headers still serve
  `stddef.h`/`stdarg.h`). Anything unresolved is a CompileError:
  `undefined symbol 'getpid': cimo's runtime does not provide it; to call a host library,
  pass e.g. link=['libc']`; a user definition of a name that a pulled-in runtime object also
  defines is an error naming both. `test_default_build_resolves_nothing_from_the_process`
  links a module using printf, malloc, qsort, strtod, ctype and libm with the loader's dlsym
  patched to fail: the merged object has no undefined symbols.
  Host *data* (`environ`, `optind`, glibc's `stdout`, `errno` via `__errno_location`) works at
  any distance: cproc marks every external-linkage object `extern` in the IR, QBE loads its
  address from a GOT slot (`adrp :got:` + `ldr`), and the loader fills the slot (defined
  objects of the module stay direct `adrp` + `add`).
- Kernels whose signature ctypes can't express (variadic, struct or union by value) are still
  compiled and callable from C; calling them from Python raises `TypeError` (struct binding
  is part of the interop API).

**M6 and later:** Python interop API polish, amd64 backend, WASM backend +
Pyodide loading, SIMD (vector types, then auto-vectorization), concurrent calls, debug mode,
then the demos.

## Risks

- **Executable memory may be blocked** where it matters most: EDR tools, AppLocker/WDAC, hosted sandboxes.
- **Register allocation** under heavy unrolling. QBE's allocator may spill more than gcc's.
- **Vectorizer quality** is the long tail; v1 vector types limit the damage.
- **Miscompilations** destroy trust. Needs heavy differential fuzzing from day one.
- **Narrow commercial path:** open-source first. Possible paid niches: GIS and plugin
  ecosystems, browser/edtech notebook platforms, enterprises with "no installs" policies.
