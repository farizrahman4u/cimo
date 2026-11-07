# cimo: requirements

**C in, machine code out.** A pure-Python C compiler for CPU kernels. Published on PyPI as `cimo`.

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
  The live demos (`docs/demo.js`, `docs/nn.js`) run Python against cimo in Pyodide, so a change to cimo's API (once:
  signature letters) can break them: `test_site_demos_in_pyodide` runs their Python on every preset, and the site deploys
  only after it passes. Each demo exports what the test needs (`globalThis.cimoDemo`, `cimoNN`) before touching the DOM.
  `docs/agent-prompt.txt` is the Prompt for Agents, behind the banner's peeling corner: it reads as for the
  released product, but agents act on it, so every instruction must work today (the GitHub install until the
  PyPI release). Update it when the API, CLI or C support changes.
- `examples/`: example kernels and the demos (see "Demos").
- `scratch/`: research code and spikes (benchmarks, hand-optimized IR, loading tests). Not part of
  the package.
- `requirements.md`: this spec.

## Engineering rules

- **Pure Python, standard library only** at runtime. No native extensions, no third-party
  dependencies. Supports CPython 3.9+ (macOS Command Line Tools still ship 3.9) and Pyodide.
- **Compiled code is a cache** (decided 2026-11-02). It's built on the machine that runs it, from source, and can always be built
  again. It is never distributed, in a package or otherwise: that would bring back binaries and a build per platform, which is what
  this project removes. A cache entry is used only with the source it was built from.
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
  - `setjmp`/`longjmp` on WebAssembly (native targets have them, 2026-11-06)
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
    CPU number; x18, the TEB, on Windows), filled by Python on a thread's first call. ~20 instructions per lookup (once
    per call, `malloc`, `free`). A thread with no context in a module that runs its code anyway (a kernel of another
    module calling it by `kernel.address`; a thread a kernel started) gets one from `malloc` through a Python callback
    (`__cimo_hook_ctx`, 2026-11-07). Its errors have no frame there, so they go to `cimo.fault`'s handler (an `udf`/`ud2`
    at `__cimo_noframe`; a module-wide `__cimo_orphan` context when the thread has none), which moves them to the
    innermost entry frame of any module on the stack (lowest saved sp at or above sp): the caller's call raises,
    naming the callee's function. Hardware faults take the same route. Open: a Python callback between two calls into
    one module (A, then Python, then B, then A by address): an error in the inner A unwinds to the outer A's frame,
    past B's and Python's; nothing on that path checks. Output from such code buffers in its module's context, which
    is flushed only by a later call from Python into that module. Rejected: x18 as a register of cimo's own (reserved on
    macOS/Windows, clobbered by host code), TLS (loader support, a TLS block per
    module), a hidden argument (changes every function's ABI and the ported frontend). The table
    only grows; Python inserts under a lock and publishes a rebuilt table after a `dmb ish`.
  - **Stack guard:** functions that can recurse (in a call-graph cycle; an indirect call counts
    as an edge to every address-taken function) call `__cimo_stkchk` in their prologue (backend
    ext `stkchk`): sp below the thread's stack bottom + 128 KB (from `pthread_getattr_np` /
    `pthread_get_stackaddr_np`) unwinds with "stack overflow". Non-recursive code pays nothing.
    VLAs/`alloca` and single frames over 128 KB reach the guard page instead: a fault, below.
  - **Hardware faults (`cimo/fault.py`):** one process-wide native handler, C compiled by cimo
    on the first load (a fault can arrive without the GIL, so never Python). pc in a registered
    module and the thread inside its trampoline: record the fault in the context and resume at
    `__cimo_fault`, the same unwind. Otherwise chain to the handler before it. A pc that equals
    the fault address is a call through a bad pointer, blamed on the return address. POSIX:
    sigaction for SIGSEGV, SIGBUS, SIGFPE, SIGILL with `SA_NODEFER` (faulthandler's `raise` must
    come straight back, to the owner-thread guard, or it loops), an alternate stack per thread;
    the handler is re-checked at every load. Windows: a vectored handler, re-added first at every
    load (faulthandler reports and carries on if it runs first), `SetThreadStackGuarantee` for
    room to run in, `_resetstkoflw` once unwound. A pc in a host library or Python is never
    unwound: their locks and state. Only a fault the CPU raised unwinds a kernel (`si_code > 0` on Linux), not one sent by
    kill. A handler passed a real fault that returns without fixing it (faulthandler's, once disabled, still installed
    under ours) brings it back forever: after 64 in a row at one pc on one thread, the default action. A module's code
    leaves the table before it is unmapped (`Image.on_close`). Windows: the new vectored handler goes in before the old
    one comes out. A thread key reused by a new thread gets its own stack limit. The page size is in the cache key, and
    an image built for another (`StaleImage`) is compiled again: data sharing a code page crashed on the first write.
    First compile in a process +6 ms (Linux arm64, then cached);
    no measurable change to a call. `CIMO_FAULTS=0` installs nothing.
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

Built: a module says what it was made from (`module.source`, `module.options`, `module.signature(name)`, `module.mutable_data`, `kernel.module`), found by running the front end when asked, so the cache format stays as it is. scramjit puts a module's source in the same C file as what it generates, so a kernel calling `k.f(x)` inlines it. Calling across modules by `kernel.address` works natively since 2026-11-07 ("Execution model"); wasm kernels have no address.

Ergonomic Python-side sugar is what makes this feel native rather than "ctypes with extra steps".
The exact design is still open. Areas to cover:

- how kernels are declared and loaded: inline source strings, `.c` files next to the `.py`, decorators
- type mapping: numpy arrays / buffers, scalars, structs, return values, heap pointers handed back to Python
- error and debug surfaces
- compile cache control and ahead-of-time packaging (e.g. shipping prebuilt cache entries)

**First external consumer: [scramjit](https://github.com/farizrahman4u/scramjit)**, a separate
pure-Python JIT for a numeric subset of Python that generates C and compiles it with cimo. It
uses only the public API and needs: in-memory compilation of a C string to callables, `#line`
honored in diagnostics and debug info, a defined convention for numpy buffers and scalars, low
compile latency with a usable cache key, and (for loop-heavy code) auto-vectorization.

Also from scramjit's first measurements (its M0, 2026-11-01):
- **FMA contraction per compile:** **Done:** `cimo.compile(..., fma=False)` (also `load`), like the
  CLI's `--no-fma`; `None` keeps the process default (`CIMO_FMA`). scramjit must reproduce
  CPython's float results bit for bit without changing other callers in the same process.
- **Compiling from several threads** is safe: the frontend and backend ports keep their state in
  module globals, so `driver._compile_lock` runs one at a time (the preprocessor, assembler and
  loader run outside it). Without the lock, 15 of 16 concurrent compiles failed.
- **Math functions against CPython:** the bundled libm differs in the last bit from the host libm
  CPython uses (glibc, arm64) for many inputs: `atan2` 18%, `hypot` 14%, `sin`/`cos` 3%,
  `exp`/`pow` under 0.1%, `sqrt`/`log` none. Worth documenting; an option to link the host's libm
  for math calls (native targets) would let generated code match CPython where that matters.
- **Fast libm (2026-11-07):** bundled functions are to be as fast as glibc's, one at a time as kernels need them.
  `hypot` is glibc's algorithm now (fma kernel where `__FP_FAST_FMA`, arm64; a correction step on x86-64 and wasm),
  under 1 ulp, 1.47x faster than musl's on a `hypot` loop. It gives glibc's bits on each target, so arm64 and
  x86-64/wasm differ in about 6% of inputs: the fma kernel is correctly rounded 87% of the time, the correction step
  99% (the correction's division is what makes it musl's speed on arm64, 1.18 ms). The same bits everywhere would
  need a correctly rounded `hypot`.

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

Small functions are inlined (`backend/inline.py`, before SSA): a function of at most 40 IR
instructions is copied into its calls in the same unit, callees first (`CIMO_INLINE=<n>` sets
the limit, 0 turns it off), so a loop calling helpers stays analyzable. The backend holds a
unit's functions until its end for this. Recursion, aggregate arguments or returns, variadic
functions and VLAs keep the call; the callee's own definition is always kept.

Repeated work QBE leaves goes too (`backend/cse.py`, `ext.cse`): repeated loads, divisions (pure on
arm64, where `sdiv` never traps; wasm's trap, so only a dominating one merges there), a remainder from
its division, a repeated branch condition and a select written twice. Costs 13-27% compile time on
`t_printf` and `t_math`. Also from LLVM and GCC, under the same flag: algebra and known bits
(`backend/simplify.py`; IR integers wrap: only `backend/nowrap.py` uses C's promise about signed overflow, and it removes it, so no other pass has to keep it true), facts from
dominating branches (same file), jump threading (`backend/thread.py`), memory (`backend/memopt.py`),
reassociation (`backend/cse.py`), unswitching (`middle/unswitch.py`). **Load hoisting runs for the first
time here:** C QBE's `loadopt` runs before
any `fillloop`, so every loop depth is 0 and its insert-a-load path (`LLoad`) never executes; the
second `loadopt` has real depths. That path had never run: it dropped `sl.off` (fixed; harmless in C
QBE, where it is dead code, as of `e786f06`, master's tip on 2026-11-01). Treat it as new code. The plan for the rest of LLVM's and GCC's optimizations is in `TODO.md`.

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
| Windows 11 arm64 (Snapdragon X, CPython 3.13 arm64) | works (cimo end to end, see "Windows arm64") | same, then `FlushInstructionCache` |
| Pyodide 0.27.7 (Node 22) | works, zero-copy | see below |
| macOS 26.7 arm64 (M4): Apple CLT Python 3.9, Homebrew 3.14, python.org 3.13 | works | `mmap(MAP_JIT)` + `pthread_jit_write_protect_np` + `sys_icache_invalidate`. W^X `mmap` RW → `mprotect` RX also works. Plain RWX `mmap` is **refused** (`EACCES`); never use it on macOS. |
| Locked-down Windows (AppLocker/WDAC, third-party EDR) | not yet tested | default Defender didn't object; enterprise policies may |

To retest a platform without opening any network access: put `loadtest.py` inline in EC2 user
data, run it at boot, write results to the serial console (Linux `/dev/ttyS0`; Windows a
`SerialPort` on COM1), and read them with `aws ec2 get-console-output`. No key pair, IAM role or
inbound rules needed.

- **arm64 needs an instruction-cache flush** after writing code.
  - macOS: `sys_icache_invalidate`.
  - Linux: `__clear_cache` is exported by `libgcc_s.so.1`, but that library isn't guaranteed
    (musl, minimal images). `cimo.loader` uses its own 25-instruction `dc cvau` / `ic ivau` routine
    (line sizes from `CTR_EL0`, loops skipped when its IDC/DIC bits say so).
  - Windows: `FlushInstructionCache(GetCurrentProcess(), ...)`.
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
  unlike Emscripten: pass `int64`/numpy buffers, not `ctypes.c_long` (32-bit there). So a heap request can exceed what
  wasm32 has: `__cimo_wasm_grow` returns NULL past 0x7fff0000 bytes (2026-11-07; a 4 GiB request wrapped to a small block
  before, and one just under 2 GiB reached Python as a negative size), and Python's grow returns NULL when it can't have
  the block. `module.free`, `reset_heap` and `heap_bytes` work in Pyodide too (`free` exported; the runtime's functions
  called as kernels, their pointers being i64).
- **ABI (cimo's own):** wasm params in order; pointer parameters, arguments and results are i32 at the boundary, as
  Emscripten's code passes them (2026-11-07: the frontend's class `p`, `wasmptr=`; i64 inside), so a kernel can call a
  host function by address (Pyodide's `frexp(double, int *)` trapped on the signature); struct returns through a hidden first pointer;
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
  1.1 ms and matches numpy bit for bit; the single file is 1.1 MB.
- **Tests:** `tests/test_wasm.py` runs modules under Node in a stand-in loader
  (`tests/wasm/host.mjs`); with `PYODIDE_JS` set, also in real Pyodide.

## LLM inference spike (2026-11-01)

karpathy's `llama2.c` (MIT, 1k lines, float32, Llama-2 architecture) compiles and generates correct
text in Pyodide with no compiler change; only file I/O, `sscanf` and `stderr` had to go (weights arrive
as a buffer). Compile: 4.5 s first, 0.7 s once the runtime is cached. Single thread, Node 22, aarch64
WSL2, mains, tok/s:

| Model | cimo wasm (Pyodide) | cimo arm64 |
|---|---|---|
| stories15M | 75 | 70 |
| stories110M | 9.0 | |

- Throughput scales with parameters: about 1 G params x tok/s today, 4-4.7 with 128-bit SIMD and
  reassociation (the ceiling at this vector width).
- Strict FP blocks vectorizing the dot-product reductions, so these loops need explicit SIMD, not
  only an auto-vectorizer.
- wasm beats our arm64 backend on this workload; unexplained.
- ggml is not an option without C++ (`gguf.cpp`, `ggml-cpu/ops.cpp`, `ggml-backend.cpp`).
- **Through scramjit:** the same forward pass as ~100 lines of plain Python (no helpers or slices, since
  neither compiles yet) runs in Pyodide at 88 tok/s, with default options or with
  `boundscheck=False, parity=False`, against 0.117 undecorated: about 750x. A
  one-line edit recompiles in 1.75 s there. It runs as float64 because scramjit rejects float32
  arithmetic. Native arm64 gets 90 to 92. It got 28 until scramjit's one-line wrapping
  helpers became macros: `CALLER_LIMIT` stopped inlining them partway through this function (see
  `TODO.md`), and wasm's defaults dropped to 13.5 the same way.
- Blockers for a real chat model: wasm SIMD, threads (Pyodide has none and its memory isn't shared,
  so a standalone module on shared memory), quantized kernels (a 1B float32 model doesn't fit in
  wasm32's 4 GB).

## Vector code (2026-11-02)

Built on arm64 and x86-64: class `q` (`ext.vector`) and the auto-vectorizer (`middle/vectorize.py`), for now only straight loops over 4- and 8-byte lanes, and `_Float16` arrays on arm64 as 4 float lanes (a `loaduh` only an `h2f` reads becomes `qldh`: `ldr d`, `fcvtl`; an `f2h` only a `storeh` reads, `qsth`: `fcvtn`, `str d`; 2026-11-07; x86-64 with F16C: `vcvtph2ps`/`vcvtps2ph $4` with memory, 8 halves a ymm with AVX2). The middle end copies the bit builtins and `h2f`/`f2h` now (`ir.PURE_SAFE`): a loop with one bailed out of every loop optimization before. GNU vector types compute in class q where it has the operation (2026-11-07, below); wasm isn't vectorized. The gap it closes: scalar code is level with gcc and clang on the 15 kernels (geomean 0.96 to 1.09x on
Neoverse N1, V1, V2), `-O3` is 1.55 to 1.73x ahead of it, and that's vectorization (`rgb2gray` up to 6.6x; `saxpy`, `jacobi2d`,
`conv3x3` 3 to 4x).

- **128 bits everywhere:** NEON, x86-64's baseline SSE2 and wasm SIMD all have it, so one IR class fits all three and nothing
  needs CPU dispatch. 256-bit AVX2 can come later, behind a check at load.
- **IR:** one class for a 128-bit value, in the FP registers, and operations that name their lanes (4 x i32, 2 x f64...), so the
  class doesn't multiply. Spill slots are 16 bytes.
- **Order:** (1) the IR class through register allocation, spilling, parsing and arm64's selection and assembler; (2) the
  auto-vectorizer on arm64, which shows whether the operations are the right ones before they're ported; (3) x86-64; (4) GNU
  vector types in the frontend (`vector_size`, operators, subscripts, `__builtin_shuffle`); (5) wasm SIMD; (6) SLP.
- **Auto-vectorizer:** `middle/inner.py` already proves what LLVM's `LoopVectorizationLegality` checks: trip count, overlap and
  overflow checks at run time, affine addresses, one versioned fast path. Where it emits 4 scalar copies, a loop with
  consecutive accesses gets one vector operation and the scalar loop as the remainder.
- **Overlap at run time:** arrays that may overlap are compared before the loop, as LLVM's `LoopVersioning`, up to 8 pairs.
  Unroll-and-jam leaves alone a loop that vectorizes, as LLVM has it off by default.
- **First result (Neoverse N1, V1, V2):** 2 to 3.4x on `saxpy`, `jacobi2d`, `conv3x3`; geomean against gcc `-O3` 1.50 to 1.53x
  down to 1.21 to 1.28x before `gemm`.
- **Float sums in C's order:** the vector loop adds its lanes one by one (gcc's fold-left reductions), only where the rest of the
  loop is worth more than those adds' latency (`vectorize._worth`), so `dot` and `prefix_sum` stay scalar. A vector division or
  square root counts as one a lane on arm64: Neoverse N1, V1, V2 ran `nbody_acc` no faster that way (V1 1.35 to 1.46x of gcc), and
  x86-64 ran it 1.4 to 1.5x faster. Polynomial sums measured on Graviton: float 0.63 to 0.66x the scalar loop's time, double 0.79 to 0.85x.
  gcc gets `nbody_acc`'s other 1.3x from vectorizing the outer loop, each lane its own sum. scramjit needs numpy's pairwise order
  for its sums, which is a loop it writes itself (`TODO.md`).
- **`#pragma STDC FP_CONTRACT` (2026-11-07):** ON, OFF, DEFAULT (the compile's `fma=`) set the frontend's
  contraction from there on; a compound statement restores the value it started with (C11 7.12.2). ON does nothing
  without a fused multiply-add (wasm, x86-64 without FMA: `fmaok`).
- **Ctrl-C (2026-11-07, POSIX):** cimo.fault's handler also takes SIGINT. A pc in a module's own code (before
  `__cimo_enter`: not the runtime, trampolines or entries, which may be mid-update) inside a kernel unwinds the innermost
  entry frame with `E_INTERRUPT`; `State.error` then runs Python's handler (`signal.raise_signal`), and raises
  `KernelError('interrupt')` if it doesn't raise. Elsewhere SIGINT goes to Python's handler. No polling in loops. Windows
  delivers Ctrl-C on its own thread: not done.
- **Inlining budget (2026-11-07):** a caller's `CALLER_LIMIT` goes to its call sites by loop weight, deepest first;
  callees no bigger than the call with its arguments are free.
- **Last values (2026-11-07):** a header phi whose latch value is from outside the loop and that nothing in the loop reads
  (`f = 1` as a flag) is set to that value by each vector iteration (`_last`), so it doesn't stop vectorizing.
- **`#pragma omp simd reduction(op:s, ...)` (2026-11-07):** `+`, `-`, `*` on float variables. The frontend marks the
  assignment's operation (IR `addassoc`, `subassoc`, `mulassoc`, `fmacassoc`; `Ins.reassoc`) in the loop the pragma
  precedes, and `_ordered` takes a marked sum or product as an unordered reduction (lanes from -0 or 1, INTERLEAVE
  copies). `fmin`/`fmax` reductions are unordered without it (fminnm/fmaxnm are associative, a quiet NaN the lane
  identity). 1M-double dot on the laptop (not a benchmark): 1.30 to 0.22 ms. Float selects (`x > 0 ? a : b`) still
  don't vectorize on x86-64, so a reduction through one stays scalar there.
- **Conditions (2026-11-05):** a comparison becomes a lane mask (`qceq`/`qcgt`/`qcge`/`qcugt`/`qcuge`, operands swapped for
  `<`, `<=`, negated for `!=`; ordered/unordered tests have none), read only by `qsel0`/`qsel1`, so a select stays exact for NaNs
  and signed zeros. `ifconv` lets an arm hold an inner diamond's select, so nested ternaries (a clamp) convert inside out.
  `qsel1` is pinned like `qfma1` and `sel1` (it wasn't: gvn merged two with different masks and gcm hoisted one, failing
  `x > 0 ? 2.0f : 1.0f` or using another select's mask; adversarial read), and arm64 isel asserts each `qsel0` its pair.
  x86-64 if-converts integer diamonds only (no scalar float select), so its float conditions stay scalar. On the Snapdragon,
  100,000 elements against the same build without the vectorizer: `relu` on doubles 1.74x, a nested float clamp 4.95x, an
  int max 2.02x. Found on the way: x86-64's 64-bit signed compare (`qlower.cgt`, no `pcmpgtq` in SSE2) used a wrong formula,
  wrong for 43% of random pairs; nothing produced one until now (`test_64_bit_compare_at_the_edges`).
- **Reductions through a condition (2026-11-05):** an integer `r = x > r ? x : r`, any comparison of x and r that orders
  them, either way round, keeps a max or min per lane from the type's identity; closeloop folds the lanes into r through the
  same comparison and select, exact since equal integers are the same. Floats are refused: per-lane maxima merged in lane
  order can pick another of two equal zeros than the scalar loop.
- **Four vectors an iteration** (`vectorize.interleave`, LLVM's interleave count) for loops whose only reductions are
  unordered and that store nothing, each copy with its own accumulators, merged on the way out. One accumulator left the
  chain of compare and select latency-bound: int64 max was no faster than scalar until then. `isum` 204 to 63 us on
  1,000,000 elements (Snapdragon), its loop 15 to 27 instructions (`bench_gate.json`).
- **Unit-stride versioning (`middle/stride.py`):** a multiply by a value the loop doesn't change, on the way to an address
  (`a[i * st]`), gets a copy of the loop with it 1, chosen before the loop (LLVM's symbolic stride speculation), using
  unswitch's `clone`. Only when the target vectorizes, at most 4 a function, never twice for one loop. scramjit's
  reductions take a stride for any axis: int64 max with stride 1 0.87 to 0.28 ms, level with numpy (0.29).
- **Loops that leave early (`middle/earlyexit.py`, 2026-11-05):** a vector loop in front of the original checks
  COPIES (4) vectors of iterations at once and goes on only if no lane leaves; otherwise the original loop runs that
  chunk from its start, so it leaves where C does with C's values, and its latch returns to the vector loop at the chunk's
  end (phis `stop`, `flag`): a page crossing doesn't leave the rest scalar. Nothing after the loop changes. Reading past
  the exit is safe only within the page of an element C reads (the chunk's first, in the body's first block, which every
  iteration runs), so a chunk whose 64 bytes cross a 4 KB page runs scalar (`test_leaving_early_reads_nothing_it_may_not`).
  The chunk bound `bound - (chunk - 1)` is computed once, with a wrap guard. With one vector a chunk the overhead lost:
  `any` on doubles 0.35 to 0.51 ms; with four, 0.35 to 0.19 (numpy 0.17), `find` on int32 3.9x (Snapdragon).
  `unswitch.clone` takes several exit targets (each reached from the loop alone, values used only there), so the
  unit-stride copy works for loops that return from inside.
- **From the adversarial read of register allocation (2026-11-05):** a cycle of parallel moves through spilled phis
  swaps a register with a stack slot, or two slots, which the emitters took as two registers (wrong values with 70
  rotating q phis; `test_phis_permuted_through_stack_slots`): now through a scratch, v31/x16 on arm64, xmm15/ymm15 on
  x86-64, and two slots 8 bytes at a time through d31 and x16 or r11. Spilling counts qfma1's and qsel1's destination as
  read (isel made it a copy of the addend or mask), or allocation ran out of registers at 30 live q values. qacrosscalls
  puts reloads before a qfma0/qsel0, not between it and its pair. arm64's ldp scheduling treats qload/qstore as memory.
- **A call keeps at most 64 bits of an FP register on arm64:** a `q` temporary live across one lives in memory
  (`isel.qacrosscalls`), and `q` temporaries stay out of v8 to v15 (`rqavoid`), whose low halves the function must keep.
- **Memory:** `qload` and `qstore` count as volatile in the memory passes, so nothing forwards across them. Safe, and
  to be tightened once something depends on it.
- **Measure on EC2:** Neoverse N1, V1 and V2 as well as this machine, whose Snapdragon hid the 52% cost of a copy in `dot`.
- **x86-64 is baseline SSE2:** no CPU check guards the code, so nothing later (`pmulld`, `pcmpgtq`, `pmovsx`, FMA). `amd64/qlower.py`
  writes the missing operations in what's there before the last cleanup, so gvn and gcm merge their masks and take them out of
  the loop. `qfma` has no form: faking it would round differently, and without `ext.fma` nothing makes one.
- **x86-64 float moves:** QBE's `movss`/`movsd` between registers keep the destination's upper bits, so they wait for whatever last
  wrote it. `movaps`, and a cleared destination before `sqrtsd` and conversions, as gcc: `nbody_acc` 6.5x to 1.0x against gcc's
  scalar code on Sapphire Rapids, `jacobi2d` 7.4x to 1.08x. Zen 4 hid most of the first.
- **Flags on x86-64:** `extern_array + 4` as a phi's argument is made at the end of the block before, after the compare the jump
  reads, so it's a `lea`, never an `add` (YARPGen: a wrong branch, then a crash). A random-loop seed differed only in a NaN's bits:
  with two NaNs `addss` keeps the first, and commuting an add picks another. C doesn't fix them, so that test ignores them on x86-64.
- **gcm's order within a block** keeps pinned instructions (calls, loads, stores) where they are and makes a pure one wait for
  what it reads; QBE pulled the definition up instead. gvn moves the pure instructions of a branch that folded away to the start
  block, simplcfg merges a call's block into it, and the call, then a load, went up past stores they read: wrong Csmith checksums
  on both targets once small functions were inlined (`tests/test_gcm_order.py`). A select whose condition folded to a constant
  becomes copies on x86-64, as on arm64.
- **Three more from Csmith and the random loops:** the value of an assignment, a compound assignment or an increment of a
  bit-field promotes as the bit-field does (`(s.f7 = 0) >= -5` is 1 for `unsigned f7 : 2`; cproc kept `unsigned int`, in C
  cproc too); jump threading refills loop depths when it changed blocks (loadopt asserted on x86-64; gcc-torture's
  `loop-12.c` shows the stale ones); a cycle of register moves with a `q` value in it swaps all 128 bits (`q | s` was 7).
  gcc `-O0` on arm64 compiles `0.0f - x` as `fneg`, -0 for x = 0, so the random-loop generator doesn't write it.
- **x86-64 for this CPU (`cimo/cpu.py`):** cpuid and xgetbv, assembled by cimo itself, run once; the features go into
  `ext.x86` and into both cache keys (a module's and the runtime's), only for this machine's own target. `CIMO_X86=baseline`
  for SSE2. FMA: `fma0`/`fma1` as afmadd with a Mem for the third, as arm64, emitted as the 231 or 213 form by which operand
  the result shares; negated operands folded (vfnmadd, vfmsub, vfnmsub); vectors keep the addend in the destination, as
  fmla. **No FMA whose addend is on a loop's recurrence** (`amd64/fmachain.py`), as gcc's avoid_fma_chains: FMA latency 4
  against an add's 2 (Sapphire Rapids) or 3 (Zen 4) made `dot` 2.0x and 1.33x slower, `iir` 1.34x and 1.22x; split, a
  negated addend is a sub again, not a neg on the path. **Loop headers on 64 bytes, not 16:** where a short loop sat in its
  line moved `gemm` 1.1 to 1.5x when one instruction changed, standalone it was faster with FMA; at 64 baseline is no slower.
  Result, against cimo's baseline build: geomean 0.967, 0.959, 0.961 (Ice Lake, Sapphire Rapids, Zen 4); `conv3x3` 0.74 to
  0.81, `saxpy` and `gemm` 0.86 to 0.89, `mandelbrot` 0.89 to 0.94. Against `gcc -O3 -march=native` 1.28, 1.18, 1.44x:
  256-bit vectors and byte lanes. The benchmark's gcc builds are cached by their flags (they weren't).
- **Fuzzing on EC2:** an ssh silent for 18 minutes never came back, from here, so run detached and poll, as `fuzz_aws.py`. A timeout under
  900 s fails YARPGen's largest programs for being slow: their compile takes about 300 s on 16 workers.
- **x86-64 scalar:** the middle end's if-conversion selects on x86-64 too, `cmov` as QBE's ifopt (`edit_distance` 2.1 to 2.4x of gcc
  down to 0.93 to 1.20x); a load, an operation and a store to one address are one instruction, and an extended load extends in the
  load (`histogram` on Zen 4 1.56 to 1.03x; gcc's own `addl $1, (mem)`).
- **x86-64 result (Ice Lake, Sapphire Rapids, Zen 4; gcc -O3 also baseline):** geomean 1.46, 1.47, 1.61x with the vectorizer off,
  1.07, 1.05, 1.08x on. Left: `nbody_acc` 1.3 to 1.4x, `gemm` 1.3 to 1.5x, `saxpy` 1.2 to 1.7x, `conv3x3`'s scalar path 2 to 4x,
  `rgb2gray` 1.0x because SSE2 gcc doesn't vectorize it either.
- **gemm, register block (2026-11-06):** unroll-and-jam by 4 around a vectorized inner loop (c's vector loaded and stored once
  per 4 rows; jam was off for vectorizing loops), with store-to-load forwarding in the vector body. Against `gcc -O3 -march=native`
  / `-mcpu=native`, gemm before and after: Graviton 2 1.47 to 0.92x, Graviton 3 1.14 to 0.79x, Graviton 4 1.50 to 1.11x, Ice Lake
  1.51 to 1.00x, Sapphire Rapids 1.20 to 0.86x, Zen 4 1.17 to 0.78x; 15-kernel geomean 1.03, 1.06, 1.08, 1.07, 1.03, 1.14x.
  Then two vectors an iteration for loops that store, offsets folded into q loads (arm64), b's loads folded into vfmadd
  (x86-64; Ice Lake didn't move: L2-bound, all of b read per row of c), and tiling of k by 16 (`middle/tile.py`; KB 16 to
  32 was best everywhere at n = 384, 64 and up no gain): gemm 0.68, 0.51, 0.73, 0.60, 0.48, 0.69x on the same six machines
  (2026-11-06); geomean 0.98, 0.99, 1.00, 1.04, 0.98, 1.14x.
- **Closing the benchmark gaps (2026-11-06), what each took:** rgb2gray: bytes in 16-bit lanes when no value can leave
  [0, 65535] (`vectorize._narrow`), pixels by one `ld3 .16b` into v28-v30 with `umull`/`umlal`(`2`) and `shrn` on arm64,
  by two byte windows and `vpshufb` masks on x86-64. nbody_acc: outer-loop vectorization (`middle/outer.py`), lanes of i
  so every sum stays in C's order; vectorizing j with in-order lane adds was slower on most machines. edit_distance:
  `a < y + 1` as `a <= y` (nowrap) and a store the next iteration loads carried in a register; x86-64 then needed the
  compare re-issued at the cmov (an add came between). mandelbrot: a product computed twice in a loop isn't fused
  (`fmashare.py`, as gcc). jacobi2d, conv3x3: `ldur`/`stur` offsets, 4 vectors an iteration for long fp chains (arm64).
  Measured and left: arm64 `fmachain` (mandelbrot +7% on V1/V2) breaks matching gcc's contracted bits on dot, iir,
  conv3x3, nbody_acc (`test_ext_kernels`); a phi's latch value sunk to the block end didn't remove mandelbrot's fmov.
- **What remains is vector width:** gcc -O3 native uses 256-bit SVE on Graviton 3 and AVX-512 on Zen 4 (zmm in nine of the
  kernels; 256-bit on Sapphire Rapids); cimo has 128-bit NEON and 256-bit AVX2. Graviton 4's SVE is 128-bit, and there the
  same kernels are level.

## GPU (planned, 2026-11-02)

Nothing here is built. Plain C runs on any GPU the machine has, including in a browser tab, with no vendor
toolkit and nothing to install beyond a driver. Tuned kernels are C too, shipped as source and compiled for
the card on first use. cimo owns the targets; scramjit exposes them.

**Not the goal:** beating tuned kernels on data-center GPUs from plain loops. A production stack read for this
(`~/code/modular`; read, not run) holds about 107k lines of GPU matmul and 120k of attention, written by hand
per GPU generation. Its plain-loop path exists only as a fallback.

### Layout

Everything GPU lives in `python/cimo/gpu/`, so one directory is all onefile has to leave out. C that runs on the GPU
or calls a driver is compiled by cimo on demand, as `python/cimo/runtime/` already is. Code generators stay in Python.

| Path, under `python/cimo/` | Holds | Language |
|---|---|---|
| `gpu/include/cimo/gpu.h` | what kernels see: thread ids, barriers, shared memory, target instructions | C |
| `gpu/lib/` | device math (`exp`, `log`, `sin`, softmax) | C |
| `gpu/kernels/<target>/` | tuned kernels (`ptx/gemm_sm90.c`) | C |
| `gpu/host/` | device discovery, buffers, dispatch: `cuda.c`, `vulkan.c`, `hip.c`, `metal.c` | C |
| `gpu/host/webgpu.py` | the browser host, through Pyodide's `js` module; a tab has no driver for C to call | Python |
| `gpu/backend/{wgsl,spirv,ptx,msl,hip}/` | IR to target code | Python |
| `backend/` | loop structuring shared with wasm, moved out of `backend/wasm/emit.py` | Python |

- **Generators stay in Python** because they read the compiler's IR, which is Python objects. In C they would need
  the IR marshalled, and would have to compile before any kernel could: on every cold start, and at about 1.5k lines
  per 2.5 s in a tab (1,611 lines measured through scramjit), a noticeable wait. Revisit only if emission shows up as
  the bottleneck.
- **Packaging:** wheels and sdists include `gpu/` (`package-data`).
- **Onefile leaves it out by default:** `cimo onefile --gpu` puts it in. CPU users keep a small `cimo.py`, and a page
  using GPU builds with the flag. Without it, `cimo.gpu` raises an error naming the flag. `import cimo` never imports
  `cimo.gpu` until it's used (module `__getattr__`), so CPU users pay nothing in either build.
- **Cache keys:** `gpu/` sources go into the key of GPU kernels only, so editing them doesn't invalidate CPU kernels,
  and a CPU kernel has the same key in both builds.
- `cimo` stays pure Python: drivers are reached from C compiled by cimo, resolved with `dlsym` at load.

### Using it

```python
gpu = cimo.gpu.device()              # finds a driver; no toolkit
k = cimo.compile(r"""
void matmul(int n, int m, int p,
            const float a[n][m], const float b[m][p], float c[n][p]) {
    #pragma cimo parallel
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            float s = 0;
            for (int t = 0; t < m; t++)
                s += a[i][t] * b[t][j];
            c[i][j] = s;
        }
}
""", device=gpu)
k.matmul(n, m, p, a, b, c)           # copies a, b in; runs; copies c back
```

- **Same source everywhere:** without `device=`, the pragma's loop runs on the CPU, so a kernel is testable as C.
- **Copies follow the signature:** `const` arrays go in, the others go in and come back. `gpu.array(a)` keeps data
  on the device across calls. Unified memory (Apple, AMD APUs) skips the copy.
- **In a tab, reading back is asynchronous** (`mapAsync`): `await k.matmul(...)`, or a synchronous call if stack
  switching works from Pyodide (to check).
- **A wrong pragma is a data race.** C keeps the user's word for it; scramjit has to prove a loop safe first.

### Kernel C

Two forms, both ordinary C compiled by the frontend.

**Parallel loops** are the easy form. `#pragma cimo parallel` covers up to 3 perfectly nested canonical loops, which
become the dispatch grid. Scalars become uniforms, array parameters become buffers.

**Explicit kernels** are the form tuned code needs: one function per thread, with the grid and block size set by the
caller.

```c
#include <cimo/gpu.h>

__gpu_kernel void gemm_block(const float *a, const float *b, float *c, int n) {
    __gpu_shared float tile[256];
    int x = gpu_thread_id(0), y = gpu_block_id(1);
    /* copy to shared memory, then the matrix instruction */
    __gpu_asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...");
    gpu_barrier();
}
```

`__gpu_asm` is the one place inline assembly exists. It is target text, passed through; a kernel using it builds only
for that target (`gpu/kernels/<target>/`), so the dispatcher picks by device.

Rejected in a kernel, with a clear error: recursion, `malloc` and other runtime calls, function pointers, storing a
buffer pointer in memory, `double` or 64-bit integers on a device without them. Never narrowed silently.

**Type recovery** is the first thing to prove. WGSL and SPIR-V want typed buffers, and cimo's IR has byte addresses.
In a kernel every load and store must trace back to one buffer parameter plus an offset divisible by its width,
which maps to `buf[i]`. What can't be traced is a compile error on those targets. Loops are rebuilt by the
structuring in `backend/wasm/emit.py` (Ramsey, "Beyond Relooper"), moved somewhere the new backends can share.

### Targets, in order

| # | Target | cimo emits | Compiled by | Needed on the machine | Matrix instructions |
|---|---|---|---|---|---|
| 1 | Browser | WGSL | browser, then driver | a WebGPU browser | no |
| 2 | Radeon, Intel, others | SPIR-V, through Vulkan | driver | Vulkan driver | `VK_KHR_cooperative_matrix` |
| 3 | NVIDIA | PTX | driver (`libcuda`, `cuModuleLoadDataEx`) | NVIDIA driver | `mma.sync`, `tcgen05.mma` by `__gpu_asm` |
| 4 | AMD Instinct | HIP source | `libhiprtc` | ROCm | `mfma` builtins, to check |
| 5 | Apple | Metal source | OS (`newLibrary` from source) | macOS | to check |

- **Browser first:** the production stack read has no WebGPU path, so that's the open ground. It also tests type
  recovery and compile time on the hardest constraints.
- **SPIR-V second:** one backend for several vendors. Cooperative matrix is exposed on RDNA3 and newer by RADV,
  by Intel's Linux driver and elsewhere; confirm per card. It reuses the structuring from WGSL.
- **PTX third:** it allows plain branches, byte addressing and 64-bit types, so it's the closest fit to cimo's IR,
  and needs no type recovery. PTX versions are capped by the driver: emit the lowest that works.
- **AMD:** the kernel driver has no PTX-style format, so the choices are `libhiprtc` (source to code object, loaded with
  `hipModuleLoadData`), comgr (GPU assembly, with registers per GPU family to manage ourselves) or Vulkan. Radeon
  takes Vulkan, the only route with nothing to install. Instinct takes `libhiprtc`, a separate library from ROCm
  7.0 that cimo can't bundle, so the pitch there is "uses the ROCm you have". Whether Instinct has a usable
  Vulkan driver is unchecked.
- **Metal:** source at runtime avoids Xcode, which AIR bitcode and `xcrun metal` would need. Shares a printer with WGSL
  and could move earlier if cheap.

### What the driver optimizes

Drivers compile, but not everything. `CU_JIT_OPTIMIZATION_LEVEL` buys instruction selection, scheduling, register
allocation and local cleanups. Inlining, moving work out of loops, loop transforms and removing checks stay ours on
PTX, since the toolkit's compiler isn't there. Nothing turns a loop into matrix instructions. Browsers and Metal
do more, varying by vendor: measure.

So kernels go through a chosen subset of cimo's passes. Picking the subset is a measurement: every pass on or off,
compile time against speed, per target. Middle-end loop transforms mustn't touch the grid loops.

### Tuned kernels

No physical ceiling: equivalent PTX reaches the same compiler and the same hardware. Whether the driver's compiler
matches NVIDIA's bundled one is unchecked (the stack read ships a cubin).

- **Size for one target** is modest, from the stack read: 2.3k lines of wrappers around NVIDIA's matrix instructions, 1.6k
  for one older-generation GEMM, 15k for all of `sm90` matmul, 3.4k of that being tuning tables.
- **Hand-written holds up:** its dispatcher records matching AMD's vendor library within 1.8% on BF16.
- **Tuning needs the card.** Each generation means rented time. Generating kernels from Python for the exact shape,
  and sweeping parameters briefly on the user's own card, could replace tables: untested.
- **Copy nothing from `~/code/modular`.** Parts aren't Apache-licensed. Write from the vendors' public docs.

Dispatch is by device and shape, in `gpu/kernels/`, falling back to the plain loop.

### scramjit's part

Through cimo's public API only:

- which Python loops are safe to run in parallel, and emitting the pragma
- real `float32` and `int32`: WGSL has no `f64` or `i64`, Metal no `double`, and scramjit rejects `float32` arithmetic today
- `@scram(device=...)`, and deciding which arrays to copy when
- recognising matmul-shaped loops and calling the tuned kernel; plain loops otherwise won't reach matrix instructions

### Real limits

- **Browser:** as far as checked, WGSL doesn't expose matrix instructions, so tuned kernels in a tab top out below native.
  `maxStorageBufferBindingSize` is 128 MiB by default and adapter-dependent above, so large weights split across
  buffers. Long dispatches can trip GPU watchdogs (seconds, by OS and browser), so work gets sliced.
- **Discrete GPUs:** copies cost PCIe bandwidth. A kernel fault can lose the whole context.
- **ROCm** isn't part of the base driver, unlike `libcuda` on NVIDIA.

### Milestones

1. **WGSL, plain.** The `matmul` above compiles and runs on WebGPU in a tab. Results match the CPU within rounding. Same GPU,
   against hand-written WGSL: speed, and compile time with passes on and off. Target: a 50-line kernel edit to dispatch
   in under 1 s. Proves type recovery, and how far plain loops get.
2. **Host in C and SPIR-V.** `gpu/host/vulkan.c` and the same kernel natively. Cold-start cost of compiling
   the host measured.
3. **Cooperative matrix.** One explicit GEMM on an RDNA3 or newer card, against the plain loop.
4. **PTX and one tuned GEMM.** `gpu/host/cuda.c`, `__gpu_asm`, one generation, one format. Same rented card: plain loop,
   ours, the stack read. This number decides how far tuned kernels go.
5. **Instinct and Metal.** As demand says.

**Testing:** software adapters (WebGPU's in headless Chromium, a software Vulkan driver) in CI for correctness; both to
confirm on arm64 Linux. Speed only on real hardware, on mains. This machine's WSL has no usable GPU: WGSL speed runs
in a Windows browser, the rest on rented cards.

### Unknown until measured

Type recovery on real kernels. Compile time through cimo's passes. The gap from plain loops to tuned. Cooperative matrix
against direct instructions. The driver's PTX compiler against NVIDIA's bundled one. `mfma` through `libhiprtc`. Stack
switching for synchronous calls in Pyodide.

Sources: [HIP runtime compiler](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.1/how-to/hip_rtc.html),
[RADV cooperative matrix](https://www.phoronix.com/news/RADV-VK_KHR_cooperative_matrix),
[Intel's driver](https://www.phoronix.com/news/Intel-ANV-Cooperative-Matrix),
[`VK_KHR_cooperative_matrix`](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html).

## Windows arm64 (2026-10-31)

Target `arm64_win` (preprocessor/backend/linker; frontend `aarch64-windows`), picked by
`driver.TARGET` when `sys.platform == 'win32'`. Same pipeline and runtime as Linux; differences:

- **Data model: LP64, as on every cimo target** (decided 2026-10-31; Windows' own C is LLP64):
  `long` is 64-bit, so kernels behave the same everywhere. Windows conventions otherwise:
  plain `char` signed, `wchar_t` = `unsigned short`, `long double` = `double`, `int64_t` =
  `long long`; `_WIN32`, `_WIN64`, `_M_ARM64`. Consequences: Python's `ctypes.c_long` is 32
  bits on Windows (as in Pyodide), so `long *` buffers must be int64; and a host DLL function
  whose signature uses `long` (CRT `strtol`, Win32 `LONG`) must be declared with `int` in the
  kernel; `compile()` warns (`UserWarning`) when a symbol resolved from a `link=` library was
  declared with `long` anywhere in its type (pointers, struct members, callbacks). (A full LLP64 variant was built and tested first: frontend `size_t`/`ptrdiff_t`
  types, headers, and `heap.c`/`abort.c`/`builtins.c` assumed `long` is pointer-sized; musl
  needed nothing else. Dropped for uniformity.)
- **ABI:** AAPCS64 as on Linux (callee-side extension of small ints, like clang), except that
  calls to and definitions of *variadic* functions pass every argument, named ones included,
  in x0-x7 then the stack: floats as their bits in integer registers, HFAs as ordinary
  composites. A variadic function saves only x0-x7, right below its stack arguments; `va_list`
  is a `char *` (Apple's `va_arg`). A struct split between x7 and the stack (MSVC allows it)
  is a compile error. x18 is the TEB: the backend's store fix-up scratch (x18 on Linux) is not
  used, and the runtime's context key is `mov x10, x18`.
- **Stack probes:** Windows commits a stack through one guard page, so a frame or VLA that
  skips a page faults. Prologues of frames over 4 KB touch every page first (x16/x17, like
  `__chkstk`); `alloca`/VLAs step sp down a page at a time. Stack limit for the recursion guard:
  `GetCurrentThreadStackLimits`.
- **Host:** `link=['libc']`/`'libm'` mean `ucrtbase.dll` (exports no variadic functions;
  `msvcrt.dll` does); `host_headers=True` is a CompileError (the SDK headers need MSVC
  extensions). Cache: `%LOCALAPPDATA%\cimo`.
- **Faults:** a vectored exception handler resumes at the unwind glue (`cimo/fault.py`). Without
  it, an access violation in a kernel reached through a trampoline ends the process silently: the
  dispatcher cannot unwind through code without unwind data to ctypes' SEH handler. A raw call of
  a kernel's address does reach ctypes, as `OSError`. Allocas must never wrap sp (a size past the
  bottom of the address space, and Windows stacks are low): the kernel needs a valid sp to
  dispatch the exception at all.
- **Testing:** a Windows-on-arm machine with WSL; a *native arm64* CPython is needed (the
  default python.org/Store install on Snapdragon is often x64 under emulation, which cannot
  call arm64 code): `uv python install cpython-3.13-windows-aarch64`. Reference compiler:
  Visual Studio's `VC/Tools/Llvm/ARM64/bin/clang.exe --target=aarch64-pc-windows-msvc`
  (`-nostdlib -fuse-ld=lld -Wl,/noentry` needs no SDK). `tests/test_windows.py`: LLP64 types
  and the Windows codegen on any host; on Windows, calls into msvcrt's `_snprintf`/`sscanf`,
  calls both ways with clang-built code (variadic, HFA, mixed and stack arguments), probes on
  a 1.6 MB frame. The gcc-based differential tests skip on Windows; the rest of the suite passes.

## macOS on Apple silicon (2026-11-01)

First run of `python/tests` on an M-series Mac (macOS 26.1, Apple CLT Python 3.9 and Python
3.13): the suite, the corpora and the examples pass; `test_abi.py` checks calls in both
directions against Apple clang (`-dynamiclib`), and `tests/test_macos.py` holds the checks below
(codegen ones run on any host). CI: `macos-15`. What it took:

- **ABI (Apple's arm64 differs from AAPCS64; the C QBE `arm64_apple` target had all three):**
  - Every *variadic* argument goes on the stack, structs included (by value, HFAs too; over 16
    bytes a pointer to a copy). `argsclass` did this for scalars only and put structs in x/v
    registers.
  - Stack arguments are packed at their own size and alignment (a `char` takes 1 byte), not 8-byte
    slots. cproc emits `w` for `char`/`short` parameters and arguments, so the backend couldn't
    tell; on Apple the frontend now emits QBE's sub-word classes (`sb`, `ub`, `sh`, `uh`), which
    the backend already lays out by size (`frontend.qbe.abiclass`; other targets keep cproc's IR).
    Python bindings map them to `c_int8`..`c_uint16` so libffi packs them the same way.
  - A 16-byte aligned composite takes the next two registers: no rounding up to an even register
    (AAPCS64 C.8); on the stack it is still 16-byte aligned.
- **Loader:** a hint alone doesn't put the data mapping after the `MAP_JIT` code in a busy
  process (the pages after a code-sized hole are taken), so the image was never snapshotted and the
  image cache always recompiled. Now one `MAP_JIT` mapping of both sizes, its tail replaced by RW
  memory with `MAP_FIXED` (`MAP_JIT` itself refuses `MAP_FIXED`: `EINVAL`).
- **`host_headers=True` with the SDK** (`xcrun --show-sdk-path`): `sys/cdefs.h` needs
  `__arm64__` (clang's names `__arm64__`, `__arm64`, `__AARCH64EL__`, `__LITTLE_ENDIAN__` are
  predefined on Apple); `_OSSwapInt16/32/64` (htonl & co.) are defined for arm64 only under
  `__GNUC__`, which cimo doesn't claim, so the `arm64_apple` target predefines them as
  `__builtin_bswap*`; the SDK has `struct {...} __attribute__((packed))` (112 times) and
  `aligned(N)` after struct bodies and stray `;` at file scope, all now accepted (as gcc/clang
  do); its "Unsupported compiler detected" `#warning` is dropped on that target. Claiming
  `__GNUC__` instead was rejected: the SDK would then use statement expressions, `__asm` and
  builtins cimo doesn't have.
- **Test references:** macOS `cc`/`cpp`/`gcc` are Apple clang (`cpp` is a traditional one, not
  GNU: the GNU cpp comparisons skip). clang vectorizes reductions such as `s += x[i] * y[i]`
  into a multiply and a separate add even with `-ffp-contract=fast`, so the fused reference is
  built with `-fno-vectorize -fno-slp-vectorize`. Asm checks use the `_` symbol prefix and `L`
  local labels.
- `corpora_expect.txt` is generated on Linux: on macOS `long double` is `double`, so 29 more
  torture programs pass there and `--update` would list different reasons. The two frontend fixes
  above made 7 Linux entries pass too (regenerated).
- macOS's bash is 3.2: `source <(...)` doesn't work (use `eval "$(cimo completion bash)"`), and a
  quoted array slice joins into one word when IFS has no space (the completion script slices
  before setting IFS).

## Threads (2026-11-05)

Calls release the GIL (ctypes), each thread has its own context, heap and alternate signal stack, and the context and
fault tables are published lock-free with a barrier before the pointer; nothing in cimo leans on the GIL.
`tests/test_threads.py` (8 threads x 1,000 calls mixing work, cross-thread malloc/free, failing asserts, faults and
printf; modules compiled, faulted in and dropped while others run; a reused pool) passes under CPython 3.13 and
free-threaded 3.13t, which keeps the GIL off. Measured on EC2 (c7g.2xlarge, c7i.2xlarge; Ubuntu's Python and uv's 3.13t):
a loop split over 1, 2, 4, 8 threads 2.0x, 4.0x, 7.7x (Graviton 3, 8 cores) and 2.0x, 3.9x, 7.0x (Sapphire Rapids, 4 cores
and 8 threads), the same free-threaded. A call: `Kernel.__call__` 0.74-1.30 us, a bare ctypes call 0.25-0.41 us, of
which releasing and taking the GIL back is 45-60 ns (15-25 ns free-threaded): the wrapper was the cost. Since then each
kernel has a class whose `__call__` is generated for its parameters, reading the error word and stdout length through a
per-thread `memoryview` of the context: a `void` call 763 to 364 ns (Graviton 3), 455 to 206 (Sapphire Rapids), against
161 and 94 for ctypes alone.

**No ctypes on a call (2026-11-07):** each kernel's class `__call__` is a builtin function whose code is a CPython
`METH_FASTCALL` entry, one per signature (`driver._pyentry`, IR compiled when a module is assembled and kept in memory;
a cached image has it). Its `self` is the address of the kernel's words (Python call, post, counts, trampoline, kinds).
`runtime/pyentry.c` converts an exact float (read in place), an exact int (`PyLong_AsLongLongAndOverflow`) and a
writable buffer of a byte or more (`PyObject_GetBuffer`), releases the GIL around the trampoline, makes the result and,
if the error word or stdout length is set, calls the kernel's `post`, which flushes and raises. Anything else (bool,
numpy scalars, `bytes`, read-only or empty buffers, ints over 64 bits, no context yet) goes to the Python call, so
results and errors are the ctypes path's. The Python C API is a table filled at load (`driver._pyapi`: the object
header's size found by reading `id(1.5)`, so free-threaded builds work), so nothing resolves at link time. A thread
reusing a dead thread's key goes through Python first: context word 120 (`CTX_PY`) is set by `State.ctx` and cleared
by a finalizer in the thread's `threading.local`. The builtin owns its `PyMethodDef` and words through its `m_module`
(its dealloc reads the method's flags before dropping that): owned by the kernel instead, the per-kernel class could
outlive them, and free-threaded 3.13 (mimalloc reuses blocks at once) crashed in GC. A `double, double, long` call 661 to 92 ns on the Snapdragon X laptop
(a bare builtin call there 39 ns; the GIL release is most of the rest). Each new set of signatures compiles once per
process, about 5 ms.

## Compile time on large functions (2026-11-05)

scramjit generates single functions of many loops; cimo's compile grew with the square of their size (a 60-loop
function every loop versioned, 4,150 lines of C: 32.6 s; 120 small loops: 21.4 s, 4x per doubling), so scramjit capped
its loop versioning. Two causes, both whole-function work repeated:
- **One transformed loop per middle-end round** (`pipeline._phase`): each round refreshed uses, re-found loops and
  recomputed alias roots over the function, ~200 times for 60 loops (61% and 25% of the time). A round now takes every
  loop no earlier change in it reached: none of its blocks, preheaders or exits changed (`_changed` looks only at the
  transformed loop's region and the blocks `ir.newblock` made), it refers to no temporary made this round, and no changed
  block uses its values (so its use lists are whole; `inner._uses_outside` walks those instead of the function). Block
  numbers only change at a refresh, and a transformation never changes what an existing temporary means, so the round's
  analyses still hold for such a loop. `memdep.Roots` takes an unseen temporary as an offset, never as unknown: a loop that
  uses one must wait, which the rule makes sure of.
- **Constants live through the whole function** (`backend/remat.py`): gvn merged equal constant broadcasts of all loops
  into one value near the start, and arm64's isel loaded each float constant once in the start block: 80% of each
  block's live-in set with 120 loops, so liveness, spill and rega were quadratic. In a function of 256 blocks or more each
  use now has a home, the preheader of the innermost loop it's in, where the constant is made once; arm64's isel then
  keeps float constants per block. Smaller functions are left alone: in the benchmark kernels, changing where a constant
  lives moved rega's choices (mandelbrot: two moves in the innermost loop). `qacrosscalls` skips liveness when there are no
  calls. Result: 32.6 to 7.9 s and 21.4 to 2.7 s; live-in entries 111,794 to 22,812 (about 16 a block, flat).
  `tests/test_compile_time.py` counts rounds and checks large functions bit for bit against the old placement.

## x86-64: Linux and Windows x64 (2026-11-02)

Targets `amd64_sysv` (Linux; frontend `x86_64-sysv`) and `amd64_windows` (frontend
`x86_64-windows`), picked by `driver.TARGET` from `platform.machine()`. Intel Macs are a clear
`CompileError`. Same pipeline, runtime and execution model as arm64; differences:

- **AVX and AVX2 in the assembler (2026-11-05):** VEX-encoded forms (`_h_avx`: three-operand arithmetic, compares,
  shifts by an immediate, shuffles, broadcasts, `vextracti128`, `vblendv*`, moves, `vpmovmskb`; FMA at 256 bits;
  `vzeroupper`), `%ymm` registers; 639 forms byte-identical to GNU as, in `data/amd64_forms.s`. A register move from a high
  register uses the store opcode, as GNU as does, for the two-byte VEX. An x86-64 GNU as on an arm64 machine without root:
  `apt-get download binutils-x86-64-linux-gnu`, `dpkg -x` it, run `x86_64-linux-gnu-as` with `LD_LIBRARY_PATH` at its
  `usr/lib/aarch64-linux-gnu` (`AS_AMD64` for the tests).
- **AVX2: 256-bit q (2026-11-05):** on x86-64 with `avx2` (`cimo.cpu`, in the cache key), class q is 256 bits
  (`ext.qbytes()` 32): the vectorizer, early exits and spill slots take their width from it, and amd64's emitter writes
  each q operation as its three-address VEX form on ymm (`_qins_y`), with AVX2's own instructions for what SSE2 has to
  have written out (`vpmulld`, `vpcmpgtq`, `vpcmpeqq`, byte to dword min and max: qlower leaves them). Per-128-bit-lane
  shuffles and unpacks keep their meaning for what uses them (lane-wise masks, sums of widened halves). `vzeroupper`
  before each call and return of a function with q values. Windows x64 keeps only 128 bits of xmm6-xmm15 across a call,
  so a q value live across one goes to memory (`qacrosscalls`, as on arm64). Tested under Windows-on-arm's x64 emulation,
  which has AVX2, and in CI; tests that drive q directly take the width from the host (`test_vector.W`).
- **No legacy SSE in a function with 256-bit code:** Sapphire Rapids ran `nbody_acc` 249x, `conv3x3` 22x and `gemm` 5.8x
  slower than gcc with SSE scalar instructions (remainder loops, reductions) next to ymm code; Ice Lake and Zen 4 didn't.
  The emitter writes those functions through `_Vex`, which turns each SSE instruction into its VEX form, the destination
  also the first source so the meaning is SSE's; the assembler has the scalar VEX forms (51, as GNU as).
- **Measured (2026-11-05, EC2 m6i/c7i/m7a, best of 7, bench/run.py):** cimo time / gcc `-O3 -march=native`, 128-bit
  build (`CIMO_X86` without avx2) then AVX2: Ice Lake 1.29 to 1.10x, Sapphire Rapids 1.16 to 1.05x, Zen 4 1.41 to 1.18x.
  `saxpy` 1.5-2.0 to 0.93-1.00x, `conv3x3` 1.1-1.7 to 0.92-0.95x, `isum` 1.1-1.9 to 0.56-0.90x (four accumulators of 8
  lanes), `jacobi2d` to 0.91-0.97x. Left: `rgb2gray` (bytes), `nbody_acc` (outer loop), `gemm` 1.1-1.5x.
- **Backend:** QBE's amd64 code (`backend/amd64/`, see its PORTING.md section), verified by
  diffing every pass against C QBE on its 76 tests. `amd64_windows` is cimo's own target over
  QBE's `amd64_win` (which stays as in C for the diffs): xmm6-xmm15 callee-saved (QBE clobbers
  them, corrupting a Windows caller such as Python), stack probes for frames and allocas over
  4 KB through r11 (reserved, as x17 on arm64), extern data through the GOT. Fixes over C QBE
  found by `tests/test_abi.py`: Windows stack arguments of a small struct took its size, not an
  8-byte slot; Windows stack float parameters were integer copies; SysV padding-only eightbytes
  (`struct { _Alignas(16) char c; }`'s second half) took a register of an undefined class
  instead of none. The frontend no longer writes tail padding QBE adds itself on x86-64 (it
  made an eightbyte of floats INTEGER).
- **Baseline x86-64 only** (SSE2; any x86-64 CPU): no FMA, no SSE4.1 rounding, no POPCNT. So:
  no FMA contraction (`fma=` is accepted and ignored: results as `-ffp-contract=off`), and
  only `sqrt`, `fabs`, `bswap`, `clz`, `ctz` become instructions (frontend `builtins=`/`bitops=`
  take the set of IR ops a backend selects). `fma`, `floor`, `ceil`, `trunc`, `round`, `fmin`,
  `fmax` (and float versions) are runtime code: musl's portable versions and an exact integer
  `fma` (`runtime/internal/softfma.h`, shared with wasm), under `__CIMO_SOFT_MATH` (the runtime
  is built without `__FP_FAST_FMA`, so musl's `pow`/`log` take their non-FMA paths, as wasm).
  CPU feature detection for those instructions is future work.
- **Struct `va_arg`:** SysV's register save area per eightbyte class (`vastruct='sysv'`);
  Windows passes structs that are not 1, 2, 4 or 8 bytes by pointer (`vastruct='win64'`).
- **Assembler:** `cimo/assembler_amd64.py`, byte for byte against GNU as 2.42 (ELF) including
  jump relaxation and gas's multi-byte NOP padding; relocations PC32, PLT32, GOTPCREL, ABS64,
  ABS32(S). Every relocated field holds 0 and the addend is in the relocation. ~150k lines/s.
- **Loader:** stubs `jmp *0(%rip); .quad addr` (no register clobbered); no cache flush.
- **Execution glue:** the context key is the thread's own control block (`%fs:0` on Linux,
  `%gs:0x30`, the TEB, on Windows); `__cimo_find` clobbers only r10, r11 and flags, so argument
  registers reach the kernel untouched. The entry trampoline pops the caller's return address
  into the frame and calls the kernel from there, so stack arguments and Windows' shadow space
  are where the caller put them, then jumps back. Windows frames add rdi, rsi and xmm6-xmm15
  (240 bytes), so contexts there are 12 KB with the frames in the third page. CAS is
  `lock cmpxchg`.
  It returns with `push; ret`, and the frame keeps the shadow stack pointer (`rdssp`, a nop without
  CET) for the unwind's `incssp`, so entry and `KernelError` stay legal under CET shadow stacks.
- **Globals:** the frontend marks every non-static global extern, so isel reaches it through the
  GOT; the loader relaxes that `movq sym@gotpcrel` to `leaq sym(%rip)` for symbols in the image.
- **Faults:** besides SIGSEGV/access violations, integer division traps on x86 (#DE: `x / 0`,
  `INT_MIN / -1`), `KernelError` kind `division fault`; on arm64 these return a value.
- **The optimizer on x86-64:** the target-independent passes (simplifier, known bits, SCCP,
  ADCE, SROA, jump threading, memory passes, inlining and IPO) run as on arm64; `divconst`,
  `bytepatterns`, `hoistcons` and sibling calls are arm64 only. They reached two more C QBE
  bugs in the amd64 instruction selection (backend/PORTING.md): flags reused from a shift by a
  variable count, and an integer constant converted to a float. Packed structs by value: SysV
  passes a struct with an unaligned member in memory, Windows x64 a 7-byte one by reference;
  the frontend emits those as opaque types (frontend/PORTING.md).
- **Measured** (2026-11-02, Windows 11 x64 / WSL Ubuntu 24.04 on the same Intel machine): the
  corpora pass in full on both (2267 programs: gcc torture, c-testsuite, musl libc-test math;
  `20101011-1.c` needs `signal.h` on x86 only); `tests/test_abi.py` 2400 random signatures on
  Linux (gcc's convention, via `zig cc`) and 1800 on Windows (MinGW-w64 gcc), 0 mismatches. The
  corpora found three wrong-code bugs: a load elimination bug of QBE's own, reachable on every
  target through `ext.cse` (backend/PORTING.md), 64-bit displacements, and the sign of a zero
  `fma` result after underflow (wasm too). The bundled headers declared Linux's `wchar_t`
  unsigned, aarch64's: on x86-64 it is `int` (`wint_t` stays unsigned).
- **Testing:** the suite runs on both (`tests/hosts.py` says what a host can run; arm64
  code-generation tests skip). References: gcc on Linux, MinGW-w64 gcc on Windows x64 (the
  Windows x64 convention, LLP64). Developed on Windows 11 x64 with WSL Ubuntu 24.04 without
  network access in WSL: packages fetched on Windows (wheels; Zig's `zig cc` as the Linux C
  compiler, which also builds C QBE against glibc 2.39; glibc headers unpacked from the .deb).
  `zig cc` is clang: it cannot build the runtime as a reference (`test_runtime` needs gcc).

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
    runtime's first build went from 8.8 s to 5.5 s. `isnan(x)` and the other header
    helpers are now inlined (`backend/inline.py`).
  - cproc's config.h defines `__FLT_EVAL_METHOD__` as 1, but cproc + QBE compute floats in
    single precision (at run time and when folding constants). `cimo.preprocessor` defines 0,
    like gcc on arm64, so `float_t` is `float` (with 1 it was `double`: slower, and results
    differing from gcc).
  - cproc has no Apple target. Apple arm64 has signed `char` and an 8-byte `long double`, while
    cproc's `aarch64` has unsigned `char` and a 16-byte `long double`. The port adds target
    `aarch64-apple` (signed `char`, `wchar_t` = `int`, 8-byte `long double`, and `va_list` =
    `char *` as Apple's ABI and QBE's `arm64_apple` vastart expect); not diffable against C.

## Single-file distribution (2026-10-30)

`cimo onefile [-o cimo.py]` (`cimo.onefile.build`) writes one `cimo.py` that works
like the package in any CPython 3.9+ with only the standard library: put it on `sys.path`,
`import cimo`.

- **Resource layer:** every read of a package file goes through `cimo.resources` (`path`,
  `isfile`, `read`, `files`): bundled headers (preprocessor search path), runtime sources,
  and the compiler sources hashed into cache keys. In the repo or a pip install they are files
  next to the package. In the onefile build `resources.ROOT` is `<cimo.py>/cimo`, a name that exists
  only in the embedded table; any other path (user and host headers, the cache) is read from
  the filesystem. The `.c`/`.h` files stay real files in the repo.
- **Format:** a license header (cimo MIT; cproc's and QBE's upstream `LICENSE` files, shipped
  as `frontend/LICENSE` and `backend/LICENSE`, and musl's `runtime/COPYRIGHT` reproduced), then `_FILES = {path relative to the package: text}` (every `.py`, `.h`, `.c`
  and `runtime/COPYRIGHT`, as readable triple-quoted literals, sorted), then a bootstrap: a
  meta-path finder/loader serving `cimo.*` from the table. Imported as `cimo`, the file
  becomes the package (`__path__ = []`) and runs `__init__.py`; run as a script it runs
  `cimo.__main__` (the CLI). Module filenames are `<cimo.py>/cimo/driver.py`, registered
  with `linecache`, so tracebacks and `inspect` show source. Deterministic (no timestamps):
  rebuilding, also from inside the onefile build, gives the same bytes.
- **Stripped:** comments, docstrings and blank lines are left out of the embedded `.py`
  (checked: same AST minus docstrings) and `.c`/`.h` (comments become a space; checked: same
  pp-tokens); C comments carrying a copyright or license notice stay (musl's math files
  have Sun/FreeBSD/Arm notices that must be preserved). 1.26 MB -> 1.05 MB.
  `--keep-comments` turns it off.
- **Cache keys are content hashes over package-relative names**, computed from the original
  sources at build time and embedded (`onefile-keys.json`), so a onefile build has the same
  `compiler_key()` and runtime key as the tree it was built from, stripped or not.
- **Sources only.** A `--prebuilt` option (embedding the compiled runtime) was tried and
  removed: it only matched the target it was built on and put generated, target-specific
  code in a file meant to be source. The first-use cost is handled by compiling runtime
  members on demand instead (M5, link model).
- **No writable cache** (`CIMO_CACHE_DIR` / `CIMO_RUNTIME_CACHE` / `~/.cache/cimo` cannot be
  created or written): the module cache and the runtime are kept in memory for the process.
- **`cache=False`** (`compile`/`load`): fully in-memory. The caches are still read; nothing is
  written (no entries, directories or intermediate files) and what is built, runtime members
  included, stays in memory for the process. A context variable, so other threads still write.
  Tests that compare two builds of one source under a patched internal (`driver._vecq`, `middle.vectorizing`) must
  change the source too: the second compile otherwise gets the first's module from the cache (found 2026-11-07,
  `tests/test_vector_q.py` compared vector code with itself until then).
- **Slots at call boundaries (2026-11-07, `backend/abislot.py`):** the ABI (abi1, after every memory pass) stores a
  struct parameter's registers, and a call's struct result, to a slot the body loads; nothing forwarded them, since
  alias analysis doesn't take the ABI's physical registers. A slot that doesn't escape, written once per offset by a
  store dominating each load, now has those loads replaced by the stored values, and the stores and slot go: an
  `__int128` add 35 instructions to 8, `struct { long x, y; }` summed in one `add`.
- **Profiling (2026-11-01):** with `CIMO_PERFMAP=1` the loader appends each image's functions
  (start, size to the next symbol, name) to `/tmp/perf-<pid>.map`, perf's convention for JIT
  code (Linux). Off by default, since it writes to disk.
- **Image cache (2026-11-01).** A module is cached as its linked image, as a shared library
  would be: code and data bytes linked at address 0, the slots that hold external addresses
  (stub literals, GOT slots, ABS64 data; resolved per process, `link=` is not in the key) and
  the 64-bit pointers into the image (rebased). Internal PC-relative references need nothing:
  code and data are contiguous and page aligned. A hit skips assembling, linking and
  relocating. First compile in a process, cache hit (on battery):

  | | Linux arm64 | Windows arm64 |
  |---|---|---|
  | printf/malloc/math kernel | 70 → 20 ms (15 ms preprocessing) | 131 → 51 ms (18 ms) |
  | 300-function file | 105 → 54 ms (47 ms preprocessing) | 180 → 110 ms (58 ms) |

  The rest on Windows is mostly hashing the compiler's and runtime's sources for the key, once
  per process (file reads are slow there); the runtime key reuses the compiler key so those
  sources are read once. An image with another load-address-dependent reference (direct adrp to
  an external, ABS32 address) is cached as assembly instead. A CRC32 over each entry: a
  truncated or corrupted file is deleted and recompiled (without it, a flipped byte loaded as
  wrong code).
- **Measured** (Linux arm64, CPython 3.13): onefile 1.05 MB; building it 1 s with stripping,
  0.05 s without; `import cimo` from it 0.05 s; first compile of a printf/sin/malloc kernel
  1.5 s with nothing cached (15 runtime members), then 0.1 s. `tests/test_onefile.py` runs it
  under `python -I -S` with no writable home.

## Command line (2026-10-30)

`cimo` (console script) = `python -m cimo` = `python cimo.py` for a onefile build. Standard library
only: `cimo/cli/parse.py` is a small declarative framework (nested commands, options with
short/long/`=`/attached forms, choices, repeatable options, a `rest` argument that passes
everything after it through), `cimo/cli/term.py` does color (`--color`, `NO_COLOR`,
`FORCE_COLOR`, `TERM=dumb`, tty), unicode symbols with ASCII fallback, wrapping to the
terminal width and a spinner.

- Commands: `run FILE [ARGS...]` (calls `main(argc, argv[, envp])` or `--entry NAME`; exit
  status = return value or `exit()` status, 134 for abort/assert/trap; `-` reads stdin),
  `compile FILE` (lists the exported functions with their Python-side types) with `-E`,
  `--ir`, `-S` / `--emit pp|ir|asm` and `-o`, `onefile`, `cache [info|clear|path]`,
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

## Conformance corpora (2026-10-31)

Most tests compare a port with its reference (cproc, QBE, gcc's cpp, gas); they show fidelity,
not correctness, and need those tools (skipped on Windows and macOS). `tests/test_corpora.py`
runs third-party programs that check their own results, on any host: gcc's
`gcc.c-torture/execute` (+ `ieee/`, 1779 programs) and c-testsuite's single-exec (220), fetched
at pinned commits by `tests/corpora.py` (not vendored: the torture tests are GPL). Each runs in
a worker process (a crash fails one test); the whole set takes about 20 s on 8 cores.
`tests/corpora_expect.txt` lists what is not expected to pass: `unsupported` (a compile error;
regenerated by `--update`), `skip` (gcc-specific behavior, written by hand), `bug`.

- **gcc.dg/vect and gcc.dg/torture** (2026-11-06, `gcc-vect`, `gcc-dg-torture`): the
  vectorizer's run tests (every `.c` with a `main` that is not `dg-do compile`; `tree-vect.h` is
  regenerated with a no-op `check_vect`) and torture's `dg-do run` programs, 2155 in all, about
  12 s on 8 cores. First run: 1429 passed, 7 aborted, 719 rejected (303 for inline assembly,
  nearly all the empty-template barrier `asm ("" ::: "memory")`, 114 for gcc's
  `__BIGGEST_ALIGNMENT__`). Fixed: 6 frontend bugs (`frontend/PORTING.md`: an empty `{}` for an
  array element, `_BitInt` wrapping in conversions, unary minus and the overflow builtins, a
  pointer-to-const member taken for a const member, `__builtin_expect` dropping its second
  argument) and a QBE fold error on a float conversion of an address. The empty `asm` is now
  accepted (it exposed the `__builtin_expect` bug in an old torture program), with
  `__BIGGEST_ALIGNMENT__` (16) and more `__builtin_` library aliases: 1865 pass. The rest are
  vector types (57), `fenv.h` (45), GNU attributes (`mode`, `scalar_storage_order`, ...),
  `__int128`/`_Float16`/`__bf16`, POSIX headers and `_Complex`; three need gcc's behavior
  (`-fsigned-char` on arm64, bit-field arithmetic in the field's width).
- **Constraint violations** (`tests/test_constraints.py`, 2026-11-01): 199 programs, at least
  one per C11 "shall" constraint (lexical elements to preprocessing), must each give a located
  error (preprocessor: a warning where gcc and clang only warn). C23 is followed where it
  differs (unnamed parameters in definitions, `f(...)`, enumerators wider than `int`), and the
  GNU zero-length array and a freestanding `main` stay accepted. First run: 33 not diagnosed,
  four of them compiler bugs (one, a duplicate label, an endless loop that ran the machine out
  of memory); see `frontend/PORTING.md`.
- **musl's libc-test** (2026-11-01, third corpus, `math/` and `functional/`): tests the runtime
  library compiled by cimo. Math programs compare against tables of correctly rounded results
  (crlibm, ucb, special cases) within each table's ulp bound. A harness generated after the
  fetch (`corpora.harness`) stubs `fenv.h`'s exceptions (the runtime's math doesn't raise musl's
  flags; since 2026-11-06 the results are checked in all four rounding modes, and pass) and drops
  libc-test's `long double` helpers. 119 pass (108 math, 11 functional: string, qsort, strtod, strtof,
  udiv), none fail; the rest need `long double` (85), `errno.h` (17), POSIX headers, threads,
  TLS or non-C11 functions (`j0`/`y0`, `sincos`, `pow10`, `lgamma_r`, `memmem`, `strlcpy`).

- First run: 1036 passed, 15 failed at run time or crashed the compiler. Fixed: 4 QBE bugs, 4
  cproc bugs (`backend/PORTING.md`, `frontend/PORTING.md`), `cimo.load` of non-UTF-8 files.
  Now 1454 pass on Linux (1406 before volatile, VLA freeing and struct `va_arg`). Windows
  found the `va_list` load-forwarding bug.
- The driver's prelude declares gcc's `__builtin_<libc function>` names (`__builtin_abort`,
  `__builtin_memcpy`, ...) as aliases of the runtime's functions (asm labels, with
  `__USER_LABEL_PREFIX__`, now predefined); `__builtin_prefetch` is a no-op. The GNU overflow
  builtins (`__builtin_{add,sub,mul}_overflow`, `_p` and typed variants) are macros over one
  runtime function given each operand's bits and signedness (`(__typeof__(x))-1 < 0`); it
  computes the exact result (sign and 128-bit magnitude), matching gcc on 3000 random calls.
  Also the quiet comparisons (`__builtin_isunordered`, `isgreater`, ...), `issignaling`,
  `nans`/`nansf` and `isinff`: 21 more torture programs. gcc's `__FLT_*` and
  `__DBL_*` macros are predefined (double constants, not gcc's `long double` casts).
- The 585 `unsupported`, by cause: K&R implicit `int` (143, not C11), inline assembly (58),
  vector types (43), other GNU builtins (42: overflow, `classify_type`, ...), `long double` (35),
  `volatile` stores (34, cproc rejects them; supported since 2026-11-01, with VLAs freed when
  their scope ends, which 3 of them needed), GNU attributes (47), nested functions (17), struct
  `va_arg` (14, supported since 2026-11-01; it exposed the struct-alignment bugs in
  `frontend/PORTING.md` and `backend/PORTING.md`), `_Complex` (13), statement expressions (10). gcc-only behavior (`skip`): bit-field
  arithmetic in the field's width when wider than `int`, `__builtin_constant_p` after inlining,
  an unspecified evaluation order.
- **Differential fuzzing** (`tests/fuzz_csmith.py`): Csmith programs (`--no-volatiles
  --no-packed-struct`) run by `gcc -O0` and by cimo, outputs compared; `tests/reduce.py` shrinks a
  mismatch with only gcc (line ddmin; candidates must be clean under ASan+UBSan and agree at
  -O0/-O2). Csmith is built from source (`~/code/csmith`, cmake from pip). About 90% of
  programs run under gcc in 5 s; first 1200 seeds: 3 cproc miscompiles (unsigned long vs long
  long, `||`/`&&` folding, bit-field promotion through `,` and `?:`), a port crash on functions with many constants, and quadratic
  constant lookup (fixed); see the PORTING files. Seeds 2000-3999 after the fixes: 1739
  pass, 0 fail (261 too slow under gcc). cimo accepts pointers differing only in
  signedness or inner qualifiers (Csmith emits them; gcc accepts them silently).
- **Calling-convention tests** (`tests/test_abi.py`, `tests/abi_gen.py`): random signatures
  (all integer widths, float, double, pointers, structs of 1-80 bytes incl. float/double
  aggregates, odd sizes, arrays, nesting, `_Alignas(16)`; up to 14 arguments; variadic with
  scalar and struct arguments)
  built by gcc (Linux) or clang (Windows) into a shared library and by cimo; each is called
  reference→cimo, cimo→reference and cimo→cimo and must match reference→reference. 4 seeds ×
  60 in the suite (a few seconds); 1800 more signatures checked on Linux and 1440 on Windows.
  Found: on Windows, a variadic call with no variadic arguments used the normal convention
  (the frontend emitted no `...`); 16-byte aligned structs (QBE: odd registers, stack
  alignment of the pointer to a large one). clang on Windows disagrees with itself on a
  16-byte aligned struct as a variadic argument (its caller aligns the slot, its callee does
  not); cimo follows the caller, and the generator leaves those out.
- **Semantics model** (`tests/test_semantics.py`, `tests/semantics_gen.py`): random expressions
  and compound assignments over every arithmetic type, bit-fields and operator, with results
  from a Python model of C11 (promotions, usual arithmetic conversions, wrapping, truncating
  division, float rounding per operation, `-0`), compared bit for bit; constants and parameters
  as operands, so folding and code generation are both covered. No undefined behavior is
  generated (it is rejected and regenerated); FMA contraction is off (the model rounds every
  operation). Needs no other compiler: runs on every host. 6 × 400 in the suite; 1240 × 400
  more were clean after fixing what it found: `_Bool` `++`/`--`, constant conversions to
  `_Bool` and of `(-1, 0)` floats to unsigned, a folded `?:` with a bit-field operand (cproc),
  and `jnz` on constants with upper bits set (QBE).
- **Backend edge cases** (`tests/test_backend_edges.py`, results computed in Python, every host):
  64-bit constants of every materialization form (movz/movn/movk, logical immediates), float
  constants (fmov immediates, literals, -0, subnormals), arithmetic immediates at the encoding
  limits, frames and locals around the load/store offset limits (4 KB to 4 MB), register
  pressure (16-64 live values of mixed classes across calls, spill slots beyond a 40 KB
  array; 120 more random cases up to 120 values checked), struct members and globals at large
  offsets, 30 arguments from a 70 KB frame. Found 2 QBE bugs (parallel moves over 64 copies;
  float slot-to-slot copies). The assembler matches GNU as on all 11252 assembly files cached
  by the corpora and fuzzing runs.
- **C23 where C11 differs (decided 2026-10-31):** as cproc, `f()` declares a prototype with no
  parameters (C11: no prototype, so `f(1)` is valid) and `true`/`false` are keywords (C11:
  identifiers). Programs relying on the C11 reading are `unsupported`.

## Plan

1. ~~Validate the architecture~~ **Done: GO.** Good IR + patched QBE is within 1.04x of gcc's
   scalar `-O3` on all 4 kernels tried on Linux arm64 (threshold was 1.2x). On Apple M4 vs clang,
   3 of 4 are within 1.11x; saxpy is at 1.43x until `ldp`/`stp` pairing lands. See "Architecture check".
2. ~~Check loading in each environment~~ **Done** for Linux arm64/x86-64, Windows x64 and
   arm64 (cimo runs end to end on Windows arm64 and macOS arm64) and Pyodide. Still open: locked-down Windows (AppLocker/WDAC/EDR), notarized macOS apps
   that embed Python without `allow-jit`.
3. **Port QBE (arm64 first) with `qbe-arm64-isel.patch` applied** (done, see M1), **then cproc**, verified by IR
   diffs against the C originals, plus differential testing against gcc using generated random programs.
4. **Middle stage loop passes:** implement the transforms that `gen_opt.py` applies by hand, plus
   explicit `fma` emission and FMA selection in the backend. Target: about 1.2x of gcc's scalar
   `-O3` on the whole suite. **Mostly done** (M4): geomean 1.11x; 12 of 15 kernels within
   1.2x; edit_distance, nbody_acc and rgb2gray wait on backend items (csel, sqrt, int
   constants/madd). Next for the middle stage: runtime alias checks (loop versioning on
   address ranges) for non-restrict pointers, multi-block loop bodies, `--report`.
5. **SIMD** (vector types, then auto-vectorization), ~~**amd64**~~ (done, see "x86-64"), **WASM**, **concurrent calls** (per-thread heaps).

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
`cimo.backend` → `cimo.assembler` → `cimo.loader`, no external tools. The linked image is cached in
`~/.cache/cimo` (`<key>.img`), keyed by the preprocessed source (without line markers, blank lines
or indentation, which place only diagnostics: moved code hits, 10 against 455 ms for 40 small
functions on arm64 Linux, unless `__LINE__`/`__FILE__` changed with it), the flags and hashes of the
compiler's own sources (assembler and loader included) and of the runtime's, so an edited or
upgraded compiler never reuses an old image. Signatures come from the QBE IR (`l` params accept buffers).
Modules link statically with the bundled runtime library (M5); nothing resolves from the
process unless named in `link=`. `examples/` uses it.
Defaults: middle end on (`CIMO_MIDDLE=0` off), FMA contraction on (`CIMO_FMA=0` for results
identical to `-ffp-contract=off`), ldp/stp pairing and loop layout on. Geomean over the 15
kernels with these defaults: 1.07x of gcc nofma, 1.13x of novec, 1.69x of `-O3` (vectorized).

**M2: encoder + loader.** arm64 machine-code encoder (no external assembler) and the JIT loader
for Linux and macOS (strategies under "Platform loading findings").
- All 15 kernels compile and run in-process from Python, with results matching gcc's.
- `scratch/qbe_eval/edge.c` cases are bit-exact.
- **Done 2026-10-30 on Linux arm64, 2026-11-01 on macOS (see "macOS on Apple silicon").**
  `python/cimo/assembler.py`
  assembles QBE's GNU-syntax output (both `arm64` and `arm64_apple` syntax) into an in-memory
  object; `python/cimo/loader.py` links it into the process. No `cc`, and no `dlopen` of generated code, for practical
  reasons: Apple Silicon kills unsigned file-backed code (anonymous JIT memory needs only
  `allow-jit`), one in-memory format replaces ELF/Mach-O/PE dynamic linking, imports resolve
  against the runtime, `link=` libraries and Python hooks (never the process), `dlclose` may not
  unmap, and Windows locks loaded DLLs. Lost: debuggers and profilers see anonymous memory.
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
- macOS: code goes in a `MAP_JIT` mapping, data in a separate RW mapping right after it (a
  `MAP_JIT` page is RX for the thread while kernels run, so writable data can't share it); if
  that fails, fall back to one W^X mapping.

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
    ignored; any other is "attribute 'X' is not supported" (`ext_vector_type`: "not supported yet").
  - **GNU vector types** (2026-11-06): `vector_size(N)` on a declaration, member, parameter, return type
    or type name makes a vector of the (innermost) integer or floating type, one type per element type
    and size (`TY.mkvector`), aligned to its size up to 16, laid out and passed as a struct of its lanes.
    The frontend lowers operators lane by lane (`qbe._v*`): arithmetic and shifts with a scalar operand
    broadcast, comparisons giving -1/0 in signed lanes of the same size, `v[i]` as an lvalue, casts of the
    bits to and from types of the same size, `__builtin_shuffle` (one or two vectors, variable masks),
    `__builtin_shufflevector`, `__builtin_convertvector`. Also GNU casts to a union (`(union U)x`).
    **Class q (2026-11-07):** vectors of `vecq` bytes (16; 32 on x86-64 with AVX2), twice or four times that compute as q values
    (frontend `vecq=`, IR `%r =q qaddw ...`): `+ - * / & | ^`, comparisons, unary `-` and `~`, shifts by a constant,
    copies and `memcpy` to or from a vector, and `v[constant]` (`qget`). `a*b + c` and funcfma's other forms on
    float vectors contract into `qfma1` as scalars do (`_qfma`). Writing `v[i]`, and reading it with a variable `i`, go
    through a slot of their own (one `qstore`, the lane, one `qload`), and so do vectors passed to and returned from calls
    (as structs of lanes): `v` itself is only ever accessed whole, so it stays a register. Calls on arm64 (2026-11-07,
    frontend `vecabi=`; the IR's `type :v = align 16 vector { s 4 }`): 8- and 16-byte vectors go in one V register as
    AAPCS64 says and gcc and clang do, named ones only (past `...` as the struct of their lanes, as `va_arg` reads them;
    gcc would use V registers there too). x86-64 SysV (2026-11-07): one xmm, class d or q (`sysv.typclass`), moved
    with `qload16`/`qstore16` when q is AVX2's 32 bytes; Windows x64 passes them by reference as any struct over 8 bytes
    and returns a 16-byte one in xmm0 (`winabi._vecret`), as MSVC and MinGW gcc do. Shifts by a variable or a
    vector on arm64 (`vecops` 'vshift'): `sshl`/`ushl` (`qsshlv`/`qushlv`), by the negated count for `>>`. `__builtin_convertvector` on arm64 (frontend
    `vecops=('convert',)`): integers widened by halves (`qsxl`/`qsxh`, `sshll`/`sshll2`), narrowed in pairs (`qnrw`, `xtn`/`xtn2`),
    floats likewise (`fcvtl`/`fcvtn`), then `scvtf`/`ucvtf`/`fcvtzs`/`fcvtzu` at one width; int64 to float stays lane by lane
    (through double it would round twice). `__builtin_shufflevector` of one-q vectors on arm64 (`vecops` 'shuffle'): a
    constant byte table per source in read-only data, `tbl` (`qtbl`; out-of-range bytes give 0), two joined by `orr`.
    The shuffles that are one instruction are matched first (`-1` matching anything; either source first): a copy,
    `zip1`/`zip2`, `uzp1`/`uzp2`, `trn1`/`trn2`, `ext #k`, `dup` of one lane. Open: the tables' `qload`s, pinned as all
    of them are, stay inside loops. A promoted vector slot starts as `qdupw 0`: a lane set in a loop reads it before any
    write, and SSA's undefined value has no q constant (gcc's vector-div-neg-pow2.c). x86-64 has none of these ops yet (`pshufb` is SSSE3, past the baseline). A slot only ever read and written whole by
    `qload`/`qstore` becomes a q temporary (`mem.promote`), so vector locals live in registers; initializer lists fill a
    slot of their own first. Lane by lane still: what q lacks (integer `/` and `%`, 64-bit `*`; on x86-64 shifts by a variable of 8- and 16-bit
    lanes and `>>` of signed 64-bit ones), `__builtin_shufflevector` and `__builtin_shuffle` of any of
    these sizes in 16-byte pieces (`_qpieces`): each piece of the result the OR of a `tbl` of each source piece it reads.
    `__builtin_shuffle` with a
    variable mask (2026-11-07, `_qvarshuffle`): each index modulo the lanes, times the lane's size, spread over its bytes
    (a constant `tbl`), plus the byte's place; bit 4, b's half, turned into bit 7, which `tbl` and `vpshufb` both read as 0,
    so `tbl(a, f(i)) | tbl(b, f(i ^ 16))`. 16-byte vectors with AVX2's 32-byte q (2026-11-07, `vecops` 'q16'): one q value,
    loaded into both halves (`qload16`, `vbroadcasti128`: the upper lanes repeat the lower, so no extra FP exception) and
    stored from the lower (`qstore16`). With AVX2 also shifts by a variable (`vpsllv`/`vpsrlv`/`vpsravd`, by direction:
    `qshlv`, `qshrv`, `qsarvw`; `vecops` 'vshiftx') and 16-byte `__builtin_shufflevector` (`qtbl` as `vpshufb`, which works
    within each half and gives 0 for an index with its top bit set, as `tbl` does past 15; 'shuffle16') and `__builtin_convertvector` ('convertx': `vpmovsx`/`vpmovzx`, through `xmm15` for a high half;
    narrowing by clearing the high bits, `vpackuswb`/`vpackusdw` and `vpermq $0xd8`, or `vpshufd`/`vpermq`/`vinserti128` for
    64 to 32 bits; `vcvtps2pd`, `vcvtpd2ps`; `vcvtdq2ps`, `vcvttps2dq`); the AVX-512 ones as exact sequences (`amd64/qlower._cvt`): an unsigned 32-bit int to float
    as its 16-bit halves, float to unsigned less 2^31 where it is that much, 64-bit ints to double as 32-bit halves made
    doubles with 2^52's bits (one rounding, in `hi * 2^32 + lo`), double to 64-bit by `vroundpd` and floor(t * 2^-32). A 16-byte vector narrowed from
    one q value lands in its lower half (the pair is the value twice). Not yet: `__int128` lanes, `ext_vector_type`.
    `aligned` after a declarator applies to the object/member, and on a typedef makes an
    over-aligned variant (glibc's `pthread.h`). The preprocessor ignores glibc's
    `#define __attribute__(x)` (made for non-GNU compilers), so attributes are never dropped.
  - Declaring `long double` (16-byte) or thread-locals is fine (glibc's headers do); defining or
    using them is an error. `_Float128`, `__float80`, decimal and fixed-point types are rejected outright; an undeclared identifier followed by a
    declarator is "unknown type name".
  - Accepted GNU extensions (2026-11-06): statement expressions (the body is emitted at parse time
    into blocks of its own and spliced in where the expression's code goes, so it runs in evaluation
    order: `a && ({...})`, a `for` loop's third expression), case ranges (one unsigned compare each,
    before the case tree), range designators (element `a`'s initializers copied to `a+1..b` once
    complete), `__label__`, `mode(QI/HI/SI/DI/word/byte/pointer/SF/DF/TI)`, `aligned` on scalar typedefs (a compatible variant, `Type.origin`) and before a
    declaration's storage class, `packed` on one struct member; `returns_twice` is ignored.
  - Rejected GNU extensions: `&&label`/`goto *`, nested functions, `__auto_type`, `__builtin_choose_expr` and other
    undeclared `__builtin_`/`__atomic_`/`__sync_` names, imaginary constants, empty structs,
    inline assembly (`asm` is a keyword, as in gcc's default mode; a statement with an empty
    template, gcc's optimization barrier, compiles to nothing). `a ?: b` works (cproc
    evaluates `a` once).
  - Accepted: `__restrict(__)`, `__const(__)`, `__volatile`, `__alignof`, `asm` labels,
    `__PRETTY_FUNCTION__`/`__FUNCTION__`, C23 `nullptr` and `nullptr_t` (cproc d1c53dd parses
    them but asserts in code generation and rejects non-constant `nullptr_t` operands; cimo
    generates a pointer-sized zero and allows `nullptr_t` expressions in assignment, `==`/`!=`
    and `?:` as C23 does).
  - Unsupported types are reported where the expression is parsed (a hook on expression
    construction), so the column points at the `long double` sub-expression. A C99 inline definition with no external definition in the
    unit becomes a local function when referenced (cproc emits nothing, so the call failed to link).
  - Driver hints: a missing header suggests `host_headers=True` (`setjmp.h` bundled since 2026-11-06);
    undeclared hosted-libc names (`fopen`, `FILE`, `time`, `pthread_*`, ...) suggest
    `host_headers=True, link=['libc']`.
  - glibc 2.39 headers (stdio, stdlib, unistd, pthread, signal, math, sys/*, ...; 70+ tried) compile
    except `glob.h` (`__SIZE_TYPE__` is not predefined).
  - **Complex numbers** (2026-11-06): `float`/`double` (and, where it is double, `long double`) `_Complex`
    are struct types of two parts (`Type.iscomplex`), so loads, stores, copies, calls and returns are a
    struct's, which is every target's C ABI for complex values (an HFA on arm64, SSE on x86-64).
    Arithmetic is lowered in the frontend (`qbe._cx*`): the parts are loaded, computed and stored to a
    slot. Annex G: a real operand keeps its domain (`x * (a+bi)` is componentwise); `*` is inline and
    calls the runtime's `__cimo_cmul` only when both parts are NaN (recovering infinities, as gcc's
    `__muldc3`); `/` calls `__cimo_cdiv`, which carries each product as a mantissa and an exponent so
    extreme quotients are exact to a few ulps (gcc's `ieee/cdivchk*.c`). Under `#pragma STDC CX_LIMITED_RANGE ON`
    (2026-11-07; kept by the preprocessor, `U.cxlimited`) both are the usual formulas, inline, as gcc's `-fcx-limited-range`. Constant folding covers static
    initializers (`1.0 + 2.0*I`). GNU `__real__`/`__imag__` (lvalues), `~` (conjugate), imaginary
    constants (`2.0i`), `__builtin_complex`. `<complex.h>` is cimo's; the functions are musl's
    `src/complex` (double and float). The runtime is now compiled in cimo's own mode (diagnostics on).
  - **`__int128`** (2026-11-06): `__int128`/`unsigned __int128` (`__int128_t`, `__uint128_t`, `mode(TI)`)
    are integer types laid out as a struct of two longs, low first (`TY.isi128`): memory, calls and
    returns are a struct's: x86-64 SysV and arm64 take it in a register pair (AAPCS64's even one, from the type's
    `align 16`), the stack 16-byte aligned, as gcc does (`tests/test_abi.py`, both directions). The frontend lowers operations to 64-bit halves (`qbe._i128*`): add/sub with a carry,
    multiplication with `umulh` on arm64 (32-bit halves elsewhere; an add's carry becomes `adds`/`adc` on arm64 by a peephole
    on the emitted function, `arm64/emit._carry`), comparisons high half first,
    shifts inline (by a variable branch-free since 2026-11-07: both halves by n mod 64, the crossing bits by a double
    shift, the halves picked by bit 6 through a mask); division and float conversions are runtime calls (as gcc's
    `__divti3`, `__floattidf`)
    (`builtins.c`, correctly rounded). Constants fold in 128 bits (`eval._i128eval`); the overflow
    builtins take `__int128` operands and results (`__cimo_overflow128`). Not supported: `__int128`
    bit-fields, `switch` on one.
  - **Half precision** (2026-11-06): `_Float16` (and `__fp16`) and `__bf16` are stored as their 16 bits
    (QBE class of `unsigned short`) and computed in float, as gcc does without half arithmetic: the usual
    arithmetic conversions make `float`, unary operators promote, and a value rounds back to nearest even
    on assignment, cast, call and return (one float operation rounded to half equals the half operation).
    Conversions (2026-11-07): `__bf16` inline on every target (a shift; round to nearest even with a NaN kept quiet,
    branch-free); `_Float16` on arm64 by `fcvt` through `h31` (backend `h2f`/`f2h`, frontend `vecops` 'half'; ARMv8.0,
    rounding in the current mode, as C asks), on x86-64 CPUs with F16C (`cimo.cpu`) by `vcvtph2ps`/`vcvtps2ph $4`
    through `xmm15`, elsewhere runtime `__cimo_f2h`/`__cimo_h2f` (nearest always). Variadic arguments go as double.
    Constants: `f16`, `bf16`, `f32`, `f64`, `f32x`, `f64x` suffixes, folded in half (`eval.halfbits`).
    `_Float32` is float, `_Float64`/`_Float32x` double; `<float.h>` has the `FLTn_` macros. Calls pass
    them in FP registers, as gcc does (2026-11-07, frontend `halfabi=`): the 16 bits are the low half of a class-s value
    (`h0` on arm64, the low word of `xmm0` on x86-64), cleared above them on the way in (`tests/test_abi.py`).
    Open: signaling-NaN constants.
  - **Floating-point environment** (2026-11-06): `<fenv.h>` (runtime `fenv.c`) on the hardware's registers
    through four backend ops (`rdfpcr`, `rdfpsr`, `wrfpcr`, `wrfpsr`: arm64 FPCR/FPSR, x86-64 MXCSR for both;
    wasm reads 0, so `fesetround` reports anything but nearest as unsupported by reading back). Comparisons
    are IEEE 754's: `fcmp`/`ucomis` for `==`, `!=`, unordered, `fcmpe`/`comis` for `<`, `<=`, `>`, `>=` (C QBE
    had one instruction for all, a different one per target), and if-conversion doesn't speculate the
    signaling ones; `iseqsig` is `x <= y && x >= y`. The entry trampoline saves FPCR and FPSR (MXCSR), clears
    the flags for the call and restores both on return and unwind: every call from Python starts like a C
    program, and a kernel's `fesetround` never changes Python's arithmetic (nor persists to the next call).
    `#pragma STDC FENV_ACCESS ON` (gcc's tests: `-frounding-math`) marks the functions it covers: the backend
    folds only exact, flag-free floating-point operations there and pins the rest (no GVN merging, no GCM
    motion across `fesetround`). Without it, constants fold in round-to-nearest, as gcc and clang do by
    default. `__int128` to float conversions round in the current mode (sticky bit, signed 64-bit convert). So does the
    software `fma` of x86-64 without FMA and wasm (`softfma.h`; found by libc-test on x86-64 CI).
    `noinline`/`noipa` are honored (the tests rely on them). The runtime's math stays pure (`FORCE_EVAL` a
    no-op), so its exception flags are not musl's; its results pass libc-test in all four rounding modes.
    Not supported: `feenableexcept` (traps), the flags of `long double` operations.
  - **`setjmp`/`longjmp`** (2026-11-06): in every module's glue (`execution._setjmp_arm64`/`_setjmp_amd64`): the
    callee-saved registers (Windows: rdi, rsi, xmm6-15 too), sp and the return address in a 32-long `jmp_buf`;
    x86-64 also the shadow stack pointer, whose entries longjmp pops as the unwind does. A function that calls
    `setjmp` has all its loads and stores made volatile by the frontend and is never inlined: the second return
    has no edge in the graph, so otherwise a variable unmodified after `setjmp` could live in a register or a
    slot coalesced with one that is dead on the graph. `_setjmp`/`_longjmp` too; not `sigsetjmp` (signal masks
    are POSIX), not GCC's `__builtin_setjmp` (its 5-word buffer can't hold the registers), not WebAssembly
    (a clear link error).

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
  fused even with `CIMO_FMA=0`, which only controls contraction. The IR tells them apart (2026-11-07): contraction
  emits `fmac` (Ins.contract on fma1, kept by copies, inlining and the loop transforms), `fma()` emits `fma`, and only
  `fmac` may be split again (`fmashare`, `amd64/fmachain`); before, both split `fma()` calls too, which broke
  error-free products (`lo = fma(x, x, -x*x)` was 0). x86-64 with FMA compiles `fma()` to `vfmadd` (was a call to
  the software `fma`).
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
