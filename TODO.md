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
7. ~~**macOS (Apple silicon).**~~ **Done:** the suite, corpora and examples pass on an M-series
   Mac, ABI checked against Apple clang, CI job `macos-arm64` (see "macOS on Apple silicon" in
   `requirements.md`).
8. **SIGSEGV/SIGBUS in a kernel kills the process.** Turn faults inside module code into
   `KernelError` (signal handler that unwinds to the entry trampoline when the pc is in a
   module; chain to Python's handler otherwise). On Windows an access violation ends the
   process silently (exit 5); a vectored exception handler that resumes at the unwind glue
   would do the same job there.
9. **`_Complex` / `complex.h`.** Not in cproc or QBE. Needs: frontend lowering (Annex G
   `__muldc3`/`__divdc3` semantics), arm64 ABI (two FP registers), musl `src/complex` in the
   runtime, numpy `complex128` binding.
10. **GPU.** Plain C on any GPU with no vendor toolkit; tuned kernels as C in `python/cimo/gpu/`. WGSL, then SPIR-V, PTX, HIP
   source, Metal source. Start with one `matmul` on WebGPU, which proves type recovery and compile time. See
   "GPU" in `requirements.md`.

## Optimizations to lift from LLVM and GCC

Everything in LLVM's and GCC's pipelines that applies to scalar C on arm64 and wasm, biggest payoff
per compile time first. Sources: `../llvm-project` (`llvm/lib/Passes/PassBuilderPipelines.cpp`,
`llvm/lib/Target/AArch64/AArch64TargetMachine.cpp`), `../gcc` (`gcc/passes.def`), both shallow
sparse clones. **Licenses:** cimo is MIT. LLVM is Apache-2.0 with the LLVM exception: porting is
fine, with its notice kept. GCC is GPL: read for what it does, never translate its code.
**Follow them by default:** read what LLVM and gcc do and emit (`clang.exe` from VS and gcc are on this
machine) before designing anything. Differ only with a measurement showing they lose, and record it.
**Rules for each:** behind `backend.ext` (stock QBE stays byte-identical), measured with
`bench/` and scramjit's `bench/gcc.py` before and after including compile time, tests, then Csmith.
A pass that doesn't pay for its compile time is recorded here and dropped; cimo compiles at run time
in Python.

**Tier 1, instruction selection and peepholes** (seen in today's output; no analysis needed):
- ~~Frame only when needed, shifts by an immediate, register-offset addressing, `madd`, `msub`,
  `cbz`, `cbnz`~~ **Done** (`ext.isel`).
- ~~Pre/post-index addressing, `ldp`/`stp` with write-back~~ **Done** (`isel.sinkinc`, `emit._writeback`).
- ~~Shifted and extended operands~~ **Done, except onto a loop's accumulator** (`isel._accumulator`), where
  LLVM folds. Measured folded / not on Snapdragon X and Neoverse N1, V1, V2: `s += x << 3` 1.82, 0.84, 0.96,
  0.95; `s -= x >> 3` 1.82, 1.19, 1.46, 1.70; through the shifted value 1.00, 0.71, 0.65, 0.55; off any chain
  1.04, 0.87, 0.76, 0.85. Folding the first on Neoverse alone (`FeatureALULSLFast`) needs per-CPU tuning: open.
- ~~`mneg`, `fnmul`~~ **Done** (`ext.isel`).
- ~~`csel`, `fcsel`~~ **Done** (`ext.isel`; policy from `SimplifyCFG`, `EarlyIfConversion`). Then
  wasm `select`.
- `cbz`, `cbnz`, `tbz`, `tbnz` (`AArch64CondBrTuning`), `tst`, `cmp`/`cmn` dropped after a flag-setting
  `subs`/`adds`/`ands` (`AArch64ConditionOptimizer`, GCC `compare_elim_after_reload`), `ccmp` for `a && b`
  (`AArch64ConditionalCompares`).
- ~~`bic`, `orn`, `eon`, `mvn`, `ubfx`, `sbfx`, `ror`, storing `xzr`, `fmov` immediates~~ **Done**
  (`ext.isel`), and `csinc`/`csneg`/`csinv`, `xzr` in `csel`, `ubfiz`, `tst`/`tbz`/`tbnz`, identical compares
  dropped, `mul` by constants near powers of two, sibling calls, self tail recursion as a loop
  (`inline.selftail`), `csetm`, `ccmp` also for values and for `&&` through a flag variable
  (`thread._ifcombine`, `isel._ccmp`), `bfi`, `bfxil` (`isel.selbfi`), flags from `ands`/`adds`/`subs`
  (`emit._flagsfrom`), `tbz`/`tbnz` on the sign. Open: `mi`/`pl` after `subs`/`adds`.
- ~~Redundant extensions (GCC `ree`)~~ **Done** where a `w` write already cleared the top (`isel._DEF32`), and
  logical immediates on `w`, which went through a register. Open: demanded bits (`BDCE`, GCC `ext_dce`).
- ~~Division by a constant~~ **Done** (`cse.divconst`). `mul` by 2^n+-1 as `add`/`sub` with a shift.
  ~~Large constants and global addresses built outside loops~~ **Done** (`cse.hoistcons`, `MachineLICM`, GCC
  `rtl_move_loop_invariants`): `H[i] += G[i]` 0.68 to 0.34 ns. Open: rematerializing when out of registers.
- Tail calls and sibling calls (`TailCallElim`, GCC `tail_recursion`, `tail_calls`), loops for
  self-recursion. ~~Jump tables for dense `switch`~~ **Done**, from 16 cases, not LLVM's 4: per element, tree /
  table, random cases: 4 cases 5.0 / 6.5 ns, 8 cases 7.5 / 7.2, 16 cases 9.0 / 7.6, 32 cases 11.3 / 7.8; a few
  dominant cases 4.5 / 5.6. Measure on other CPUs. ~~Value tables~~ **Done** (`emit._valuetable`), LLVM's 4 cases and 40% density: 0.85 ns
  per element from 4 cases on against 3.8 to 7.8 for the tree. Open: bit tests for `c == 1 || c == 2 || c == 5`.

**Tier 2, scalar IR passes:**
- Algebraic simplification: `InstCombine`, `AggressiveInstCombine`, GCC `forwprop`/`match.pd`.
- ~~Value ranges and known conditions~~ **Done** (`simplify.srange`, `simplify.decide`, known bits): facts
  from dominating branches, counters that only grow. `SCCP`, `ADCE` **done** (`backend/scalar.py`).
- ~~`nsw` from the frontend~~ **Done** (`backend/nowrap.py`, `compile(wrapv=True)` for `-fwrapv`): comparisons,
  divisions and LLVM's `WidenIV`; wasm doesn't get it yet. Open: LFTR and the middle end's overflow guards,
  `nsw` on `shl`, `add` with an invariant step.
- ~~Jump threading~~ **Done** (`backend/thread.py`), for joins that only branch. Open: threading through
  blocks with other work, by copying them (`DFAJumpThreading`).
- ~~Loop unswitching~~ **Done** (`middle/unswitch.py`), small innermost loops with one exit block.
- If-conversion with a cost model: `SimplifyCFG` (speculate at most 4 cost units), `SelectOptimize`
  (select back to a branch when predictable), `EarlyIfConversion` (trace metrics); GCC `phiopt`,
  `cselim`, `tree_ifcombine`. Replaces the one-block-body rule in `middle/ifconv.py`.
- ~~`Reassociate`~~ **Done** for constants, sums and `a * b + a * c` (`cse.reassoc`), range checks as one
  comparison (`cse.ranges`). Open: `StraightLineStrengthReduce`/GCC `strength_reduction`, `SpeculativeExecution`.
- ~~Local `DSE`, `MergedLoadStoreMotion`, GVN hoisting, GCC `store_merging`, small `memcpy`, `memset`, `strlen`
  of a literal, `pow(x, 2)`, `ExpandMemCmp` for `== 0`, loads of constant globals~~ **Done** (`backend/memopt.py`,
  `frontend.qbe`, `inline.constglobals`), `SROA` (`backend/sroa.py`), load PRE for one missing way in and `DSE`
  across a branch (`memopt.pre`, `memopt.dse`). Open: `MemCpyOpt`, `MergeICmps`.
- ~~GCC `optimize_bswap`, load combining~~ **Done** on arm64 (`cse.bytepatterns`); wasm could have the loads.
  Open: GCC `cse_reciprocals`, `cse_sincos`, `expand_pow`.
- ~~`DeadArgumentElimination`, `IPSCCP`, `GlobalDCE`, readonly from `FunctionAttrs`~~ **Done** (`inline.ipo`,
  `inline.expand_all`). Open: nocapture, `ModRef`/GCC `ipa_modref`, `PartialInliner`, `ConstantMerge`,
  `MergeFunctions`/GCC `ipa_icf`. Inlining with a cost model (`InlineCost`): no benchmark moves between size
  limits 40, 100 and 225 (only 0 is worse: scramjit's viewshed 3.48 against 4.05 ms), so 40 stays for now.

**Tier 3, loops** (`middle/`, which has unrolling, unroll-and-jam, scalar replacement, strength reduction):
- `LoopRotate`, `IndVarSimplify`/GCC `iv_canon`, `LoopSimplifyCFG`, `LICM` scalar promotion. ~~`LoopDeletion`~~
  **Done** for loops that count (`scalar._deadloops`).
- `LoopIdiomRecognize`: memset, memcpy, popcount, ctz. GCC `crc_optimization`.
- `LoopStrengthReduce`/GCC `iv_optimize`, with `AArch64` addressing modes as its cost.
- Store-to-load across iterations: `LoopLoadElimination`, GCC `predcom`. scramjit does the one-store case
  itself (`ema` 3.09 to 2.39 ns, where the same loop as hand C through cimo stays at 3.06).
- `LoopInterchange`/GCC `linterchange`, `LoopDistribute`/GCC `loop_distribution`, `LoopFuse`, `LoopFlatten`,
  `LoopVersioningLICM`/GCC `loop_versioning`, `IRCE`/GCC `loop_split`, `LoopSink`.
- Vectorization is its own item below (`LoopVectorize`, `SLPVectorizer`, `VectorCombine`).

**Tier 4, machine level:**
- Register allocation: live-range splitting and rematerialization (`RegAllocGreedy`, GCC `ira`, `early_remat`),
  coalescing (`RegisterCoalescer`, GCC `regrename`, `web`): QBE's allocator spills in the hot paths of
  branchy code. `MachineCopyPropagation`/GCC `cprop_hardreg`, `AArch64RedundantCopyElimination`.
- Shrink wrapping (GCC `thread_prologue_and_epilogue`), `MachineSink`, `MachineLICM`, `MachineCSE`.
- Instruction scheduling (`MachineScheduler`, GCC `sched`, `sched2`, `sched_fusion`), fetch alignment of
  loops, `MachineBlockPlacement`/GCC `reorder_blocks`, `AArch64CodeLayoutOpt`, `TailDuplication`/GCC `tracer`.
- `AArch64LoadStoreOpt` beyond today's pairs, `AArch64PromoteConstant`, `GlobalMerge`, `AArch64CollectLOH`.

**Not applicable:** coroutines, OpenMP, sanitizers, CFI and hardening, PGO and sample profiles, LTO and
devirtualization, matrix intrinsics, SVE and SME, per-CPU workarounds, `long double`.

## Performance

- **A faster `Kernel.__call__`** (requested by scramjit, 2026-11-01). On quiet arm64 Linux with ints and an
  int address for the pointer, a call costs about 905 ns where the underlying ctypes call is about 450, so the
  wrapper is about 450 ns, for any caller making many small calls. Passing a buffer costs more: a numpy array
  about 1,200 ns more (`__array_interface__`; `addressof(c_char.from_buffer(a))` is much less, falling back for
  read-only arrays), a `bytearray` more again (`memoryview` and `from_buffer`). Only those totals were measured;
  the likely pieces, from reading the code, are the argument list comprehension, `self._state.ctx()` and the
  `try`/`finally` for stdout. Idea: choose a per-kernel call function when it's created, from its parameters
  (no pointers means no list; no stdout means no `finally`). In Pyodide a call is about 13 us, not broken down;
  the JavaScript wrapper converting i64 parameters to BigInt is a suspect. Reproduce with a loop over
  `kernel(1, 2, addr)`. **When done, note it in scramjit's `TODO.md`.**
- **Say whether a compile came from the cache** (requested by scramjit, 2026-11-01): a public attribute on the
  module, or a compile time. `scramjit preflight --run` wants to show `disk` instead of a time, and can't tell.
  **When done, note it in scramjit's `TODO.md`.**
- **WebAssembly: a call through a function pointer that takes a pointer traps** (found by scramjit, 2026-11-01).
  Calling Pyodide's libm `frexp` by address, cast to `double (*)(double, int *)`, gives "null function or
  function signature mismatch"; doubles and ints are fine (`ldexp`, `sin`). cimo's wasm pointers don't match
  the main module's `i32`. It affects any kernel calling into a host library by pointer, or a callback taking
  one. scramjit worked around it by computing `frexp` from the float's bits. **When done, note it in
  scramjit's `TODO.md`.**
- **Make the CLI framework public, or a small package** (requested by scramjit, 2026-11-01, low priority).
  scramjit's command line copies `cli/term.py`, `cli/parse.py` and `cli/completion.py`, because they aren't
  public API, so a fix to one needs copying by hand.
- **Ctrl-C can't stop a running kernel** (found by scramjit, 2026-11-01): Python's handler only runs between
  bytecodes, so `KeyboardInterrupt` waits until the kernel returns, and an infinite loop can only be killed.
  Every cimo user has this. scramjit is adding a check of its own to the C it generates; if cimo offers a
  general way (an interruptible loop flag in the execution context, set from the entry path), scramjit would
  use that instead.
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
  - ~~No common subexpression elimination~~ **Done** (`backend/cse.py`, `ext.cse`): loads merge after
    gvn, divisions are pure on arm64 and a remainder shares its division, a repeated condition and
    `y < 0 ? y + n : y` merge. Generated flood_fill 3.60 to 2.78 ms against gcc's 1.82, viewshed
    (parity off) 7.30 to 6.98 against 3.41. What's left there: the items under "Optimizations to lift".
  - `middle/ifconv.py` only converts loops with one exit and no call: measured, still right with
    `csel` (viewshed 8.17 against 14.38 ms). `backend/ifopt.py` has no such rule, and scramjit's
    flood_fill lost 2.47 to 2.72 ms from it: needs `SelectOptimize`'s model (Tier 2).
  **When fixed, add a note to scramjit's `TODO.md`** so it can drop the workarounds listed there.

- ~~**Inline small functions**~~ **Done** (`backend/inline.py`): helpers now run as fast as macros
  (`ema` 5.12 to 3.33 ns, a wrapping-add sum 2.05 to 0.34 ns). Costs 13-38% compile time and about
  25-30% more code on `t_printf` and `t_math`. Open: drop a static function once every call is
  inlined (its definition is still compiled and emitted); honor `noinline`.
- **Inlining stops by position in a large function** (from scramjit): `inline.CALLER_LIMIT` (4000 IR
  instructions) is spent in source order, so later code keeps its calls however hot it is and however
  small the callee. A llama2.c forward pass as one function (scramjit's `bench/llm.py`) called
  one-instruction helpers three times per element in its matrix loops: 28 tok/s, against 89 with the
  limit lifted (compile 0.8 to 1.4 s). Exempt callees no bigger than the call they replace, which
  shrinks the caller, and spend the rest of the budget on the deepest loops first. scramjit made its
  four wrapping helpers macros because of this; its other helpers still hit the limit.
- **FMA contraction slows reductions on arm64**: `s += a[i] * b[i]` runs 2.15 ns per element with `fmadd`,
  1.81 with `fmul` and `fadd` (`-ffp-contract` shows the same in gcc: `bench/llm.py`'s C at `-O3`, 63 tok/s
  with, 82 without), probably the latency on the sum (not measured on other cores). Contraction is on
  by default. Measure on other arm64 cores, then consider not contracting when `c` is carried round the loop.
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

- Investigate multiprocessing to speed up large compilations (one compile uses one core today).
