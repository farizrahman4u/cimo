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

## Optimizations to lift from LLVM and GCC

Everything in LLVM's and GCC's pipelines that applies to scalar C on arm64 and wasm, biggest payoff
per compile time first. Sources: `../llvm-project` (`llvm/lib/Passes/PassBuilderPipelines.cpp`,
`llvm/lib/Target/AArch64/AArch64TargetMachine.cpp`), `../gcc` (`gcc/passes.def`), both shallow
sparse clones. **Licenses:** cimo is MIT. LLVM is Apache-2.0 with the LLVM exception: porting is
fine, with its notice kept. GCC is GPL: read for what it does, never translate its code.
**Rules for each:** behind `backend.ext` (stock QBE stays byte-identical), measured with
`bench/` and scramjit's `bench/gcc.py` before and after including compile time, tests, then Csmith.
A pass that doesn't pay for its compile time is recorded here and dropped; cimo compiles at run time
in Python.

**Tier 1, instruction selection and peepholes** (seen in today's output; no analysis needed):
- ~~Frame only when needed, shifts by an immediate, register-offset addressing, `madd`, `msub`,
  `cbz`, `cbnz`~~ **Done** (`ext.isel`).
- Pre/post-index addressing (`AArch64LoadStoreOpt`, GCC `inc_dec`).
- Shifted and extended operands: `add x0, x0, x1, lsl #4`, `add x0, x0, w1, sxtw`.
- `mneg`, `fnmul` (`MachineCombiner`, GCC `combine`).
- ~~`csel`, `fcsel`~~ **Done** (`ext.isel`; policy from `SimplifyCFG`, `EarlyIfConversion`). Then
  `csinc`, `csneg`, `cneg`, `csetm`. wasm `select`.
- `cbz`, `cbnz`, `tbz`, `tbnz` (`AArch64CondBrTuning`), `tst`, `cmp`/`cmn` dropped after a flag-setting
  `subs`/`adds`/`ands` (`AArch64ConditionOptimizer`, GCC `compare_elim_after_reload`), `ccmp` for `a && b`
  (`AArch64ConditionalCompares`).
- `bic`, `orn`, `eon`, `mvn`, `ubfx`, `sbfx`, `ubfiz`, `bfi`, `bfxil`, `extr`/`ror`.
- `xzr` and `wzr` as operands: `str xzr, [x0]`; `movi`/`fmov #imm` for FP constants (2.0, 0.0).
- Redundant extensions (`ldrb` then `uxtb`; GCC `ree`, `ext_dce`).
- Division by a constant: multiply by the inverse (`TargetLowering::BuildSDIV`/`BuildUDIV`, Hacker's
  Delight), signed powers of two with shifts. `mul` by 2^n+-1 as `add`/`sub` with a shift.
- Tail calls and sibling calls (`TailCallElim`, GCC `tail_recursion`, `tail_calls`), loops for
  self-recursion. Jump tables for dense `switch` (`lower_switch`, `if_to_switch`).

**Tier 2, scalar IR passes:**
- Algebraic simplification: `InstCombine`, `AggressiveInstCombine`, GCC `forwprop`/`match.pd`.
- Value ranges and known conditions: `ConstraintElimination`, `CorrelatedValuePropagation`, GCC `vrp`,
  `dominator`; removes bounds checks from what dominating conditions prove (`top > 0` then
  `top - 1 >= 0`). `SCCP`/GCC `ccp`, `BDCE` and known bits, `ADCE`/GCC `cd_dce`.
- Jump threading (`JumpThreading`, GCC `thread_jumps`, `DFAJumpThreading`) and loop unswitching
  (`SimpleLoopUnswitch`, GCC `tree_unswitch`).
- If-conversion with a cost model: `SimplifyCFG` (speculate at most 4 cost units), `SelectOptimize`
  (select back to a branch when predictable), `EarlyIfConversion` (trace metrics); GCC `phiopt`,
  `cselim`, `tree_ifcombine`. Replaces the one-block-body rule in `middle/ifconv.py`.
- `Reassociate`/GCC `reassoc`, `StraightLineStrengthReduce`/GCC `strength_reduction`, `SpeculativeExecution`.
- Memory: `SROA`/GCC `sra`, `DSE`, `MemCpyOpt`, `MergedLoadStoreMotion`, `GVN` load PRE/GCC `pre`, GCC
  `store_merging`, `sink_code`, `MergeICmps`, `ExpandMemCmp`.
- GCC `cse_reciprocals`, `cse_sincos`, `expand_pow`, `optimize_bswap`, `optimize_widening_mul`, `strlen`.
- Inlining with a real cost model (`InlineCost`), `DeadArgumentElimination`, `IPSCCP`, `GlobalOpt`,
  `FunctionAttrs` (pure, nocapture: loads and stores survive calls), `ModRef`/GCC `ipa_modref`,
  `PartialInliner`, `GlobalDCE`, `ConstantMerge`, `MergeFunctions`/GCC `ipa_icf`.

**Tier 3, loops** (`middle/`, which has unrolling, unroll-and-jam, scalar replacement, strength reduction):
- `LoopRotate`, `IndVarSimplify`/GCC `iv_canon`, `LoopDeletion`, `LoopSimplifyCFG`, `LICM` scalar promotion.
- `LoopIdiomRecognize`: memset, memcpy, popcount, ctz. GCC `crc_optimization`.
- `LoopStrengthReduce`/GCC `iv_optimize`, with `AArch64` addressing modes as its cost.
- Store-to-load across iterations: `LoopLoadElimination`, GCC `predcom` (scramjit's `ema`).
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
