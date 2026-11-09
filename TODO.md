# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Big

- **Before the next release: a custom SIGINT handler mustn't unwind a kernel** (found by scramjit, 2026-11-09). Since
  Ctrl-C stops kernels (2026-11-07), SIGINT during a kernel unwinds it whatever Python's handler is. With a handler of
  the program's own that doesn't raise (`signal.signal(SIGINT, lambda *a: seen.append(1))`), plain Python would keep
  going; the call now dies with `KernelError: interrupted in sj_kernel()`, and no handler can resume an unwound kernel.
  Proposed rule, for every caller: if Python's SIGINT handler is `signal.default_int_handler`, unwind and raise
  `KeyboardInterrupt` as now (the program gets the exception it was getting, sooner); otherwise (a handler of its own,
  or `SIG_IGN`) record the signal as Python's C handler does and let the kernel finish, so the handler runs when the
  call returns, as after any slow C call. Snapshot the handler at kernel entry: `signal.signal()` works only on the main
  thread, and while a kernel runs there no Python code can change it, so the snapshot is exact and the signal handler
  needs no Python. Reproduce: scramjit's `tests/test_interrupt.py::test_a_custom_handler_doesnt_stop_the_kernel`
  (and `test_a_handler_set_after_the_first_compile_is_respected` must stay passing). scramjit needs no change: its own
  stop flag stays as the fallback where cimo can't unwind (Windows, time spent in the runtime).

- **High priority: x86-64 for this machine's CPU.** Detection, the cache key, FMA (scalar and vector, negations
  folded, none on a recurrence through its addend) and `popcnt`/`lzcnt`/`tzcnt` **done** (`cimo/cpu.py`; see "x86-64
  for this CPU" in `requirements.md`). 256-bit AVX2 vectors **done** (`ext.qbytes()`; see "AVX2" in `requirements.md`): against gcc `-O3 -march=native`
  1.05x (Sapphire Rapids), 1.10x (Ice Lake), 1.18x (Zen 4), from 1.16 to 1.41x with 128 bits. Open: `vpblendvb` for selects
  (and/andn/or today), byte lanes (`rgb2gray` 3.7 to 8.2x), `nbody_acc` (1.9 to 2.6x: gcc's outer-loop vectorization).

- **High priority: scavenge the test suites of other C compilers.** Miscompiles keep turning up as side findings, which
  isn't acceptable. In CI now (`tests/corpora.py`, fetched at pinned commits, never copied: several are GPL): gcc's
  `c-torture/execute`, `c-torture/compile` (compile only), `gcc.dg` run tests, `gcc.dg/torture` and `gcc.dg/vect`,
  c-testsuite and musl's libc-test, 7587 programs with no known miscompile (2026-11-09); plus Csmith, YARPGen, random
  loops and the ABI tests. Left, in this order:
  - **Other targets:** the corpora run natively only, so x86-64 is covered by the linux-x64 and windows-x64 CI jobs
    at whatever ISA level the runner has. Not yet: x86-64 baseline and AVX2 forced (`CIMO_X86`), and wasm (node).
    The 2026-11-09 round found four bugs only other platforms ran (float constants rounded twice, two x86-64
    backend crashes): every suite on every backend.
  - **LLVM test-suite** `SingleSource/UnitTests`, `SingleSource/Regression/C` and the small `SingleSource/Benchmarks`,
    with their reference outputs (and Benchmarks doubling as performance checks).
  - **cproc's** and **QBE's** own tests (cimo is a port of both: anything they fixed since the port), **tcc** `tests/tests2`
    (numbered programs with `.expect` files), **chibicc** and **slimcc** `test/`, **lacc**, **kefir**, **arocc**, SDCC's
    portable regression tests, and CompCert's `test/c`.
  - The rest of gcc's: `gcc.dg/vect`'s subdirectories (`costmodel/`, `tsvc/`, ...) and the programs that include
    `../../c-c++-common/` files; `gcc.dg`'s subdirectories (`ipa/`, `tree-ssa/`, ...: mostly dump scans, some run).
  - Features the corpora reject today (clear errors, not miscompiles): non-ASCII and UCN identifiers; bit-fields that
    `#pragma pack` or `packed` make cross a storage unit; x86-64 sibling calls (`skip:amd64`); compile time and memory
    of huge functions (100000 case labels: 255 s, 5.5 GB; three `skip`s).
  Each suite: a fetcher entry in `corpora.py`, entries in `corpora_expect.txt` (what's known to fail and why: an
  extension, a rejected feature, a gcc option), CI, and every new failure triaged into a bug fixed here or a recorded
  exclusion; a miscompile is always fixed, never recorded. Done when every suite above runs on every backend in CI with
  no unexplained failure.

0. ~~**A fault in a kernel must not kill the Python process.**~~ **Done** (`cimo/fault.py`,
   `tests/test_faults.py`; see "Hardware faults" in `requirements.md`).

1. ~~**Single-file distribution.**~~ **Done:** `cimo onefile [-o cimo.py]`
   (`cimo/onefile.py`, `cimo/resources.py`; `.c`/`.h` stay real files). See "Single-file
   distribution" in `requirements.md`.
2. **Python interop API (M6).** Struct/union arguments and returns (ctypes structures, dicts),
   VLA shape checks at call time (README promises them). The CLI is done (`cimo run`,
   `compile`, `onefile`, `cache`, `completion`) except `cimo report kernels.c`: per-loop
   optimization remarks from the middle end (README promises it). zsh/fish completion
   scripts are untested.
3. ~~**amd64 backend.**~~ **Done** (branch `amd64-backend`): Linux x86-64 and Windows x64 run end to end. Corpora and `tests/test_abi.py` pass on both (see "x86-64" in `requirements.md`). Open: sibling calls (arm64's `b` for a tail call; mutual tail recursion still uses stack on x86-64), and the arm64-only `ext.cse` passes (`divconst`, `bytepatterns`, `hoistcons`) for x86-64; SSE4.1/POPCNT/FMA with CPU detection; `conv3x3`'s scalar path, 2 to 4x behind gcc's; `edit_distance` on Neoverse V2, 1.3x; hardware faults (#DE on INT_MIN/-1 or x/0 kills the process, as SIGSEGV does). A wrong checksum on Windows x64 only (Csmith seed 17936855311685): right on Linux x86-64, Linux and Windows arm64, wrong under Windows on Arm's x64 emulation; real x64 Windows would tell an ABI bug from the emulator. Two more from Csmith, in the code before the vector work too: seed 17936895825451 trips `load.def_`'s loop-depth assertion on x86-64 (the `ext.cse` loadopt), and seed 17936895823073 never finishes on either target, middle end and inlining off or not (gcc's build runs in 2 ms).
4. ~~**WASM backend + Pyodide loading.**~~ **Done:** `cimo.backend.wasm`, `cimo.wasm` (see "WASM
   backend" in `requirements.md`). Open: host JS libraries via `link=`, async loading for large
   modules on a browser main thread, wasm SIMD.
5. **SIMD.** Auto-vectorizer **running on arm64 and x86-64** (`middle/vectorize.py`; see "Vector code" in `requirements.md`). Open, in
   order: reductions through a chain (`s += x * x + 3`), outer-loop vectorization (`nbody_acc`, gcc 1.3x ahead), byte and short lanes with widening and narrowing (scramjit's `image[i] + k` on `uint8`: 0.24 ns per element, `int32` 0.12), interleaved access (`rgb2gray`),
   downward steps; GNU vector types' operators as class-`q` instructions **done** (2026-11-07; the rest under 5b), wasm, SLP; AVX2 behind a check at load. ~~Conditions~~ **done** (comparison masks and selects, nested ones too; see "Vector code"). **Float conditions on x86-64:** `ifconv` makes selects of integers only there, since x86-64's scalar code has no float select; a float one (and/andn/or of a `cmpsd` mask, or a branch) would let the vectorizer take `relu` there too. Integer `max`/`min` through a condition **done**, with four accumulators and the unit-stride fast path (`middle/stride.py`): int64 max 0.87 to 0.28 ms on 1,000,000 elements (numpy 0.29), int32 min 0.86 to 0.14 ms (Snapdragon). **Next: the rest of the reductions with a condition (scramjit 2026-11-05, arm64, 1,000,000 elements, against numpy's own SIMD): `max` and `min` of `float64` 1.28 ms against 0.15, of `float32` 1.29 against 0.09, of `int64` 0.86 against 0.15, of `int32` 0.86 against 0.09; `argmax` 1.70 against 0.39; `any` and `all`, which also leave the loop early, 0.43 against 0.20 (`float32` 0.12). A `float` `max` must keep what scramjit's `max` does with NaN and signed zeros: per-lane maxima merged in lane order can pick another of two equal zeros than the scalar loop, which keeps the first. scramjit's float `max`/`min` also leave the loop at the first NaN (`i < n && r == r`), and `argmax` returns from inside it. Loops that exit early **done** for search loops (`middle/earlyexit.py`: `any`, `all`, find; scramjit's strided `any` 0.40 to 0.20 ms, numpy 0.19); open: early exits that also reduce (`argmax`, float `max` with its NaN exit).** An `if` whose arm loads (`x <= a ? y[i] * 2 : x`, `y[i]` read only there) isn't converted: speculating the load needs it proved safe.
5b. **Types the IR has no class for: native code instead of lowering.** Class `q` (128 bits, 256 with AVX2) exists, but only
   the auto-vectorizer emits it. The frontend lowers what it can't express in w/l/s/d: GNU vector types run lane by lane
   through memory, `__int128` as pairs of 64-bit words (multiplication by `umulh`; division, shifts and conversions by
   runtime calls), `_Float16`/`__bf16` as `unsigned short` with a runtime call per conversion, `_Complex` as two scalars.
   The results are correct; the code is slow, and `__int128` and half arguments don't follow the platform calling
   convention (gcc: register pairs, FP registers). In order:
   - GNU vector types as class-`q` values: operators, copies, `v[constant]`, and vector locals in registers **done**
     (2026-11-07, `tests/test_vector_q.py`; "GNU vector types" in `requirements.md`). `convertvector` and one-q `shufflevector` (`tbl`) on arm64 **done**.
     One-instruction shuffles on arm64 **done**. x86-64 AVX2 shuffles of 16 bytes and variable shifts **done**. x86-64 AVX2 `convertvector` **done**. `shuffle` with a variable mask, lane writes and variable shifts **done**. Lane by lane by design: integer
     division and 64-bit `*` (no instruction on either; gcc does the same). Shuffles of any width, and x86-64's unsigned 32-bit and 64-bit int
     to or from float conversions (sequences in place of AVX-512's), **done**. Vectors in calls as gcc passes them **done** on every target. Measure a hand-vectorized `Q8_0`
     dot and a 4x4 matrix kernel on mains power (not yet measured).
   - Half conversions without calls: arm64 `fcvt` and `__bf16` inline everywhere **done** (2026-11-07,
     `tests/test_half_hw.py`). Half loads and stores in the vectorizer on arm64 **done**
     (`ldr d` + `fcvtl`, `fcvtn` + `str d`; `test_vectorize.py`). x86-64 F16C conversions (`cimo.cpu`) and half arrays in
     the vectorizer (`vcvtph2ps`/`vcvtps2ph` with memory: 8 halves a ymm) **done**. Half and bf16 arguments in FP registers **done**.
   - `__int128`: passed as gcc does (even pair on arm64, two registers on SysV; `tests/test_abi.py`), halves in registers
     within a function (sroa), `umulh`, variable shifts inline **done** (2026-11-07). Struct and `__int128` values at call boundaries in registers **done**
     (`backend/abislot.py`: an `__int128` add 35 instructions to 8). An `__int128` add as `adds`/`adc` on arm64 **done**
     (`arm64/emit._carry`). Open: the same in a multiply-accumulate (the low word made by `madd`, needing a scratch).
   - `_Complex` stays two scalars (what gcc does too); multiply and divide inline under `CX_LIMITED_RANGE` **done** (2026-11-07).
6. **Debug mode.** `debug=True`: bounds checks against Python buffers and heap blocks
   (`cimo.BoundsError`, as in the README), use-after-free, double free, leak report.
7. ~~**macOS (Apple silicon).**~~ **Done:** the suite, corpora and examples pass on an M-series
   Mac, ABI checked against Apple clang, CI job `macos-arm64` (see "macOS on Apple silicon" in
   `requirements.md`).
8. **Adversarial code reading before a public launch.** First pass **done** (2026-11-05), five independent reads, each
   finding reproduced against GNU as, a Python model or the scalar build before it was fixed:
   loader/execution/faults (7 found: a hang when a displaced handler returns, kill-sent signals unwinding kernels, a stale
   fault-table slot, cached images from another page size crashing on load, Windows handler gap and stack limits; fixed
   but the three listed in "From the adversarial read" below), the arm64 assembler (6, input checks: xzr where 31 is sp,
   truncated immediates, NEON arrangements, shifts and lanes), the x86-64 assembler (2 and validation gaps: absolute
   addresses from 0x80000000 sign-extended, xmm/ymm widths; branch relaxation listed below), the vectorizer and loop
   passes (2: qsel1 unpinned, an early exit with a narrower test), register allocation (4: reg/slot swaps miscompiled,
   spill pressure at qfma1/qsel1, qacrosscalls splitting a pair, ldp across qload/qstore). Each read's checked-and-sound
   list is in its commit. Not yet read: frontend, preprocessor, the ext passes, gvn/gcm, the wasm backend, runtime C.

9. ~~**`_Complex` / `complex.h`.**~~ **Done** (2026-11-06; see "Complex numbers" in `requirements.md`). Open: a
   complex parameter or return of a kernel called from Python (Python `complex`, numpy `complex128`), and
   `long double _Complex` on Linux (no `long double` there).
10. **GPU.** Plain C on any GPU with no vendor toolkit; tuned kernels as C in `python/cimo/gpu/`. WGSL, then SPIR-V, PTX, HIP
   source, Metal source. Start with one `matmul` on WebGPU, which proves type recovery and compile time. See
   "GPU" in `requirements.md`.

## Directions (priority order; weeks each; no C original to diff against)

1. **Derivatives of the bundled libm.** scramjit's autodiff (its `TODO.md`, "Directions" #1) needs
   `d/dx` of every `math.h` function cimo provides, as C in the runtime, with the same bit-exactness
   discipline as the functions themselves. Also an `FP_CONTRACT`-respecting way for generated
   derivative code to opt out of reassociation. Small; blocks the item above.
2. **Be the compiler behind tools that generate C.** Devito, CasADi/acados, OSQP's codegen,
   PyMC/PyTensor's C backend, Brian2, libRoadRunner: each emits C and shells out to gcc/MSVC. The
   work per tool: compile its output unchanged (expect `#pragma omp`, `restrict`, `static inline`,
   math-heavy expressions, large files), a `CC`-shaped entry point or a drop-in for its compiler
   hook, and a measurement against gcc on the tool's own benchmark. Start with Devito (stencils,
   multicore, runs in a notebook) and CasADi (self-contained C, no Python API), one each. Then
   a doc page "cimo as a backend for code generators."
3. **GPU: WGSL first** (also Big #10). Moved up: the shape is a per-element kernel from C with an
   index-space launch, WebGPU in the browser and wgpu natively, no vendor toolkit. First
   milestone: one matmul and one stencil from `kernels.c`, compiled to WGSL by a new backend over
   the same IR, with type recovery (QBE IR loses C types; WGSL needs them) as the first problem.
   SPIR-V, PTX, HIP and Metal source only after WGSL works end to end.
4. **Autotuning in-process.** Because a compile is milliseconds and nothing leaves the process:
   compile a kernel in N variants over the middle end's existing knobs (unroll factor, jam depth,
   vector width, alias-check vs. versioned, inline limit), time each on the host CPU, keep the
   best in the cache keyed by `(source hash, CPU model)`. API: `cimo.compile(src, tune=True)`;
   report what won. Measure on the 15 kernels first: if nothing beats the defaults by more than
   10%, record that and stop.
5. **Sandbox limits for untrusted kernels on the WASM target.** What scramjit's sandbox (its
   `TODO.md`, "Directions" #5) needs from cimo: a fuel counter or epoch interruption for time
   limits, a memory cap enforced by the `grow` hook, and no host imports beyond memory. Natively,
   nothing: native is for trusted code only, say so.

## Adoption (priority order, below Directions; what scramjit's "Adoption" items need from cimo)

1. ~~**Threads: what `@scram(parallel=True)` needs.**~~ **Done** (2026-11-05): `tests/test_threads.py` (8 threads x
   1,000 calls with the heap, asserts, faults and printf crossing threads; modules compiled and dropped while others
   run; a reused pool) passes, also under free-threaded CPython 3.13t (20 runs in a row, and the execution, fault, runtime,
   loader, vector and compile-time tests); nothing needed changing. Scaling 7.7x on 8 threads (Graviton 3), 7.0x (Sapphire
   Rapids, 4 cores and 8 threads); a call about 1 us, of which the GIL's release is 45-60 ns. CI runs those files under
   3.13t (`linux-arm64-free-threaded`).
2. ~~Public cache export and import~~ **Done** (2026-11-07) as `module.cache_entry()` and `cimo.add_cache_entry()`, without `prebuilt=` (compiled code is a cache, never shipped). The plan was: API:
   `cimo.export_cache(module, path)` writes the linked image plus the signatures, the source hash,
   the target (`os-arch` or `wasm32`), the cimo version and the options; `cimo.import_cache(path)`
   returns a module without compiling when the host matches, and raises `cimo.CacheMismatch` naming
   what differs. `cimo.load(src, prebuilt=dir)` looks in `dir/<target>/<hash>.img` before compiling.
   No code signing: documented as trusted input, same as a wheel. Tests: round trip on every CI
   platform; a mismatch on each field; a corrupted file is a mismatch, not a crash. **Done when**
   scramjit's packaging item can drop its use of the private cache layout.
3. **Notebook environments.** Confirm `micropip.install("cimo")` from a stock Pyodide index resolves
   and imports (pure-Python wheel; check the wheel has no `platform` tag and that `onefile` isn't
   needed), in `tests/wasm/`. Then a `marimo` WASM notebook and a JupyterLite config in
   `examples/notebooks/`, as in scramjit's item 3, with one `cimo.compile` kernel each.
4. **Arrow buffers.** Nothing needed beyond the buffer protocol, which `execution` already accepts;
   add a test that a `pyarrow` buffer (dev extra) is passed zero-copy, and document the validity
   bitmap as an ordinary `const uint8_t *` parameter.
5. **Determinism across targets.** The compiler-side half of scramjit's item 5: a `deterministic`
   compile option (`fma=False`, no reassociation, scalar libm paths where the vector path differs),
   and `tests/test_deterministic.py` hashing the 15 kernels' outputs on fixed inputs into
   `tests/data/deterministic.json`, compared across all CI platforms and wasm. The first run is a
   survey: list every function whose bits differ between arm64, x86-64 and wasm (expect
   `pow`, `exp`, `log` between the bundled libm's scalar and vectorized code, and `sqrt`/`fma`), fix
   or exclude each with a reason in `faq.md`. Not the host libm: `link=["libm"]` is outside the promise.
6. **Half types and int8 dot products** (for scramjit's Directions #6, quantized inference kernels).
   `_Float16` and `__bf16` are accepted (2026-11-06) as storage types computed in `float`, converted by
   runtime calls. Open: hardware conversions (arm64 `fcvt`/`bfcvt`, x86-64 F16C behind a check at
   load; wasm stays in software), then vectorized loads and widening in the vectorizer. Int8 dot products
   (`sdot`/`udot` on arm64, VNNI behind a check on x86-64, wasm relaxed-SIMD dot) for a reduction of
   `int8 * int8` into `int32`. Oracle: gcc and clang on the same C. Measure a `Q8_0` block dot against
   ggml's before and after.

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
- ~~FP accumulator copies~~ **Done for phis with their own hint** (`rega`): `s += x[i] * y[i]` had one `fmov` on the chain,
  52 to 56% on Neoverse, nothing on Snapdragon X. Open: phis that feed each other share a hint and still swap through
  `s31` (`iir`, `conv3x3`), for the register allocation work below.
- ~~Benchmark gate~~ **Done** (`tests/test_bench_gate.py`): counts in the kernels' loops; timing stays a manual EC2 run.
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
  per element from 4 cases on against 3.8 to 7.8 for the tree.

**Tier 2, scalar IR passes:**
- Algebraic simplification: `InstCombine`, `AggressiveInstCombine`, GCC `forwprop`/`match.pd`.
- ~~Value ranges and known conditions~~ **Done** (`simplify.srange`, `simplify.decide`, known bits): facts
  from dominating branches, counters that only grow. `SCCP`, `ADCE` **done** (`backend/scalar.py`).
- ~~`nsw` from the frontend~~ **Done** (`backend/nowrap.py`, `compile(wrapv=True)` for `-fwrapv`): comparisons,
  divisions and LLVM's `WidenIV`; wasm doesn't get it yet. Open: LFTR and the middle end's overflow guards,
  `nsw` on `shl`, `add` with an invariant step.
- ~~Bit tests~~ **Done for `||` and `&&` chains** (`thread._bittest`), from 6 values and branch-free, where LLVM
  branches from 3 (measured in its docstring). Open: `switch` with several cases to one target, which reaches
  the backend as a tree.
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
- ~~A local assigned a constant in a loop stops vectorizing~~ **Done** (2026-11-07): a header phi set to a value from outside the loop each iteration, read nowhere in it, takes that value after each vector iteration (`vectorize._last`); `s += x[i]; f = 1;` runs as fast as the plain sum.
- `LoopRotate`, `IndVarSimplify`/GCC `iv_canon`, `LoopSimplifyCFG`, ~~`LICM` scalar promotion~~ (fields, above). ~~`LoopDeletion`~~
  **Done** for loops that count (`scalar._deadloops`).
- `LoopIdiomRecognize`: memset, memcpy, popcount, ctz. GCC `crc_optimization`.
- `LoopStrengthReduce`/GCC `iv_optimize`, with `AArch64` addressing modes as its cost.
- ~~SIMD targets from scramjit~~ **Done** (2026-11-09; float64 and int64, 1M elements, Snapdragon X): float maximum,
  minimum and argmax, int argmax vectorize exactly (`vectorize._select`, `Index`, `_closeindexed`; a float maximum in a
  loop that stores nothing runs the scalar loop again on a zero result instead of keeping an index); select reductions
  keep 8 accumulators on arm64; `qsel1` is `bit`, in place. scramjit's own C as it is: `max_i` 218 to 122 us (numpy
  121), `argmax_i` 677 to 178 (172), `all` 153 (148). Its float `max`/`argmax` stop at a NaN, which no vectorizer takes:
  rewritten as a strict maximum plus `z += e - e` under `#pragma omp simd reduction(+:z)`, the original loop only when
  z is NaN: `max` 1016 to 176 (numpy 113), `argmax` 1366 to 258 (numpy 315). Open: x86-64 float selects (the scalar
  `if` isn't if-converted there), 32-bit floats with an index, downward loops.
- Store-to-load across iterations: `LoopLoadElimination`, GCC `predcom`. scramjit does the one-store case
  itself (`ema` 3.09 to 2.39 ns, where the same loop as hand C through cimo stays at 3.06).
- ~~Tiling~~ **Done** for i, k, j nests (`middle/tile.py`: blocks of 16 values of k outside i; gemm). Open: other
  shapes, a block size from the row's bytes, tiling j.
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

- ~~Large blocks: keep freed ones for reuse~~ **Done** (2026-11-07, found by scramjit): up to 8 per thread, 128 MB,
  reused within 2x of the request (`heap.c`); a 10-round loop over 1M doubles 30.1 to 4.1 ms on the laptop.
  `MADV_HUGEPAGE` not tried.

- **A fast bundled libm, for scramjit's `parity=False`** (found by scramjit, 2026-11-07). scramjit's `parity=False`
  calls cimo's C library for `math` (its `parity=True` keeps the host's, as CPython does, and its own CPython-exact
  code). Agreed direction: cimo's functions as fast as glibc's or faster, which beats calling the host library because
  they inline, keep their constants in registers and vectorize (a call to the host blocks all three), and give the
  same results on every target, Pyodide and Windows included. One function at a time, as kernels need them, each timed
  against glibc in a loop. First, `hypot` is 1.4x glibc's (Snapdragon X): a viewshed loop (96x96, one `hypot` per
  inner step, about 280,000 calls) takes 2.19 ms compiled by cimo with its own `hypot`, and 1.53 ms compiled by
  gcc -O2 with glibc 2.39's; with `sqrt(a*a + b*b)` instead, cimo 0.98 and gcc 1.12 ms, so the loop itself isn't it.
  glibc's is one `sqrt` of an fma and one correction step. Plain C semantics (errno, IEEE results); CPython-exact
  variants stay in scramjit. When done, note it in scramjit's `TODO.md`. **`hypot` done** (2026-11-07): glibc's
  algorithm (Borges 2019), the fma kernel on arm64 and the correction step elsewhere, under 1 ulp (checked against
  exact results over all exponent ranges); a `hypot` loop 1.14 to 0.77 ms on the laptop, which puts it level with glibc
  by scramjit's ratio (to re-measure on mains power). Still a call: the runtime doesn't inline into kernels yet.

- ~~**The inliner sizes a function before cleaning it up, and grows small functions with their cold paths**~~
  **Done** (2026-11-07, found by scramjit): loads and stores of slots only ever loaded and stored, their constant
  offsets, copies and extensions of constants and `fma0` don't count toward `ext.inline` (`inline._free`); a function
  declared `inline` may be `HINT` (2) times the limit; a `static` function called once inlines only at that call, and
  not into a small caller it would push past that caller's own limit (`test_inline.py`). `sj_hypot2` counts 57, not
  over 150; scramjit's viewshed 3.44 to 3.12 ms. Noted in scramjit's `TODO.md`.

- ~~**Keep struct fields in registers across a loop**~~ **Done** (2026-11-05): a field the loop reads and writes at an
  address fixed for the loop (`p->x`), with one size and class, that no other access can touch (other offsets of the
  same base, or bases the alias analysis tells apart: `restrict`, or no other pointer at all) is a state variable of the
  fast loop, loaded before it and stored after it (`Fast.find_promoted`); a field only read (`p->k`) is hoisted, since
  offsets of one base no longer count as overlapping (`_apart`). The spring step through `p` is now the same loop as with
  locals, instruction for instruction, without `restrict`; through two pointers it needs `restrict` on both
  (`test_struct_fields_stay_in_registers_through_a_loop`, `fields_*` in `tests/middle/loops.c`). Open: fields of a
  base that changes in an outer loop are promoted per inner loop only; vectorized loops don't promote.

- ~~**A small local array passed to an inlined call stays in memory**~~ **Done** (2026-11-05): the inlined callee's
  pointer parameter is a copy of the array's address, which `sroa` took as the address escaping; it follows copies now,
  and runs again after promotion (which removes the slot the parameter lived in). scramjit's io[] is registers: no store
  left in the loop (`test_inlined_callees_result_array_is_registers`). The 16.5 against 11.7 ns didn't reproduce for a
  plain function call (scramjit's generated C, laptop: 6.16 ns both, before and after: the store was off the critical
  path); the method case may be the struct one below. Noted in scramjit's `TODO.md`.

- ~~**A faster `Kernel.__call__`**~~ **Done** (2026-11-05): each kernel gets a class of its own whose `__call__` is
  written for its parameters (pointers converted where they are, no list), and the error word and the stdout length are
  read from a per-thread `memoryview` of the context, with no ctypes object made; buffers go through
  `c_char.from_buffer` first. EC2, before to after: a `void` call 763 to 364 ns (Graviton 3) and 455 to 206 (Sapphire
  Rapids), one int 1,098 to 609 and 677 to 376, three longs 1,602 to 966 and 1,046 to 621, a `bytearray` 2,187 to 1,409
  and 1,373 to 881; ctypes alone, no arguments, 161 and 94. Pyodide's 13 us not looked at. Noted in scramjit's `TODO.md`.
- ~~**Large `malloc` on wasm**~~ (found by scramjit) **Done 2026-11-07:** NULL past what wasm32 has (glue and Python's grow),
  and `module.free`/`reset_heap`/`heap_bytes` in Pyodide (`tests/test_wasm.py`).
- ~~**`k.f(np.int64(3))` to a 64-bit parameter passes the scalar's address**~~ **Fixed** (2026-11-05): signatures
  mark pointer parameters `p` (from the front end's C types), so an `l` parameter takes a numpy integer as its value, on
  native and in Pyodide. Noted in scramjit's `TODO.md`.
- ~~Say whether a compile came from the cache~~ **Done** (2026-11-07, requested by scramjit): `module.cached`, also
  in Pyodide.
- ~~**WebAssembly: a call through a function pointer that takes a pointer traps**~~ (found by scramjit) **Done 2026-11-07:**
  pointers are i32 at a wasm function's boundary, as Emscripten's code passes them. A host `long` parameter still differs
  (Emscripten's is 32-bit, cimo's 64).
- ~~**A call by address into another module crashes once the callee uses `malloc`**~~ (found by scramjit) **Done
  2026-11-07:** contexts made on demand, errors moved to the innermost entry frame (`tests/test_cross_module.py`).
- ~~`#pragma STDC FP_CONTRACT`~~ **Done** (2026-11-07, requested by scramjit): ON, OFF and DEFAULT (the compile's
  `fma=`), to the end of the block or the next pragma at file scope.
- ~~Make the CLI framework public~~ **Done** (2026-11-07, requested by scramjit): `cimo.cli.parse`, `cimo.cli.term`,
  `cimo.cli.completion.script(shell, prog)` are public, documented in the CLI page; scramjit's color and encoding
  fixes folded in, the C-file hint a root parameter (`unknown_hint`).
- ~~Ctrl-C can't stop a running kernel~~ **Done on Linux and macOS** (2026-11-07, found by scramjit): SIGINT in a
  kernel's own code unwinds it and runs Python's handler (`KeyboardInterrupt`). Open: Windows (Ctrl-C arrives on its
  own thread: suspend the main thread and redirect it, as the fault path does), and a kernel spending all its time in
  the runtime.
- ~~Public cache export and import~~ **Done** (2026-11-07, requested by scramjit): `module.cache_entry()` (bytes)
  and `cimo.add_cache_entry(data)`, native and Pyodide; used only by a compile of the same source and options.
- ~~A public way to get a module's WebAssembly~~ **Done** (2026-11-07, requested by scramjit): `module.wasm` in Pyodide,
  `cimo.wat.disassemble` (was `docs/nn/wat.py`), `cimo.wasm.build(src)` documented.

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
  - **2026-11-07 (Snapdragon X, mains):** scramjit's `bench/gcc.py` now has gray_scott level, viewshed 0.64x of gcc
    (cimo faster; `hypot`, inlining, CSE), hand flood_fill level; only scramjit's generated flood_fill is behind, 2.10 to
    2.18 against 1.84 ms. Found and fixed on it: a select whose condition waits on an integer division is kept a branch
    (`ifopt.slowcond`, `middle/ifconv`; the floor-division sign fix as an `if`, 2.47 to 2.00 ms, gcc 1.82), dead `cmp`s
    dropped (`arm64/emit._deadcmp`), `divrem` again after loads merge (`p[i] / w` and `p[i] % w` shared one sdiv only
    when the loads were already one). Left: gcc unswitches the loop on `w` (cimo's `unswitch` takes loops of at most 50
    instructions with one way out; the generated loop has many `SJ_FAIL` exits), and computes the sign fix as
    `cbz r; eor r, w; tbz` where cimo materializes booleans through branches.
  **When fixed, add a note to scramjit's `TODO.md`** so it can drop the workarounds listed there.

- ~~**Inline small functions**~~ **Done** (`backend/inline.py`): helpers now run as fast as macros
  (`ema` 5.12 to 3.33 ns, a wrapping-add sum 2.05 to 0.34 ns). Costs 13-38% compile time and about
  25-30% more code on `t_printf` and `t_math`. Open: drop a static function once every call is
  inlined (its definition is still compiled and emitted). `noinline` and `noipa` are honored (2026-11-06).
- ~~`ratio_sum` in scramjit's `bench/m1.py` slower at `79d11d6`~~ **Not a regression: this machine only.** On Snapdragon X
  it runs at 1.02 ns per element before that commit and 1.05 to 1.11 after, moving with nops before the loop, though the
  hot loop's instructions are the same. On Neoverse N1, V1, V2 both commits take 2.03, 0.98 and 0.91 ns, with under
  0.5% spread at any offset or alignment. Check timing differences of a few percent on EC2 before chasing them here.
- **Loop alignment (LLVM's `alignBlocks`, 16 to 32 bytes by CPU) tried and left out:** on the 15 kernels the geomean is
  0.997 to 1.005 at both sizes on Snapdragon X and Neoverse N1, V1, V2, at 62 directives in 1639 instructions.
- ~~Inlining stops by position in a large function~~ **Done** (2026-11-07, from scramjit): `CALLER_LIMIT` is spent
  on the deepest loops first (`fillloop`'s weights; inlined blocks take the call's), and a callee no bigger than its call
  and arguments inlines without counting. llama2 (`bench/llm.py`, stories15M, mains): 90.5 tok/s, was 28 (89 with no limit).
- ~~FMA contraction slows reductions on arm64~~ **Done** (2026-11-07): `fmachain` runs on arm64 for accumulator chains
  (addend the loop's phi or another fma on it): Snapdragon X `dot` 1.47 to 1.12 ns an element, `bench/run.py` dot 273.8
  to 207.1 ms (gcc -O3 275), iir 830 to 768, mandelbrot unchanged. To confirm on Neoverse and Apple cores.
- ~~**Linking the host libm for math calls**~~ **Done** (2026-11-09, requested by scramjit): `compile(libm='host')`,
  CLI `--host-libm`; results equal CPython's `math` bit for bit (see "Math functions against CPython" in
  `requirements.md`). Noted in scramjit's `TODO.md`.

- `edit_distance` 2.5x of gcc: needs `csel` (select) in the backend.
- `rgb2gray` 1.3x of gcc: constant materialization / `madd`.
- Runtime: per-module copy of what it uses (printf ~20 KB); fine for now.

## Compile time

cimo compiles at run time, in Python, so its own speed is a feature. scramjit's first call is 0.5 to 3 s
and up to 15 s on large generated functions (its `TODO.md`, "Compile time"), and most of that is cimo.
Profile `cimo.compile` on `scratch/qbe_eval/kernels.c`, scramjit's `bench/` and its LLM forward pass
(`cProfile`, native and Pyodide) and record the split by stage here: preprocessor, front end, middle
end, each backend pass, encoder, link. Then, biggest first. Candidates: a per-process cache of parsed
bundled headers and the runtime (the prelude is the same in every scramjit compile), passes that walk
the whole function for one block, the `ext` passes that don't pay for themselves on the measured
inputs (rule from "Optimizations to lift": drop them), and the cost of `onefile` imports in Pyodide.
Target: a 100-line kernel in under 0.3 s warm natively and under 1 s in Pyodide.

- **Optimization sprint: cimo's own speed.** scramjit's first call is 2 to 8x Numba's (0.1 to 1 s against 0.04 to
  0.12 s, scramjit's `bench/compare.py`, 2026-11-07), and about 99% of it is cimo. Profile of a cold compile of
  viewshed and Gray-Scott (cProfile, so inflated, but the shares hold): backend 64%, frontend 12%, middle end 5%,
  preprocessor 3%, assembler 1%, scramjit 1%. The hottest functions are C-shaped: `rtype` and `rval` (1.6M and 1.1M
  calls, unpacking a ref from an int), the bit sets (`bselems`, `bset`), `filluse`, `filllive`, `lex`, rega's
  `hint`/`prio2`. Measure each strategy before and after, on the inputs above, and keep the output byte-identical:
  - **Cheaper Python where it's hot:** inline `rtype`/`rval` and other one-line helpers into the passes that call them
    millions of times, bit sets as Python ints (`&`, `|`, `int.bit_count`) where they're lists, no whole-function walks
    for one block, and drop `ext` passes that don't pay for themselves.
  - **Cache what repeats:** the prelude and bundled headers parsed once per process (a warm scramjit compile is mostly
    preprocessing the same prelude).
  - **Compile hot loops with scramjit when it's installed.** cimo looks for scramjit at run time, never as a dependency:
    without it, nothing changes. The hot passes (liveness, uses, bit sets, the lexer) are written within what scramjit
    compiles: integers, arrays, records with references, no lists that grow in the inner loops. scramjit compiles them
    to C, which cimo compiles and caches like any kernel, built on the user's machine and never shipped. What it needs:
    the import happens at the first compile, not when cimo is imported (scramjit imports cimo); a per-thread flag runs
    the Python passes while one of them is being compiled, which also bootstraps the first compile; cimo's output is
    byte-identical with and without it, so its cache keys don't change, checked by running the suite both ways and
    comparing the machine code (a scramjit miscompile would otherwise reach every C compile); an environment variable
    turns it off; in Pyodide the compiled passes are wasm, cached the same way. Try `filllive` and `filluse` first, as
    written: what scramjit refuses, and what they gain on a real compile, before reshaping more passes for it.

- ~~**Compile time is quadratic in a function's size**~~ **mostly done** (2026-11-05; see "Compile time on large
  functions" in `requirements.md`): scramjit's 60-loop function, every loop versioned (4,150 lines of C), 32.6 to 7.9 s;
  a generated function of 120 small loops 21.4 to 2.7 s. What's left is spread over QBE's passes, each about 3x per
  doubling of the function, none dominant: SSA construction (its liveness and phi insertion, 1.3 s of the 7.9), gvn,
  `filluse`, spill, rega. scramjit has been told it can raise `VERSION_TOTAL`.

## Experiments

- **Compile Cython's generated C.** If cimo can build a Cython module from its C, every existing
  Cython package installs from source with no compiler on the machine: no "Microsoft Visual C++ 14.0
  or greater is required," no wheel matrix, Windows arm64 included. A distribution story, not a
  performance one: typed Cython loops are plain C on unboxed data and would run near gcc's speed
  as the kernels do; the `PyObject` glue runs once per call; untyped loops are slow under any
  compiler. It does nothing for scramjit, which never boxes inside a loop. What it needs, in order:
  1. Parse the host's `Python.h` (`host_headers=True`) on Linux, macOS and Windows (MSVC-isms), and
     link `libpython` / `python3.dll` (`link=`). Measure how much of the C API's header surface cimo
     accepts; `__attribute__`, `__builtin_expect`, `inline`, `__int128` are the likely compile errors.
  2. Build one small typed module (a `cdef` loop over a memoryview) from `cython`'s `.c` output,
     map it, call `PyInit_<name>`, and import it through a `sys.meta_path` hook that replaces
     `build_ext`. Compare its results and speed with the MSVC/gcc build.
  3. Then a real package without Windows arm64 wheels. Record compile time: Cython modules are tens
     of thousands of lines, so expect tens of seconds, cached after the first install. If that's
     unusable, the experiment ends here with a number.
  Not a product claim until step 2 works on all three OSes.

## Small

- **Show the full support matrix everywhere.** The README's platform table lists OS and loader; the
  site and `platforms.md` differ from it and from each other. One source of truth, generated into
  README, `docs/docs/content/platforms.md` and the site's index: rows for Linux x86-64, Linux arm64,
  macOS Apple silicon, macOS Intel (unsupported: say so), Windows x64, Windows arm64, Pyodide / browsers
  (which browsers, main thread vs worker), Node; columns for backend, loader, what CI runs, what is run
  by hand, and the Python versions (3.9 to 3.14, PyPy unsupported). Mark FreeBSD, musl/Alpine, Android
  (Termux) and iOS (no executable memory: wasm only, untested) explicitly rather than leaving them out.

- **The compile cache is checked for integrity, not authenticity.** `.img` files in `CIMO_CACHE_DIR`
  are machine code mapped executable, protected by a CRC32 (`driver._image_get`). Anyone who can
  write to the user's cache directory can run code in the process, the same threat model as
  `site-packages`. Say so in `files-and-cache.md`, and keep the cache directory's permissions to the
  user (`0700`) when creating it.
- **The README's performance table predates the vectorizer** (scalar numbers from 2026-10-30;
  the x86-64 vectorizer landed 2026-11-02 with 1.15 to 1.20x of `gcc -O3`). Regenerate the table
  from `bench/` with the vectorizer on, on arm64 and x86-64, and quote the geomeans against `-O3`.
- **Commit subjects are paragraphs.** `git log --oneline` is unreadable. One line of at most 72
  characters, then a blank line, then the detail (add the rule to `AGENTS.md`).

- **Backend speed on huge functions:** a Csmith function with a 125 KB initializer compiles in
  18 s (spill, liveness bitsets, `filluse`); C QBE takes well under a second.
- **Middle end on functions with hundreds of loops** (YARPGen, `tests/fuzz_yarpgen.py`): a
  474 KB program took 141 s, 77 s since `memdep.Roots` is computed once per round instead of
  once per loop tried. Still quadratic: after each transformed loop, `_phase` refreshes uses
  (`filluse`), re-finds loops and recomputes `Roots` over the whole function (~480 rounds).
  Fix: keep `Roots` across transforms (temporaries it has not seen must count as unknown, and
  rewritten uses must keep their roots) and continue with the loops still valid.

- **From the adversarial read of loader, execution and faults (2026-11-05), not fixed:** an error in module code called
  through a raw `address` from a Python callback, while a kernel of that module runs on the thread, unwinds past the
  callback's frames (documented in errors.md; the entry frame could record a marker); unwinding restores neither FPCR nor
  MXCSR (a kernel that changed rounding through host libc leaves it changed); the x86 unwind doesn't `vzeroupper` after a
  fault in AVX2 code (the glue is assembled outside the compile's options); on macOS a SIGSEGV sent with kill while a
  kernel runs unwinds it (no telling it from a fault there).

- **x86-64 branch relaxation with alignment in the span** (adversarial read, 2026-11-05): `_layout` decides every
  branch from the previous pass's addresses and only grows them, so a backward `jmp` across a `.p2align 6` that a
  growing earlier branch shrinks stays long (`e9` where GNU as has `eb`): correct, 3 bytes larger, 5 of 400 random
  programs. GNU's relax_segment walks in address order within a pass.

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
