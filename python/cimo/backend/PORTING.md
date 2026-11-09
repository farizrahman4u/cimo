# Porting conventions for the QBE port

Source: QBE `e786f06` at `~/code/qbe` with `scratch/qbe_eval/qbe-arm64-isel.patch` applied
(the working tree there). One Python module per C file, same function names, same order,
same control flow. The goal is output identical to C QBE (debug dumps and assembly), so
port literally; don't "improve" anything.

## Data structures (`defs.py`, from `all.h`)

- **Ref** is an int `(type << 29) | val`. `R == 0`, so `req(a, b)` is `a == b`.
  `rtype(r)` (returns -1 for R), `rval(r)` is `r.val`, `rsval(r)`, constructors `TMP`, `CON`,
  `SLOT`, `TYPE`, `CALL`, `MEM`, `INT`. `r.type` (raw, without the R special case) is `rkind(r)`.
- Structs are classes with `__slots__`. **C struct copies (`*a = *b`, passing a struct by
  value, `Ins x = *i`) must be explicit**: `.copy()` or `.set(other)` (Ins, Con, Alias, Addr,
  Sym, Lnk). Forgetting this aliases objects and is the most likely bug.
- Field renames: `Tmp.def` is `def_`; `Tmp.hint.{r,w,m}` are `hint_r/hint_w/hint_m`;
  `Blk.in` is `in_`; `Alias.u.sym` is `sym`, `Alias.u.loc.{sz,m}` are `loc_sz/loc_m`;
  `Con.bits.i` is `c.i` with `c.d` / `c.s` as float views of the same bits;
  `Dat.u.*` are flat fields (`num`, `fltd`, `flts`, `str`, `ref_name`, `ref_off`).
- Arrays (`b->ins`, `fn->tmp`, `fn->con`, `fn->mem`, `b->pred`, `p->arg`, ...) are Python
  lists, with the C count fields (`nins`, `ntmp`, `ncon`, `npred`, `narg`, ...) kept as
  plain ints exactly as in C. A list may be longer than its count. `vgrow(lst, n, factory)`
  grows a list.
- A C pointer into an array becomes an index (or the element, when only the element is
  used). Pointer-to-pointer idioms (`Blk **p`, `Phi **pp`) become explicit prev/next or
  `(obj, attr)` pairs.
- Unsigned fields set to `-1u` use `U32` (0xffffffff) so unsigned comparisons still work
  (`b->id = -1u`, `b->visit = -1u`, `phiargn` result).
- Integer arithmetic that relies on C wrap-around must be wrapped: `s64()`, `s32()`, masks
  with `M64` / `U32`. Constant folding and immediate encoding need this.
- Float32 arithmetic must round to float32 after every operation: `defs.f32()`.
- **BSet** holds a Python int in `.t` (all words) and the word count in `.nt`. Use the
  `util.bs*` functions. `bsiter(bs, i)` returns the next element >= i or -1; the C loop
  `for (i=0; bsiter(bs, &i); i++)` reads the live set, so if the body modifies the set use
  `i = bsiter(bs, 0); while i >= 0: ...; i = bsiter(bs, i + 1)`. `bselems(bs)` iterates a
  snapshot.
- Globals shared between C files live in `defs`: `defs.T` (target), `defs.debug` (dict of
  flag letters), `defs.insb`, `defs.curi`, `defs.typ`. Import the module (`from . import defs
  as D`) and write `D.curi`, never `from .defs import curi`.
- The instruction buffer `D.insb` is a fixed list of `NIns` slots and `D.curi` an int index,
  so C pointer arithmetic on `curi` maps directly: `&insb[NIns] - curi` is `NIns - D.curi`;
  `curi->arg[0]` is `D.insb[D.curi].arg[0]`; `emit()` writes `D.insb[D.curi - 1]`.
  `idup(b, s, n)` copies `insb[s:s+n]` into `b.ins`; `icpy(dst, di, src, si, n)` copies
  between lists.
- Functions with out-parameters return tuples instead: `iscmp(op) -> (k, c) | None`,
  `clsmerge(k1, k) -> (newk1, result)`, `isconbits(fn, r) -> bits | None`,
  `igroup(b, i) -> (i0, i1)` (indices), `cfg.ifgraph(b) -> (then, else, join) | None`.
  Document any new ones in the docstring.
- `err(fmt, *args)` and `die(fmt, *args)` (in `util`) raise `defs.BackendError`.
- Debug dumps go to `sys.stderr` with the same text as C (`printfn` is in `parse`).
- Recursion that can go deep on large inputs (CFG walks) can be made iterative, but must
  visit in the same order.

## Fixes over C QBE

Bugs of the reference are fixed in the port, marked `fix over C QBE` in the code (C QBE's
output for these is broken, so no comparison test depends on it):

- Float comparisons (`arm64.isel.selcmp`, `amd64.isel.selcmp`): arm64 compared with `fcmpe` and x86-64 with
  `ucomis` whatever the relation; IEEE 754 (C Annex F) has `==`, `!=` and unordered quiet and `<`, `<=`, `>`,
  `>=` raise invalid on a quiet NaN. Now `fcmp`/`ucomis` for the first and `fcmpe`/`comis` (new ops `afcmpq`,
  `xcmpe`) for the second; `ifopt` doesn't run a signaling comparison on the path that doesn't ask for it
  (`!isunordered(x, y) && x > y`). Found by gcc's `inf-compare-*.c`.

- `amd64.isel` (`seljmp`, `selsel`): a `jnz` (or select) on a value whose definition set the flags reused
  them even when that instruction was 64-bit: `jnz` tests the low 32 bits, so a w lane loaded (and forwarded)
  from the l `or` that stored two lanes was tested whole (gcc pr94412). Only w instructions' flags are reused.
- `amd64` ABIs: a struct passed or returned by value of `ext.bigcopy` bytes or more is copied by a call to
  `memcpy` (as on arm64), not an unrolled blit: 128 KB by value, 8 calls, was 9.6 MB of moves (pr23135).
- `fold.opfold`: a float operation on a symbol's address (`swtof $P`, after a load forwarded the
  stored address: gcc.dg/torture pr65077) stopped compilation with "invalid address operand"; it
  is left unfolded, as an integer operation on one is.
- `gvn.assoccon`: reassociating a `w` add/sub chain folded the constants in 64 bits, so
  `x + -1 + -30` became `sub w0, w0, #18446744065119617055` (rejected by the assembler).
  The folded constant is now masked to 32 bits, as `fold.opfold` does.
- `arm64.isel.fixarg`: an extern symbol's address plus an offset added the offset as an
  immediate whatever its value (`a[i - 2000000000]`: an assertion in emit). Offsets without an
  `add` encoding go through a register.
- `mem.promote`: a load from a slot that is never stored was an error ("slot is read but never
  stored to"), so C reading an uninitialized local did not compile. It reads 0.
- `mem.coalesce`: a never-stored slot passed by value (`argc`, an uninitialized struct
  argument) was killed and address 0 passed (a segfault). Such slots are kept.
- `rega.pmadd`: a parallel move held at most 64 copies ("no more pm slots"); a loop keeping
  more values in registers needs more on an edge. The list grows.
- `arm64.emit` (`copy` to a slot from a slot): the value went through x17 whatever its class,
  so a float was "invalid class"; floats go through v31. Reached once a large parallel move
  copies spilled values. Both found by `tests/test_backend_edges.py` (64 live values).
- `gvn`: `jnz` on a constant was decided on all 64 bits, but `jnz` tests a word (isel compares
  the w register): `x && (int)0x8000000000000LL` (cproc leaves narrowing a constant to the
  word use) took the branch. Only the low 32 bits count.
- `parse.typecheck`: C indexes `fn->tmp` with any argument's value (a constant's too) and reads
  it only in an error message; the port indexed the list first, an `IndexError` once a function
  had more constants than temporaries. (Not a C bug; a porting one.) All 65 `tmp[rval(r)]`
  sites were audited: the others check `rtype(r) == RTmp` first or index a phi/instruction
  destination. Guarded besides: `gvn.assoccon` (C passes `&fn->tmp[val]` of any ref to
  `adduse`, which would corrupt a tmp's uses if the operand were a constant), `util.salloc`,
  the phi operand error message, and `middle.ir.defblock`.
- `util.newcon`/`getcon`: QBE searches the constants linearly, quadratic in their number; the
  port keeps an index (`Fn.concache`), same results (constants are only appended). A 125 KB
  local initializer took 223 s, now 18 s.
- Quadratic in a long chain of ifs (gcc's `gcc.c-torture/compile/20001226-1.c`, 8192 compare-and-branch pairs in
  one function, over 120 s): `cfg.fillfron` asked `sdom` (a walk) at every step of each walk up the dominator
  tree; it now walks to `idom(s)` and stops at a runner that already has `s` in its frontier (same frontiers, same
  order). `gvn.zeroval` looks at 128 dominating `jnz` at most (unknown, past that), and `load.def` gives up
  (keeps the load) after 20,000 instructions and blocks for one load. Now 99 s; the rest is linear-ish.
- `cse.reassoc` (`_adddef`) took `x - $sym` for a subtraction of an integer constant and crashed
  (`gcc.c-torture/compile/pr34029-2.c`).
- `load.load`: `rebase` moves a load's constant offset into `Slice.off`, which the address of
  an inserted load ignored (as it did a blit source's offset), so a load of `p+24` hoisted out
  of a loop read `p`. Reachable once
  `ext.cse` runs load elimination after gvn (gcc's `20011008-3.c` on every target; on arm64 it
  only made a loop run with a garbage bound). No QBE test exercises it: the dumps still match.
- `load.def`: `vaarg`/`vastart` (lowered only by `abi1`, after load elimination) were not seen
  as writing their `va_list`, so a store was forwarded across them: with Apple's and Windows'
  `char *` `va_list`, a parameter passed on after `va_arg` was passed unchanged.

- `arm64/abi.typclass`/`argsclass`: a 16-byte aligned composite was refused ("alignments
  larger than 8"); now it starts at an even register (the skipped one is set to 0 before the
  call, so liveness holds, and is not a parameter register of the callee) or a 16-byte
  aligned stack offset, while a pointer replacing a large one keeps 8-byte alignment. A
  zero-size composite (GNU C) was passed by pointer; gcc passes nothing. Found with
  `tests/test_abi.py` (structs with `_Alignas(16)`) and gcc's `va-arg-22.c`.
- Recursion: the passes recurse as deep as the program is long (gcm's `schedearly`/`earlyins`
  along a dependency chain, `cfg.reachrec` along a chain of blocks), so a 3000-statement
  chain overflowed Python's recursion limit. `cimo.driver` runs the backend under the
  frontend's raised limit (1,000,000 from Python 3.11, whose Python calls do not use the C
  stack; 20,000 before), and `reachrec` is iterative, as `porec` already was. Found by
  YARPGen.
- `arm64/emit` (Apple in C QBE; Windows in cimo, both without x18): a store to a stack slot
  beyond the immediate offset range, whose value is in x17, goes through s31/d31; the width
  came from `i.cls`, which is `Kw` for the `storel` of a spill, so 4 of the 8 bytes were
  stored. Found by gcc's `pr28982b.c` on Windows (a big frame and many live pointers).
- `arm64/abi.argsclass` (Windows variadic call): a composite that does not fit in the
  remaining registers is split between them and the stack (x7 is saved right below the
  stack arguments, so the callee sees it whole), as clang's and cimo's `va_arg` read it; it
  was an error. A struct parameter split that way in a variadic function is still an error.
- `arm64/abi.argsclass` (Apple, as C QBE's `arm64_apple` had it): a struct argument after the
  `...` of a call went in registers; Apple passes every variadic argument on the stack. And a
  16-byte aligned composite skipped to an even register pair (the AAPCS64 fix above), which
  Apple does not do. Found by `tests/test_abi.py` against Apple clang and `va_arg` of structs.

## amd64 (`amd64/`, 2026-11-01)

Ported from the same QBE `e786f06` (`amd64/{targ,sysv,winabi,isel,emit}.c`; `runmatch` from
`util.c` is in `amd64/isel.py`, the X(nmem, zflag, lflag) column of `ops.h` in
`amd64/targ.py`). Every `-d` dump and the assembly match C QBE on its 76 tests for
`amd64_sysv` and `amd64_apple`; for `amd64_win` too, except the fix below (and `tls.ssa`, which
both refuse, C QBE losing its buffered output when it aborts). Conventions as above; `Num.nl`
and `Num.nr` are `uchar` in C, so `refn`'s INT_MIN is stored as 0 (`& 0xff`). The diff needs a
C QBE built with glibc (WSL); the arm64 targets need it built with the arm64 patch, the amd64
ones are the same either way.

- `winabi.lower_func_parameters` (fix over C QBE): a float parameter passed on the stack (the
  fifth or later) was a `Kl` copy into the float temporary (`%d2 =l copy S-12`), so it could
  land in an integer register (`addss %ecx, %xmm0`); it is now a load of its own class, as in
  `sysv.c`. Changes the `amd64_win` dumps of QBE's `abi6`, `abi8` and `vararg2`.
- `winabi.lower_call` (fix over C QBE): a small struct passed on the stack advanced the
  argument area by its size (1 to 8), not an 8-byte slot, so the arguments after it were
  misplaced. Found by `tests/test_abi.py` against MinGW gcc.
- `sysv` (fix over C QBE): an eightbyte with no field (`Kx`: the second half of
  `struct { _Alignas(16) char c; }`) took a register of an undefined class (a negative shift in
  `retr`, a class assertion in `rega`); as gcc (NO_CLASS), it takes none (`sysv._second`).
  Found by `tests/test_abi.py` once the frontend stopped writing such padding as `b` fields.
- `isel.seladdr` (fix over C QBE): an address whose constant does not fit a signed 32-bit
  displacement stays an `add` (gcc's `pr94734.c`: `arr[0xdeadbeefbeefdead]` once inlined).
- `isel.flagi` (fix over C QBE): a branch or select reused the zero flag of a shift, but x86
  leaves the flags alone when the count (masked to 5 or 6 bits) is 0, so `sarl %cl, %edx; jnz`
  tested the instruction before. Only a shift by a constant count that is not 0 is reused now
  (`_shiftsflags`). Reachable once `ext.cse`'s simplifier dropped an `& 1` after the shift
  (gcc's `builtin-bitops-1.c`).
- `isel.sel` (fix over C QBE): `swtof`/`sltof`/`cast` of an integer constant to a float has no
  immediate form (`cvtsi2sd $0, %xmm1`); the constant goes to a register first. C QBE's fold
  never leaves such a conversion, `ext.cse`'s passes can.

cimo's own Windows x64 target is `amd64_windows` (`targ.T_amd64_windows`), not QBE's
`amd64_win`, which stays as in C for the diffs: xmm6-xmm15 are callee-saved (QBE uses them as
scratch, which corrupts a Windows caller's registers: Python's), saved by the prologue below the
pushes (`E.xsave`, `E.xarea`; xmm15 when the emitter uses it as scratch); frames and allocas over
a page are probed (`emit._probe`) through r11, which is in `rglob` (never allocated); extern
data goes through the GOT; ELF syntax with `.L` labels.

## cimo extensions (`ext.py`, off by default)

- `volatile`: `vloadsb`...`vload`, `vstoreb`...`vstored` (and `vloadw`/`vloadl`/`vloads`/
  `vloadd`), pinned and unknown to `mem`, `load`, `gvn` and `gcm`, so nothing deletes, merges,
  moves, promotes or forwards them; `load.def` stops at a volatile store that may alias.
  `defs.unvolatile` makes them plain before isel; an unused volatile load's result goes to x17
  (IP1, in `rglob`, never treated as dead), for which `spill` skips the "dead reg" check.
- `vla`: `getsp` / `setsp` read and reset the stack pointer (arm64 `mov`, wasm `__stack_pointer`),
  pinned, mark the function `dynalloc`, and are scheduling barriers for `ldp` pairing.
- `math` also covers the bit builtins: `bclz`, `bctz`, `bpopcnt` (through v31: `fmov`,
  `cnt`, `addv`), `bswap`, `bswap16`; and the floating-point environment: `%r =w rdfpcr`, `rdfpsr`,
  `wrfpcr %v`, `wrfpsr %v` (arm64 `mrs`/`msr` FPCR and FPSR; x86-64 `stmxcsr`/`ldmxcsr` through the 8 bytes
  below rsp, moved with `lea` to keep the flags; wasm: 0 and nothing), pinned.
- `#pragma STDC FENV_ACCESS ON` (the frontend's `paraminfo['#fenv']`, through `middle.params`): in those
  functions `fold` only folds floating-point operations whose result is exact and raises nothing
  (`fold._fenvexact`), and `gcm.pinned` holds their floating-point operations, so `gvn` doesn't merge them and
  `gcm` doesn't move them across the calls that change the rounding mode. `paraminfo['#noinline']`: functions
  declared `noinline` or `noipa`, which `inline` neither copies nor gives constant parameters.
- `bigcopy`: a struct argument or return value passed by reference and at least that large is
  copied by a call to `memcpy`, emitted already lowered (before any argument register is set),
  instead of an unrolled blit.
- `fma`, `math`, `ldp`, `layout`, `trap`, `stkchk`: see `ext.py`.

## Checking

`python tests/backend_diff.py -f <flags>` (from `python/`) runs C QBE and the port on QBE's tests
and `kernels.ssa`, one `-d` flag at a time, plus `.` for the final assembly, and reports the
first differing line. A dump can only match once every earlier pass is ported.
