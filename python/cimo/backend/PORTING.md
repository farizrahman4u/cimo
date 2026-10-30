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

- `gvn.assoccon`: reassociating a `w` add/sub chain folded the constants in 64 bits, so
  `x + -1 + -30` became `sub w0, w0, #18446744065119617055` (rejected by the assembler).
  The folded constant is now masked to 32 bits, as `fold.opfold` does.

## Checking

`python tests/backend_diff.py -f <flags>` (from `python/`) runs C QBE and the port on QBE's tests
and `kernels.ssa`, one `-d` flag at a time, plus `.` for the final assembly, and reports the
first differing line. A dump can only match once every earlier pass is ported.
