# Porting conventions for the cproc port

Source: cproc `d1c53dd` at `~/code/cproc`. Everything that goes into `cproc-qbe` is ported,
one Python module per C file (`scan`, `token`, `pp`, `decl`, `expr`, `stmt`, `init`, `eval`,
`type`, `scope`, `map`, `tree`, `attr`, `utf`, `util`, `targ`, `qbe`, `main`), same function
names, same order, same control flow. `driver.c` is not ported. The goal is output identical
to C `cproc-qbe` (IR, `-E` output, error text and exit status), so port literally and don't
"improve" anything; cproc's bugs are kept (see "Quirks kept" below).

Same spirit as `python/cimo/backend/PORTING.md`.

## Data structures (`cc.py`, from `cc.h`, `tokens.h`, `util.h`)

- Token kinds, type kinds, etc. are module-level ints with the C names (`TIDENT`, `TYPEINT`,
  `EXPRCONST`, ...). Import them with `from .cc import ...`.
- Structs are classes with `__slots__`, fields keep their C names. Fields C leaves
  uninitialized default to 0/None.
- C unions are flattened into separate attributes, named in the class docstring:
  `Expr` (`decl`, `init`, `u`, `ssize`/`sdata`, `args`/`nargs`, `bits`, `post`, `l`/`r`,
  `ct`/`cf` for `cond.t`/`cond.f`, `bkind`, `sztype`, `temp`), `Type` (arith, array with
  `u.array.size` as `asize`, func, structunion), `Decl` (`align`/`storage`,
  `inlinedefn`/`isnoreturn`, `enumconst`, `builtin`). Nothing in cproc reads a union member
  other than the one last written, except the constant views below.
- `expr->u.constant` is one unsigned 64-bit int `e.u`; `e.i` (signed) and `e.f` (double, via
  `struct`) are properties over the same bits, so `u.constant.u` of a float constant reads
  its bit pattern as in C (e.g. `-0.0 || x`).
- Small structs copied by value are immutable tuples: `Location` (file, line, col) and
  `Bitfield` (before, after). Assign a new one instead of mutating. `Token.set()` /
  `.copy()` are explicit struct copies (`tok = *t`).
- The current token `token.tok` is a single object updated in place (C writes into `&tok`
  directly in several places). Always use `tok.kind`, never cache it across `next()`.
- Out-parameters (`enum typequal *`, `char **name`, `int *align`, `struct scope **`,
  `unsigned long long *offset`) are `cc.Ref` objects with a `.v` field, or None for NULL.
  `commonreal()` returns `(t, e1, e2)`; `decodechar()` returns `(chr, hexoct, length)`.
- `struct list` is `cc.ListNode` with an `owner` (replaces `listelement`).
- Pointer-to-next-pointer list building (`**end`, `**last`, `tentativedefnsend`, `paramend`,
  `b->last`, `p->last`) is "the last node, or None for the list head".
- Maps (`map.c`) are dicts: they are only used for lookups. `stringdecl` keys on
  `bytes(data[:size])`, i.e. the first `size` *bytes* as C does (see quirks).
- String literal data (`stringlit.data`) is a bytearray of the raw element bytes
  (little-endian for wide strings); `ssize` counts elements.
- The init parser's `obj[32]` is a list of `Object` with `cur`/`sub` as indices (`cur` None
  for NULL). `Object.u` is one attribute for the union `{mem, idx}`.
- pp: token arrays are lists of Token objects; a context frame is `[toks, index, ntoken,
  macro]`; `MacroArg` holds `(toks, start, ntoken, str)`. Token objects are shared exactly as
  the C pointers are (e.g. `ctxpush` sets `space` on the first replacement token).
- `tree.c` slots (`void **`) are `(container, index)` pairs.
- C globals are module state. Mutable ones that later modules read are accessed through the
  module (`scope.filescope`, `targ.targ`, `type.typeadjvalist`, `pp.ppflags`), never with
  `from x import name`. `main._reset()` resets all per-compile state, so `compile()` can be
  called repeatedly; `targinit()` rebuilds the target's va_list types (they get `.value` from
  `emittype`).

## Integers, floats, text

- Wrap explicitly: `& M64` for `unsigned long long`, `& M32` for `unsigned`. "-1" in an
  unsigned parameter is `M32` (`bitfieldwidth`) or `M64` (`addmember` width).
- Constant folding (`eval.py`) follows aarch64, where the reference C build runs: integer
  division by zero gives 0, remainder by zero gives the dividend, `INT64_MIN / -1` gives
  `INT64_MIN`, float-to-int of NaN gives 0. Float division by zero gives inf/nan as in C.
  `(float)x` rounds through `struct` (`eval.f32`, inf on overflow).
- `strtod`/`strtoull` are emulated for the strings that can reach them (pp-numbers);
  `%g`/`%.17g` print `nan`/`-nan`/`inf` as glibc does.
- Input is bytes decoded as latin-1, so one char is one byte (`isalnum` etc. are ASCII
  only, as in the C locale) and UTF-8 is handled byte-wise as in C. EOF is `''`. When C reads
  the NUL terminator past the end of a token, the port does too (`'\0'`). Output and
  messages are re-encoded latin-1; `compile()` returns str (UTF-8 decode, surrogateescape).
- stdout is the list `util.out`; `error()`/`fatal()` raise `FrontendError` (message = cproc's
  stderr line without the newline, `status` = 1). `#warning` writes to stderr directly.

## Recursion

Recursive functions stay recursive (expressions, declarators, `emittype`, `casesearch`);
`compile()`/`main()` raise the recursion limit to 20000 while running.

## Deviations (intentional)

- Default target is `aarch64` (C `cproc-qbe` defaults to `x86_64-sysv`; the C driver always
  passes `-t`). All of cproc's targets are kept: `x86_64-sysv`, `aarch64`, `riscv64`.
- Extra target `aarch64-apple` (cproc has none; its driver maps `aarch64-*` to `aarch64`):
  `va_list` is `char *`, plain `char` is signed, `wchar_t` is `int`, `long double` is 8
  bytes. Pair it with QBE's `arm64_apple`.
- Extra target `aarch64-windows`: as `aarch64-apple`, but `wchar_t` is `unsigned short`. LP64
  (Windows' own C is LLP64; cimo keeps `long` 64-bit everywhere). Pair it with the backend's
  `arm64_win`.
- Line markers: the file name's `\\` and `\"` escapes are decoded (cproc has an "XXX" and
  takes the text up to the next quote), so Windows paths in diagnostics are right.
- Bugs fixed (marked `cimo:` in the code; found by gcc's torture tests): a block-scope `extern`
  that hides a local took the local's absent linkage and crashed (C11 6.2.2p4); the size of a
  variably modified typedef was computed at its first use, which need not dominate the others
  (now at the typedef, C11 6.7.8p3); a parameter's array length with side effects
  (`int a[i++]`) was dropped (now evaluated on entry, C11 6.9.1p10); a local of size 0 (GNU
  zero-length array) was taken for a VLA (an assertion). Found by Csmith: the common type of
  `unsigned long` and `long long` was `long long` (C11 6.3.1.8: `unsigned long long`), so
  `uint64_t` against a `LL` constant compared signed; constant `||`/`&&` folded to an operand
  (`5L || x` was the `long` 5, `1 && 7` was 7) instead of the `int` 0 or 1; `?:` with two
  operands of the same arithmetic type skipped the usual arithmetic conversions (`c ? ch : ch`
  was a `char`, a bit-field was not promoted), and a bit-field as a comma operator's last
  operand did not promote as one (`(0, s.f2) < -8` was unsigned; gcc and clang: signed).
  Found by the semantics model (`tests/test_semantics.py`): `++`/`--` on a `_Bool` stored
  `x + 1`/`x - 1` unconverted (2, 255); constant conversion to `_Bool` kept the low bit of an
  integer (`_Bool b = 2;` was 0; C cproc even stores `_Bool b = 3;` as 3) and truncated a
  float (`(_Bool)0.5` was 0, `(_Bool)-0.5` an error); `(unsigned)-0.5` was rejected (its
  integer part, 0, is representable); folding `1 ? b.f : 0u` returned the bit-field itself,
  which then promoted to `int` again.
- Found by `tests/test_constraints.py` (C11 constraint violations): a duplicate label linked
  its block in twice, a cycle that made `emitfunc` produce output without end (it exhausted the
  machine's memory); a struct as `?:`'s condition, `int a = {1, 2}` and a function defined
  with a typedef'd type tripped assertions; `++` on a struct, `&`/`|`/`^` on floating
  operands and a typedef name used as a value reached the backend; an initializer for a VLA
  of complete element type was accepted (C23's `= {}` is not supported either).
- `compile(diagnostics=True)` also diagnoses the C11 constraints cproc leaves out (a `cimo:`
  comment names the clause): a declaration declaring nothing, `inline`/`_Noreturn` on an
  object or typedef, `_Alignas` with `register`, duplicate members, enumerators and parameter
  names, a flexible array member as the only member, `void` with other parameters, `static`
  in a non-parameter array declarator, incomplete parameter types in a definition, `&` of a
  `register` object, casts between pointers and floating types, assignment to a struct with a
  `const` member, a string longer than its array, an integer constant too large for every
  type, non-`auto` objects in a `for` declaration, a `goto` to an undefined label, and (with
  `vlafree`) jumps into the scope of a VLA by `goto` or a `case` label. The preprocessor warns
  about `__VA_ARGS__` in a non-variadic macro and `#undef` of a predefined macro, as gcc does.
- `emittype` (cproc's XXX): QBE types get `align N` and `b N` padding where QBE's layout of
  the fields differs from C's (`_Alignas`, trailing padding of an over-aligned struct), and
  no field for a flexible or zero-length array (`b 0` for an empty struct); cproc's types
  were too small, so calls copied part of the struct. `zero()` indexed its store table with 16
  for a 16-byte aligned local (crash). Found with `va_arg` of structs and `tests/test_abi.py`.
- Target `aarch64-apple`: `char` and `short` parameters and named call arguments use QBE's
  sub-word ABI classes (`sb`, `ub`, `sh`, `uh`; cproc emits `w`), because Apple packs stack
  arguments at their own size and the backend can only tell from the class (`abiclass`).
- Accepted as gcc and clang do (the macOS SDK's headers have both): a stray `;` at file scope,
  and `__attribute__((aligned(N)))` after a struct or union body, which raises its alignment.
- Target `aarch64-windows`: a call to a variadic function gets the `...` marker even when no
  variadic argument follows (cproc emits it before the first one), since Windows passes all
  arguments of a variadic call its own way. Found by `tests/test_abi.py`.
- `compile(diagnostics=True)` accepts pointers whose targets differ only in integer signedness
  or in qualifiers below the top level (assignment, `==`, `!=`), as gcc does without a warning;
  cproc calls them incompatible (`diag.ptrsimilar`).
- GNU attributes are accepted among a pointer declarator's qualifiers (`int *__attribute__((x))
  f(void)`). The math builtins are recognized by symbol name, so the prelude's
  `__builtin_sqrt` (declared as an alias of `sqrt`) is inlined too.
- Where C cproc trips an `assert`, the CLI prints a similar message (naming the Python
  source) and `abort()`s, losing stdout as C does. Where C would crash (NULL dereference,
  e.g. `extern int x;` redeclared in a nested block), the port raises a Python exception;
  the CLI reports `internal error` and exits with status 70. `compile()` lets both propagate.
- `delexpr`, `macrodel`, `delfunc` free nothing (garbage collection); nothing reads freed
  objects.

## cimo extensions (off by default, so the CLI stays cproc's)

Switched on by `compile()` keyword arguments (module globals reset after each call), all
passed by `cimo.compile`:

- `fma=True` (`qbe.contract`): `a*b+c` within one expression becomes the `fma` instruction.
- `builtins=True` (`qbe.builtins`): `sqrt`, `fabs`, `fmin`, `fmax`, `floor`, `ceil`, `trunc`,
  `round` (and `f` versions), `fma` become single instructions.
- `qbe.paraminfo = {}`: `emitfunc` records parameter temporaries and `restrict` flags.
- `diagnostics=True` (`util.diagnostics`, code in `diag.py`, hooked into the ported modules
  with `if U.diagnostics:`): errors with a location for unsupported C (unknown attributes,
  `vector_size`, `long double` and `__int128` at use or definition, thread-local storage,
  GNU extensions, unknown type names); GNU keyword spellings (`asm`, `__restrict`, `__const`,
  `__alignof`, `__PRETTY_FUNCTION__`); keywords as attribute names; `aligned` after any
  declarator; `nullptr` as `(void *)0`; C99 inline definitions emitted as local functions
  when referenced (or exported when a later declaration makes them external).
- `volatile=True` (`qbe.volatile`): accesses to volatile objects (loads, stores, bit-field
  read-modify-writes, aggregate copies from or to them, initializers) use the backend's
  `vload*`/`vstore*`; cproc rejects volatile stores and emits volatile loads as plain ones.
- `bitops=True` (`qbe.bitops`): `__builtin_clz`/`ctz`/`popcount` (and `l`, `ll`) and
  `bswap16/32/64`, as declared by the prelude, become the backend's bit instructions.
- `bigcopy=N` (`qbe.bigcopy`): aggregate copies and zero-fills of N bytes or more call
  `memcpy`/`memset` (not for volatile objects) instead of a load/store pair per 8 bytes, whose
  backend time grows faster than linearly (a 256 KB `= {}` took a minute).
- `vastruct='aapcs'|'slots'` (`qbe.vastruct`): `va_arg` of a struct or union, lowered to
  loads and stores of the `va_list` (`funcvaargstruct`), following gcc's `va_arg`: an HFA from
  the FP register save area (members 16 bytes apart, gathered into a copy), other composites
  over 16 bytes through a pointer, else from the general register save area, never split
  with the stack; `slots` (Apple, Windows, wasm) walks 8-byte slots, and Windows treats HFAs
  as other composites. cproc rejects it (todo.sr.ht/~mcf/cproc/52).
- `vlafree=True` (`qbe.vlafree`): `getsp` before each VLA allocation and `setsp` when control
  leaves its scope: end of block or `for`, `break`/`continue` (`Scope.breakvla`/`contvla`), and
  `goto` (backward: the label's VLAs are a prefix; forward: a `goto_vla` block filled in by
  `emitfunc` once the label is known). cproc frees VLAs only on return, so one in a loop grows
  the stack each iteration.

## Quirks kept (match C)

Examples, all reproduced on purpose: `enum E : unsigned char { A }` errors ("no unsigned
integer type can represent enumerator value"); `(_Bool)256` folds to 0 via an 8-bit mask but
`_Bool b = 2` stores 2 in data; char data is printed as the 64-bit value (`b
18446744073709551615`); multiple initialized union members assert; wide string literals are
deduplicated on their first `size` bytes; the `__VA_ARGS__`/nested same-macro argument
handling of cproc's pp; `nextinto()` tests the global `tok` for "start of line".

## Checking

`python tests/test_frontend.py [-v] [files...]` (from `python/`; also runs under pytest, skipped
without C cproc) runs C `cproc-qbe` and the port on cproc's test suite (as `runtests` does),
on `scratch/qbe_eval/*.c` and `tests/frontend/*.c` preprocessed with `cproc -E` for every target,
and on the raw snippets in `tests/frontend/snippets.txt` (also with `-E`), comparing stdout,
stderr and exit status byte for byte.
