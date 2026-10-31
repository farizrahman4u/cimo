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
