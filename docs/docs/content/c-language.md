# The C you can use

cimo compiles C11. Kernels can use the whole language: structs, unions, bitfields, pointers of any depth, function pointers, recursion, variadic functions, variable-length arrays, `goto`, `switch`, globals and `const` tables.

## The bundled C library

cimo ships its own C library, so a kernel behaves the same on every platform, the browser included, and doesn't depend on the host's libc. These headers are available with no setup:

| Header | Contents |
|---|---|
| `stdio.h` | `printf`, `puts`, `putchar`, `snprintf`, `sprintf` and the `v` variants |
| `stdlib.h` | `malloc`, `calloc`, `realloc`, `free`, `qsort`, `bsearch`, `strtol`, `strtod`, `atoi`, `abs`, `div`, `rand`, `abort`, `exit` |
| `string.h` | `memcpy`, `memmove`, `memset`, `memcmp`, `strlen`, `strcmp`, `strcpy`, `strchr`, `strstr`, ... |
| `math.h` | the C99 math functions, `float` and `double` versions |
| `ctype.h` | `isdigit`, `isalpha`, `toupper`, ... |
| `inttypes.h` | `PRId64` and friends |
| `assert.h` | `assert` |

and the freestanding headers: `stdint.h`, `stddef.h`, `stdbool.h`, `stdarg.h`, `limits.h`, `float.h`, `stdalign.h`, `stdnoreturn.h`, `iso646.h`.

`printf` writes to Python's `sys.stdout`. `sqrt`, `fabs`, `floor`, `ceil`, `trunc`, `round`, `fmin` and `fmax` compile to single instructions.

GNU bit builtins work as in gcc: `__builtin_popcount`, `__builtin_clz`, `__builtin_ctz`, `__builtin_bswap32` and their `l`/`ll` forms.

## What's not available

Each of these is a compile error that names the problem:

- inline assembly
- `setjmp` / `longjmp`
- 128-bit `long double`
- the hosted parts of the C library: files, time, threads, sockets, `getenv`. To use them, link the host's C library; see [Calling host libraries](host-libraries.md).

> [!PLANNED]
> GNU vector types (`__attribute__((vector_size(16)))`) for hand-written SIMD; today they are a compile error. `_Complex` and `complex.h`.

## Type sizes

| Type | Size |
|---|---|
| `char` | 1 byte, unsigned (as on Linux arm64) |
| `short` | 2 |
| `int` | 4 |
| `long`, `long long`, pointers, `size_t` | 8 |
| `float` / `double` | 4 / 8 |

These are the same on every target, including WebAssembly, where cimo keeps `long` and pointers 64-bit.

## Floating point

By default cimo contracts `a * b + c` within one expression into a fused multiply-add, as clang does with `-ffp-contract=on`. That is faster and slightly more accurate, and it means results can differ in the last bit from a compiler that doesn't contract. Set `CIMO_FMA=0` for results identical to `-ffp-contract=off`.

There is no floating-point environment: rounding is to nearest and there are no exception flags.
