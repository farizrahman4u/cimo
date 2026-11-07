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
| `complex.h` | `float` and `double` `_Complex` arithmetic and functions (`cexp`, `csqrt`, `cabs`, ...), `I`, `CMPLX` |
| `setjmp.h` | `setjmp`, `longjmp` |
| `fenv.h` | rounding modes and exception flags: `fesetround`, `fetestexcept`, `feholdexcept`, ... |
| `ctype.h` | `isdigit`, `isalpha`, `toupper`, ... |
| `inttypes.h` | `PRId64` and friends |
| `assert.h` | `assert` |

and the freestanding headers: `stdint.h`, `stddef.h`, `stdbool.h`, `stdarg.h`, `limits.h`, `float.h`, `stdalign.h`, `stdnoreturn.h`, `iso646.h`.

`printf` writes to Python's `sys.stdout`. `sqrt` and `fabs` compile to single instructions, and `fma` too where the CPU has a fused multiply-add (arm64, x86-64 with FMA; always fused, one rounding); on arm64 so do `floor`, `ceil`, `trunc`, `round`, `fmin` and `fmax` (on x86-64 they are library calls).

GNU C extensions that work as in gcc: statement expressions (`({ ... })`), case ranges (`case 'a' ... 'z':`), range designators (`[0 ... 7] = x`), local labels (`__label__`), and the attributes `aligned`, `packed`, `mode` and the usual hints.

GNU bit builtins work as in gcc: `__builtin_popcount`, `__builtin_clz`, `__builtin_ctz`, `__builtin_bswap32` and their `l`/`ll` forms. So do the overflow builtins (`__builtin_add_overflow` and friends, `_BitInt` results included), `__builtin_expect`, `__builtin_assume_aligned`, and the `__builtin_` names of the C library functions cimo provides (`__builtin_memcpy`, `__builtin_fma`, `__builtin_powf`, `__builtin_isdigit`, ...).

## What's not available

Each of these is a compile error that names the problem:

- inline assembly (an empty `asm`, such as the barrier `asm volatile ("" ::: "memory")`, is accepted and compiles to nothing)
- `setjmp` / `longjmp` on WebAssembly (they work on the native targets)
- 128-bit `long double`
- the hosted parts of the C library: files, time, threads, sockets, `getenv`. To use them, link the host's C library; see [Calling host libraries](host-libraries.md).

GNU vector types (`__attribute__((vector_size(N)))`) work and match gcc's results. On arm64 and x86-64, vectors of 16, 32 and 64 bytes (16 to 128 on x86-64 with AVX2) keep their values in vector registers: arithmetic, bitwise operators, comparisons, negation, shifts by a constant, copies and lanes read by a constant index are vector instructions, and writing a lane or indexing by a variable keeps the vector in a register, and so are shifts by a variable and `__builtin_shufflevector` of 16-byte vectors (on x86-64 with AVX2), and `__builtin_convertvector` (widening, narrowing, and between integers and floats; on x86-64 with AVX2 not for 64-bit or unsigned 32-bit integers to or from floats).

> [!PLANNED]
> Vector instructions for the rest: integer division, 64-bit multiplication, `__builtin_shuffle` (a variable mask), wider shuffles, and on x86-64 vectors passed in vector registers as gcc does (arm64 does).

## Type sizes

| Type | Size |
|---|---|
| `char` | 1 |
| `short` | 2 |
| `int` | 4 |
| `long`, `long long`, pointers, `size_t` | 8 |
| `float` / `double` | 4 / 8 |
| `_Float16`, `__bf16` | 2 (computed in `float`) |
| `__int128`, `unsigned __int128` | 16 |
| `float _Complex` / `double _Complex` | 8 / 16 |

These are the same on every target, including WebAssembly, where cimo keeps `long` and pointers 64-bit.

A few things follow the platform's C ABI, so that kernels can call its libraries and use its headers:

| | Linux, Pyodide | macOS | Windows |
|---|---|---|---|
| plain `char` | unsigned | signed | signed |
| `wchar_t` | 4 bytes | 4 bytes | 2 bytes |
| `long double` | compile error | 8 bytes, same as `double` | 8 bytes, same as `double` |

Write `signed char` or `unsigned char` when the sign matters: `char c = 200; return c > 0;` returns 1 on Linux and 0 on macOS and Windows.

## Signed overflow

Signed integer overflow is undefined, and the compiler assumes it doesn't happen: `x + 1 > x` is always true. Pass `wrapv=True` to `cimo.compile` (or `--wrapv`, or set `CIMO_WRAPV=1`) for code that relies on wrapping, as with `-fwrapv`. `unsigned` arithmetic always wraps.

## Floating point

By default cimo contracts `a * b + c` within one expression into a fused multiply-add (on GNU vector types too), as clang does with `-ffp-contract=on`, except in a loop when the same product is computed more than once (then it's one multiply and plain adds, as gcc does). That is faster and slightly more accurate, and it means results can differ in the last bit from a compiler that doesn't contract. Set `CIMO_FMA=0` for results identical to `-ffp-contract=off`. x86-64 CPUs without FMA and WebAssembly never contract, so results there are always those of `-ffp-contract=off`.

`<fenv.h>` controls the hardware's rounding mode and exception flags. Comparisons follow IEEE 754: `==`, `!=` and `isgreater` and its family are quiet, while `<`, `<=`, `>`, `>=` and `iseqsig` raise `FE_INVALID` on a NaN. Each call from Python starts with the flags clear and in the rounding mode Python uses, and leaves Python's mode as it was, even after `fesetround` or an `abort()`. Floating-point operations on constants are folded at compile time in round-to-nearest; write `#pragma STDC FENV_ACCESS ON` before code that changes the rounding mode or reads the flags, and those operations run at run time in the program's mode. WebAssembly has only round-to-nearest and no flags: `fesetround` to another mode returns nonzero. The math library returns correctly rounded or near results in every mode, but doesn't promise exact exception flags.

`__attribute__((noinline))` and `noipa` keep a function from being inlined into its callers.
