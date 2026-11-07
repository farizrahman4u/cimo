# Python API

```python
import cimo
```

## cimo.compile

```python
cimo.compile(src, include_dirs=(), link=(), host_headers=False, defines=(), cache=True, fma=None)
```

Compiles C source and returns a [module](#module). Raises [`CompileError`](#compileerror).

| Parameter | |
|---|---|
| `src` | C source, as a string |
| `include_dirs` | directories searched for `#include`, before cimo's bundled headers |
| `link` | host libraries that may resolve functions cimo's C library doesn't provide: `"libc"`, `"libm"`, `"python"`, a soname, a short name or a path. See [Calling host libraries](host-libraries.md). Not in Pyodide. |
| `host_headers` | also search the host's C headers. Not in Pyodide. |
| `defines` | macros: a dict or `(name, value)` pairs; `None` means `1` |
| `cache` | `False`: read the compile cache but write nothing to disk |
| `fma` | `False`: no fused multiply-add contraction, for results identical to `-ffp-contract=off`. `True`: contract `a*b+c` within one expression. `None` (default): on, unless `CIMO_FMA=0`. `#pragma STDC FP_CONTRACT ON`, `OFF` or `DEFAULT` in the source overrides it to the end of the block, or to the next pragma at file scope. wasm, and x86-64 CPUs without FMA, never contract. |

## cimo.load

```python
cimo.load(path, include_dirs=(), link=(), host_headers=False, defines=(), cache=True, fma=None)
```

Compiles a `.c` file. The file's directory is searched for `#include "..."`. Other parameters as for `compile`.

## cimo.add_cache_entry

```python
cimo.add_cache_entry(data)
```

Puts a module's cache entry (`module.cache_entry()`) into the cache, on disk if it is writable, else in memory. A later `compile()` of the same source and options loads it instead of compiling. Raises `ValueError` for bytes that aren't an intact entry for this platform.

## Module

What `compile` and `load` return. Every function in the source that isn't `static` is an attribute.

| | |
|---|---|
| `module.name(...)` | call the C function `name` |
| `module(...)` | call `main`, or the only function if there is just one |
| `dir(module)` | the functions callable from Python |
| `module.free(ptr)` | `free()` a pointer a kernel returned from `malloc` and friends |
| `module.reset_heap()` | free every block of the module's heap at once; no kernel of the module may be running |
| `module.heap_bytes()` | bytes currently mapped for the module's heap |
| `module.cache_entry()` | the module's compile cache entry as bytes, for `cimo.add_cache_entry()` |
| `module.cached` | `True` if the module came from the compile cache with nothing compiled, `False` if this call compiled it |
| `module.source` | the C text it was compiled from |
| `module.options` | the arguments of `compile()` or `load()` that weren't the defaults |
| `module.signature(name)` | a function's C types: `('double', ('const double *', 'int64_t'))`. Integers by size, a VLA as a pointer to its elements |
| `module.mutable_data` | names of data that isn't `const`, static locals included: what compiling the source again wouldn't share with this module |

## Kernels

A module's functions. Calling one converts the arguments by the C signature:

| C parameter | Pass |
|---|---|
| integer types | `int`, or a numpy integer |
| `float`, `double` | `float` or `int` |
| pointer | a buffer (numpy array, `bytearray`, `memoryview`, ctypes array or structure, `bytes`) or an address as an `int` |

| C return type | Result |
|---|---|
| integer types, pointers | `int` |
| `float`, `double` | `float` |
| `void` | `None` |

The call releases the GIL and costs about 0.2 to 0.6 µs from Python (on Sapphire Rapids, 0.4 to 1 µs on Graviton 3), a little more with a buffer to convert. A function that is variadic, or passes or returns a struct or union by value, can't be called from Python: calling it raises `TypeError` (other C functions in the module can still use it).

| Attribute | |
|---|---|
| `kernel.address` | the function's address, as an `int` (native builds); a kernel of another module can call it, and its errors raise from that call |
| `kernel.function` | the WebAssembly export, callable from JavaScript (Pyodide): a `long` argument is a BigInt, a pointer a Number (an i32 address) |
| `kernel.__name__` | the C name |
| `kernel.module` | its module |

## CompileError

`cimo.CompileError(Exception)`: the source didn't compile, a header wasn't found, a symbol couldn't be linked. The message starts with `file:line:column:` when there is a position.

## KernelError

`cimo.KernelError(Exception)`: a kernel called `abort()` or `exit()`, failed an `assert`, trapped, overflowed its stack or faulted. See [Errors](errors.md).

| Attribute | |
|---|---|
| `kind` | `'abort'`, `'assert'`, `'exit'`, `'trap'`, `'stack overflow'`, `'nesting'` |
| `kernel` | the function Python called |
| `function` | where it happened, if known |
| `expr`, `file`, `line` | a failed `assert` |
| `status` | the status passed to `exit()` |
