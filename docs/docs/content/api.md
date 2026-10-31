# Python API

```python
import cimo
```

## cimo.compile

```python
cimo.compile(src, include_dirs=(), link=(), host_headers=False, defines=())
```

Compiles C source and returns a [module](#module). Raises [`CompileError`](#compileerror).

| Parameter | |
|---|---|
| `src` | C source, as a string |
| `include_dirs` | directories searched for `#include`, before cimo's bundled headers |
| `link` | host libraries that may resolve functions cimo's C library doesn't provide: `"libc"`, `"libm"`, `"python"`, a soname, a short name or a path. See [Calling host libraries](host-libraries.md). Not in Pyodide. |
| `host_headers` | also search the host's C headers. Not in Pyodide. |
| `defines` | macros: a dict or `(name, value)` pairs; `None` means `1` |

## cimo.load

```python
cimo.load(path, include_dirs=(), link=(), host_headers=False, defines=())
```

Compiles a `.c` file. The file's directory is searched for `#include "..."`. Other parameters as for `compile`.

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

## Kernels

A module's functions. Calling one converts the arguments by the C signature:

| C parameter | Pass |
|---|---|
| integer types | `int` |
| `float`, `double` | `float` or `int` |
| pointer | a buffer (numpy array, `bytearray`, `memoryview`, ctypes array or structure, `bytes`) or an address as an `int` |

| C return type | Result |
|---|---|
| integer types, pointers | `int` |
| `float`, `double` | `float` |
| `void` | `None` |

The call releases the GIL. A function that is variadic, or passes or returns a struct or union by value, can't be called from Python: calling it raises `TypeError` (other C functions in the module can still use it).

| Attribute | |
|---|---|
| `kernel.address` | the function's address, as an `int` (native builds) |
| `kernel.function` | the WebAssembly export, callable from JavaScript (Pyodide) |
| `kernel.__name__` | the C name |

## CompileError

`cimo.CompileError(Exception)`: the source didn't compile, a header wasn't found, a symbol couldn't be linked. The message starts with `file:line:column:` when there is a position.

## KernelError

`cimo.KernelError(Exception)`: a kernel called `abort()` or `exit()`, failed an `assert`, trapped or overflowed its stack. See [Errors](errors.md).

| Attribute | |
|---|---|
| `kind` | `'abort'`, `'assert'`, `'exit'`, `'trap'`, `'stack overflow'`, `'nesting'` |
| `kernel` | the function Python called |
| `function` | where it happened, if known |
| `expr`, `file`, `line` | a failed `assert` |
| `status` | the status passed to `exit()` |
