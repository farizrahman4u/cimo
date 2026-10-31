# Errors

## Compile errors

Errors in the C raise `cimo.CompileError` with the position and a message:

```python
cimo.compile("int f(void) { return undefined_thing; }")
# cimo.CompileError: <string>:1:22: error: undeclared identifier: undefined_thing
```

`<string>` is the file name for source passed as a string; `cimo.load("kernels.c")` reports `kernels.c` instead. Messages about the parts of C that cimo doesn't bundle say what to do, for example to pass `host_headers=True` for a POSIX header.

## Errors at run time

`abort()`, `exit()`, a failed `assert` and traps don't take your Python process down. They unwind back to the caller and raise `cimo.KernelError`:

```python
k = cimo.compile(r"""
#include <assert.h>
#include <stdlib.h>

void check(int x) { assert(x > 0); }
void quit(int status) { exit(status); }
""")

k.check(-1)
# cimo.KernelError: assertion failed: x > 0, at <string>:4 in check() (calling check)

k.quit(3)
# cimo.KernelError: exit(3) called in quit() (calling quit)
```

The exception has the details as attributes:

| Attribute | Meaning |
|---|---|
| `kind` | `'abort'`, `'assert'`, `'exit'`, `'trap'`, `'stack overflow'` or `'nesting'` |
| `kernel` | the function Python called |
| `function` | the function where it happened, if known |
| `expr`, `file`, `line` | for a failed `assert` |
| `status` | for `exit()` |

The module stays usable after an error; call it again.

## Crashes

Deep recursion is caught as a `'stack overflow'` KernelError. A kernel that reads or writes outside its memory can still crash the process, as a bad pointer in a C extension would.

> [!WARNING]
> Kernels run with the full privileges of your Python process, like ctypes. There is no sandbox. In the browser, WebAssembly is the sandbox.

> [!PLANNED]
> `debug=True`: bounds checks on every access, against Python buffers and heap blocks, plus use-after-free, double-free and leak detection, reported as a `cimo.BoundsError` at the line that caused it. Turning memory faults inside a kernel into a KernelError instead of a crash.
