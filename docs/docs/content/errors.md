# Errors

## Compile errors

Errors in the C raise `cimo.CompileError` with the position and a message:

```python
cimo.compile("int f(void) { return undefined_thing; }")
# cimo.CompileError: <string>:1:22: error: undeclared identifier: undefined_thing
```

`<string>` is the file name for source passed as a string; `cimo.load("kernels.c")` reports `kernels.c` instead. Messages about the parts of C that cimo doesn't bundle say what to do, for example to pass `host_headers=True` for a POSIX header.

## Errors at run time

`abort()`, `exit()`, a failed `assert`, traps, stack overflow and hardware faults don't take your Python process down. They unwind back to the caller and raise `cimo.KernelError`:

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
| `kind` | `'abort'`, `'assert'`, `'exit'`, `'trap'`, `'stack overflow'`, `'nesting'`, `'memory fault'`, `'division fault'` or `'illegal instruction'` |
| `kernel` | the function Python called |
| `function` | the function where it happened, if known |
| `expr`, `file`, `line` | for a failed `assert` |
| `status` | for `exit()` |
| `address` | for a memory fault, the address |

The module stays usable after an error; call it again.

## Faults

A bad pointer, a call through one, and running off the end of the stack raise `KernelError`, on any thread:

```python
k = cimo.compile("""
int read(int *p) { return *p; }
int bad_call(int (*f)(void)) { return f(); }
int dv(int a, int b) { return a / b; }
""")

k.read(8)
# cimo.KernelError: memory fault at 0x8 in read() (calling read)

k.bad_call(0)
# cimo.KernelError: memory fault at 0x0 in bad_call() (calling bad_call)

k.dv(1, 0)
# cimo.KernelError: integer division by zero or overflow in dv() (calling dv)
```

`kind` is `'memory fault'`, `'stack overflow'`, `'division fault'` (x86-64 only: arm64's division doesn't trap, and `a / 0` gives 0) or `'illegal instruction'`.

The module stays usable, but look at what the error means:

- The kernel stopped at the faulting instruction, so anything it was writing may be half-written, including its heap if the fault was in `malloc` or `free` called with a bad pointer. If in doubt, `reset_heap()` or compile the module again.
- Only an access to memory the process doesn't have faults. A write past the end of an array that lands on other memory of the process corrupts it, with no error, and can show up later as a crash anywhere.
- A fault inside a host library called from a kernel (`link=`) or inside Python ends the process: unwinding past that code could leave its locks held and its state half-updated. So does a fault in a function called through its raw `address`.

Other fault handlers keep working. cimo passes on any fault that isn't a kernel's, so `faulthandler` still reports crashes elsewhere. Handlers are checked each time a module loads: one installed later, such as a call to `faulthandler.enable()`, takes over until the next load. `CIMO_FAULTS=0` installs nothing, so a fault ends the process with a core dump.

In the browser, a memory fault is an access beyond the end of WebAssembly memory. Low addresses are ordinary memory there: reading through a null pointer returns what's at address 0.

> [!WARNING]
> Kernels run with the full privileges of your Python process, like ctypes. There is no sandbox. In the browser, WebAssembly is the sandbox.

> [!PLANNED]
> `debug=True`: bounds checks on every access, against Python buffers and heap blocks, plus use-after-free, double-free and leak detection, reported as a `cimo.BoundsError` at the line that caused it.
