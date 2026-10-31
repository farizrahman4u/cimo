# Files, includes and the cache

## Kernels in a file

Keep larger kernels in a `.c` file next to your Python code:

```python
import os
import cimo

k = cimo.load(os.path.join(os.path.dirname(__file__), "kernels.c"))
```

`cimo.load` compiles the file. Errors name the file, and `#include "..."` finds headers next to it.

## Include paths and macros

Both `compile` and `load` take:

```python
k = cimo.compile(src,
                 include_dirs=["include", "third_party/foo"],
                 defines={"N": 256, "USE_FAST_PATH": None})
```

- `include_dirs`: searched for `#include`, after the including file's own directory and before cimo's bundled headers.
- `defines`: macros, as a dict or `(name, value)` pairs. A value of `None` defines the macro as `1`.

## The cache

cimo caches compiled code by a hash of the preprocessed source (and of the compiler itself). Compiling the same code again, in the same process or a later one, loads the cached result and skips compilation entirely. Editing the C, a header it includes, or a macro value makes a new entry.

The cache lives in the first of:

1. `$CIMO_CACHE_DIR`
2. `$XDG_CACHE_HOME/cimo`
3. `~/.cache/cimo`

If none is writable, cimo caches in memory for the life of the process.

```console
$ cimo cache info     # where it is and how big
$ cimo cache clear    # delete it
$ cimo cache path     # print the directory
```

The parts of cimo's C library a program uses are compiled the first time they're needed and cached in the same place, so the very first compile after installing (or after an upgrade) takes a little longer.
