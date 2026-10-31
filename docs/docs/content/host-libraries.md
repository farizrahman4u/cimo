# Calling host libraries

By default a module links only against cimo's own C library: nothing is resolved from the host. When a kernel needs the operating system (files, clocks, sockets, threads) or an existing library, name it with `link=`:

```python
k = cimo.compile(r"""
#include <stdio.h>
#include <time.h>

long now(void) { return time(0); }

int first_byte(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int c = fgetc(f);
    fclose(f);
    return c;
}
""", link=["libc"], host_headers=True)
```

- `link`: libraries whose functions may resolve what cimo's library doesn't provide. Accepted forms: `"libc"`, `"libm"`, `"python"` (the Python C API), a soname such as `"libz.so.1"`, a short name such as `"z"`, or a path.
- `host_headers=True`: also search the host's C headers (after your `include_dirs`, before cimo's), so the declarations of those libraries are available.

cimo's own functions still win: `printf`, `malloc`, `memcpy` and the math functions come from cimo's library even when libc is linked, so they behave the same everywhere.

## Example: zlib

```python
k = cimo.compile(r"""
#include <zlib.h>
unsigned long crc(const unsigned char *p, long n) { return crc32(0, p, n); }
""", link=["libz.so.1"], host_headers=True)

k.crc(b"hello", 5)
```

## Without link=

Using a hosted function without linking is a compile error that says so:

```text
<string>:2:31: error: undeclared identifier: fopen; fopen is not in cimo's runtime:
pass host_headers=True to use the host's C headers (and link=['libc'] to call their functions)
```

> [!WARNING]
> Linked libraries run with your process's full privileges, like anything loaded with ctypes.

`link=` and `host_headers=` aren't available in the browser, where there is no host C library.
