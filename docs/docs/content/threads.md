# Threads and the heap

## Kernels run in parallel

Calls into a kernel release the GIL, and kernels are safe to call from several threads at once. There is no launch API: use Python threads and give each one a part of the work.

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cimo

k = cimo.compile(r"""
void blur_rows(int lo, int hi, int h, int w, const float src[h][w], float dst[h][w]) {
    for (int i = lo; i < hi; i++)
        for (int j = 1; j < w - 1; j++)
            dst[i][j] = 0.2f * (src[i][j] + src[i-1][j] + src[i+1][j]
                                          + src[i][j-1] + src[i][j+1]);
}
""")

h, w = 4096, 4096
src = np.random.rand(h, w).astype(np.float32)
dst = np.zeros_like(src)

n = 16
rows = [(1 + i * (h - 2) // n, 1 + (i + 1) * (h - 2) // n) for i in range(n)]
with ThreadPoolExecutor() as pool:
    list(pool.map(lambda r: k.blur_rows(r[0], r[1], h, w, src, dst), rows))
```

Threads must not write to the same elements, as in any C.

Kernels scale with the cores: a loop split across 8 threads runs 7.7 times as fast as on one on an 8-core Graviton 3, and 7.0 times on a 4-core, 8-thread Sapphire Rapids. Each call into a kernel costs about 0.1 µs with float, int and writable-buffer arguments, and up to 1 µs with others (releasing and reacquiring the GIL is about 50 ns of that), so give each thread work that takes much longer than that.

Free-threaded CPython (3.13t and later, no GIL) works the same way, and cimo doesn't turn the GIL back on.

An error in a kernel, an `assert`, `abort()` or a memory fault, raises `KernelError` in the thread that called it; kernels running on other threads carry on. Any thread can compile a module and call it, and modules can be dropped while other threads run kernels of other modules.

## malloc

`malloc`, `calloc`, `realloc`, `aligned_alloc` and `free` work as usual. Each module has its own heap, separate from Python's, and each thread gets its own part of it, so allocation doesn't take a lock. Memory freed by another thread is handed back safely.

A kernel can return memory it allocated; Python releases it with `free`:

```python
k = cimo.compile(r"""
#include <stdlib.h>
double *ramp(long n) {
    double *p = malloc(n * sizeof *p);
    for (long i = 0; i < n; i++) p[i] = i;
    return p;
}
""")

p = k.ramp(1000)   # an address (int)
# ... read it with ctypes or numpy, then:
k.free(p)
```

## Managing a module's heap

| Method | |
|---|---|
| `module.free(ptr)` | `free()` a pointer a kernel returned |
| `module.reset_heap()` | free every block at once, e.g. between calls that use the heap as scratch space |
| `module.heap_bytes()` | bytes currently mapped for the module's heap |

`reset_heap()` must not run while a kernel of the module is running, and pointers into the heap are invalid afterwards.
