# Passing arrays

A pointer parameter takes a buffer: a numpy array, a `bytearray`, a `memoryview`, a ctypes array or `bytes`. The kernel gets the buffer's address and works on its memory in place.

```python
import numpy as np
import cimo

k = cimo.compile(r"""
void scale(long n, double *x, double a) {
    for (long i = 0; i < n; i++)
        x[i] *= a;
}
""")

x = np.arange(5, dtype=np.float64)
k.scale(len(x), x, 2.0)
x   # array([0., 2., 4., 6., 8.])
```

Nothing is copied: the C code reads and writes the array's own memory.

## Match the element type

cimo passes the address; it does not check that the array's dtype matches the C type. A `float64` array passed as `float *` gives wrong results, not an error.

| C type | numpy dtype |
|---|---|
| `float` | `float32` |
| `double` | `float64` |
| `int` | `int32` |
| `long`, `long long`, `size_t` | `int64`, `uint64` |
| `unsigned char`, `uint8_t` | `uint8` |
| `short` | `int16` |

`long` is 64 bits on every platform cimo targets, including Windows and the browser.

The array must also be contiguous: pass `np.ascontiguousarray(a)` if it might be a slice or a transposed view.

## Strings and bytes

`bytes` objects are passed in place and keep their terminating NUL, so `const char *` parameters accept them directly:

```python
k = cimo.compile(r"""
long count(const unsigned char *s, long n, int c) {
    long k = 0;
    for (long i = 0; i < n; i++)
        k += s[i] == c;
    return k;
}
""")

k.count(b"banana", 6, ord("a"))   # 3
```

Read-only buffers other than `bytes` are passed as a copy, so writes to them are lost.

## Multidimensional arrays

C99 variable-length array parameters describe the shape, and index like numpy:

```python
k = cimo.compile(r"""
void blur(int h, int w, const float src[h][w], float dst[h][w]) {
    for (int i = 1; i < h - 1; i++)
        for (int j = 1; j < w - 1; j++)
            dst[i][j] = 0.2f * (src[i][j] + src[i-1][j] + src[i+1][j]
                                          + src[i][j-1] + src[i][j+1]);
}
""")

src = np.random.rand(512, 512).astype(np.float32)
dst = np.zeros_like(src)
k.blur(*src.shape, src, dst)
```

The C sees the same row-major layout numpy uses.

> [!PLANNED]
> Checking at call time that an array matches a variable-length array parameter's shape.

## Returning arrays

Kernels write into arrays you pass; allocate the output in Python with `np.empty` or `np.zeros`. A kernel can also `malloc` memory and return the pointer; see [Threads and the heap](threads.md).

## Structs

Struct parameters can be passed by pointer, as a ctypes structure or a numpy structured array. Passing or returning a struct by value isn't supported from Python yet; calling such a function raises a `TypeError` (C callers in the module can still use it).

> [!PLANNED]
> Struct arguments by value, from a ctypes structure or a `dict`.
