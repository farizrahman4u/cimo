# In the browser

In Pyodide, JupyterLite and marimo, `cimo.compile` produces WebAssembly instead of arm64 and loads it into Pyodide's own memory. The same code works unchanged:

```python
import micropip
await micropip.install("cimo")

import numpy as np
import cimo

k = cimo.compile(r"""
void blur(int h, int w, const float src[h][w], float dst[h][w]) {
    for (int i = 1; i < h - 1; i++)
        for (int j = 1; j < w - 1; j++)
            dst[i][j] = 0.2f * (src[i][j] + src[i-1][j] + src[i+1][j]
                                          + src[i][j-1] + src[i][j+1]);
}
""")

src = np.random.rand(1024, 1024).astype(np.float32)
dst = np.zeros_like(src)
k.blur(*src.shape, src, dst)   # numpy arrays used in place
```

Everything runs in the page: no server compiles anything. The [demos on the homepage](../) are cimo running this way.

## What's the same

- numpy arrays, `bytearray`, `memoryview` and ctypes buffers are passed in place.
- `printf` goes to `sys.stdout`; `abort`, `assert` and `exit` raise `KernelError`.
- The bundled C library is the same, so results match the native build. The one known exception: `pow` can differ in the last bit for a few inputs, because WebAssembly has no fused multiply-add.

## What's different

- **`long` is 64 bits, but `ctypes.c_long` is 32 bits in Pyodide.** For `long *` parameters, pass `int64` numpy arrays or `ctypes.c_int64` buffers, not `ctypes.c_long`.
- **No host libraries:** `link=` and `host_headers=` raise a `CompileError`.
- **No threads:** Pyodide is single-threaded.
- **A `goto` into the middle of a loop** is a compile error; WebAssembly needs structured control flow.
- **The first compile** in a page also builds the parts of the C library the code uses, so it takes a second or two. After that, small kernels compile in well under a second.

## Calling kernels from JavaScript

Each kernel object has `.function`, the raw WebAssembly export, for calling from JavaScript in a hot loop (for example once per animation frame) without going through Python. Pointer and `long` arguments are BigInts there.
