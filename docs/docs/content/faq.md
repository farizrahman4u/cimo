# Troubleshooting

### My kernel returns wrong numbers

Most often the array's dtype doesn't match the C type: a `float64` array passed as `float *`, or `int64` as `int *`. cimo passes the address and can't check. See the table in [Passing arrays](arrays.md#match-the-element-type). Also check the array is contiguous (`np.ascontiguousarray`).

### Results differ slightly from gcc

cimo contracts `a * b + c` into a fused multiply-add by default. Set `CIMO_FMA=0` to compare with `gcc -ffp-contract=off`; results are then bit-identical on the benchmarks.

### The first compile is slow

The first compile after installing builds the parts of cimo's C library your code uses, and caches them. Later compiles skip that, and unchanged code isn't compiled at all.

### "fopen is not in cimo's runtime"

cimo's bundled C library has no files, clocks or sockets. Link the host's: `cimo.compile(src, link=["libc"], host_headers=True)`. See [Calling host libraries](host-libraries.md).

### "cannot be called from Python: it passes or returns a struct or union by value"

Pass a pointer to the struct instead. Struct values from Python are [planned](arrays.md#structs).

### "cimo generates arm64 code only"

See [Platforms](platforms.md): x86-64 is planned.

### My process crashed

A kernel that writes outside its arrays can crash the process, as in any C. `assert`, `abort` and deep recursion are caught; bad pointers aren't yet. Check array lengths and the indices your loops use.

### In Pyodide, a `long *` array gives garbage

`ctypes.c_long` is 32 bits in Pyodide, while cimo's `long` is 64 bits. Use `ctypes.c_int64` or `np.int64` arrays. See [In the browser](browser.md).

### Stale results after changing a header

The cache key includes every header the source includes, so this shouldn't happen. If it does, `cimo cache clear` and please [report it](https://github.com/farizrahman4u/cimo/issues).
