# Introduction

cimo is a C compiler written in pure Python. It compiles C to machine code inside your Python process and hands you the functions as Python callables. It needs no system compiler, SDK, LLVM or binary wheels, so it works wherever CPython or Pyodide runs.

```python
import cimo

k = cimo.compile(r"""
double dot(long n, const double *x, const double *y) {
    double s = 0;
    for (long i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}
""")
```

`k.dot` is now a function backed by native code. Call it on numpy arrays and they are used in place, never copied.

## When to use it

- **Loops numpy can't express.** Stencils, recurrences, flood fills, dynamic programming: code where pure Python is 100x too slow and vectorized numpy doesn't fit.
- **Native code generated at runtime.** If your program writes C (from a model, a query, a formula) and needs it compiled on the user's machine, cimo is the compiler, with nothing to install.
- **Places without a toolchain.** Pyodide, JupyterLite and marimo in the browser, plugin repositories that accept no binaries, locked-down machines.

## How these docs are organized

The first three parts are a walkthrough: each page builds on the one before, from your first kernel to linking host libraries and running in the browser. Read them in order the first time; **Next** at the bottom of each page takes you on.

The **Reference** part is for looking things up: the full Python API, the command line, environment variables, platforms and troubleshooting. Press <kbd>/</kbd> anywhere to search.

> [!NOTE]
> cimo is under active development. Features that are designed but not built yet are marked **Planned**.
