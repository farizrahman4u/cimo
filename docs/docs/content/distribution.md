# Shipping code that uses cimo

## As a dependency

List `cimo` in your package's dependencies. Since it is pure Python, your package stays pure Python too: one wheel for every platform, no build step, and the C kernels ship as source in your `.py` files or as `.c` package data.

```python
# mypkg/fast.py
import os, cimo

_k = cimo.load(os.path.join(os.path.dirname(__file__), "kernels.c"))
smooth = _k.smooth
```

Users get native speed on first import (after a short compile), and from the cache after that.

## As one vendored file

Where you can't add dependencies (a QGIS plugin, a script that must run on a locked-down machine, an embedded interpreter), vendor cimo as a single file:

```console
$ cimo bundle -o mypkg/_vendor/cimo.py
```

The file holds all of cimo, including its headers and C library sources, and imports as `cimo`. It works in any Python 3.9+ and in Pyodide. The licenses of the code it includes (cimo: MIT; the ported cproc: ISC; QBE and musl: MIT) are kept at the top of the file.

## In desktop applications

Apps frozen with PyInstaller, Briefcase, Nuitka or similar can include cimo like any pure-Python package. There is no LLVM to bundle and no C compiler needed on the build machines. Make sure the cache directory is writable (set `CIMO_CACHE_DIR` if the default isn't), or cimo falls back to caching in memory and recompiles on each start.
