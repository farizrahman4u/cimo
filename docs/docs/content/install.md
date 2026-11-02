# Install

```console
$ pip install cimo
```

cimo is pure Python with no dependencies outside the standard library. It needs CPython 3.9 or newer, or Pyodide.

## Check it works

```console
$ cimo run hello.c
hello, world
```

where `hello.c` is:

```c
#include <stdio.h>

int main(void) {
    printf("hello, world\n");
    return 0;
}
```

The first compile also builds the parts of cimo's C library that the program uses and caches them, so it takes a moment longer than the ones after it.

## Without pip

All of cimo also comes as a single Python file that works like the installed package:

```console
$ wget https://github.com/farizrahman4u/cimo/releases/download/onefile/cimo.py
```

Put `cimo.py` next to your code, or anywhere on `sys.path`, and `import cimo`. In a notebook:

```python
!wget -q https://github.com/farizrahman4u/cimo/releases/download/onefile/cimo.py
import cimo
```

The file is rebuilt from `main` on every change. To write one from an installed cimo, run `cimo onefile -o cimo.py`. See [Shipping code that uses cimo](distribution.md).

## In Pyodide, JupyterLite or marimo

```python
import micropip
await micropip.install("cimo")
import cimo
```

The browser has its own page: [In the browser](browser.md).

## Platforms

cimo generates arm64 code on Linux, macOS (Apple silicon) and Windows, x86-64 code on Linux and Windows, and WebAssembly for Pyodide; see [Platforms](platforms.md).
