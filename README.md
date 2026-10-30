# cimo

**C in, machine code out.** A C compiler written in pure Python that compiles freestanding C
kernels to optimized machine code and loads them into the running Python process: no compiler,
SDK, wheel or native dependency on the target machine. The same compiler targets WebAssembly for
Python in the browser (Pyodide).

Status: early development. See [requirements.md](requirements.md) for the spec, measurements and plan.

## Layout

- `python/cimo/`: the compiler and Python runtime
- `js/`: browser/JS side (later)
- `docs/`: documentation
- `examples/`: example kernels and demos
