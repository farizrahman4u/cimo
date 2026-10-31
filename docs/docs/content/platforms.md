# Platforms

| Platform | Code | Status |
|---|---|---|
| Linux, arm64 | arm64 | supported: CPython 3.9 and newer |
| Pyodide, JupyterLite, marimo | WebAssembly | supported: Pyodide 0.27 |
| macOS, Apple silicon | arm64 | planned: loading is prototyped; needs testing on M-series Macs |
| Linux, x86-64 | x86-64 | planned |
| Windows, x64 | x86-64 | planned: loading is prototyped |

On an unsupported machine, `cimo.compile` raises `CompileError: cimo generates arm64 code only; this machine is x86_64`.

## How code is loaded

| Platform | |
|---|---|
| Linux | an anonymous memory map, written, then made executable and read-only (W^X); the instruction cache flushed |
| macOS | `MAP_JIT` with per-thread write protection, falling back to W^X |
| Windows | `VirtualAlloc` and `VirtualProtect` |
| Pyodide | a WebAssembly side module, linked into Pyodide's own memory |

Nothing is written to disk except the cache, and no system linker or loader is involved.
