# Platforms

| Platform | Code | Status |
|---|---|---|
| Linux, arm64 | arm64 | supported: CPython 3.9 and newer |
| Pyodide, JupyterLite, marimo | WebAssembly | supported: Pyodide 0.27 |
| Windows, arm64 | arm64 | supported: arm64 builds of CPython |
| macOS, Apple silicon | arm64 | supported: CPython 3.9 and newer (Apple's Command Line Tools, python.org, Homebrew) |
| Linux, x86-64 | x86-64 | supported: code uses FMA, POPCNT, LZCNT and TZCNT when the CPU running it has them, and baseline SSE2 otherwise (`CIMO_X86`) |
| Windows, x64 | x86-64 | supported: x64 builds of CPython; uses the CPU's features as on Linux |

On an unsupported machine (for example an Intel Mac), `cimo.compile` raises `CompileError`.

## How code is loaded

| Platform | |
|---|---|
| Linux | an anonymous memory map, written, then made executable and read-only (W^X); the instruction cache flushed |
| macOS | `MAP_JIT` with per-thread write protection, data right after it; falling back to W^X |
| Windows | `VirtualAlloc` and `VirtualProtect` |
| Pyodide | a WebAssembly side module, linked into Pyodide's own memory |

Nothing is written to disk except the cache, and no system linker or loader is involved.
