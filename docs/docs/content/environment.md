# Environment variables

| Variable | Default | |
|---|---|---|
| `CIMO_CACHE_DIR` | `$XDG_CACHE_HOME/cimo`, else `~/.cache/cimo` | where compiled code is cached. If no directory is writable, cimo caches in memory. |
| `CIMO_FMA` | `1` | `0`: no fused multiply-add contraction, for results identical to `-ffp-contract=off` (wasm, and x86-64 CPUs without FMA, never contract) |
| `CIMO_X86` | detected | x86-64 features to compile for: `baseline` (SSE2 only), or a comma list of `fma`, `popcnt`, `lzcnt`, `bmi1`, `bmi2`, `avx`, `avx2`. By default, what this CPU has |
| `CIMO_MIDDLE` | `1` | `0`: turn off the loop optimizer |
| `CIMO_WRAPV` | `0` | `1`: signed integer overflow wraps, like `-fwrapv` |
| `CIMO_COLOR` | `auto` | the command line's colors: `auto`, `always`, `never` |

Changing `CIMO_FMA`, `CIMO_X86`, `CIMO_MIDDLE` or `CIMO_WRAPV` changes the cache key, so cached code compiled with other settings isn't reused.
