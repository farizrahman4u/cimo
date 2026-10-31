# Environment variables

| Variable | Default | |
|---|---|---|
| `CIMO_CACHE_DIR` | `$XDG_CACHE_HOME/cimo`, else `~/.cache/cimo` | where compiled code is cached. If no directory is writable, cimo caches in memory. |
| `CIMO_FMA` | `1` | `0`: no fused multiply-add contraction, for results identical to `-ffp-contract=off` |
| `CIMO_MIDDLE` | `1` | `0`: turn off the loop optimizer |
| `CIMO_COLOR` | `auto` | the command line's colors: `auto`, `always`, `never` |

Changing `CIMO_FMA` or `CIMO_MIDDLE` changes the cache key, so cached code compiled with other settings isn't reused.
