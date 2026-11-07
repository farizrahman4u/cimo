# Command line

Installing cimo adds a `cimo` command. `cimo <command> --help` shows every option.

## cimo run

```console
$ cimo run program.c [args...]
```

Compiles the file and calls its `main` in this process, with the arguments as `argv[1:]`. Exits with the value `main` returns, or the status passed to `exit()`. `-` reads the source from stdin.

| Option | |
|---|---|
| `-e`, `--entry <name>` | call this function instead of `main` |
| `--time` | print compile and run times to stderr |

## cimo compile

```console
$ cimo compile kernels.c              # check it compiles; list the callable functions
$ cimo compile -E kernels.c           # stop after preprocessing
$ cimo compile --ir kernels.c         # stop after the front end: QBE IR
$ cimo compile -S kernels.c -o k.s    # stop after code generation: assembly
```

`--emit pp|ir|asm` is the same as `-E`, `--ir`, `-S`. `-o` writes to a file.

## Options for run and compile

| Option | |
|---|---|
| `-I`, `--include <dir>` | add an `#include` directory (repeatable) |
| `-D`, `--define <name[=value]>` | define a macro (repeatable) |
| `-l`, `--link <lib>` | link a host library: `libc`, `libm`, `python`, a soname or a path (repeatable) |
| `--host-headers` | also search the host's C headers |
| `-O0`, `--no-opt` | turn off the loop optimizer |
| `--no-fma` | no fused multiply-add contraction |
| `--wrapv` | signed integer overflow wraps, instead of being undefined |

## cimo onefile

```console
$ cimo onefile [-o cimo.py] [--keep-comments] [--compress]
```

`--compress` stores the files zlib-compressed, decompressed in memory when first used: about 460 KB instead of 1.2 MB, but not readable.

Writes all of cimo as one Python file (also published at `https://github.com/farizrahman4u/cimo/releases/download/onefile/cimo.py`). See [Shipping code that uses cimo](distribution.md).

## cimo cache

```console
$ cimo cache info    # location and size
$ cimo cache clear   # delete cached files
$ cimo cache path    # print the directory
```

## cimo completion

```console
$ cimo completion bash >> ~/.bashrc
$ cimo completion zsh > "${fpath[1]}/_cimo"
$ cimo completion fish > ~/.config/fish/completions/cimo.fish
```

Every command takes `--color auto|always|never`.

## Building your own command line

The parser, help, suggestions, colors and shell completion behind `cimo` are a small standard-library framework you can use for your own program's command line:

```python
from cimo.cli.parse import Command, parse, UsageError
from cimo.cli.term import Terminal
from cimo.cli import completion

root = Command('tool', 'What the tool does')
check = root.command('check', 'Check files', func=run_check)
check.arg('paths', 'Files to check', nargs='*')
check.opt('-q', '--quiet', flag=True, help='Less output')

p = parse(root, sys.argv[1:])        # raises UsageError, with suggestions and a hint
completion.script('bash', 'tool')    # a completion script that calls `tool __complete ...`
```

`Command(..., unknown_hint=f)` on the root adds a hint when a word isn't a command. `Terminal(stream, 'auto')` colors output when the stream is a terminal, and honors `NO_COLOR`, `FORCE_COLOR` and `CLICOLOR_FORCE`.

