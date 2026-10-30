"""The `cimo` command: its command tree and what each command does."""

import ctypes
import os
import platform
import sys
import time

from .parse import Command, UsageError
from . import report

# IR classes as Python sees them: 'l' is a C long or any pointer (an int or a buffer)
_PY = {'w': 'int32', 'l': 'int64|pointer', 's': 'float32', 'd': 'float64', '...': '...'}


def _pytype(k):
    return _PY.get(k, 'struct' if k.startswith(':') else k)


def build_tree():
    import cimo
    root = Command('cimo', 'cimo %s, a C compiler written in pure Python' % cimo.__version__,
                   description='Compiles C to machine code inside Python, with no toolchain, '
                               'and runs it. Use it from Python with `import cimo`, or from '
                               'here.')
    root.opt('-h', '--help', flag=True, help='Show help for a command')
    root.opt('--version', flag=True, help='Show the version and exit')
    root.opt('--color', metavar='when', choices=('auto', 'always', 'never'), default='auto',
             help='Colored output')
    root.example('cimo run hello.c', 'compile and run a program')
    root.example('cimo compile -S kernel.c -o kernel.s', 'look at the generated assembly')
    root.example('cimo bundle -o cimo.py', 'one file to drop into any Python')

    # run
    run = root.command(
        'run', 'Compile a C program and run its main()', func=cmd_run,
        description='Compiles the file and calls its `main` in this process, with the '
                    'arguments after it as `argv[1:]`. Exits with the value `main` returns, or '
                    'the status passed to `exit()`. Output from `printf` goes to stdout.')
    run.arg('file', 'C source file (`-` reads stdin)')
    run.arg('args', 'Arguments for the program', nargs='rest', complete=None)
    _compile_options(run)
    run.opt('-e', '--entry', metavar='name', default='main',
            help='Function to call instead of `main`')
    run.opt('--time', flag=True, help='Print compile and run times to stderr')
    run.example('cimo run hello.c')
    run.example('cimo run wc.c -l notes.txt', 'everything after the file goes to the program')
    run.example('echo \'int main(){return 3;}\' | cimo run -; echo $?')

    # compile
    comp = root.command(
        'compile', 'Compile without running: check, or show the code at any stage',
        func=cmd_compile, aliases=('cc',),
        description='Without an output stage, compiles and loads the file and lists the functions '
                    'Python can call. `-E`, `--ir` and `-S` stop after preprocessing, after '
                    'the frontend (QBE IR) or after code generation (arm64 assembly).')
    comp.arg('file', 'C source file (`-` reads stdin)')
    comp.opt('-E', dest='emit', flag=True, const='pp', group='Output:',
             help='Stop after preprocessing')
    comp.opt('--ir', dest='emit', flag=True, const='ir', group='Output:',
             help='Stop after the frontend: QBE IR')
    comp.opt('-S', dest='emit', flag=True, const='asm', group='Output:',
             help='Stop after code generation: assembly')
    comp.opt('--emit', metavar='stage', choices=('pp', 'ir', 'asm'), group='Output:',
             help='The same, by name')
    comp.opt('-o', '--output', metavar='file', complete='file', group='Output:',
             help='Write the output here (default: stdout)')
    _compile_options(comp)
    comp.example('cimo compile kernel.c', 'does it compile? which functions are exported?')
    comp.example('cimo compile -S kernel.c -o kernel.s')
    comp.example('cimo compile --ir -DN=64 kernel.c')

    # bundle
    b = root.command(
        'bundle', 'Write cimo as one self-contained .py file', func=cmd_bundle,
        description='Writes a single Python file holding all of cimo: the compiler, its '
                    'headers and runtime library sources. Drop it next to your code in any '
                    'Python 3.9+ (embedded interpreters, locked-down machines, Pyodide) and '
                    '`import cimo` works as with the installed package.')
    b.opt('-o', '--output', metavar='file', default='cimo.py', complete='file',
          help='Output file (default: `cimo.py`)')
    b.opt('--keep-comments', flag=True,
          help='Keep comments and docstrings (by default only license notices stay)')
    b.example('cimo bundle')
    b.example('cimo bundle -o vendor/cimo.py')

    # cache
    cache = root.command(
        'cache', 'Inspect or clear the compile cache', func=cmd_cache_info,
        description='cimo caches generated assembly per source file and the compiled runtime '
                    'library, keyed by content, so unchanged code is never compiled twice.')
    cache.command('info', 'Show where the cache is and how big it is', func=cmd_cache_info)
    clear = cache.command('clear', 'Delete cached files', func=cmd_cache_clear)
    clear.opt('--modules', flag=True, help='Only compiled modules')
    clear.opt('--runtime', flag=True, help='Only compiled runtime libraries')
    cache.command('path', 'Print the cache directory', func=cmd_cache_path)

    # completion
    comp_ = root.command(
        'completion', 'Print a shell completion script', func=cmd_completion,
        description='Prints a script that makes your shell complete cimo commands, options '
                    'and files.')
    comp_.arg('shell', 'bash, zsh or fish', complete=None)
    comp_.example('cimo completion bash >> ~/.bashrc')
    comp_.example('cimo completion zsh > "${fpath[1]}/_cimo"')
    comp_.example('cimo completion fish > ~/.config/fish/completions/cimo.fish')

    # help
    h = root.command('help', 'Show help for a command', func=cmd_help)
    h.arg('command', 'Command name, e.g. `run` or `cache clear`', nargs='*', complete=None)

    root.command('__complete', 'Shell completion backend', func=cmd_complete, hidden=True) \
        .arg('words', nargs='rest', complete=None)
    root.command('__update-runtime-symbols', 'Regenerate runtime/symbols.txt (development)',
                 func=cmd_update_symbols, hidden=True)
    return root


def _compile_options(c):
    g = 'Compilation:'
    c.opt('-I', '--include', metavar='dir', multiple=True, complete='dir', group=g,
          help='Add a directory to the `#include` search path')
    c.opt('-D', '--define', metavar='name[=value]', multiple=True, group=g,
          help='Define a macro')
    c.opt('-l', '--link', metavar='lib', multiple=True, group=g,
          help='Make a host library callable: `libc`, `libm`, `python`, a soname or a path')
    c.opt('--host-headers', flag=True, group=g,
          help="Also search the host's C headers (with `--link libc` for the OS API)")
    c.opt('-O0', '--no-opt', dest='no_opt', flag=True, group=g,
          help='Turn off the loop optimizer')
    c.opt('--no-fma', flag=True, group=g,
          help='No fused multiply-add contraction (results identical to `-ffp-contract=off`)')


# ---------------------------------------------------------------------------------------------
# helpers

class Failure(Exception):
    """A command failed; the message was printed. status: the exit status."""

    def __init__(self, status=1):
        Exception.__init__(self, status)
        self.status = status


def _read_source(ctx, path):
    if path == '-':
        return sys.stdin.read(), '<stdin>'
    try:
        with open(path) as f:
            return f.read(), path
    except FileNotFoundError:
        near = _near_files(path)
        raise UsageError("no such file: '%s'" % path, ctx.command, near) from None
    except IsADirectoryError:
        raise UsageError("'%s' is a directory" % path, ctx.command) from None
    except UnicodeDecodeError:
        raise UsageError("'%s' is not a text file" % path, ctx.command) from None


def _near_files(path):
    import difflib
    d = os.path.dirname(path) or '.'
    try:
        names = [n for n in os.listdir(d) if n.endswith(('.c', '.h'))]
    except OSError:
        return []
    hits = difflib.get_close_matches(os.path.basename(path), names, n=3, cutoff=0.6)
    return [os.path.join(os.path.dirname(path), h) for h in hits]


def _defines(ctx):
    out = []
    for d in ctx.values.get('define') or ():
        name, eq, val = d.partition('=')
        if not name:
            raise UsageError("bad macro definition '-D%s'" % d, ctx.command)
        out.append((name, val if eq else None))
    return out


class _Settings:
    """Apply --no-opt / --no-fma to the driver for the duration of a command."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        from .. import driver
        self.old = driver.MIDDLE, driver.FMA
        if self.ctx.values.get('no_opt'):
            driver.MIDDLE = False
        if self.ctx.values.get('no_fma'):
            driver.FMA = False

    def __exit__(self, *exc):
        from .. import driver
        driver.MIDDLE, driver.FMA = self.old


def _module(ctx, src, filename):
    from .. import driver
    v = ctx.values
    try:
        with _Settings(ctx), _runtime_spinner(ctx):
            return driver._module(src, filename, v.get('include') or (), v.get('link') or (),
                                  v.get('host_headers'), _defines(ctx))
    except driver.CompileError as e:
        report.compile_error(ctx.err, e, {filename: src})
        raise Failure(1)


class _runtime_spinner:
    """Show a spinner instead of the runtime library's first-use build messages."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        from .. import linker
        self.linker = linker
        self.old = getattr(linker, 'progress', None)
        self.spinner = None
        if hasattr(linker, 'progress'):
            linker.progress = self._progress

    def _progress(self, event, **info):
        from .term import Spinner
        e = self.ctx.err
        if event == 'file':
            text = 'Compiling the runtime library: %s' % info.get('name', '')
            if self.spinner is None:
                self.spinner = Spinner(e, 'Compiling the runtime library (first use only)')
                self.spinner.__enter__()
            self.spinner.update(text)
        elif event == 'done':
            if self.spinner:
                self.spinner.__exit__(None, None, None)
                self.spinner = None
            n = info.get('files', 0)
            e.write(e.style('%s Compiled %d runtime library file%s in %.1f s' % (
                e.symbol('ok'), n, '' if n == 1 else 's', info.get('seconds', 0)), 'muted'))

    def __exit__(self, *exc):
        if self.spinner:
            self.spinner.__exit__(None, None, None)
        if hasattr(self.linker, 'progress'):
            self.linker.progress = self.old


def _write_output(ctx, text):
    out = ctx.values.get('output')
    if out in (None, '-'):
        sys.stdout.write(text)
        return
    with open(out, 'w') as f:
        f.write(text)
    ctx.err.write('%s Wrote %s' % (ctx.err.style(ctx.err.symbol('ok'), 'ok'),
                                   ctx.err.style(out, 'path')))


# ---------------------------------------------------------------------------------------------
# commands

def cmd_run(ctx):
    from .. import KernelError
    src, filename = _read_source(ctx, ctx.values['file'])
    t0 = time.time()
    m = _module(ctx, src, filename)
    t1 = time.time()
    entry = ctx.values['entry']
    k = m._kernels.get(entry)
    if k is None:
        import difflib
        names = sorted(m._kernels)
        raise UsageError("'%s' has no function %s()" % (filename, entry), ctx.command,
                         difflib.get_close_matches(entry, names, n=3, cutoff=0.5),
                         hint=('exported functions: %s' % ', '.join(names)) if names
                         else 'it exports no functions')
    argv = [filename if filename != '<stdin>' else '-'] + list(ctx.values['args'])
    params = k._params
    if k._f is None:
        ctx.err.write('%s %s() cannot be called: %s' % (ctx.err.style('error:', 'error'), entry,
                                                        k._unsupported))
        raise Failure(1)
    if params == []:
        args = ()
    elif params in (['w', 'l'], ['w', 'l', 'l']):
        arr = (ctypes.c_char_p * (len(argv) + 1))(
            *[a.encode('utf-8', 'surrogateescape') for a in argv], None)
        env = [('%s=%s' % kv).encode('utf-8', 'surrogateescape') for kv in os.environ.items()]
        envp = (ctypes.c_char_p * (len(env) + 1))(*env, None)
        args = (len(argv), ctypes.addressof(arr), ctypes.addressof(envp))[:len(params)]
    else:
        ctx.err.write('%s %s() must take no arguments or (int argc, char **argv)'
                      % (ctx.err.style('error:', 'error'), entry))
        raise Failure(1)
    status = 0
    try:
        r = k(*args)
        status = (r or 0) & 0xff if isinstance(r, int) else 0
    except KernelError as e:
        sys.stdout.flush()
        if e.kind == 'exit':
            status = e.status & 0xff
        else:
            report.kernel_error(ctx.err, e, {filename: src})
            status = 134
    finally:
        sys.stdout.flush()
        if ctx.values.get('time'):
            t2 = time.time()
            ctx.err.write(ctx.err.style('compile %.3f s, run %.3f s' % (t1 - t0, t2 - t1),
                                        'muted'))
    return status


def cmd_compile(ctx):
    from .. import driver
    v = ctx.values
    src, filename = _read_source(ctx, v['file'])
    emit = v.get('emit') or None
    if emit is None:
        t0 = time.time()
        m = _module(ctx, src, filename)
        dt = time.time() - t0
        e = ctx.err
        names = sorted(m._kernels)
        e.write('%s %s compiled in %.2f s, %d function%s' % (
            e.style(e.symbol('ok'), 'ok'), e.style(filename, 'path'), dt, len(names),
            '' if len(names) == 1 else 's'))
        for n in names:
            k = m._kernels[n]
            params = ', '.join(e.style(_pytype(p), 'muted') for p in k._params)
            ret = getattr(k, '_ret', None)
            sig = '%s(%s)%s' % (e.style(n, 'bold'), params,
                                ' %s %s' % (e.symbol('arrow'), _pytype(ret)) if ret else '')
            note = '' if k._f else e.style('  not callable from Python: %s' % k._unsupported,
                                           'muted')
            e.write('  %s %s%s' % (e.style(e.symbol('bullet'), 'muted'), sig, note))
        return 0
    try:
        with _Settings(ctx):
            driver._check_machine() if emit == 'asm' else None
            pp = driver._preprocess(src, filename, v.get('include') or (),
                                    v.get('host_headers'), _defines(ctx))
            if emit == 'pp':
                return _write_output(ctx, pp) or 0
            ssa, params = driver._ir(pp, filename, v.get('host_headers'))
            if emit == 'ir':
                return _write_output(ctx, ssa) or 0
            from .. import execution
            text = driver._qbe(ssa, params, stkchk=execution.stack_checked(ssa))
            return _write_output(ctx, text) or 0
    except driver.CompileError as e:
        report.compile_error(ctx.err, e, {filename: src})
        raise Failure(1)


def cmd_bundle(ctx):
    try:
        from .. import bundle
    except ImportError:
        ctx.err.write('%s bundling is not available in this build' % ctx.err.style('error:',
                                                                                   'error'))
        raise Failure(1)
    from .term import Spinner
    out = ctx.values['output']
    e = ctx.err
    with Spinner(e, 'Bundling cimo into %s' % out) as sp:
        stats = bundle.build(out, progress=sp.update, strip=not ctx.values.get('keep_comments'))
    size = os.path.getsize(out)
    e.write('%s Wrote %s (%s) in %.1f s' % (e.style(e.symbol('ok'), 'ok'), e.style(out, 'path'),
                                            _size(size), sp.elapsed))
    details = []
    for key, label in (('modules', 'modules'), ('resources', 'resource files')):
        if isinstance(stats, dict) and stats.get(key):
            details.append('%d %s' % (stats[key], label))
    if details:
        e.write('  ' + e.style(', '.join(details), 'muted'))
    e.write('  %s %s' % (e.style('use:', 'hint'),
                         e.code('put `%s` next to your code and `import cimo`'
                                % os.path.basename(out))))
    return 0


def _size(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return ('%d %s' % (n, unit)) if unit == 'B' else ('%.1f %s' % (n, unit))
        n /= 1024.0


def _cache_dirs():
    from .. import driver, linker
    mods = driver._cache_dir()
    root = linker._cache_root() if hasattr(linker, '_cache_root') else mods
    return mods, root


def _cache_entries():
    """(module files, runtime dirs) as lists of (path, bytes)."""
    mods_dir, rt_root = _cache_dirs()
    mods, rts = [], []
    if mods_dir and os.path.isdir(mods_dir):
        for n in sorted(os.listdir(mods_dir)):
            p = os.path.join(mods_dir, n)
            if os.path.isfile(p) and n.endswith(('.s', '.sig')):
                mods.append((p, os.path.getsize(p)))
    if rt_root and os.path.isdir(rt_root):
        for n in sorted(os.listdir(rt_root)):
            p = os.path.join(rt_root, n)
            if os.path.isdir(p) and n.startswith('runtime-'):
                total = sum(os.path.getsize(os.path.join(d, f))
                            for d, _, fs in os.walk(p) for f in fs)
                rts.append((p, total))
    return mods, rts


def cmd_cache_info(ctx):
    e = ctx.out
    mods_dir, rt_root = _cache_dirs()
    mods, rts = _cache_entries()
    nmods = sum(1 for p, _ in mods if p.endswith('.s'))
    e.write(e.style('Cache', 'heading'))
    e.write('  %-18s %s' % ('modules', e.style(mods_dir or '(in memory)', 'path')))
    if rt_root != mods_dir:
        e.write('  %-18s %s' % ('runtime libraries', e.style(rt_root or '(in memory)', 'path')))
    e.write('')
    e.write('  %-18s %s in %s' % ('compiled modules', e.style(str(nmods), 'number'),
                                  _size(sum(n for _, n in mods))))
    e.write('  %-18s %s in %s' % ('runtime builds', e.style(str(len(rts)), 'number'),
                                  _size(sum(n for _, n in rts))))
    if len(rts) > 1:
        e.write('  ' + e.code('older runtime builds are left by previous cimo versions; '
                              '`cimo cache clear --runtime` removes them'))
    return 0


def cmd_cache_clear(ctx):
    import shutil
    v = ctx.values
    both = not v.get('modules') and not v.get('runtime')
    mods, rts = _cache_entries()
    freed = 0
    nm = nr = 0
    if both or v.get('modules'):
        for p, n in mods:
            try:
                os.remove(p)
                freed += n
                nm += p.endswith('.s')
            except OSError:
                pass
    if both or v.get('runtime'):
        for p, n in rts:
            shutil.rmtree(p, ignore_errors=True)
            freed += n
            nr += 1
        try:
            from .. import linker
            linker._runtime = None
        except Exception:
            pass
    e = ctx.err
    e.write('%s Removed %d module%s and %d runtime build%s, freed %s' % (
        e.style(e.symbol('ok'), 'ok'), nm, '' if nm == 1 else 's', nr, '' if nr == 1 else 's',
        _size(freed)))
    return 0


def cmd_cache_path(ctx):
    mods_dir, _ = _cache_dirs()
    if mods_dir is None:
        ctx.err.write('%s no writable cache directory: cimo caches in memory for this process'
                      % ctx.err.style('error:', 'error'))
        return 1
    sys.stdout.write(mods_dir + '\n')
    return 0


def cmd_help(ctx):
    from .parse import render_help
    cmd = ctx.command.root
    for w in ctx.values.get('command') or ():
        sub = cmd.find_command(w)
        if sub is None:
            raise UsageError("unknown command '%s'" % w, cmd,
                             __import__('difflib').get_close_matches(
                                 w, [c.name for c in cmd.visible_commands()], n=3, cutoff=0.6))
        cmd = sub
    ctx.out.write(render_help(cmd, ctx.out))
    return 0


def cmd_update_symbols(ctx):
    from .. import linker
    from .term import Spinner
    with Spinner(ctx.err, 'Compiling every runtime source') as sp:
        text = linker.symbol_table()
    with open(linker.SYMBOLS, 'w') as f:
        f.write(text)
    ctx.err.write('%s Wrote %s in %.1f s' % (ctx.err.style(ctx.err.symbol('ok'), 'ok'),
                                             linker.SYMBOLS, sp.elapsed))
    return 0


def cmd_completion(ctx):
    from . import completion
    shell = ctx.values['shell']
    scripts = {'bash': completion.BASH, 'zsh': completion.ZSH, 'fish': completion.FISH}
    if shell not in scripts:
        import difflib
        raise UsageError("unsupported shell '%s'" % shell, ctx.command,
                         difflib.get_close_matches(shell, list(scripts), n=1, cutoff=0.5),
                         hint='supported: bash, zsh, fish')
    sys.stdout.write(scripts[shell])
    return 0


def cmd_complete(ctx):
    from .parse import complete
    cands, kind = complete(ctx.command.root, ctx.values['words'] or [''])
    for c in cands:
        sys.stdout.write(c + '\n')
    if kind:
        sys.stdout.write('__%s__\n' % kind)
    return 0


def version_text():
    import cimo
    return 'cimo %s (Python %s, %s %s)' % (cimo.__version__, platform.python_version(),
                                           sys.platform, platform.machine())
