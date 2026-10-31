"""The `cimo` command line (cimo.cli): parsing, help, suggestions, commands, completion."""

import os
import platform
import re
import subprocess
import sys

import pytest

from cimo import driver
from cimo.cli import main
from cimo.cli import parse as P
from cimo.cli.commands import build_tree

ARM64 = platform.machine().lower() in ('arm64', 'aarch64')
needs_arm64 = pytest.mark.skipif(not ARM64, reason='arm64 only')


def run(capsys, *argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        import io
        monkeypatch.setattr(sys, 'stdin', io.StringIO(stdin))
    status = main(['--color=never'] + list(argv))
    out, err = capsys.readouterr()
    return status, out, err


@pytest.fixture
def prog(tmp_path):
    def write(name, text):
        p = tmp_path / name
        p.write_text(text)
        return str(p)
    return write


# ---------------------------------------------------------------------------------------------
# parsing and help

def test_parse_options_and_rest():
    root = build_tree()
    p = P.parse(root, ['run', '-I', 'a', '-Ib', '--define=X=1', '-DY', 'prog.c', '-x', '--y',
                       'z'])
    assert p.command.name == 'run'
    assert p.values['include'] == ['a', 'b']
    assert p.values['define'] == ['X=1', 'Y']
    assert p.values['file'] == 'prog.c'
    assert p.values['args'] == ['-x', '--y', 'z']  # everything after the file is the program's
    p = P.parse(root, ['compile', '-S', '-O0', 'k.c', '-o', 'k.s'])
    assert (p.values['emit'], p.values['no_opt'], p.values['output']) == ('asm', True, 'k.s')
    p = P.parse(root, ['cache', 'clear', '--runtime'])
    assert p.command.path == 'cimo cache clear' and p.values['runtime']


@pytest.mark.parametrize('argv,msg,suggestion', [
    (['rnu'], "unknown command 'rnu'", 'run'),
    (['run', '--inclde', 'x', 'a.c'], "unknown option '--inclde'", '--include'),
    (['compile', '--emit', 'assm', 'a.c'], "invalid value 'assm' for --emit", 'asm'),
    (['cache', 'claer'], "unknown command 'claer'", 'clear'),
    (['help', 'compil'], "unknown command 'compil'", 'compile'),
])
def test_did_you_mean(capsys, argv, msg, suggestion):
    status, out, err = run(capsys, *argv)
    assert status == 2
    assert 'error: ' + msg in err
    assert 'Did you mean %s' % suggestion in err
    assert 'Usage: cimo' in err


def test_usage_errors(capsys):
    status, _, err = run(capsys, 'run')
    assert status == 2 and 'missing argument <file>' in err and 'cimo run --help' in err
    status, _, err = run(capsys, 'compile', '-o')
    assert status == 2 and "option '-o' needs a value <file>" in err
    status, _, err = run(capsys, 'prog.c')
    assert status == 2 and 'cimo run prog.c' in err  # hint for a C file given as a command
    status, _, err = run(capsys, 'cache', 'info', 'extra')
    assert status == 2 and "unexpected argument 'extra'" in err


def test_help_at_every_level(capsys):
    root = build_tree()

    def walk(c):
        yield c
        for s in c.commands.values():
            if not s.hidden:
                yield from walk(s)

    for c in walk(root):
        words = c.path.split()[1:]
        for form in (words + ['--help'], words + ['-h'], ['help'] + words):
            status, out, err = run(capsys, *form)
            assert status == 0, (form, err)
            assert out.startswith(c.summary), form
            assert 'Usage: %s' % c.path in out
            if c.commands:
                assert 'Commands:' in out
    status, out, _ = run(capsys)  # no arguments: the top-level help
    assert status == 0 and 'Commands:' in out and 'Examples:' in out
    status, out, _ = run(capsys, 'cache', '--help')
    assert 'clear' in out and '--version' not in out


def test_version(capsys):
    import cimo
    status, out, _ = run(capsys, '--version')
    assert status == 0 and out.startswith('cimo %s (Python ' % cimo.__version__)


def test_help_wraps_to_terminal_width(capsys, monkeypatch):
    monkeypatch.setenv('COLUMNS', '60')
    _, out, _ = run(capsys, 'run', '--help')
    assert max(len(line) for line in out.splitlines()) <= 60


def test_color(capsys, monkeypatch):
    status = main(['--color=always', 'run', '--help'])
    out, _ = capsys.readouterr()
    assert '\x1b[1m' in out and '\x1b[36m' in out
    monkeypatch.setenv('NO_COLOR', '1')
    main(['run', '--help'])
    out, _ = capsys.readouterr()
    assert '\x1b[' not in out


# ---------------------------------------------------------------------------------------------
# commands

@needs_arm64
def test_run_argv_and_status(capsys, prog):
    p = prog('hello.c', r'''#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    for (int i = 0; i < argc; i++) printf("[%s]", argv[i]);
    printf("\n");
    return argc > 1 ? atoi(argv[argc - 1]) : 0;
}''')
    status, out, _ = run(capsys, 'run', p, '-x', '--flag', '42')
    assert status == 42
    assert out == '[%s][-x][--flag][42]\n' % p


@needs_arm64
def test_run_stdin_exit_and_entry(capsys, prog, monkeypatch):
    status, _, _ = run(capsys, 'run', '-', stdin='int main(void){ return 3; }',
                       monkeypatch=monkeypatch)
    assert status == 3
    p = prog('e.c', '#include <stdlib.h>\nvoid quit(void){ exit(5); }\nint main(void){ quit(); }')
    assert run(capsys, 'run', p)[0] == 5
    p = prog('f.c', '#include <stdio.h>\nvoid go(void){ puts("go"); }\n')
    status, out, _ = run(capsys, 'run', '-e', 'go', p)
    assert (status, out) == (0, 'go\n')
    status, _, err = run(capsys, 'run', p)
    assert status == 2 and 'has no function main()' in err and 'exported functions: go' in err


@needs_arm64
def test_run_defines_and_includes(capsys, prog, tmp_path):
    inc = tmp_path / 'inc'
    inc.mkdir()
    (inc / 'k.h').write_text('#define K 7\n')
    p = prog('d.c', '#include <k.h>\nint main(void){ return N + K; }')
    assert run(capsys, 'run', '-I', str(inc), '-DN=30', p)[0] == 37


@needs_arm64
def test_compile_error_excerpt(capsys, prog):
    # long double is binary64 (allowed) on Apple and Windows, 128-bit (rejected) on Linux
    ty, msg = (('long double', 'long double is not supported') if driver.TARGET == 'arm64'
               else ('__int128', 'cast type must be scalar'))
    p = prog('bad.c', 'int main(void) {\n\tdouble y = 1;\n\ty = y * (%s)2;\n'
                      '\treturn 0;\n}\n' % ty)
    status, _, err = run(capsys, 'run', p)
    assert status == 1
    lines = err.splitlines()
    assert lines[0] == 'error: ' + msg
    assert re.match(r'  -> .*bad\.c:3:\d+$', lines[1]) or re.match(r'  → .*bad\.c:3:\d+$',
                                                                    lines[1])
    assert '    y = y * (%s)2;' % ty in err  # tabs expanded
    caret = [l for l in lines if l.rstrip().endswith('^')]
    assert caret


@needs_arm64
def test_compile_error_hint_uses_flags(capsys, prog):
    p = prog('h.c', '#include <time.h>\nint main(void){ return 0; }\n')
    _, _, err = run(capsys, 'run', p)
    assert 'hint:' in err and '--host-headers' in err and 'host_headers=True' not in err


@needs_arm64
def test_assert_failure(capsys, prog):
    p = prog('a.c', '#include <assert.h>\nint main(void){\n    assert(1 + 1 == 3);\n}\n')
    status, _, err = run(capsys, 'run', p)
    assert status == 134
    assert 'assertion failed: 1 + 1 == 3' in err and 'a.c:3' in err and 'in main()' in err
    assert 'assert(1 + 1 == 3);' in err


def test_missing_file(capsys, prog):
    p = prog('hello.c', 'int main(void){ return 0; }')
    status, _, err = run(capsys, 'run', p[:-3] + 'llo.c')
    assert status == 2 and 'no such file' in err and 'Did you mean' in err


@needs_arm64
def test_compile_stages(capsys, prog, tmp_path):
    p = prog('k.c', '#define TWICE(x) ((x) * 2)\nint f(int x){ return TWICE(x) + N; }\n')
    status, out, _ = run(capsys, 'compile', '-E', '-DN=1', p)
    assert status == 0 and '((x) * 2) + 1' in out
    status, out, _ = run(capsys, 'compile', '--ir', '-DN=1', p)
    assert status == 0 and 'export' in out and 'function w $f' in out
    out_s = str(tmp_path / 'k.s')
    status, _, err = run(capsys, 'compile', '-S', '-DN=1', p, '-o', out_s)
    assert status == 0 and 'Wrote' in err
    assert '.globl f' in open(out_s).read()
    status, _, err = run(capsys, 'compile', '-DN=1', p)
    assert status == 0 and 'compiled' in err and 'f(int32)' in err


@needs_arm64
def test_compile_lists_signatures(capsys, prog):
    p = prog('l.c', 'float k(double d, long *p){ return d; }\nint v(int n, ...){ return n; }\n')
    _, _, err = run(capsys, 'compile', p)
    assert 'k(float64, int64|pointer)' in err and 'float32' in err
    assert 'not callable from Python: it is variadic' in err


def test_cache(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path / 'c'))
    os.makedirs(tmp_path / 'c', exist_ok=True)
    (tmp_path / 'c' / 'abc.s').write_text('x' * 100)
    (tmp_path / 'c' / 'abc.sig').write_text('f - ')
    status, out, _ = run(capsys, 'cache')
    assert status == 0 and 'compiled modules' in out and str(tmp_path / 'c') in out
    status, out, _ = run(capsys, 'cache', 'path')
    assert out.strip() == str(tmp_path / 'c')
    status, _, err = run(capsys, 'cache', 'clear', '--modules')
    assert status == 0 and 'Removed 1 module' in err
    assert not (tmp_path / 'c' / 'abc.s').exists()


# ---------------------------------------------------------------------------------------------
# completion

def test_complete():
    root = build_tree()
    assert P.complete(root, ['r'])[0] == ['run']
    assert P.complete(root, ['co'])[0] == ['compile', 'completion']
    assert '--include' in P.complete(root, ['run', '--in'])[0]
    assert P.complete(root, ['run', '--version'])[0] == []  # global, root only
    assert P.complete(root, ['compile', '--emit', ''])[0] == ['pp', 'ir', 'asm']
    assert P.complete(root, ['run', ''])[1] == 'file'
    assert P.complete(root, ['run', '-I', ''])[1] == 'dir'
    assert P.complete(root, ['cache', ''])[0] == ['clear', 'info', 'path']
    assert P.complete(root, ['--color', ''])[0] == ['auto', 'always', 'never']


def test_completion_scripts(capsys):
    for shell in ('bash', 'zsh', 'fish'):
        status, out, _ = run(capsys, 'completion', shell)
        assert status == 0 and 'cimo __complete' in out
    status, _, err = run(capsys, 'completion', 'bsah')
    assert status == 2 and 'Did you mean bash?' in err


@pytest.mark.skipif(not os.path.exists('/bin/bash'), reason='needs bash')
def test_bash_completion(tmp_path):
    here = os.path.dirname(os.path.abspath(__file__))
    (tmp_path / 'cimo').write_text('#!/bin/sh\nPYTHONPATH=%s exec %s -m cimo "$@"\n'
                                   % (os.path.dirname(here), sys.executable))
    os.chmod(tmp_path / 'cimo', 0o755)
    (tmp_path / 'hello.c').write_text('')
    script = r'''
source <(cimo completion bash)
t() { COMP_WORDS=("$@"); COMP_CWORD=$(( ${#COMP_WORDS[@]} - 1 )); _cimo; echo "${COMPREPLY[*]}"; }
t cimo co
t cimo compile --emit a
t cimo run hel
'''
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, cwd=tmp_path,
                       env=dict(os.environ, PATH='%s:%s' % (tmp_path, os.environ['PATH'])))
    assert r.stdout.splitlines() == ['compile completion', 'asm', 'hello.c'], r.stderr


def test_python_m_cimo():
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, '-m', 'cimo', '--version'], capture_output=True,
                       text=True, cwd=os.path.dirname(here))
    assert r.returncode == 0 and r.stdout.startswith('cimo ')


def test_bundle_command(capsys, tmp_path):
    out = tmp_path / 'one.py'
    status, _, err = run(capsys, 'bundle', '-o', str(out))
    assert status == 0 and 'Wrote' in err and out.exists()
    r = subprocess.run([sys.executable, '-I', '-S', str(out), '--version'], capture_output=True,
                       text=True, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.startswith('cimo ')
