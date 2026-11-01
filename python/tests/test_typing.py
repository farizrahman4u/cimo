"""The user-facing API is typed (py.typed): mypy checks the modules listed in pyproject.toml,
and a small program using the API type-checks (skipped without mypy, a dev dependency)."""

import os
import subprocess
import sys

import pytest

pytest.importorskip('mypy')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def mypy(*args):
    return subprocess.run([sys.executable, '-m', 'mypy', '--no-incremental', *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_api_modules_typecheck():
    p = mypy()
    assert p.returncode == 0, p.stdout + p.stderr


def test_user_code_typechecks(tmp_path):
    good = tmp_path / 'good.py'
    good.write_text('''
import cimo
m = cimo.compile("long f(long x) { return x; }", defines={"N": 3}, cache=False)
r = m.f(3)
n: int = m.heap_bytes()
try:
    cimo.load("k.c", include_dirs=["inc"], link="libc")
except cimo.KernelError as e:
    s: str | None = e.kind
except cimo.CompileError:
    pass
''')
    bad = tmp_path / 'bad.py'
    bad.write_text('import cimo\ncimo.compile(42)\ncimo.compile("", cache="no")\n')
    p = mypy('--follow-imports=silent', str(good))
    assert p.returncode == 0, p.stdout
    p = mypy('--follow-imports=silent', str(bad))
    assert p.returncode != 0 and p.stdout.count('error:') == 2, p.stdout
