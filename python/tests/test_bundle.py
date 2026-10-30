"""The single-file bundle (cimo.bundle): one cimo.py that works like the package in a bare
interpreter (`python -I -S`: no site-packages, so the installed cimo is not importable)."""

import ast
import json
import os
import subprocess
import sys

import pytest

from cimo import bundle, linker

pytestmark = pytest.mark.skipif(
    __import__('platform').machine().lower() not in ('arm64', 'aarch64'),
    reason='cimo generates arm64 code only')

KERNEL = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double kernel(int n) {
    double v[n];  /* a VLA */
    double *h = malloc(n * sizeof *h);
    double s = 0;
    for (int i = 0; i < n; i++) {
        v[i] = sin(i);
        h[i] = 2 * v[i];
        s += h[i];
    }
    printf("n=%d last=%.6f\n", n, v[n - 1]);
    free(h);
    return s;
}
'''

# Run in the subprocess, with the bundle's directory as argv[1].
SCRIPT = r'''
import importlib.util, json, sys
assert importlib.util.find_spec('cimo') is None, 'the installed cimo is importable'
sys.path.insert(0, sys.argv[1])
import cimo
from cimo import bundle, linker, resources
m = cimo.compile(sys.argv[2])
s = m.kernel(10)
try:
    cimo.compile('int f(void) { return y; }')
except cimo.CompileError:
    import traceback
    tb = traceback.format_exc()
out = {'sum': s, 'file': cimo.__file__, 'root': resources.ROOT,
       'driver': cimo.driver.__file__, 'traceback': tb,
       'compiler_key': linker.compiler_key(), 'runtime_key': linker.runtime().key}
if len(sys.argv) > 3:
    bundle.build(sys.argv[3])  # the bundle rebuilds itself
print('RESULT ' + json.dumps(out))
'''


def _run(directory, env, rebuild=None):
    args = [sys.executable, '-I', '-S', '-c', SCRIPT, str(directory), KERNEL]
    if rebuild:
        args.append(str(rebuild))
    r = subprocess.run(args, cwd=str(directory), env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == 'n=10 last=0.412118', r.stdout
    return json.loads(lines[-1][len('RESULT '):]), r.stderr


def _bare_env(home, cache):
    env = {'HOME': str(home), 'CIMO_CACHE_DIR': str(cache)}
    if 'PATH' in os.environ:
        env['PATH'] = os.environ['PATH']
    return env


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp('bundle')
    plain, pre = d / 'plain', d / 'prebuilt'
    plain.mkdir()
    pre.mkdir()
    stats = bundle.build(str(plain / 'cimo.py'))
    stats_pre = bundle.build(str(pre / 'cimo.py'), prebuilt=True)
    return d, stats, stats_pre


def test_one_file_deterministic(built, tmp_path):
    d, stats, stats_pre = built
    assert set(os.listdir(str(d / 'plain'))) - {'__pycache__'} == {'cimo.py'}
    assert stats['modules'] > 50 and stats['resources'] > 200 and stats['runtime_files'] == 0
    assert stats_pre['runtime_files'] == len(linker.runtime().index) + 1  # + index.json
    again = tmp_path / 'cimo.py'
    bundle.build(str(again))
    assert again.read_bytes() == (d / 'plain' / 'cimo.py').read_bytes()
    tree = ast.parse((d / 'prebuilt' / 'cimo.py').read_text(encoding='utf-8'))
    files = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and n.targets[0].id == '_FILES')
    assert 'include/stdio.h' in files and 'runtime/math/sin.c' in files
    assert 'driver.py' in files and 'backend/arm64/isel.py' in files
    assert not any('__pycache__' in k or k.startswith('tests/') for k in files)
    key = linker.runtime().key
    assert json.loads(files['prebuilt/runtime-%s/index.json' % key]) == linker.runtime().index


def test_bundle_without_cache_dir(built, tmp_path):
    """No writable cache: the runtime is built and the module cached in memory."""
    d = built[0] / 'plain'
    blocker = tmp_path / 'file'  # directories under a file cannot be created
    blocker.write_text('')
    out, err = _run(d, _bare_env(blocker / 'home', blocker / 'cache'),
                    rebuild=tmp_path / 'rebuilt.py')
    assert 'building the runtime library' in err
    assert out['file'] == str(d / 'cimo.py')
    assert out['root'] == '<cimo.py>/cimo' and out['driver'] == '<cimo.py>/cimo/driver.py'
    assert 'File "<cimo.py>/cimo/driver.py", line' in out['traceback']
    assert out['compiler_key'] == linker.compiler_key()
    assert out['runtime_key'] == linker.runtime().key
    assert abs(out['sum'] - 2 * sum(__import__('math').sin(i) for i in range(10))) < 1e-12
    assert (tmp_path / 'rebuilt.py').read_bytes() == (d / 'cimo.py').read_bytes()


def test_bundle_prebuilt_read_only(built, tmp_path):
    """--prebuilt: the runtime is never built, and nothing needs to be written."""
    home = tmp_path / 'home'
    home.mkdir()
    blocker = tmp_path / 'file'
    blocker.write_text('')
    os.chmod(str(home), 0o555)
    try:
        out, err = _run(built[0] / 'prebuilt', _bare_env(home, blocker / 'cache'))
    finally:
        os.chmod(str(home), 0o755)
    assert 'building the runtime' not in err, err
    assert os.listdir(str(home)) == []
    assert out['runtime_key'] == linker.runtime().key


def test_strip_keeps_code_and_notices():
    """Stripped sources: same Python AST (minus docstrings), same C tokens; license comments
    stay. Every file the bundle embeds is checked."""
    import ast
    from cimo import bundle, resources
    from cimo.preprocessor import lex, FileInfo
    files = bundle._package_files()
    for rel, text in sorted(files.items()):
        if rel.endswith('.py'):
            out = bundle.strip_py(text)  # raises if the AST changes
            assert bundle.strip_py(out) == out, rel  # idempotent: a bundle rebuilds itself
        elif rel.endswith(('.c', '.h')):
            out = bundle.strip_c(text)
            key = lambda t: (t.kind, t.text, t.bol, t.space)
            assert [key(t) for t in lex(out, FileInfo(rel))] == \
                [key(t) for t in lex(text, FileInfo(rel))], rel
            assert bundle.strip_c(out) == out, rel
            for m in bundle._C_TOKEN.finditer(text):
                c = m.group('block') or m.group('line')
                if c and bundle._NOTICE.search(c):
                    assert c in out, rel
    # docstrings and comments are gone, strings that look like them are not
    src = ('"""doc"""\n# comment\nX = """\n\n  kept\n"""  # trailing\n'
           'def f():\n    """only a docstring"""\n\nclass C:\n    "doc"\n    y = 1\n')
    out = bundle.strip_py(src)
    assert 'doc' not in out and 'comment' not in out and 'trailing' not in out
    assert 'X = """\n\n  kept\n"""' in out and 'def f():\n    pass' in out
    c = '/* Copyright (C) 1993 Sun */\nint a; /* gone */ // gone\n#define S "/* not a comment */"\n'
    out = bundle.strip_c(c)
    assert 'Copyright' in out and 'gone' not in out and '"/* not a comment */"' in out
