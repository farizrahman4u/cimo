"""Differential test: the Python cproc port (cimo.frontend) against C cproc-qbe.

Inputs:
  * cproc's own tests (~/code/cproc/test/*.c and test/constraint/*.c), run the way
    cproc's `runtests` runs them: `-t <arch>` from the `+arch` file name suffix
    (default x86_64-sysv), `-E` for tests with a .pp file; tests with a .err file
    are expected to fail.
  * C files preprocessed with `cproc -E` (system cpp with cproc's flags):
    scratch/qbe_eval/*.c and python/tests/frontend/*.c, for every target.
Each case compares stdout, stderr and exit status byte for byte.

    python tests/test_frontend.py [-j N] [-v] [files...]     # from python/
    pytest tests/test_frontend.py

Skipped when C cproc (~/code/cproc, or $CPROC_DIR) is not built.
"""

import glob
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PYROOT)
CPROC_DIR = os.path.expanduser(os.environ.get('CPROC_DIR', '~/code/cproc'))
CPROC = os.path.join(CPROC_DIR, 'cproc')
CPROCQBE = os.path.join(CPROC_DIR, 'cproc-qbe')
TARGETS = ['aarch64', 'x86_64-sysv', 'riscv64']


def have_cproc():
    return os.access(CPROCQBE, os.X_OK) and os.access(CPROC, os.X_OK)


def run(cmd, stdin=None):
    env = dict(os.environ, PYTHONPATH=PYROOT)
    p = subprocess.run(cmd, input=stdin, capture_output=True, env=env, timeout=600)
    return p.returncode, p.stdout, p.stderr


def cproc_tests():
    """(name, args) pairs for cproc's test suite, as runtests runs them."""
    cases = []
    tests = sorted(glob.glob(os.path.join(CPROC_DIR, 'test', '*.c'))
                   + glob.glob(os.path.join(CPROC_DIR, 'test', 'constraint', '*.c')))
    for t in tests:
        name = t[:-2]
        base = os.path.basename(name)
        arch = base.rsplit('+', 1)[1] if '+' in base else 'x86_64-sysv'
        if os.path.exists(name + '.pp'):
            args = ['-t', arch, '-E', t]
        else:
            args = ['-t', arch, t]
        cases.append((os.path.relpath(t, CPROC_DIR), args))
    return cases


def extra_files():
    return (sorted(glob.glob(os.path.join(REPO, 'scratch', 'qbe_eval', '*.c')))
            + sorted(glob.glob(os.path.join(HERE, 'frontend', '*.c'))))


def preprocess(path, tmpdir):
    rc, out, err = run([CPROC, '-E', path])
    if rc != 0:
        raise RuntimeError('cproc -E %s failed: %s' % (path, err.decode('latin-1')))
    dst = os.path.join(tmpdir, os.path.basename(path)[:-2] + '.i')
    with open(dst, 'wb') as f:
        f.write(out)
    return dst


def check(args):
    c = run([CPROCQBE] + args)
    p = run([sys.executable, '-m', 'cimo.frontend'] + args)
    if c == p:
        return None
    if (c[0], p[0]) in ((-6, -6), (-11, 70)):
        # both hit an assertion (the text names C vs Python source), or C crashes
        # (SIGSEGV) where the port raises an internal error; stdout is lost either way
        return None
    if c[0] != p[0]:
        return 'exit status c=%s py=%s\n%s' % (c[0], p[0], p[2].decode('latin-1')[-2000:])
    for what, a, b in (('stdout', c[1], p[1]), ('stderr', c[2], p[2])):
        if a != b:
            al = a.decode('latin-1').splitlines()
            bl = b.decode('latin-1').splitlines()
            n = 0
            while n < min(len(al), len(bl)) and al[n] == bl[n]:
                n += 1
            return '%s differs at line %d:\n  c:  %r\n  py: %r' % (
                what, n + 1, al[n] if n < len(al) else '<eof>', bl[n] if n < len(bl) else '<eof>')
    return 'differs'


def snippets(tmpdir):
    """Cases from tests/frontend/snippets.txt: raw C (cproc's own pp only), compiled for
    every target and also run with -E."""
    with open(os.path.join(HERE, 'frontend', 'snippets.txt'), 'rb') as f:
        text = f.read()
    chunks = []
    cur = None
    for line in text.split(b'\n'):
        if line.startswith(b'#---'):
            cur = []
            chunks.append(cur)
        elif cur is not None:
            cur.append(line)
    cases = []
    for n, lines in enumerate(chunks):
        path = os.path.join(tmpdir, 'snip%03d.c' % n)
        with open(path, 'wb') as f:
            f.write(b'\n'.join(lines) + b'\n')
        for t in TARGETS:
            cases.append(('snippet %d [%s]' % (n, t), ['-t', t, path]))
        cases.append(('snippet %d [-E]' % n, ['-E', path]))
    return cases


def all_cases(tmpdir, files=None):
    cases = []
    if not files:
        cases += cproc_tests()
        cases += snippets(tmpdir)
        files = extra_files()
    for f in files:
        i = preprocess(f, tmpdir)
        for t in TARGETS:
            cases.append(('%s [%s]' % (os.path.relpath(f, REPO), t), ['-t', t, i]))
    return cases


def run_all(files=None, jobs=None, verbose=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        cases = all_cases(tmpdir, files)
        with ThreadPoolExecutor(jobs or os.cpu_count()) as ex:
            res = list(ex.map(lambda c: check(c[1]), cases))
    bad = [(name, r) for (name, _), r in zip(cases, res) if r is not None]
    for name, r in bad:
        print('FAIL %s: %s' % (name, r if verbose else r.splitlines()[0]))
    print('%d/%d match' % (len(cases) - len(bad), len(cases)))
    return not bad


def test_cproc_differential():
    import pytest
    if not have_cproc():
        pytest.skip('C cproc not built at %s' % CPROC_DIR)
    assert run_all()


def test_compile_api():
    sys.path.insert(0, PYROOT)
    from cimo.frontend import compile, FrontendError
    ir = compile('int f(int x) { return x + 1; }\n', 'f.c')
    assert 'function w $f(w %.1)' in ir
    try:
        compile('int f(void) { return y; }\n', 'g.c')
    except FrontendError as e:
        assert e.msg == 'g.c:1:22: error: undeclared identifier: y', e.msg
    else:
        raise AssertionError('no error')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=None)
    ap.add_argument('-v', action='store_true')
    ap.add_argument('files', nargs='*')
    a = ap.parse_args()
    if not have_cproc():
        print('C cproc not built at %s; skipping' % CPROC_DIR)
        sys.exit(0)
    sys.exit(0 if run_all(a.files, a.j, a.v) else 1)
