"""Differential checks of cimo.preprocessor against GNU cpp and C cproc (used by test_preprocessor.py).

Run directly for a report over the whole corpus:  python3 python/tests/preprocessor_compare.py
"""

import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cimo import preprocessor  # noqa: E402

CPROC_DIR = os.environ.get('CIMO_CPROC_DIR', os.path.expanduser('~/code/cproc'))
REPO = os.path.dirname(os.path.dirname(HERE))
CASES = os.path.join(HERE, 'preprocessor_cases')


def gcc_flags(target='arm64'):
    """`cpp` flags equivalent to cimo.preprocessor's predefines (cproc's config.h list)."""
    flags = ['-undef', '-ffreestanding', '-U__STDC_VERSION__', '-D__STDC_VERSION__=201112L']
    for name, val in preprocessor.TARGETS[target]:
        flags.append('-D%s' % name if val is None else '-D%s=%s' % (name, val))
    return flags


def corpus():
    """C files for the comparisons: cproc's tests, scratch/qbe_eval, our edge cases."""
    files = sorted(glob.glob(os.path.join(CPROC_DIR, 'test', '*.c')))
    files += sorted(glob.glob(os.path.join(CPROC_DIR, 'test', 'constraint', '*.c')))
    files += sorted(glob.glob(os.path.join(REPO, 'scratch', 'qbe_eval', '*.c')))
    files += sorted(glob.glob(os.path.join(CASES, '*.c')))
    return files


_MARKER = re.compile(r'^\s*#\s*(?:line\s+)?\d+(?:\s+"(?:[^"\\]|\\.)*")?(?:\s+\d+)*\s*$')
_PRAGMA = re.compile(r'^\s*#\s*pragma\b(.*)$')


def tokens(text):
    """The pp-token values of preprocessor output, minus line markers and non-omp pragmas."""
    lines = []
    for line in text.split('\n'):
        if _MARKER.match(line):
            continue
        m = _PRAGMA.match(line)
        if m and not m.group(1).split()[:1] == ['omp']:
            continue
        lines.append(line)
    return [t.value for t in preprocessor.lex('\n'.join(lines), preprocessor.FileInfo('<out>'))]


def run_gcc(path, include_dirs=(), target='arm64'):
    cmd = ['cpp'] + gcc_flags(target) + ['-I%s' % d for d in include_dirs] + [path]
    p = subprocess.run(cmd, capture_output=True, text=True, errors='surrogateescape')
    return p.returncode, p.stdout, p.stderr


def run_ours(path, include_dirs=(), target='arm64'):
    with open(path, 'rb') as f:
        src = f.read().decode('utf-8', 'surrogateescape')
    try:
        return 0, preprocessor.preprocess(src, path, include_dirs, target=target, warn=lambda m: None), ''
    except preprocessor.PreprocessError as e:
        return 1, '', str(e)


_SYS_INCLUDE = re.compile(r'^\s*#\s*include\s*<', re.M)


def uses_system_headers(path):
    with open(path, 'rb') as f:
        return _SYS_INCLUDE.search(f.read().decode('utf-8', 'replace')) is not None


def missing_headers(path):
    """System headers `path` includes that cimo does not bundle."""
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', 'replace')
    names = re.findall(r'^\s*#\s*include\s*<([^>]+)>', text, re.M)
    return [n for n in names if not os.path.exists(os.path.join(preprocessor.BUNDLED_INCLUDE, n))]


def compare_tokens(path, include_dirs=(CASES,)):
    """Returns None if equal (or both fail), else a description of the first difference."""
    rc1, out1, err1 = run_gcc(path, include_dirs)
    rc2, out2, err2 = run_ours(path, include_dirs)
    if rc1 != 0 or rc2 != 0:
        if (rc1 != 0) == (rc2 != 0):
            return None
        return 'gcc rc=%d ours rc=%d: %s%s' % (rc1, rc2, err1.strip()[:300], err2.strip()[:300])
    t1 = tokens(out1)
    t2 = tokens(out2)
    if t1 == t2:
        return None
    i = 0
    while i < min(len(t1), len(t2)) and t1[i] == t2[i]:
        i += 1
    return 'token %d: gcc %r vs ours %r' % (i, t1[max(0, i - 3):i + 5], t2[max(0, i - 3):i + 5])


def system_include_dirs():
    """GNU cpp's <...> search list (as `cproc -E` sees it)."""
    p = subprocess.run(['cpp', '-v', '/dev/null'], capture_output=True, text=True)
    lines = p.stderr.splitlines()
    try:
        i = lines.index('#include <...> search starts here:')
        j = lines.index('End of search list.')
    except ValueError:
        return []
    return [line.strip() for line in lines[i + 1:j]]


def cproc_available():
    return os.path.exists(os.path.join(CPROC_DIR, 'cproc-qbe')) and \
        os.path.exists(os.path.join(CPROC_DIR, 'cproc')) and shutil.which('cpp') is not None


def qbe_ir(pp_text, target='aarch64'):
    """Run C cproc-qbe on preprocessed text; return (rc, stdout, stderr)."""
    p = subprocess.run([os.path.join(CPROC_DIR, 'cproc-qbe'), '-t', target],
                       input=pp_text, capture_output=True, text=True, errors='surrogateescape')
    return p.returncode, p.stdout, p.stderr


_CIMO_FUNC = re.compile(r'^(?:export\n)?function [^\n]*\$__cimo_\w+\(.*?^}\n', re.M | re.S)


def strip_header_functions(ir):
    """Drop the static inline helpers of the bundled math.h (cproc emits unused statics)."""
    return _CIMO_FUNC.sub('', ir)


def renumber_labels(ir):
    """cproc numbers block labels and string literals across the whole file (header functions
    take numbers too); renumber both in order of appearance."""
    seen = {}

    def sub(m):
        return '%s%s.%d' % (m.group(1), m.group(2), seen.setdefault(m.group(0), len(seen) + 1))
    return re.sub(r'(@|\$\.L)(\w+)\.(\d+)', sub, ir)


def compare_ir(path, include_dirs=(CASES,), strip=True):
    """cproc-qbe output on `cproc -E` vs on cimo.preprocessor output. None if identical.

    Files that include system headers get glibc's headers on the cproc side and the bundled ones
    on ours; the IR must still match once the bundled math.h helpers are removed."""
    cmd = [os.path.join(CPROC_DIR, 'cproc'), '-E'] + ['-I%s' % d for d in include_dirs] + [path]
    p = subprocess.run(cmd, capture_output=True, text=True, errors='surrogateescape')
    rc2, out2, err2 = run_ours(path, include_dirs)
    if p.returncode != 0 or rc2 != 0:
        if (p.returncode != 0) == (rc2 != 0):
            return None
        return 'cpp rc=%d ours rc=%d: %s%s' % (p.returncode, rc2, p.stderr[:300], err2[:300])
    a = qbe_ir(p.stdout)
    b = qbe_ir(out2)
    if strip and uses_system_headers(path):
        a = (a[0], renumber_labels(a[1]), a[2])
        b = (b[0], renumber_labels(strip_header_functions(b[1])), b[2])
    if a == b:
        return None
    if a[1] != b[1]:
        return 'IR differs (rc %d vs %d)' % (a[0], b[0])
    return 'diagnostics differ: %r vs %r' % (a[2][:300], b[2][:300])


# --- Random macro programs ---------------------------------------------------------------------

_FUZZ_NAMES = ['A', 'B', 'C', 'f', 'g', 'h', 'k']


def _fuzz_body(rng, params, variadic):
    atoms = _FUZZ_NAMES + ['x1', '1', '+', '(', ')', ',', '-', 'y'] + list(params) * 3
    if variadic:
        atoms += ['__VA_ARGS__'] * 2
    out = []
    for _ in range(rng.randint(0, 6)):
        r = rng.random()
        if params and r < 0.1:
            out.append('#' + rng.choice(params))
        elif out and r < 0.2:
            out.append('##')
            out.append(rng.choice(atoms))
        else:
            out.append(rng.choice(atoms))
    return ' '.join(out)


def fuzz_program(rng):
    """A random set of mutually referring macros and some text invoking them."""
    lines = []
    for n in _FUZZ_NAMES:
        if rng.random() < 0.8:
            if rng.random() < 0.6:
                params = ['p%d' % i for i in range(rng.randint(0, 2))]
                variadic = rng.random() < 0.2
                plist = ', '.join(params + (['...'] if variadic else []))
                lines.append('#define %s(%s) %s' % (n, plist, _fuzz_body(rng, params, variadic)))
            else:
                lines.append('#define %s %s' % (n, _fuzz_body(rng, [], False)))
    atoms = _FUZZ_NAMES + ['(', ')', ',', '1', 'z', '(', ')']
    for _ in range(3):
        lines.append(' '.join(rng.choice(atoms) for _ in range(rng.randint(1, 12))))
    return '\n'.join(lines) + '\n'


def fuzz(seed, count, workdir):
    """Compare random macro programs with GNU cpp. Returns (failures, valid_programs)."""
    import random
    rng = random.Random(seed)
    path = os.path.join(workdir, 'fuzz.c')
    failures = []
    valid = 0
    for _ in range(count):
        src = fuzz_program(rng)
        with open(path, 'w') as f:
            f.write(src)
        g = subprocess.run(['cpp', '-undef', '-P', path], capture_output=True, text=True)
        gerr = g.returncode != 0 or 'error' in g.stderr
        try:
            out = preprocessor.preprocess(src, path, warn=lambda m: None)
            oerr = False
        except preprocessor.PreprocessError:
            out = ''
            oerr = True
        valid += not gerr
        if gerr != oerr or (not gerr and tokens(g.stdout) != tokens(out)):
            failures.append(src)
    return failures, valid


def main():
    bad = 0
    files = corpus()
    for f in files:
        missing = missing_headers(f)
        if missing:
            # Not buildable with the bundled headers: check the preprocessor on glibc's instead.
            dirs = system_include_dirs()
            r = compare_tokens(f, dirs)
            if r:
                bad += 1
                print('TOKENS %s (system headers): %s' % (os.path.relpath(f), r))
            if cproc_available():
                r = compare_ir(f, dirs, strip=False)
                if r:
                    bad += 1
                    print('IR     %s (system headers): %s' % (os.path.relpath(f), r))
            continue
        r = None if uses_system_headers(f) else compare_tokens(f)
        if r:
            bad += 1
            print('TOKENS %s: %s' % (os.path.relpath(f), r))
        if cproc_available():
            r = compare_ir(f)
            if r:
                bad += 1
                print('IR     %s: %s' % (os.path.relpath(f), r))
    print('%d files, %d differences' % (len(files), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
