"""Third-party C test programs that check their own results, run by test_corpora.py:

  gcc-torture  gcc's gcc.c-torture/execute (and its ieee/): each program calls abort() when
               the compiler got something wrong, and exits 0 otherwise.
  c-testsuite  c-testsuite's single-exec tests: main returns 0, stdout matches .expected.

They are not copied into this repository (the torture tests are GPL); this script fetches
them at pinned commits with git (sparse, blobs of the needed directories only):

    python tests/corpora.py            # into python/tests/corpora/ (or $CIMO_CORPORA)
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get('CIMO_CORPORA') or os.path.join(HERE, 'corpora')

# name: (repository, commit, directory with the tests, glob patterns under it)
CORPORA = {
    'gcc-torture': ('https://github.com/gcc-mirror/gcc', '99126c684223cabab03e41e806616160b4058f73',
                    'gcc/testsuite/gcc.c-torture/execute', ['*.c', 'ieee/*.c']),
    'c-testsuite': ('https://github.com/c-testsuite/c-testsuite',
                    '5c7275656d751de0e68b2d340a95b5681858ed07', 'tests/single-exec', ['*.c']),
}


def root(name):
    """The directory of a corpus's tests, or None if it has not been fetched."""
    d = os.path.join(DIR, name, CORPORA[name][2])
    return d if os.path.isdir(d) else None


def files(name):
    """The corpus's test programs, as paths relative to root(name)."""
    import glob
    d = root(name)
    if d is None:
        return []
    out = []
    for pat in CORPORA[name][3]:
        out += [os.path.relpath(p, d).replace(os.sep, '/') for p in glob.glob(os.path.join(d, pat))]
    return sorted(out)


def fetch(name):
    url, commit, sub, _ = CORPORA[name]
    d = os.path.join(DIR, name)
    git = ['git', '-C', d]
    if not os.path.isdir(os.path.join(d, '.git')):
        os.makedirs(d, exist_ok=True)
        subprocess.check_call(git + ['init', '-q'])
        subprocess.check_call(git + ['remote', 'add', 'origin', url])
    subprocess.check_call(git + ['sparse-checkout', 'set', sub])
    subprocess.check_call(git + ['fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin',
                                 commit])
    subprocess.check_call(git + ['checkout', '-q', commit])
    print('%s: %d programs in %s' % (name, len(files(name)), root(name)))


if __name__ == '__main__':
    for n in sys.argv[1:] or CORPORA:
        fetch(n)
