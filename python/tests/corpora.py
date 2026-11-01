"""Third-party C test programs that check their own results, run by test_corpora.py:

  gcc-torture  gcc's gcc.c-torture/execute (and its ieee/): each program calls abort() when
               the compiler got something wrong, and exits 0 otherwise.
  c-testsuite  c-testsuite's single-exec tests: main returns 0, stdout matches .expected.
  libc-test    musl's libc-test: math/ (results against tables of correctly rounded values,
               within each table's ulp bound) and functional/ (string, stdio, stdlib, ...),
               which test cimo's runtime library and the compiler compiling it. A harness
               (generated next to it, see harness()) stubs fenv.h: cimo has no rounding modes
               or exception flags, so only round-to-nearest results are checked.

They are not copied into this repository (the torture tests are GPL); this script fetches
them at pinned commits with git (sparse, blobs of the needed directories only):

    python tests/corpora.py            # into python/tests/corpora/ (or $CIMO_CORPORA)
"""

import os
import subprocess
import sys
import threading

_lock = threading.Lock()
HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get('CIMO_CORPORA') or os.path.join(HERE, 'corpora')

# name: (repository, commit, directory with the tests, glob patterns under it)
CORPORA = {
    'gcc-torture': ('https://github.com/gcc-mirror/gcc', '99126c684223cabab03e41e806616160b4058f73',
                    'gcc/testsuite/gcc.c-torture/execute', ['*.c', 'ieee/*.c']),
    'c-testsuite': ('https://github.com/c-testsuite/c-testsuite',
                    '5c7275656d751de0e68b2d340a95b5681858ed07', 'tests/single-exec', ['*.c']),
    'libc-test': ('https://repo.or.cz/libc-test.git', '7b95dfa5f5d5ca4d949221e0228ccc290bacc14e',
                  'src', ['math/*.c', 'functional/*.c']),
}

_FENV = '''/* cimo has no floating-point environment (no rounding modes, no exception flags): without
   the FE_ macros, libc-test skips the other rounding modes and does not check exceptions */
#ifndef CIMO_FENV_H
#define CIMO_FENV_H
static int fesetround(int r) { return r ? -1 : 0; }
static int feclearexcept(int e) { return 0; }
static int fetestexcept(int e) { return 0; }
#define FE_ALL_EXCEPT 0
#endif
'''

_PRELUDE = '''#include <stdio.h>
#include <stdarg.h>
#include "test.h"
volatile int t_status = 0;
int t_printf(const char *s, ...)
{
	va_list ap;
	char buf[512];
	int n;
	t_status = 1;
	va_start(ap, s);
	n = vsnprintf(buf, sizeof buf, s, ap);
	va_end(ap);
	printf("%s", buf);
	return n;
}
'''


def harness(name, rel):
    """(prelude file compiled in front of program rel, include directories), or None: libc-test's
    common code, adapted (generated from the fetched sources: no long double, no unistd.h)."""
    if name != 'libc-test' or root(name) is None:
        return None
    d = os.path.join(DIR, 'libc-test-harness')
    src = os.path.join(root(name), 'common')
    with _lock:
        if not os.path.isdir(d):
            _make_harness(d, src)
    return os.path.join(d, rel.split('/')[0] + '.c'), [d]


def _make_harness(d, src):
    import re
    os.makedirs(d)
    with open(os.path.join(src, 'mtest.h')) as f:  # long double structs and ulperrl out
        mh = ''.join(l.replace('long double', 'double') for l in f
                     if not ('long double' in l and l.startswith(('struct', 'float'))))
    mh = '#ifndef CIMO_MTEST_H\n#define CIMO_MTEST_H\n%s#endif\n' % mh
    with open(os.path.join(src, 'test.h')) as f:
        th = f.read().replace('#include <unistd.h>\n', '')
    th = '#ifndef CIMO_TEST_H\n#define CIMO_TEST_H\n%s#endif\n' % th
    with open(os.path.join(src, 'mtest.c')) as f:
        mc = re.sub(r'\n(int eulpl|float ulperrl)\(.*?\n}\n', '\n', f.read(), flags=re.S)
    with open(os.path.join(src, 'rand.c')) as f:
        rc = re.sub(r'\nstatic long double frandl\(\).*?\n}\n', '\n', f.read(), flags=re.S)
    for n, text in (('mtest.h', mh), ('test.h', th), ('mtest.c', mc), ('fenv.h', _FENV),
                    ('rand.c', rc), ('math.c', _PRELUDE + '#include "mtest.c"\n'),
                    ('functional.c', _PRELUDE + '#include "rand.c"\n')):
        with open(os.path.join(d, n), 'w') as f:
            f.write(text)


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
    # keep the files byte for byte (expected outputs are compared exactly; Git for Windows
    # converts to CRLF by default)
    subprocess.check_call(git + ['config', 'core.autocrlf', 'false'])
    subprocess.check_call(git + ['sparse-checkout', 'set', sub])
    subprocess.check_call(git + ['fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin',
                                 commit])
    subprocess.check_call(git + ['checkout', '-q', commit])
    harness(name, 'math/')  # generate it now (libc-test)
    print('%s: %d programs in %s' % (name, len(files(name)), root(name)))


if __name__ == '__main__':
    for n in sys.argv[1:] or CORPORA:
        fetch(n)
