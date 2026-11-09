"""Third-party C test programs that check their own results, run by test_corpora.py:

  gcc-torture  gcc's gcc.c-torture/execute (and its ieee/): each program calls abort() when
               the compiler got something wrong, and exits 0 otherwise.
  c-testsuite  c-testsuite's single-exec tests: main returns 0, stdout matches .expected.
  gcc-vect     gcc's gcc.dg/vect run tests (the vectorizer's: loops whose results the program checks,
               abort() on a wrong one), with tree-vect.h's check_vect (signals, cpuid) made a no-op.
  gcc-dg-torture  gcc's gcc.dg/torture run tests (`dg-do run`).
  gcc-dg       gcc's gcc.dg run tests (`dg-do run`, the top directory; its subdirectories are mostly
               compile-only tests of gcc's own dumps).
  gcc-torture-compile  gcc's gcc.c-torture/compile: compiled and assembled, not linked or run (crashes,
               assertions and assembler errors in the compiler; a program cimo rejects is listed).
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
import re
import subprocess
import sys
import threading

_lock = threading.Lock()
HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get('CIMO_CORPORA') or os.path.join(HERE, 'corpora')

GCC = ('https://github.com/gcc-mirror/gcc', '99126c684223cabab03e41e806616160b4058f73')

# name: (repository, commit, directory with the tests, glob patterns under it[, which files are run
# tests: 'dg-run' for those saying `dg-do run`, 'dg-default' for all but compile-only ones, 'compile' for every
# file, compiled only])
CORPORA = {
    'gcc-torture': GCC + ('gcc/testsuite/gcc.c-torture/execute', ['*.c', 'ieee/*.c']),
    'gcc-vect': GCC + ('gcc/testsuite/gcc.dg/vect', ['*.c'], 'dg-default'),
    'gcc-dg-torture': GCC + ('gcc/testsuite/gcc.dg/torture', ['*.c'], 'dg-run'),
    'gcc-dg': GCC + ('gcc/testsuite/gcc.dg', ['*.c'], 'dg-run'),
    'gcc-torture-compile': GCC + ('gcc/testsuite/gcc.c-torture/compile', ['*.c'], 'compile'),
    'c-testsuite': ('https://github.com/c-testsuite/c-testsuite',
                    '5c7275656d751de0e68b2d340a95b5681858ed07', 'tests/single-exec', ['*.c']),
    'libc-test': ('https://repo.or.cz/libc-test.git', '7b95dfa5f5d5ca4d949221e0228ccc290bacc14e',
                  'src', ['math/*.c', 'functional/*.c']),
}

_FENV = '''/* cimo's rounding modes, but not its exception flags: the runtime's math is pure (musl's
   FORCE_EVAL is a no-op), so the flags it raises are not musl's; without FE_ALL_EXCEPT, libc-test
   checks the results in every rounding mode and not the exceptions */
#ifndef CIMO_FENV_H
#define CIMO_FENV_H
#ifdef __x86_64__
#define FE_TONEAREST 0
#define FE_DOWNWARD 0x400
#define FE_UPWARD 0x800
#define FE_TOWARDZERO 0xc00
#else
#define FE_TONEAREST 0
#define FE_UPWARD 0x400000
#define FE_DOWNWARD 0x800000
#define FE_TOWARDZERO 0xc00000
#endif
int fesetround(int);
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


_NORUN = re.compile(rb'dg-do (compile|assemble|link|preprocess)')
_MAIN = re.compile(rb'\bmain\s*\(')
# a second source file, or code in GIMPLE or RTL (gcc's own IR, behind -fgimple)
_NOTC = re.compile(rb'dg-additional-sources|__GIMPLE|__RTL|-fgimple')


def _is_run(path, kind):
    with open(path, 'rb') as f:
        text = f.read()
    if _NOTC.search(text):
        return False
    if kind == 'compile':
        return True
    if not _MAIN.search(text):
        return False
    if kind == 'dg-run':
        return b'dg-do run' in text
    return not _NORUN.search(text)


def files(name):
    """The corpus's test programs, as paths relative to root(name)."""
    import glob
    d = root(name)
    if d is None:
        return []
    kind = CORPORA[name][4] if len(CORPORA[name]) > 4 else None
    out = []
    for pat in CORPORA[name][3]:
        out += [os.path.relpath(p, d).replace(os.sep, '/') for p in glob.glob(os.path.join(d, pat))
                if kind is None or _is_run(p, kind)]
    return sorted(out)


_CHECK_VECT = '''/* cimo's test_corpora.py: gcc's tree-vect.h with check_vect a no-op (it probes the CPU with signals,
   cpuid and inline assembly; cimo's vector code needs nothing the host may lack). Generated by corpora.py. */
extern void abort (void);
extern void exit (int);
static void check_vect (void) {}
#define VECTOR_BITS 128
#undef OFFSET
'''


def compile_only(name):
    return len(CORPORA[name]) > 4 and CORPORA[name][4] == 'compile'


def fetch(name):
    url, commit, sub = CORPORA[name][:3]
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
    if subprocess.call(git + ['fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', commit]):
        # repo.or.cz serves no commit by its hash, or partial clones: the whole history there
        subprocess.check_call(git + ['fetch', '-q', 'origin'])
    subprocess.check_call(git + ['checkout', '-q', '-f', commit])
    if name == 'gcc-vect':
        with open(os.path.join(root(name), 'tree-vect.h'), 'w', newline='\n') as f:
            f.write(_CHECK_VECT)
    harness(name, 'math/')  # generate it now (libc-test)
    print('%s: %d programs in %s' % (name, len(files(name)), root(name)))


if __name__ == '__main__':
    for n in sys.argv[1:] or CORPORA:
        fetch(n)
