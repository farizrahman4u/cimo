"""Differential test: the Python QBE port against C QBE.

For each input and each debug flag (and the final assembly), runs both and
compares stdout, stderr and exit status.

    python tests/backend_diff.py [-f FLAGS] [-t target] [-j N] [-v] [files...]

FLAGS is a string of QBE -d flags to check one at a time, plus 'S' style
letters; '.' means the plain assembly output. Default: "PMNCGAILSR.".
Default files: ~/code/qbe/test/*.ssa and scratch/qbe_eval/build/kernels.ssa.
"""

import argparse
import difflib
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PYROOT)
CQBE = os.path.expanduser(os.environ.get('CQBE', '~/code/qbe/qbe'))


def run(cmd, timeout=300):
    env = dict(os.environ, PYTHONPATH=PYROOT)
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        return p.returncode, p.stdout.decode('latin-1'), p.stderr.decode('latin-1')
    except subprocess.TimeoutExpired:
        return 'timeout', '', ''


def check(path, flag, target):
    args = ['-t', target] + (['-d', flag] if flag != '.' else []) + [path]
    c = run([CQBE] + args)
    p = run([sys.executable, '-m', 'cimo.backend'] + args)
    if c == p:
        return None
    # describe the first difference
    for name, a, b in (('stdout', c[1], p[1]), ('stderr', c[2], p[2])):
        if a != b:
            al = a.splitlines()
            bl = b.splitlines()
            n = 0
            while n < min(len(al), len(bl)) and al[n] == bl[n]:
                n += 1
            ctx = list(difflib.unified_diff(al[max(0, n - 3):n + 5], bl[max(0, n - 3):n + 5],
                                            'c', 'py', lineterm='', n=3))
            tb = ''
            if 'Traceback' in p[2]:
                tb = p[2][p[2].index('Traceback'):][-1500:]
            return '%s line %d\n%s\n%s' % (name, n + 1, '\n'.join(ctx), tb)
    return 'exit status: c=%s py=%s\n%s' % (c[0], p[0], p[2][-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--flags', default='PMNCGAILSR.')
    ap.add_argument('-t', '--target', default='arm64')
    ap.add_argument('-j', type=int, default=os.cpu_count())
    ap.add_argument('-v', action='store_true', help='print each failure in full')
    ap.add_argument('files', nargs='*')
    a = ap.parse_args()
    files = a.files or (sorted(glob.glob(os.path.expanduser('~/code/qbe/test/*.ssa')))
                        + [os.path.join(REPO, 'scratch/qbe_eval/build/kernels.ssa')])
    jobs = [(f, fl) for fl in a.flags for f in files]
    with ThreadPoolExecutor(a.j) as ex:
        res = list(ex.map(lambda j: check(j[0], j[1], a.target), jobs))
    bad = 0
    perflag = {}
    for (f, fl), r in zip(jobs, res):
        ok, tot = perflag.get(fl, (0, 0))
        perflag[fl] = (ok + (r is None), tot + 1)
        if r is not None:
            bad += 1
            if a.v:
                print('FAIL %s -d %s: %s\n' % (os.path.basename(f), fl, r))
            else:
                print('FAIL %s -d %s: %s' % (os.path.basename(f), fl, r.splitlines()[0]))
    print(' '.join('%s:%d/%d' % (fl, ok, tot) for fl, (ok, tot) in perflag.items()))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
