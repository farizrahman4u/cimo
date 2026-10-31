"""Differential fuzzing: random C programs from Csmith, run by gcc -O0 and by cimo; their
outputs (a checksum of every global) must match.

    python tests/fuzz_csmith.py [-n COUNT] [--seed FIRST] [-j JOBS] [--out DIR]

Needs Csmith (built from source: $CSMITH_DIR, default ~/code/csmith, with build/src/csmith and
the generated build/runtime/safe_math.h) and gcc. Programs that gcc cannot run within the time
limit are discarded. A mismatch, a compile error or a crash is saved under --out as
<seed>.c with <seed>.c.expected (gcc's output) and <seed>.txt (cimo's result); shrink it with
tests/reduce.py, then add a small test for the bug to tests/test_regressions.py.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CSMITH_DIR = os.path.expanduser(os.environ.get('CSMITH_DIR', '~/code/csmith'))
CSMITH = os.path.join(CSMITH_DIR, 'build', 'src', 'csmith')
INCLUDE = [os.path.join(CSMITH_DIR, 'runtime'), os.path.join(CSMITH_DIR, 'build', 'runtime')]
# what cimo does not support: volatile stores (cproc), packed bit-fields
CSMITH_ARGS = ['--no-volatiles', '--no-packed-struct']
GCC_TIMEOUT = 5


def generate(seed, d):
    """Csmith's program for seed and gcc's output for it, or None if gcc cannot run it."""
    src = os.path.join(d, '%d.c' % seed)
    with open(src, 'w') as f:
        subprocess.check_call([CSMITH, '--seed', str(seed)] + CSMITH_ARGS, stdout=f)
    exe = src[:-2] + '.exe'
    subprocess.check_call(['gcc', '-O0', '-w', '-o', exe, src] + ['-I' + i for i in INCLUDE])
    try:
        p = subprocess.run([exe], capture_output=True, timeout=GCC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.remove(exe)
    if p.returncode != 0:
        return None
    with open(src + '.expected', 'wb') as f:
        f.write(p.stdout)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-n', type=int, default=100, help='programs to try')
    ap.add_argument('--seed', type=int, default=int(time.time()), help='first seed')
    ap.add_argument('-j', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--out', default=os.path.join(HERE, 'fuzz_failures'))
    a = ap.parse_args()
    if not os.access(CSMITH, os.X_OK):
        sys.exit('Csmith not found at %s (set CSMITH_DIR)' % CSMITH)
    from test_corpora import Worker
    env = dict(os.environ, CIMO_INCLUDE=os.pathsep.join(INCLUDE))
    import threading
    local = threading.local()
    workers = []
    counts = {'pass': 0, 'skip': 0, 'fail': 0}
    lock = threading.Lock()

    def one(seed):
        w = getattr(local, 'w', None)
        if w is None:
            w = local.w = Worker(env)
            workers.append(w)
        src = generate(seed, tmp)
        if src is None:
            res = 'SKIP'
        else:
            res = w.run(src)
        with lock:
            if res == 'SKIP':
                counts['skip'] += 1
            elif res == 'PASS':
                counts['pass'] += 1
            else:
                counts['fail'] += 1
                os.makedirs(a.out, exist_ok=True)
                for ext in ('.c', '.c.expected'):
                    shutil.copy(src[:-2] + ext, os.path.join(a.out, '%d%s' % (seed, ext)))
                with open(os.path.join(a.out, '%d.txt' % seed), 'w') as f:
                    f.write(res + '\n')
                print('seed %d: %s' % (seed, res[:200]), flush=True)
        if src:
            for ext in ('.c', '.c.expected'):
                os.remove(src[:-2] + ext)

    tmp = tempfile.mkdtemp(prefix='cimo-fuzz-')
    t0 = time.time()
    try:
        with ThreadPoolExecutor(a.j) as ex:
            list(ex.map(one, range(a.seed, a.seed + a.n)))
    finally:
        for w in workers:
            w.close()
        shutil.rmtree(tmp, ignore_errors=True)
    print('seeds %d..%d: %d pass, %d fail, %d skipped (gcc timeout) in %.0f s'
          % (a.seed, a.seed + a.n - 1, counts['pass'], counts['fail'], counts['skip'],
             time.time() - t0))
    return 1 if counts['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
