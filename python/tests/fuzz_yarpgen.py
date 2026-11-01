"""Differential fuzzing with YARPGen: random loop-heavy C programs (nested loops over arrays,
the shapes cimo's middle end transforms, which Csmith rarely produces), run by gcc -O0 and
by cimo; their outputs (a hash of every result) must match.

    python tests/fuzz_yarpgen.py [-n COUNT] [--seed FIRST] [-j JOBS] [--out DIR]

Needs YARPGen 2 (built from source: $YARPGEN, default ~/code/yarpgen/build/yarpgen) and gcc.
Its two translation units become one file, and its `max`/`min` macros (GNU statement
expressions) plain `?:`: YARPGen's expressions have no side effects, so the value is the same.
Failures are saved under --out as y<seed>.c (see fuzz_csmith.py).
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fuzz_csmith  # noqa: E402

YARPGEN = os.path.expanduser(os.environ.get('YARPGEN', '~/code/yarpgen/build/yarpgen'))
ARGS = ['--std=c', '--emit-align-attr=none', '--emit-pragmas=none', '--omp-simd=off',
        '--max-array-dims=3']  # larger ones can exceed 4 GB
GCC_TIMEOUT = 10
_MINMAX = re.compile(r'#define (max|min)\(a,b\) \\\n(?:.*\\\n)*.*\n')


def generate(seed, d):
    """YARPGen's program for seed as one file, and gcc's output for it; None if gcc cannot
    run it."""
    sd = os.path.join(d, 'y%d' % seed)
    os.makedirs(sd)
    subprocess.check_call([YARPGEN, '-s', str(seed), '-o', sd] + ARGS, stdout=subprocess.DEVNULL)
    with open(os.path.join(sd, 'func.c')) as f:
        func = _MINMAX.sub(lambda m: '#define %s(a,b) ((a) %s (b) ? (a) : (b))\n'
                           % (m.group(1), '>' if m.group(1) == 'max' else '<'), f.read())
    with open(os.path.join(sd, 'driver.c')) as f:
        driver = f.read()
    with open(os.path.join(sd, 'init.h')) as f:
        init = f.read()
    for n in ('func.c', 'driver.c', 'init.h'):
        os.remove(os.path.join(sd, n))
    src = os.path.join(d, '%d.c' % seed)
    with open(src, 'w') as f:
        f.write(driver + '\n' + func.replace('#include "init.h"', init))
    os.rmdir(sd)
    exe = src[:-2] + '.exe'
    if subprocess.run(['gcc', '-O0', '-w', '-o', exe, src]).returncode:
        return None
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
    a = fuzz_csmith.args(__doc__)
    a.prefix = 'y'
    if not os.access(YARPGEN, os.X_OK):
        sys.exit('YARPGen not found at %s (set YARPGEN)' % YARPGEN)
    return fuzz_csmith.run(generate, a)


if __name__ == '__main__':
    sys.exit(main())
