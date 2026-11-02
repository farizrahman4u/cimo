"""A gate for the benchmark kernels: timing on CI machines is too noisy to fail on, so this counts what's in their
loops instead, and fails if a count grows. The moves count would have caught dot's fmov, which cost 52 to 56% on Neoverse
and nothing on the machine it was written on. It compiles for one target, so every host gets the same numbers.

After an intended change: python tests/test_bench_gate.py, which rewrites the baseline; say why in the commit.
Timing is still a manual run of bench/run.py on EC2."""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cimo import driver  # noqa: E402

BASELINE = os.path.join(HERE, 'bench_gate.json')
KERNELS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'scratch', 'qbe_eval', 'kernels.c')
MOVE = re.compile(r'(fmov\s+[dsq]\d+, [dsq]\d+|mov\s+[xw]\d+, [xw]\d+)$')
STACK = re.compile(r'\[(sp|x29)\b')
BRANCH = re.compile(r'\b(b(?:\.?[a-z]{2})?|cbn?z\s+\w+,|tbn?z\s+\w+, #\d+,)\s*(\.?L\d+)$')


def measure(text):
    """{kernel: [instructions in loops, register moves in loops, stack accesses in loops]}."""
    found = {}
    for m in re.finditer(r'^_?(\w+):\n(.*?)^/\* end function', text, re.S | re.M):
        lines = [x.strip() for x in m.group(2).split('\n')]
        labels = {x[:-1]: n for n, x in enumerate(lines) if re.fullmatch(r'\.?L\d+:', x)}
        inloop = [False] * len(lines)
        for n, x in enumerate(lines):
            b = BRANCH.search(x)
            if b and labels.get(b.group(2), n + 1) <= n:
                inloop[labels[b.group(2)]:n + 1] = [True] * (n + 1 - labels[b.group(2)])
        code = [x for x, s in zip(lines, inloop) if s and x and not x.endswith(':') and not x.startswith('.')]
        found[m.group(1)] = [len(code), sum(1 for x in code if MOVE.match(x)), sum(1 for x in code if STACK.search(x))]
    return found


def current():
    old = driver.TARGET
    driver.TARGET = 'arm64'
    try:
        with open(KERNELS) as f:
            pp = driver._preprocess(f.read(), KERNELS, (), False, ())
        return measure(driver._compile(pp, KERNELS, False)[0])
    finally:
        driver.TARGET = old


def test_loops_of_the_kernels_did_not_grow():
    with open(BASELINE) as f:
        base = json.load(f)
    now = current()
    names = ('instructions', 'moves', 'stack accesses')
    worse = ['%s: %s %d, was %d' % (k, name, now[k][n], base[k][n])
             for k in base for n, name in enumerate(names) if now[k][n] > base[k][n]]
    assert not worse, 'loops of the benchmark kernels grew (intended? python tests/test_bench_gate.py):\n' + '\n'.join(worse)


def test_same_source_same_code():
    """Scalar replacement ordered its variables by a name holding id(chain), so conv3x3's registers depended on where
    the allocator happened to put that object: on what the process had compiled before."""
    old = driver.TARGET
    driver.TARGET = 'arm64'
    try:
        with open(KERNELS) as f:
            pp = driver._preprocess(f.read(), KERNELS, (), False, ())
        first = re.sub(r'\.?L\d+', 'L', driver._compile(pp, KERNELS, False)[0])
        keep = []
        for n in range(6):  # moves what's allocated next, as a compile in between would
            keep.append([object() for _ in range(5000 * (n + 1))])
            if n % 2:
                del keep[0]
            assert re.sub(r'\.?L\d+', 'L', driver._compile(pp, KERNELS, False)[0]) == first, n
    finally:
        driver.TARGET = old


def test_a_copy_on_the_chain_is_seen():
    with_copy = 'f:\n.L1:\n\tldr\td1, [x0]\n\tfmov\td2, d0\n\tfmadd\td0, d1, d1, d2\n\tbne\t.L1\n\tret\n/* end function f */\n'
    without = 'f:\n.L1:\n\tldr\td1, [x0]\n\tfmadd\td0, d1, d1, d0\n\tbne\t.L1\n\tret\n/* end function f */\n'
    assert measure(with_copy)['f'] == [4, 1, 0]
    assert measure(without)['f'] == [3, 0, 0]


if __name__ == '__main__':
    numbers = current()
    with open(BASELINE, 'w') as f:
        json.dump(numbers, f, indent=0, sort_keys=True)
        f.write('\n')
    print('wrote', BASELINE)
