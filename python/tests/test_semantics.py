"""C semantics against a Python model (semantics_gen.py): random expressions over every
arithmetic type and operator, results compared bit for bit. Needs no other compiler, so it runs
on every host. FMA contraction is off (C allows it; the model rounds every operation).

    python tests/test_semantics.py [SEED [COUNT]]      # one more seed
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import semantics_gen  # noqa: E402
import cimo  # noqa: E402
from cimo import driver  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
COUNT = 400
CHAR_SIGNED = driver.TARGET != 'arm64'  # plain char: unsigned on Linux arm64


def check(seed, count):
    """Mismatches as (C function, args, expected, got)."""
    src, cases = semantics_gen.cases(seed, count, CHAR_SIGNED)
    old = driver.FMA
    driver.FMA = False
    try:
        m = cimo.compile(src)
    finally:
        driver.FMA = old
    bad = []
    funcs = {}
    for chunk in src.split('\nunsigned long long ')[1:]:
        funcs[chunk.split('(', 1)[0]] = 'unsigned long long ' + chunk.strip()
    for name, args, want in cases:
        args = [a - (1 << 64) if isinstance(a, int) and a >= 1 << 63 else a for a in args]
        got = getattr(m, name)(*args) & 0xffffffffffffffff
        if got != want:
            bad.append((funcs[name], args, want, got))
    return bad


def report(bad):
    return '\n\n'.join('%s\nargs %r: want %#x, got %#x' % b for b in bad[:10])


@pytest.mark.parametrize('seed', SEEDS)
def test_random_expressions(seed):
    bad = check(seed, COUNT)
    assert not bad, '%d mismatches:\n%s' % (len(bad), report(bad))


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    count = int(sys.argv[2]) if len(sys.argv) > 2 else COUNT
    bad = check(seed, count)
    print(report(bad))
    print('seed %d: %d expressions, %d mismatches' % (seed, count, len(bad)))
    sys.exit(1 if bad else 0)
