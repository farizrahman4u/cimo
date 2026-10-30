"""Constant reassociation in gvn (assoccon): chains of add/sub with constants must fold to
in-range immediates of the right width. C QBE folded 32-bit chains in 64 bits
(`x + -1 + -30` became `sub w0, w0, #18446744065119617055`)."""

import platform
import random

import pytest

import cimo
from cimo import driver

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                                reason='arm64 only')

M32, M64 = (1 << 32) - 1, (1 << 64) - 1


def _consts(rnd):
    pick = [0, 1, -1, 30, -30, 4095, -4095, 4096, 1 << 12 | 1, 0x7fffffff, -0x80000000,
            0xffffffff, 1 << 31, 1 << 40, -(1 << 40), (1 << 63) - 1]
    return rnd.choice(pick) if rnd.random() < 0.6 else rnd.randrange(-(1 << 33), 1 << 33)


@pytest.mark.parametrize('middle', [False, True])
def test_add_sub_chains(monkeypatch, middle):
    monkeypatch.setattr(driver, 'MIDDLE', middle)
    rnd = random.Random(7)
    funcs, expect = [], []
    for n in range(120):
        ty, mask, bits = (('int', M32, 32), ('long', M64, 64))[n % 2]
        ops = [(rnd.choice('+-'), _consts(rnd)) for _ in range(rnd.randrange(2, 5))]
        body = 'x'
        for op, c in ops:
            body = '(%s %s (%s)%dL)' % (body, op, ty, c) if abs(c) >= 1 << 31 else \
                '(%s %s %d)' % (body, op, c)
        funcs.append('%s f%d(%s x) { return %s; }' % (ty, n, ty, body.replace('L)', 'LL)')))
        expect.append((n, ty, mask, bits, ops))
    src = '\n'.join(funcs)
    m = cimo.compile(src)
    for n, ty, mask, bits, ops in expect:
        for x in (0, 5, -7, 1 << 30, -(1 << 31)):
            want = x
            for op, c in ops:
                c &= mask  # the C cast to ty
                want = (want + c if op == '+' else want - c) & mask
            if want >> (bits - 1):
                want -= 1 << bits
            assert getattr(m, 'f%d' % n)(x) == want, (funcs[n], x)
