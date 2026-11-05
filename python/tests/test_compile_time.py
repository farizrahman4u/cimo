"""Compile time on large functions (scramjit generates them): what made it quadratic in the number of loops, checked by
counting rather than timing, and that the large-function paths compute what the old ones did."""

import random

import numpy as np
import pytest

import cimo
import hosts

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='cimo does not run here')


def loops(n, seed=0):
    """One function of n small loops of the kinds scramjit writes: elementwise, a select, sums, a nest, ints, a branch;
    constants from a pool, so equal ones recur in many loops."""
    r = random.Random(seed)
    lines = ['double f(double *restrict o, const double *restrict a, const float *restrict c, int *restrict q, long m) {',
             '  double s = 0; float t = 0; long z = 0;']
    for k in range(n):
        c1 = r.choice(['2.0', '-1.5', '0.1', '3.14159', '1e-3', '0.0', '7.25', '-0.0'])
        c2 = r.choice(['1.0', '0.3', '-2.75', '1e5'])
        lines.append([
            '  for (long i = 0; i < m; i++) o[i] = a[i] * %s + o[i] * %s;' % (c1, c2),
            '  for (long i = 0; i < m; i++) { double v = o[i] - %s; o[i] = v > 0.0 ? v : %s; }' % (c1, c2),
            '  for (long i = 0; i < m; i++) s += o[i] * %s;' % c1,
            '  for (long j = 0; j < 3; j++) for (long i = 0; i < m; i++) t += c[i] * %sf + (float)j;' % c2,
            '  for (long i = 0; i < m; i++) { q[i] = q[i] * 3 + %d; if (q[i] > %d) z += q[i]; }' % (k, 50 * k),
            '  if (s > %s) s = s * %s; else s = s + %s;' % (c1, c2, c1)][k % 6])
    lines += ['  return s + t + (double)z;', '}']
    return '\n'.join(lines)


def test_rounds_take_every_loop_they_can(monkeypatch):
    """The middle end computed the function's alias roots (and refreshed its uses) once per transformed loop: 199 times
    for 60 loops. A round now takes every loop no earlier change in it reached."""
    from cimo.middle import memdep
    count = [0]
    real = memdep.Roots

    def counted(*a):
        count[0] += 1
        return real(*a)
    monkeypatch.setattr(memdep, 'Roots', counted)
    cimo.compile(loops(60), cache=False)
    assert count[0] <= 20, count[0]


def test_constants_get_homes_in_a_large_function(monkeypatch):
    """A broadcast of a constant merged across loops, and (arm64) a float constant loaded in the start block, were live
    through the whole function: 80% of each block's live-in set with 120 loops."""
    from cimo.backend import remat
    seen = []
    real = remat.run

    def run(fn, floats):
        changed = real(fn, floats)
        seen.append((fn.nblk >= remat.LARGE, changed))
        return changed
    monkeypatch.setattr(remat, 'run', run)
    cimo.compile(loops(40), cache=False)
    assert seen == [(True, True)]


@pytest.mark.parametrize('seed', [0, 1])
def test_large_function_computes_what_it_did(seed, monkeypatch):
    """Bit for bit what the same function gives with constants left where they were, and its integers what it gives
    with no middle end."""
    from cimo.backend import remat
    import cimo.middle as M
    src = loops(40, seed)
    on = cimo.compile(src, cache=False)
    monkeypatch.setattr(remat, 'LARGE', 1 << 30)
    ref = cimo.compile(src + '\nint ref_marker;', cache=False)
    monkeypatch.setattr(M, 'enabled', False)
    plain = cimo.compile(src + '\nint plain_marker;', cache=False)
    for m in (5, 17, 64):
        out = []
        for mod in (on, ref, plain):
            rng = np.random.default_rng(seed + m)
            o, a = rng.standard_normal(m), rng.standard_normal(m)
            c, q = rng.standard_normal(m).astype(np.float32), rng.integers(-100, 100, m).astype(np.int32)
            v = mod.f(o, a, c, q, m)
            out.append((np.float64(v).tobytes(), o.tobytes(), q.tobytes()))
        assert out[0] == out[1], m
        assert out[0][2] == out[2][2], m
