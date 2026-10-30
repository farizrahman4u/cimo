"""Random loop programs (tests/middle_gen.py): cimo with and without the middle end against
gcc -O0 -fwrapv -ffp-contract=off, bit for bit. CIMO_RANDOM_SEEDS=a:b runs more seeds."""

import ctypes
import os

import pytest

np = pytest.importorskip('numpy')

from test_middle import pytestmark, _compile, _buffers, _call, CC  # noqa: E402,F401
import middle_gen  # noqa: E402
import cimo  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def _no_fma():
    """These tests compare bit for bit against gcc -ffp-contract=off."""
    import cimo.driver
    old = cimo.driver.FMA
    cimo.driver.FMA = False
    yield
    cimo.driver.FMA = old

_r = os.environ.get('CIMO_RANDOM_SEEDS', '0:6').split(':')
SEEDS = list(range(int(_r[0]), int(_r[1])))
CALLS = [(n, k, None) for n in (0, 1, 2, 3, 4, 5, 7, 8, 13, 31, 64, 100) for k in (0, -3, 4)]
ALIAS = [(n, 1, d) for n in (3, 9, 33) for d in (-2, 0, 1, 3)]


@pytest.mark.parametrize('seed', SEEDS)
def test_random_loops(tmp_path, seed):
    src, names = middle_gen.program(seed)
    path = tmp_path / ('r%d.c' % seed)
    path.write_text(src)
    import subprocess
    so = str(tmp_path / 'ref.so')
    subprocess.run([CC, '-O0', '-ffp-contract=off', '-fwrapv', '-shared', '-fPIC', str(path),
                    '-o', so], check=True)
    ref = ctypes.CDLL(so)
    on = _compile(src, True, tmp_path)
    try:
        off = _compile(src, False, tmp_path)
    except (cimo.CompileError, AssertionError) as e:
        # known backend bug, independent of the middle end (which works around it): QBE's
        # gvn folds 32-bit constant chains without truncating, and the arm64 isel emits
        # the result as an immediate (assembler error, or an assertion in emit)
        assert isinstance(e, AssertionError) or 'immediate out of range' in str(e), e
        off = None
    vp = ctypes.c_void_p
    bad = []
    for name in names:
        fr = getattr(ref, name)
        fr.restype = ctypes.c_double
        fr.argtypes = [ctypes.c_int, ctypes.c_int, vp, vp, vp, vp, vp, vp]
        cases = CALLS + (ALIAS if name.endswith('_alias') else [])
        for j, (n, k, alias) in enumerate(cases):
            res = [_call(f, n, k, _buffers(j), alias)
                   for f in (getattr(on, name), getattr(off or on, name), fr)]
            if res[1] != res[2]:
                bad.append('%s n=%d k=%d alias=%r: middle end off differs' % (name, n, k, alias))
                break
            if res[0] != res[2]:
                bad.append('%s n=%d k=%d alias=%r: middle end on differs' % (name, n, k, alias))
                break
    assert not bad, '\n'.join(bad) + '\n(source: %s)' % path
