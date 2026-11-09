"""compile(libm='host'): the math library's functions are the host's libm, the ones CPython's math module calls, so
results match Python's bit for bit; the bundled libm (the default) differs in the last bit for many inputs."""

import math
import random

import pytest

import cimo
from cimo import driver

pytestmark = pytest.mark.skipif(driver._PYODIDE, reason='native targets only')

FNS = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'exp', 'expm1',
       'log', 'log1p', 'log10', 'log2', 'erf', 'erfc']
FNS2 = ['atan2', 'pow', 'fmod', 'copysign']  # not hypot: CPython's is its own, not libm's

SRC = '#include <math.h>\n' + ''.join('double f_%s(double x) { return %s(x); }\n' % (n, n) for n in FNS) \
    + ''.join('double g_%s(double x, double y) { return %s(x, y); }\n' % (n, n) for n in FNS2)


def _domain(n, x):
    if n in ('asin', 'acos', 'atanh'):
        return math.sin(x)
    if n == 'acosh':
        return 1 + abs(x)
    if n in ('log', 'log10', 'log2'):
        return abs(x) + 1e-300
    if n == 'log1p':
        return abs(x)
    return x


@pytest.fixture(scope='module')
def host():
    return cimo.compile(SRC, libm='host', cache=False)


def test_same_bits_as_python(host):
    rnd = random.Random(7)
    xs = [rnd.uniform(-20, 20) for _ in range(3000)] + [rnd.uniform(-1e-3, 1e-3) for _ in range(300)] + [0.0, -0.0, 1.0]
    for n in FNS:
        f, py = getattr(host, 'f_' + n), getattr(math, n)
        for x in xs:
            x = _domain(n, x)
            try:
                want = py(x)
            except (OverflowError, ValueError):
                continue
            got = f(x)
            assert got == want and math.copysign(1, got) == math.copysign(1, want), (n, x, got, want)
    for n in FNS2:
        g, py = getattr(host, 'g_' + n), getattr(math, n)
        for _ in range(3000):
            x, y = rnd.uniform(0.01, 30), rnd.uniform(-4, 4)
            try:
                want = py(x, y)
            except (OverflowError, ValueError):
                continue
            assert g(x, y) == want, (n, x, y)


def test_the_bundled_libm_differs():
    """Why the option exists: cimo's own atan2 is off in the last bit for many inputs (and is the same on every target)."""
    m = cimo.compile(SRC, cache=False)
    rnd = random.Random(3)
    pairs = [(rnd.uniform(-5, 5), rnd.uniform(-5, 5)) for _ in range(2000)]
    assert sum(m.g_atan2(x, y) != math.atan2(x, y) for x, y in pairs) > 0


def test_from_the_cache(tmp_path, monkeypatch):
    """A module whose image comes from the compile cache resolves its math in the host libm too; the cache keeps the
    two builds apart."""
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    src = '#include <math.h>\ndouble f(double x, double y) { return atan2(x, y); } // hostlibm cache\n'
    a = cimo.compile(src, libm='host')
    b = cimo.compile(src, libm='host')
    c = cimo.compile(src)
    assert not a.cached and b.cached and not c.cached
    rnd = random.Random(11)
    for _ in range(500):
        x, y = rnd.uniform(-5, 5), rnd.uniform(-5, 5)
        assert a.f(x, y) == b.f(x, y) == math.atan2(x, y)
    assert b.options == {'libm': 'host'} and c.options == {}


def test_long_functions_stay_bundled():
    """lrint and friends return long: 32-bit in Windows DLLs, 64-bit in cimo."""
    m = cimo.compile('#include <math.h>\nlong r(double x) { return lrint(x); }\n', libm='host', cache=False)
    assert m.r(3e12 + 0.5) == 3000000000000 and m.r(2.5) == 2
    assert 'lrint' not in m._libsyms


def test_bad_value():
    with pytest.raises(ValueError, match='bundled'):
        cimo.compile('int f(void) { return 0; }', libm='glibc')
