"""The 15 kernels and edge.c with the backend extensions (cimo.backend.ext) switched on.

- ldp + layout (no fma): bit-identical to cc -O2 -ffp-contract=off, like test_m2.
- fma + ldp + layout: bit-identical to cc -O2 with explicit contraction where it is identical
  by construction; checked per kernel against both cc -ffp-contract=off and =fast (gcc's
  default "fast" contracts across statements, clang's "on" does not; the kernels where the two
  choices give the same instructions match =fast, the others must match one of the two).
- the extensions really fire on these kernels (pairs, single-branch loop latches, fmadd).
"""

import ctypes
import os
import subprocess

import pytest

import extcompile as X
import test_m2 as M

pytestmark = M.pytestmark

FLAGSETS = {'ldp+layout': ('ldp', 'layout'), 'all': ('fma', 'ldp', 'layout'),
            'gen_opt+ldp+layout': ('ldp', 'layout')}
# kernels whose fused result equals gcc's -ffp-contract=fast result by construction
FAST_SAME = {'saxpy', 'gemm', 'conv3x3', 'dot', 'nbody_acc', 'iir'}


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('ext')
    with open(M.KERNELS) as f:
        src = f.read()
    mods = {k: X.module(src, fl, X.gen_opt if k.startswith('gen_opt') else None)
            for k, fl in FLAGSETS.items()}
    refs = {}
    for name, opt in (('off', '-ffp-contract=off'), ('fast', '-ffp-contract=fast')):
        so = str(tmp / ('ref-%s.so' % name))
        subprocess.run([M.CC, '-O2', opt, '-shared', '-fPIC', M.KERNELS, '-o', so, '-lm'],
                       check=True)
        refs[name] = ctypes.CDLL(so)
    return mods, refs, tmp, src


def _run(fn, args, restype=None, ref=False):
    if ref:
        fn.restype = restype
        fn.argtypes = [ctypes.c_void_p if isinstance(a, M.np.ndarray) else
                       (M.F if isinstance(a, M.F) else M.I) for a in args]
        return fn(*[a.ctypes.data if isinstance(a, M.np.ndarray) else a for a in args])
    return fn(*[a.value if isinstance(a, M.F) else a for a in args])


def _copy(args):
    return [a.copy() if isinstance(a, M.np.ndarray) else a for a in args]


def _state(r, args):
    return [M._bits(r)] + [a.tobytes() for a in args if isinstance(a, M.np.ndarray)]


@pytest.mark.parametrize('flags', sorted(FLAGSETS))
@pytest.mark.parametrize('name', sorted(M.CASES))
def test_kernel(built, flags, name):
    mods, refs, _, _ = built
    restype, factories = M.CASES[name]
    args = [f() for f in factories]
    mine = _copy(args)
    got = _state(_run(getattr(mods[flags], name), mine), mine)
    want = {}
    for rn, lib in refs.items():
        a = _copy(args)
        want[rn] = _state(_run(getattr(lib, name), a, restype, ref=True), a)
    if 'fma' not in FLAGSETS[flags]:
        assert got == want['off']
    elif name in FAST_SAME:
        assert got == want['fast']
    else:
        assert got in (want['off'], want['fast'])


@pytest.mark.parametrize('flags', sorted(FLAGSETS))
def test_edge(built, flags):
    mods, _, tmp, _ = built
    if 'fma' in FLAGSETS[flags]:
        pytest.skip('edge.c checks against unfused C')
    mod = mods[flags]
    shim = tmp / 'edge_shim.c'
    shim.write_text('#define main edge_main\n'
                    '#define gemm (*p_gemm)\n#define saxpy (*p_saxpy)\n'
                    '#define jacobi2d (*p_jacobi2d)\n#define conv3x3 (*p_conv3x3)\n'
                    '#include "%s"\n' % os.path.join(M.EVAL, 'edge.c'))
    so = str(tmp / ('edge-%s.so' % flags))
    subprocess.run([M.CC, '-O0', '-ffp-contract=off', '-shared', '-fPIC', str(shim), '-o', so,
                    '-lm'], check=True)
    lib = ctypes.CDLL(so)
    for k in ('gemm', 'saxpy', 'jacobi2d', 'conv3x3'):
        ctypes.c_void_p.in_dll(lib, 'p_' + k).value = getattr(mod, k).address
    lib.edge_main.restype = ctypes.c_int
    assert lib.edge_main() == 0


def test_extensions_fire(built):
    src = built[3]
    s = X.asm(src, ('fma', 'ldp', 'layout'))
    stock = X.asm(src)
    assert 'fmadd' in s and 'fmadd' not in stock
    # loop latches end in one conditional branch back instead of "b loop" to the test
    body = s.split('\nsaxpy:')[1].split('/* end function')[0]
    assert '\tb\t' not in body and '\tblt\t' in body
    assert s.count('\n\tb\t') * 3 < stock.count('\n\tb\t') * 2
    pairs = sum(1 for l in s.splitlines() if l.startswith(('\tldp', '\tstp'))
                and 'x29, x30' not in l)
    assert pairs > 0
