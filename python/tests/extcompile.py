"""Test helper: compile C with explicit cimo extension flags (cimo.backend.ext), bypassing the
driver's choice and its cache. Same pipeline as cimo.driver._build."""

import io
import sys

from cimo import driver
from cimo.preprocessor import preprocess
from cimo import frontend
from cimo.backend import ext

FLAGS = ('fma', 'ldp', 'layout')
APPLE = sys.platform == 'darwin'
TARGET = driver.TARGET  # this machine's: arm64, arm64_apple or arm64_win


def ir(src, fma=False, filename='t.c'):
    pp = preprocess(driver._PRELUDE + '#line 1\n' + src, filename, target=TARGET)
    return frontend.compile(pp, filename, target=driver._FRONTEND[TARGET], fma=fma)


def backend(ssa, flags=(), target=None):
    """QBE IR text -> assembly text with the given extension flags."""
    from cimo.backend import defs as D
    from cimo.backend.main import _Driver, _targets
    from cimo.backend.parse import parse
    old = D.T
    D.T = next(t for t in _targets() if t.name == (target or TARGET))
    out = io.StringIO()
    try:
        with ext.options(**{f: f in flags for f in FLAGS}):
            drv = _Driver(out, 0)
            parse(ssa.encode('latin-1'), '<test>', drv.dbgfile, drv.data, drv.func)
            D.T.emitfin(out)
    finally:
        D.T = old
    return out.getvalue()


def asm(src, flags=()):
    return backend(ir(src, 'fma' in flags), flags)


def gen_opt(ssa):
    """Replace saxpy, gemm, jacobi2d, conv3x3 in cproc's IR with the hand-optimized IR of
    scratch/qbe_eval/gen_opt.py (loop-transformed IR like the middle stage's)."""
    import os
    import re
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__)))), 'scratch', 'qbe_eval')
    if d not in sys.path:
        sys.path.insert(0, d)
    import gen_opt as G
    for name in G.KERNELS:
        ssa = re.sub(r'export\nfunction (\w+ )?\$' + name + r'\(.*?\n}\n', '', ssa, flags=re.S)
    return ssa + '\n' + '\n\n'.join('\n'.join(f()) for f in G.KERNELS.values()) + '\n'


def module(src, flags=(), ssa_filter=None):
    """C source -> cimo.driver.Module, compiled with exactly these flags; ssa_filter
    rewrites the IR before the backend (signatures come from the unfiltered IR)."""
    ssa = ir(src, 'fma' in flags)
    text = backend(ssa_filter(ssa) if ssa_filter else ssa, flags)
    sigs = '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                   ','.join(a.split()[0] for a in m.group(3).split(',')
                                            if a.strip()))
                     for m in driver._SIG.finditer(ssa))
    return driver.Module(text, sigs)
