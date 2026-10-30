"""cimo's middle end: loop optimizations on the backend's SSA form (`backend.defs.Fn`).

Off by default, so the ported QBE (`python -m cimo.backend`) and its byte-for-byte
differential tests are unaffected; `cimo.driver` switches it on. `backend/main.py` calls
`run(fn)` after GCM (clean SSA, loop invariants already hoisted by GCM, critical edges not yet
split by the ABI and isel passes) and then re-runs QBE's cleanup passes.

    from cimo import middle
    with middle.options(enabled=True, params={...}):
        ...  # parse + emit

`params` maps a function name to its parameter info from the frontend
(`frontend.qbe.paraminfo`): a list of (temporary name, is_restrict) per parameter. Restrict
parameters let the alias analysis prove that loads and stores through different pointers
don't overlap.

Passes (see each module): `loops` (natural loops, preheaders), `affine` (induction
variables and affine forms), `memdep` (base-object alias analysis), `fullunroll` (small
constant-trip loops), `jam` (unroll-and-jam of a loop around an inner loop), `inner`
(innermost loops: versioning with overflow checks, rotation, strength reduction, LICM,
unrolling with a remainder loop, scalar replacement).
"""

import contextlib

enabled = False
params = {}      # function name -> [(tmp name, restrict)]
unroll = 4       # main unroll factor for innermost loops (1 disables unrolling)
debug = False    # print the IR before/after to stderr
stats = None     # optional dict: counters of transformations applied

_NAMES = ('enabled', 'params', 'unroll', 'debug', 'stats')


def configure(**kw):
    g = globals()
    old = {k: g[k] for k in _NAMES}
    for k, v in kw.items():
        if k not in _NAMES:
            raise TypeError('unknown middle-end option %r' % k)
        g[k] = v
    return old


@contextlib.contextmanager
def options(**kw):
    old = configure(**kw)
    try:
        yield
    finally:
        configure(**old)


def count(name, n=1):
    if stats is not None:
        stats[name] = stats.get(name, 0) + n


def run(fn):
    """Optimize fn in place. Requires and preserves: SSA, fillcfg, filluse, filldom."""
    from .pipeline import optimize
    return optimize(fn)
