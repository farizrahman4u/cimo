"""cimo extensions to the QBE port, each off by default.

The ported passes reproduce C QBE byte for byte; everything that deviates from it is gated on
a flag here, so `python -m cimo.backend` and the differential tests see stock QBE unless a
caller switches a flag on. `cimo.driver` switches them on.

Flags:
  fma     accept the `fma` instruction in the IR text (`%r =s fma %a, %b, %c` is a*b + c,
          rounded once) and select fmadd/fmsub/fnmadd/fnmsub for it on arm64
  ldp     arm64: pair adjacent loads/stores off the same base into ldp/stp
  layout  arm64: lay loops out so the back edge is one conditional branch

    from cimo.backend import ext
    with ext.options(fma=True, ldp=True, layout=True):
        ...  # parse + emit
    ext.configure(**ext.ALL)   # or set them globally

IR representation of fma: QBE instructions have two arguments, so a three-operand op is a
pair of adjacent instructions, like QBE's own blit0/blit1 and sel0/sel1:

    fma0 %c            (to = R, the addend)
    %r =s fma1 %a, %b  (the product operands and the result)

Both are pinned (gcm does not move them, gvn does not merge them, if-conversion leaves them
alone) and `igroup` keeps them together when gcm reorders a block. Every pass that walks
instruction arguments therefore sees all three uses without change. Instruction selection
turns the pair into one machine instruction (see arm64/isel.py).
"""

import contextlib

fma = False
ldp = False
layout = False

# alias hook for ldp scheduling: disjoint(fn, base1, base2) -> True if accesses off the two
# base temps (SSA, after isel) can never overlap, e.g. distinct restrict pointers. None: they
# may, so accesses never move past accesses off another base.
disjoint = None

ALL = {'fma': True, 'ldp': True, 'layout': True}
_NAMES = tuple(ALL)


def configure(**kw):
    """Set flags globally; returns the previous values."""
    g = globals()
    old = {k: g[k] for k in _NAMES}
    for k, v in kw.items():
        if k not in _NAMES:
            raise TypeError('unknown backend extension %r' % k)
        g[k] = bool(v)
    return old


@contextlib.contextmanager
def options(**kw):
    """Set flags for the duration of a with block."""
    old = configure(**kw)
    try:
        yield
    finally:
        configure(**old)


def current():
    g = globals()
    return {k: g[k] for k in _NAMES}
