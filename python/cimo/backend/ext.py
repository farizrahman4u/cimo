"""cimo extensions to the QBE port, each off by default.

The ported passes reproduce C QBE byte for byte; everything that deviates from it is gated on
a flag here, so `python -m cimo.backend` and the differential tests see stock QBE unless a
caller switches a flag on. `cimo.driver` switches them on.

Flags:
  fma     accept the `fma` instruction in the IR text (`%r =s fma %a, %b, %c` is a*b + c,
          rounded once) and select fmadd/fmsub/fnmadd/fnmsub for it on arm64
  ldp     arm64: pair adjacent loads/stores off the same base into ldp/stp
  layout  arm64: lay loops out so the back edge is one conditional branch
  math    accept the math instructions in the IR text (`%r =d fsqrt %x`; also fabs, fmin,
          fmax, ffloor, fceil, ftrunc, fround, frint, fnearbyint), which cimo.frontend
          emits for calls to the C functions of those names; arm64 selects fsqrt, fabs,
          fminnm, fmaxnm, frintm, frintp, frintz, frinta, frintx, frinti; and the bit
          builtins (`%r =w bclz %x`; also bctz, bpopcnt, bswap, bswap16), emitted for
          __builtin_clz/ctz/popcount/bswap*: clz, rbit+clz,
          cnt (through v31), rev, rev16
  volatile  accept the volatile loads and stores in the IR text (`vloadw`, `vstorel`, ... as
          the plain ones), which cimo.frontend emits for accesses to volatile objects; no pass
          deletes, merges, moves or forwards them, and instruction selection turns them into
          plain loads and stores
  bigcopy None, or a size: a struct argument of at least that many bytes is copied by a call
          to memcpy instead of an unrolled blit (arm64)
  isel    arm64 selection beyond QBE's: shifts by an immediate, register-offset addressing
          (`ldr x0, [x1, x2, lsl #3]`, `[x1, w2, sxtw #2]`), madd/msub/mneg, cbz/cbnz, csel/fcsel,
          shifted and extended operands outside loops, bic/orn/eon/mvn, ubfx/sbfx, ror, fnmul,
          storing xzr, fmov immediates, no frame in a leaf function that needs no stack
  cse     load elimination again after gvn, identical integer divisions merged, remainders
          computed from the division of the same operands on arm64 (cimo.backend.cse)
  inline  0, or a size: copy functions of at most that many instructions into the callers
          that follow them in the module (cimo.backend.inline)
  vla     accept getsp/setsp in the IR text (`%s =l getsp`, `setsp %s`), which cimo.frontend
          emits to free variable-length arrays when their scope ends
  trap    arm64: unreachable code (QBE's `hlt`) calls __cimo_trap instead of `brk #1000`,
          so it unwinds to Python (cimo.execution) instead of killing the process
  stkchk  None, or a set of function names whose prologue calls __cimo_stkchk (the stack
          guard, cimo.execution); not a bool, and not part of ALL

    from cimo.backend import ext
    with ext.options(fma=True, ldp=True, layout=True, math=True):
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
math = False
trap = False
volatile = False
vla = False
cse = False
isel = False
bigcopy = None
inline = 0
stkchk = None

# alias hook for ldp scheduling: disjoint(fn, base1, base2) -> True if accesses off the two
# base temps (SSA, after isel) can never overlap, e.g. distinct restrict pointers. None: they
# may, so accesses never move past accesses off another base.
disjoint = None

ALL = {'fma': True, 'ldp': True, 'layout': True, 'math': True, 'trap': True, 'volatile': True, 'vla': True, 'cse': True, 'isel': True}
_NAMES = tuple(ALL) + ('stkchk', 'bigcopy', 'inline')


def configure(**kw):
    """Set flags globally; returns the previous values."""
    g = globals()
    old = {k: g[k] for k in _NAMES}
    for k, v in kw.items():
        if k not in _NAMES:
            raise TypeError('unknown backend extension %r' % k)
        g[k] = ((frozenset(v) if v is not None else None) if k == 'stkchk'
                else v if k == 'bigcopy' else int(v or 0) if k == 'inline' else bool(v))
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
