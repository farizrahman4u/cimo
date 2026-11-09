"""LLVM's LoopVectorize, on what inner.py has already proved: a counted loop of one straight run of instructions,
whose accesses are consecutive, becomes one vector operation per instruction, and the scalar loop stays as the
remainder and as the way out when the guards fail. Lanes are 128 bits' worth, so 4 floats or ints, or 2 doubles or longs.

Legal when, as LoopVectorizationLegality and LoopAccessAnalysis ask:
- every instruction has a vector form, with lanes of one width;
- each load and store steps by its own size, upwards;
- two accesses of one base and step are far enough apart that a vector's worth of iterations can't change what
  either sees: see _distance;
- nothing is carried round the loop but the induction variables and integer reductions (_reduction), whose order C
  doesn't fix, and float sums (_ordered), whose order it does: the vector loop adds its lanes one by one, in order, as
  gcc's fold-left reductions, when what else the loop does is worth it (_worth); in any order where
  `#pragma omp simd reduction` names them (Ins.reassoc), and fmin/fmax ones always;
- operands are computed in the loop by a vectorized instruction, or are the same on every iteration.

Each lane does what the scalar instruction did, in the same rounding, so results are bit for bit the scalar loop's.
Arrays that may overlap are compared before the loop (overlapchecks), up to MAXCHECKS pairs.
Conditions: a comparison becomes a mask of its lanes (all ones where it holds), which only a select may read, and the
select takes each lane's bits from one side or the other; so `x > 0 ? x : 0` is exact, NaNs and signed zeros included.
Integer reductions through a condition (a running max or min) too (_select), and a loop of unordered reductions does
INTERLEAVE vectors an iteration, each with its own accumulators (interleave).
Not yet: other widening and narrowing, float reductions through a condition, early exits, downward steps, SLP.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ks, Kd, Kq, Ins, optab, rval
from ..backend.util import newtmp
from . import ir
from ..backend import ext as _ext

_LANE = {Ks: 's', Kd: 'd', Kw: 'w', Kl: 'l'}
_SIZE = {Ks: 4, Kd: 8, Kw: 4, Kl: 8}
_FPONLY = frozenset(('s', 'd'))
MAXCHECKS = 8  # pairs of arrays compared at run time: LLVM's runtime-memory-check-threshold


def _opname(i):
    """The q instruction that does what i does to each lane, or None."""
    k = i.cls
    x = _LANE.get(k)
    if x is None:
        return None
    op = i.op
    if op in (D.Oadd, D.Osub, D.Oneg):
        return {D.Oadd: 'qadd', D.Osub: 'qsub', D.Oneg: 'qneg'}[op] + x
    if op == D.Omul:
        return 'qmul' + x if x != 'l' else None  # NEON has none
    if op == D.Odiv:
        return 'qdiv' + x if x in _FPONLY else None
    if x in _FPONLY and op in (D.Ofsqrt, D.Ofabs, D.Ofmin, D.Ofmax):
        return {D.Ofsqrt: 'qsqrt', D.Ofabs: 'qabs', D.Ofmin: 'qmin', D.Ofmax: 'qmax'}[op] + x
    if x not in _FPONLY and op in (D.Oand, D.Oor, D.Oxor):
        return {D.Oand: 'qand', D.Oor: 'qor', D.Oxor: 'qxor'}[op]
    if x not in _FPONLY and op in (D.Oshl, D.Oshr, D.Osar):
        return {D.Oshl: 'qshl', D.Oshr: 'qshr', D.Osar: 'qsar'}[op] + x
    return None


# a scalar comparison as a vector one: (q operation without its lane, swap the operands, negate the mask)
_CMP = {'ceq': ('qceq', 0, 0), 'cne': ('qceq', 0, 1),
        'cgt': ('qcgt', 0, 0), 'cge': ('qcge', 0, 0), 'clt': ('qcgt', 1, 0), 'cle': ('qcge', 1, 0),
        'csgt': ('qcgt', 0, 0), 'csge': ('qcge', 0, 0), 'cslt': ('qcgt', 1, 0), 'csle': ('qcge', 1, 0),
        'cugt': ('qcugt', 0, 0), 'cuge': ('qcuge', 0, 0), 'cult': ('qcugt', 1, 0), 'cule': ('qcuge', 1, 0)}
_CMPCLS = {'w': Kw, 'l': Kl, 's': Ks, 'd': Kd}


def _compare(i):
    """(q operation, swap, negate, the operands' class) for a comparison with a vector form, or None. Its result is
    a w; the lanes are its operands'."""
    name = optab[i.op].name or ''
    c = _CMP.get(name[:-1])
    k = _CMPCLS.get(name[-1:])
    if c is None or k is None or (k in (Ks, Kd)) != (len(name) == 4 and name[1] in 'gl' or name[1:3] in ('eq', 'ne')
                                                     and k in (Ks, Kd)):
        return None
    return c + (k,)


def _halves(fast):
    """_Float16 arrays: {id(loaduh): its h2f} for a half load whose only reader is a widening to float, and {id(storeh):
    the f2h} for a half store of a narrowing that nothing else reads; and the ids of those h2f and f2h. A vector loop
    loads and stores them 4 floats at a time (qldh, qsth)."""
    uses = {}
    defs = {}
    for i in fast.cloned:
        if ir.istmp(i.to):
            defs[rval(i.to)] = i
        for a in i.arg[:2]:
            if ir.istmp(a):
                uses.setdefault(rval(a), []).append(i)
    for p in ir.phis(fast.sh.H):
        for a in p.arg[:p.narg]:
            if ir.istmp(a):
                uses.setdefault(rval(a), []).append(None)
    loads, stores, inner = {}, {}, set()
    for i in fast.cloned:
        if i.op == D.Oloaduh and ir.istmp(i.to):
            u = uses.get(rval(i.to), [])
            if len(u) == 1 and u[0] is not None and u[0].op == D.Oh2f:
                loads[id(i)] = u[0]
                inner.add(id(u[0]))
        if i.op == D.Ostoreh and ir.istmp(i.arg[0]):
            d = defs.get(rval(i.arg[0]))
            if d is not None and d.op == D.Of2h and len(uses.get(rval(i.arg[0]), [])) == 1:
                stores[id(i)] = d
                inner.add(id(d))
    return loads, stores, inner


def _loadsize(i):
    if i.op == D.Oload and i.cls in _SIZE:
        return _SIZE[i.cls]
    if i.op in (D.Oloadsw, D.Oloaduw) and i.cls == Kw:
        return 4
    return None


def _storeclass(i):
    return {D.Ostorew: Kw, D.Ostorel: Kl, D.Ostores: Ks, D.Ostored: Kd}.get(i.op)


def _way(fast, step):
    """Whether an access stepping by step goes the way the loop's others do (cimo.middle: fast.vdown, a loop whose
    accesses step down); the first one decides."""
    down = step < 0
    if fast.vdown is None:
        fast.vdown = down
    return fast.vdown == down


def _at(fast, k0, j=0, lanes=None):
    """The iteration whose element is at a vector's lowest address: j vectors of `lanes` elements into the copy that
    starts at iteration k0. Stepping down, lanes hold the elements in memory order, the copy's last iteration first;
    every access does the same, so each lane is one iteration throughout."""
    lanes = lanes or fast.vf
    if fast.vdown:
        return k0 + fast.vf - 1 - j * lanes
    return k0 + j * lanes


def _distance(store, load, vf, storeat, loadat):
    """store and load are forms with one base and step. d = store's address - load's. Nothing changes if they're the
    same element, or a vector's worth apart. A load ahead of a store sees what it saw before only if the vector load
    comes first, since the vector store would write what the scalar loop hasn't yet."""
    d = store.L.c - load.L.c
    size = abs(store.S.c)
    if d == 0 or abs(d) >= vf * size:
        return True
    return (d < 0) == (store.S.c > 0) and loadat < storeat  # the load ahead, in the loop's order


class Reduction:
    """s = s op v, round the loop: t is s's phi, op the instruction, v the operand, ext how v comes from 32-bit lanes."""
    __slots__ = ('t', 'i', 'op', 'v', 'ext', 'acc0', 'in0', 'in1', 'out0', 'out1', 'cls', 'ordered', 'third',
                 'cmp', 'sel0', 'identity', 'copies', 'index', 'later', 'flt', 'restart')


class Index:
    """The position of a select reduction's value (Reduction.index): q, the phi of `q = cond ? b : q` beside the value's
    select on the same condition (None: kept only to order equal values: float zeros), b an induction variable stepping
    by step, sel0 and sel1 its select; its lanes in body and closeloop."""
    __slots__ = ('q', 'b', 'step', 'sel0', 'sel1', 'bfirst', 'acc0', 'in0', 'out0', 'copies', 'rel')


_REDOPS = {D.Oadd: 'qadd', D.Osub: 'qsub', D.Oand: 'qand', D.Oor: 'qor', D.Oxor: 'qxor'}
_FREDOPS = {**_REDOPS, D.Omul: 'qmul', D.Ofmin: 'qmin', D.Ofmax: 'qmax'}  # and float reductions in any order
# each lane's starting value, as float bits: -0 is the identity of a sum (+0 isn't, for -0), a quiet NaN of fmin and
# fmax (which take the number over it)
_FIDENT = {(Ks, D.Oadd): 0x80000000, (Kd, D.Oadd): 1 << 63, (Ks, D.Omul): 0x3f800000, (Kd, D.Omul): 0x3ff0000000000000,
           (Ks, D.Ofmin): 0x7fc00000, (Kd, D.Ofmin): 0x7ff8000000000000}


def _reduction(fast, p, cloned):
    """p as a Reduction, or None: nothing in the loop reads its value but one instruction, whose result only the phi reads."""
    A = fast.A
    fn = fast.fn
    if p.cls not in (Kw, Kl):
        return None
    t = rval(p.to)
    nxt = ir.phiargof(p, fast.sh.latch)
    if not ir.istmp(nxt):
        return None
    i = None
    for x in cloned:
        if x.to == nxt:
            i = x
    if i is None or i.op not in _REDOPS or i.cls != p.cls:
        return None
    if i.arg[0] == p.to and i.arg[1] != p.to:
        v = i.arg[1]
    elif i.arg[1] == p.to and i.arg[0] != p.to and i.op != D.Osub:
        v = i.arg[0]
    else:
        return None
    reads = 0
    for x in cloned:
        reads += sum(1 for a in x.arg[:2] if a == p.to)
        if x is not i and nxt in x.arg[:2]:
            return None
    if reads != 1:
        return None
    for q in ir.phis(fast.sh.H):
        if q is not p and (p.to in q.arg[:q.narg] or nxt in q.arg[:q.narg]):
            return None
    r = Reduction()
    r.t, r.i, r.op, r.v, r.cls = t, i, i.op, v, p.cls
    r.ext = r.third = None
    r.ordered = False
    return r


def _select(fast, p, cloned):
    """p as a Reduction through a condition, or None: an integer s = (x > s) ? x : s, or any comparison of x and s that
    orders them, either way round, so a maximum or a minimum. Each lane keeps its own from the type's identity, and the
    lanes join the scalar value in closeloop through the same comparison and select. Equal integers are the same
    integer, so the order doesn't matter. A float one (a comparison false on NaN: NaNs never win) keeps an index beside
    each lane (plan: Index), since its equal values aren't all the same (two zeros): the lanes join by value, then the
    earlier or later index, as the loop would have."""
    if p.cls not in (Kw, Kl, Ks, Kd):
        return None
    nxt = ir.phiargof(p, fast.sh.latch)
    if not ir.istmp(nxt):
        return None
    n = next((n for n, x in enumerate(cloned) if x.to == nxt), None)
    if n is None or n < 2 or cloned[n].op != D.Osel1 or cloned[n].cls != p.cls or cloned[n - 1].op != D.Osel0:
        return None
    sel1, sel0 = cloned[n], cloned[n - 1]
    cmp = next((x for x in cloned if x.to == sel0.arg[0]), None) if ir.istmp(sel0.arg[0]) else None
    c = _compare(cmp) if cmp is not None else None
    if c is None or c[0] == 'qceq' or c[3] != p.cls:
        return None
    if p.to not in sel1.arg[:2] or sel1.arg[0] == sel1.arg[1] or p.to not in cmp.arg[:2] or cmp.arg[0] == cmp.arg[1]:
        return None
    x = sel1.arg[1] if sel1.arg[0] == p.to else sel1.arg[0]
    if x not in cmp.arg[:2]:
        return None
    # who wins: x when the condition holds if it's sel1's first operand; the condition says x > s, after the swaps
    xfirst = (cmp.arg[0] == x) != bool(c[1])
    greater = xfirst == (sel1.arg[0] == x)
    reads = sum(1 for y in cloned for a in y.arg[:2] if a == p.to)
    if reads != 2 or any(y is not sel1 and nxt in y.arg[:2] for y in cloned):
        return None
    if any(cmp.to in y.arg[:2] for y in cloned if y is not sel0 and y.op != D.Osel0):
        return None  # other sel0s: an index's (plan checks)
    for q in ir.phis(fast.sh.H):
        if q is not p and (p.to in q.arg[:q.narg] or nxt in q.arg[:q.narg]):
            return None
    flt = p.cls in (Ks, Kd)
    if flt and (c[2] or sel1.arg[0] != x):
        return None  # !(x <= s), or s > x ? s : x: a NaN would win
    bits = 8 * _SIZE[p.cls]
    unsigned = c[0].startswith('qcu')
    if flt and p.cls == Ks:
        identity = (0xff800000 if greater else 0x7f800000)
    elif flt:
        identity = (0xfff0000000000000 if greater else 0x7ff0000000000000)  # -inf, +inf: nothing passes them strictly
    elif greater:
        identity = 0 if unsigned else -(1 << (bits - 1))
    else:
        identity = -1 if unsigned else (1 << (bits - 1)) - 1
    # does x win when it equals s? the condition holds on equality (>=), and x is what it picks then
    eqtrue = c[0] in ('qcge', 'qcuge')
    later = eqtrue == (sel1.arg[0] == x)
    r = Reduction()
    r.t, r.i, r.op, r.v, r.cls = rval(p.to), sel1, D.Osel1, x, p.cls
    r.ext = r.third = None
    r.ordered = False
    r.cmp, r.sel0, r.identity = cmp, sel0, identity
    r.index, r.later, r.flt = None, later, flt
    return r


def _index(fast, q, cloned, sels):
    """q as the Index of one of the select reductions sels: q = cond ? b : q (or the other way round) on the same
    condition, b an induction variable with a constant step; or None."""
    if q.cls != Kl:
        return None
    nxt = ir.phiargof(q, fast.sh.latch)
    if not ir.istmp(nxt):
        return None
    n = next((n for n, x in enumerate(cloned) if x.to == nxt), None)
    if n is None or n < 1 or cloned[n].op != D.Osel1 or cloned[n].cls != Kl or cloned[n - 1].op != D.Osel0:
        return None
    sel1, sel0 = cloned[n], cloned[n - 1]
    r = next((r for r in sels if r.cmp.to == sel0.arg[0] and r.index is None), None)
    if r is None or q.to not in sel1.arg[:2] or sel1.arg[0] == sel1.arg[1]:
        return None
    b = sel1.arg[1] if sel1.arg[0] == q.to else sel1.arg[0]
    step = _bivstep(fast, b)
    if step is None:
        return None
    reads = sum(1 for y in cloned for a in y.arg[:2] if a == q.to)
    if reads != 1 or any(nxt in y.arg[:2] for y in cloned):
        return None
    for o in ir.phis(fast.sh.H):
        if o is not q and (q.to in o.arg[:o.narg] or nxt in o.arg[:o.narg]):
            return None
    # the index follows the value: it takes b when the value takes x
    x = r.v
    if (sel1.arg[0] == b) != (r.i.arg[0] == x):
        return None
    ix = Index()
    ix.q, ix.b, ix.step, ix.sel0, ix.sel1, ix.bfirst = q, b, step, sel0, sel1, sel1.arg[0] == b
    ix.rel = r.cls in (Kw, Ks)  # 32-bit lanes: an iteration count beside each, the position from it (_closeindexed)
    return ix


def _bivstep(fast, b):
    """The constant step of induction variable b (a phi of the header), or None."""
    if not ir.istmp(b) or rval(b) not in fast.A.biv:
        return None
    st = fast.A.biv[rval(b)][1]
    return st.c if st.isconst() and st.c else None


def _counter(fast):
    """(address, step) of a load of the loop with a constant step: a float select's hidden index, which only orders
    the iterations (an induction variable the loop doesn't otherwise use has no value in the vector loop); or None."""
    for i in fast.cloned:
        if _loadsize(i) is not None and ir.istmp(i.arg[0]):
            f = fast.A.form(i.arg[0], 64)
            if f is not None and f.S.isconst() and f.S.c:
                return i.arg[0], f.S.c
    return None


def _last(fast, p, cloned):
    """Whether p is set to the same value every iteration (`f = 1`, or one from before the loop) and nothing in the loop
    reads it, so the loop leaves with that value once it has run: a vector iteration sets it as a scalar one does."""
    v = ir.phiargof(p, fast.sh.latch)
    if ir.istmp(v) and fast.A.inloop(v) and v != p.to:  # p = phi(x, p) is x (load elimination's, carrying a load past
        return False                                    # the loop to one after it)
    if any(p.to in x.arg[:2] for x in cloned):
        return False
    return not any(q is not p and p.to in q.arg[:q.narg] for q in ir.phis(fast.sh.H))


def _ordered(fast, p, cloned):
    """p as a float Reduction, or None: s = s + v, s - v or fma(a, b, s), nothing else reading s or the new value but
    the phi, done in order; in any order if `#pragma omp simd reduction` names s (Ins.reassoc), and then s * v too.
    s = fmin(s, v) and fmax always in any order: fminnm and fmaxnm give the same in any order (NaNs and zeros too)."""
    if p.cls not in (Ks, Kd):
        return None
    nxt = ir.phiargof(p, fast.sh.latch)
    if not ir.istmp(nxt):
        return None
    i = third = None
    for n, x in enumerate(cloned):
        if x.to == nxt:
            i = x
            if x.op == D.Ofma1 and n and cloned[n - 1].op == D.Ofma0:
                third = cloned[n - 1]
    if i is None or i.cls != p.cls:
        return None
    if i.op == D.Ofma1:
        if third is None or third.arg[0] != p.to or p.to in i.arg[:2]:
            return None
        v = (i.arg[0], i.arg[1])
    elif i.op == D.Oadd and i.arg[0] == p.to and i.arg[1] != p.to:
        v = i.arg[1]
    elif i.op == D.Oadd and i.arg[1] == p.to and i.arg[0] != p.to:
        v = i.arg[0]  # v + s: s + v, the same for floats
    elif i.op == D.Osub and i.arg[0] == p.to and i.arg[1] != p.to:
        v = i.arg[1]
    elif (i.op in (D.Ofmin, D.Ofmax) or i.op == D.Omul and i.reassoc) and p.to in i.arg[:2] and i.arg[0] != i.arg[1]:
        v = i.arg[1] if i.arg[0] == p.to else i.arg[0]
    else:
        return None
    reads = 0
    for x in cloned:
        reads += sum(1 for a in x.arg[:2] if a == p.to)
        if x is not i and nxt in x.arg[:2]:
            return None
    if reads != 1:
        return None
    for q in ir.phis(fast.sh.H):
        if q is not p and (p.to in q.arg[:q.narg] or nxt in q.arg[:q.narg]):
            return None
    r = Reduction()
    r.t, r.i, r.op, r.v, r.cls, r.third = rval(p.to), i, i.op, v, p.cls, third
    r.ext = None
    r.ordered = not (i.reassoc or i.op in (D.Ofmin, D.Ofmax))
    r.identity = _FIDENT[p.cls, {D.Osub: D.Oadd, D.Ofma1: D.Oadd, D.Ofmax: D.Ofmin}.get(i.op, i.op)]
    return r


_SLOW = {D.Odiv, D.Ofsqrt}  # one at a time, on a vector as on one lane


def _cycles(i, lanes=1):
    """lanes: a vector's, if a division or square root on it costs one a lane (arm64: Neoverse N1, V1 and V2 run
    nbody_acc no faster on 2 x f64, measured; x86-64's divpd and sqrtpd do, 1.4x)."""
    if i.op in (D.Onop, D.Odbgloc, D.Ofma0):
        return 0
    if i.op in _SLOW and i.cls in (Ks, Kd):
        return (8 if i.cls == Kd else 5) * lanes
    return 0.5


def _worth(fast, vf, skip):
    """Whether in-order float sums pay: each vector iteration still adds its vf lanes one after the other, so the
    loop gains only when the rest of it took longer than those adds' latency, as gcc's cost model has it. Estimates in
    cycles: two operations a cycle, a division or square root at a time, 4 a dependent add."""
    rest = [i for i in fast.cloned if id(i) not in skip]
    per = vf if D.T.name.startswith('arm64') else 1
    chain = 4
    ordered = [r for r in fast.vreds if r.ordered]
    scalar = vf * max(sum(_cycles(i) for i in rest) + 0.5 * len(ordered), chain)
    vector = max(sum(_cycles(i, per) for i in rest) + vf * len(ordered), vf * chain)  # a lane out and an add, for each
    return vector < 0.8 * scalar


def _widening(fast, r, cloned):
    """long s; s += (long)x: returns the extension if v is one, of something only this reads, for the op that adds it."""
    if r.op != D.Oadd or r.cls != Kl or not ir.istmp(r.v):
        return None
    for x in cloned:
        if x.to == r.v and x.op in (D.Oextsw, D.Oextuw) and x.cls == Kl:
            if sum(1 for y in cloned for a in y.arg[:2] if a == r.v) == 1:
                return x
    return None


_LOWBITS = {D.Oadd: 'qadd', D.Osub: 'qsub', D.Omul: 'qmul', D.Oand: 'qand', D.Oor: 'qor', D.Oxor: 'qxor', D.Oshl: 'qshl',
            D.Oneg: 'qneg'}
_IDENT = (D.Ocopy, D.Oextub, D.Oextsb, D.Oextuh, D.Oextsh, D.Oextuw, D.Oextsw)
_NARROWLD = {1: (D.Oloadub, D.Oloadsb), 2: (D.Oloaduh, D.Oloadsh)}
_NARROWST = {1: D.Ostoreb, 2: D.Ostoreh}


def _bytes(fast, rts):
    """Loops over bytes or shorts whose values only reach their stores through operations whose low bits depend only on
    their operands' low bits (+, -, *, &, |, ^, << a constant, negation; extensions and copies change none of them):
    brighten's `a[i] = (uint8_t)(a[i] + k)`. Then lanes as wide as the stores (16 bytes a vector, 8 shorts; twice that
    with AVX2) give the bits C stores, whatever the values in between, k included (its low bits, on every lane). Returns
    vf, or 0 (cimo.middle.vectorize._narrow then tries 16-bit lanes with value ranges)."""
    if _ext.qbytes() == 32 or D.T.name.startswith('amd64'):
        if 'avx' not in _ext.x86 and _ext.qbytes() == 32:
            return 0
    elif not D.T.name.startswith('arm64'):
        return 0
    if fast.vreds or any(rval(p.to) not in fast.A.biv for p in ir.phis(fast.sh.H)):
        return 0
    A = fast.A
    fn = fast.fn
    width = None
    made = set()
    mem = []
    x86 = D.T.name.startswith('amd64')
    for n, i in enumerate(fast.cloned):
        if i.op in (D.Onop, D.Odbgloc):
            continue
        w = next((w for w, ops in _NARROWLD.items() if i.op in ops), None)
        st = next((w for w, op in _NARROWST.items() if i.op == op), None)
        if w or st:
            w = w or st
            if width not in (None, w):
                return 0
            width = w
            f = A.form(i.arg[1] if st else i.arg[0], 64)
            if f is None or not f.S.isconst() or abs(f.S.c) != w or A.avlevel(f) == 'X' or not _way(fast, f.S.c):
                return 0
            if st and not _operand(fast, i.arg[0], made):
                return 0
            mem.append((i, f, n, bool(st)))
            if not st:
                made.add(rval(i.to))
            continue
        if not ir.istmp(i.to) or i.cls not in (Kw, Kl):
            return 0
        if i.op in _IDENT:
            if not _operand(fast, i.arg[0], made):
                return 0
            made.add(rval(i.to))
            continue
        name = _LOWBITS.get(i.op)
        if name is None:
            return 0
        if i.op == D.Oshl:
            c = ir.conval(fn, i.arg[1])
            if c is None or not 0 <= c < 16:
                return 0
            args = [i.arg[0]]
        else:
            args = [i.arg[0]] if i.arg[1] == R else i.arg[:2]
        for a in args:
            if not _operand(fast, a, made):
                return 0
        made.add(rval(i.to))
    if width is None or not any(st for _, _, _, st in mem):
        return 0
    # x86-64 has no byte multiply or shift: bytes in 16-bit lanes there, widened after loading and narrowed to store
    lw = 2 if x86 and width == 1 and any(i.op in (D.Omul, D.Oshl) for i in fast.cloned) else width
    vf = _ext.qbytes() // width
    far = set()
    for i, f, n, isstore in mem:
        if not isstore:
            continue
        for j, g, m, other in mem:
            if j is i or (other and m < n):
                continue
            if not rts.may_alias(i.arg[1], j.arg[1] if other else j.arg[0]):
                continue
            if f.L.noconst().ident() != g.L.noconst().ident():
                far.add(tuple(sorted((f.L.noconst().ident(), g.L.noconst().ident()), key=str)))
            elif other:
                d = f.L.c - g.L.c
                if d != 0 and abs(d) < vf * f.S.c:
                    return 0
            elif not _distance(f, g, vf, n, m):
                return 0
    if len(far) > MAXCHECKS:
        return 0
    fast.vchecks = _checks(mem, far)
    fast.vmem = mem
    fast.vnarrow = ('bytes', width, lw)
    return vf


def _bytes_body(fast, em, state, copy):
    """body() for _bytes: each value a vector of lanes as wide as the stores."""
    fn = fast.fn
    k0 = copy * fast.vf
    w, lw = fast.vnarrow[1], fast.vnarrow[2]
    x = 'b' if lw == 1 else 'h'
    dup = D.Oqdupb if lw == 1 else D.Oqduph
    wide = lw != w  # bytes in halfword lanes: two vectors each, in order
    env = {}

    def vec(a):
        if ir.istmp(a) and rval(a) in env:
            return env[rval(a)]
        key = ('lowdup', lw, a)
        if key not in fast.memo:
            c = ir.conval(fn, a)
            if c is not None:
                s = fast.fp.con(c & (0xff if lw == 1 else 0xffff))
            else:
                em0, fast.em = fast.em, fast.fp
                s = fast.opmap(a, 0, {}, state, None)
                fast.em = em0
            fast.memo[key] = fast.fp.raw(dup, Kq, s)
        return [fast.memo[key]] * (2 if wide else 1)

    def each(op, *vs):
        return [em.raw(op, Kq, *xs) for xs in zip(*vs)]

    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc):
            continue
        if i.op in _NARROWLD[w]:
            v = em.raw(D.Oqload, Kq, fast.opmap(i.arg[0], _at(fast, k0), env, state, None))
            env[rval(i.to)] = [em.raw(D.Oquxlb, Kq, v), em.raw(D.Oquxhb, Kq, v)] if wide else [v]
            continue
        if i.op == _NARROWST[w]:
            vs = vec(i.arg[0])
            v = em.raw(D.Oqnrwh, Kq, vs[0], vs[1]) if wide else vs[0]
            em.ins(Ins(D.Oqstore, Kq, R, v, fast.opmap(i.arg[1], _at(fast, k0), env, state, None)))
            continue
        if i.op in _IDENT:
            env[rval(i.to)] = vec(i.arg[0])
            continue
        name = _LOWBITS[i.op]
        if i.op == D.Oneg:
            env[rval(i.to)] = each(getattr(D, 'Oqsub' + x), [_zero(fast, dup)] * len(vec(i.arg[0])), vec(i.arg[0]))
        elif i.op == D.Oshl:
            env[rval(i.to)] = [em.raw(getattr(D, 'Oqshl' + x), Kq, v, i.arg[1]) for v in vec(i.arg[0])]
        elif i.op in (D.Oand, D.Oor, D.Oxor):
            env[rval(i.to)] = each(getattr(D, 'O' + name), vec(i.arg[0]), vec(i.arg[1]))
        else:
            env[rval(i.to)] = each(getattr(D, 'O' + name + x), vec(i.arg[0]), vec(i.arg[1]))
    return dict(state)


def _zero(fast, dup):
    if ('zero', dup) not in fast.memo:
        fast.memo[('zero', dup)] = fast.fp.raw(dup, Kq, fast.fp.con(0))
    return fast.memo[('zero', dup)]


def _memop(i):
    """(element size, signed, class of the value, store) of a load or store, or None. A load narrower than its class
    extends (signed or not)."""
    if i.op == D.Oload and i.cls in _SIZE:
        return _SIZE[i.cls], True, i.cls, False
    ext = {D.Oloadub: (1, False), D.Oloadsb: (1, True), D.Oloaduh: (2, False), D.Oloadsh: (2, True),
           D.Oloaduw: (4, False), D.Oloadsw: (4, True)}.get(i.op)
    if ext is not None and i.cls in (Kw, Kl) and ext[0] <= _SIZE[i.cls]:
        return ext[0], ext[1], i.cls, False
    st = {D.Ostoreb: 1, D.Ostoreh: 2, D.Ostorew: 4, D.Ostorel: 8, D.Ostores: 4, D.Ostored: 8}.get(i.op)
    if st is not None:
        return st, False, {D.Ostores: Ks, D.Ostored: Kd}.get(i.op), True
    return None


_CONV = {D.Oswtof: ('qitofs', Kw, Ks), D.Ouwtof: ('quitofs', Kw, Ks), D.Osltof: ('qitofd', Kl, Kd),
         D.Oultof: ('quitofd', Kl, Kd)}
_SUBEXT = {D.Oextub: (8, False), D.Oextsb: (8, True), D.Oextuh: (16, False), D.Oextsh: (16, True)}


def _mixed(fast, rts):
    """Loops over elements of several widths (a byte array into ints, bytes summed into a long, ints shifted into
    bytes): each value in lanes of its own class, as C computes it, as many vectors as that takes for the narrowest
    element's vector of iterations (an int, for 16 bytes: four), so every operation is C's, right shifts and comparisons
    of what it can't bound included. An extending load widens its vector (ushll and ushll2, sshll; punpck on x86-64), a
    narrower store narrows pairs (qnrw: the low bits, as C's conversion), extsw and extuw widen a value's vectors, an
    integer reduction adds them all. Arrays that may overlap get run-time checks. Returns vf, or 0."""
    if not (D.T.name.startswith('arm64') or D.T.name.startswith('amd64')):
        return 0
    A = fast.A
    fn = fast.fn
    x86 = D.T.name.startswith('amd64')
    mem = []
    sizes = set()
    for n, i in enumerate(fast.cloned):
        mo = _memop(i)
        if mo is None:
            continue
        e, signed, k, st = mo
        f = A.form(i.arg[1] if st else i.arg[0], 64)
        if f is None or not f.S.isconst() or abs(f.S.c) != e or A.avlevel(f) == 'X' or not _way(fast, f.S.c):
            return 0
        mem.append((i, f, n, st))
        sizes.add(e)
    if not mem or not any(st for _, _, _, st in mem) and not any(True for p in ir.phis(fast.sh.H)
                                                                    if rval(p.to) not in A.biv):
        return 0
    w = min(sizes)
    reds = []
    for p in ir.phis(fast.sh.H):
        if rval(p.to) in A.biv:
            continue
        r = _reduction(fast, p, fast.cloned)
        if r is None:
            if _last(fast, p, fast.cloned):
                continue
            return 0
        reds.append(r)
    redins = {id(r.i) for r in reds}
    cls = {}  # tmp -> class of its lanes
    small = set()  # zero-extended bytes and shorts: never negative
    for n, i in enumerate(fast.cloned):
        if i.op in (D.Onop, D.Odbgloc) or id(i) in redins:
            continue
        mo = _memop(i)
        if mo is not None:
            e, signed, k, st = mo
            if st:
                v = i.arg[0]
                if not (ir.istmp(v) and rval(v) in cls) and not _operand(fast, v, set(cls)):
                    return 0
                vk = _stclass(fn, cls, v, e, k)
                if _SIZE[vk] < e or (k is not None and vk != k) or (k is None and vk not in (Kw, Kl)):
                    return 0
                continue
            cls[rval(i.to)] = k
            if not signed and e < 4:
                small.add(rval(i.to))
            continue
        if not ir.istmp(i.to) or i.cls not in _SIZE or _SIZE[i.cls] % w:
            return 0
        made = set(cls)
        if i.op in (D.Oextsw, D.Oextuw):
            if i.cls != Kl or not ir.istmp(i.arg[0]) or cls.get(rval(i.arg[0])) != Kw:
                return 0
        elif i.op in _SUBEXT:  # within the operand's lanes (w or l), then to i's class
            src = cls.get(rval(i.arg[0]), i.cls) if ir.istmp(i.arg[0]) else i.cls
            if not _operand(fast, i.arg[0], made) or src not in (Kw, Kl) or _SIZE[src] > _SIZE[i.cls]:
                return 0
            cls[rval(i.to)] = i.cls
            continue
        elif i.op == D.Omul and i.cls == Kl and _pow2(ir.conval(fn, i.arg[1])) is not None:
            if not _operand(fast, i.arg[0], made):  # no 64-bit multiply in NEON: a shift
                return 0
        elif i.op == D.Ocopy:
            if not _operand(fast, i.arg[0], made):
                return 0
        elif i.op in _CONV:
            name, src, dst = _CONV[i.op]
            if i.cls != dst or not ir.istmp(i.arg[0]) or cls.get(rval(i.arg[0])) != src:
                return 0
            if x86 and name != 'qitofs' and not (name == 'quitofs' and rval(i.arg[0]) in small):
                return 0  # x86-64: signed int to float only (unsigned from a byte or short is never negative)
        else:
            name = _opname(i)
            if name is None:
                return 0
            args = [i.arg[0]] if i.arg[1] == R or i.op in (D.Oshl, D.Oshr, D.Osar) else i.arg[:2]
            if i.op in (D.Oshl, D.Oshr, D.Osar):
                c = ir.conval(fn, i.arg[1])
                if c is None or not 0 < c < 8 * _SIZE[i.cls]:
                    return 0
            for a in args:
                if not _operand(fast, a, made) or (ir.istmp(a) and rval(a) in cls and cls[rval(a)] != i.cls):
                    return 0
        cls[rval(i.to)] = i.cls
    for r in reds:
        if not ir.istmp(r.v) or cls.get(rval(r.v)) != r.cls:
            if not _operand(fast, r.v, set(cls)):
                return 0
    if max((_SIZE[k] for k in cls.values()), default=w) == w and all(e == w for e in sizes):
        return 0  # one width: plan's own path
    vf = _ext.qbytes() // w
    far = set()
    for i, f, n, isstore in mem:
        if not isstore:
            continue
        for j, g, m, other in mem:
            if j is i or (other and m < n):
                continue
            if not rts.may_alias(i.arg[1], j.arg[1] if other else j.arg[0]):
                continue
            if f.L.noconst().ident() != g.L.noconst().ident():
                far.add(tuple(sorted((f.L.noconst().ident(), g.L.noconst().ident()), key=str)))
            elif f.S.c != g.S.c:
                return 0
            elif other:
                d = f.L.c - g.L.c
                if d != 0 and abs(d) < vf * f.S.c:
                    return 0
            elif not _distance(f, g, vf, n, m):
                return 0
    if len(far) > MAXCHECKS:
        return 0
    fast.vchecks = _checks(mem, far)
    fast.vmem = mem
    fast.vreds = reds
    fast.vnarrow = ('mixed', w, cls, small)
    return vf


_UX = {1: 'b', 2: 'h', 4: 'w'}


def _stclass(fn, cls, v, e, k):
    """The class of the lanes a store of e bytes stores v from: a value of the loop's, its own; an invariant, its tmp's;
    a constant, the store's (an 8-byte integer store's constant is an l: pr70021's storel of one beside a storew)."""
    if ir.istmp(v):
        return cls.get(rval(v), fn.tmp[rval(v)].cls)
    return k or (Kl if e == 8 else Kw)


def _pow2(c):
    """k if c is 2**k (0 < k < 63), else None"""
    if c is None or c <= 1 or c & (c - 1) or c >= 1 << 63:
        return None
    return c.bit_length() - 1


def _mixed_body(fast, em, state, copy):
    """body() for _mixed: env maps a value to its list of vectors, lane j of the iteration in vector j // lanes."""
    fn = fast.fn
    _, w, cls, small = fast.vnarrow
    k0 = copy * fast.vf
    q = _ext.qbytes()
    env = {}

    def parts(k):
        return _SIZE[k] // w

    def vec(a, k):
        if ir.istmp(a) and rval(a) in env:
            return env[rval(a)]
        key = ('mixdup', a, k)
        if key not in fast.memo:
            em0, fast.em = fast.em, fast.fp
            s = fast.opmap(a, 0, {}, state, None) if ir.istmp(a) else a
            fast.em = em0
            fast.memo[key] = fast.fp.raw(getattr(D, 'Oqdup' + _LANE[k]), Kq, s)
        return [fast.memo[key]] * parts(k)

    def widen(vs, e, signed, to):
        """vectors of e-byte lanes, in order, as vectors of `to`-byte lanes"""
        while e < to:
            op = 'qsx' if signed else 'qux'
            out = []
            for v in vs:
                out.append(em.raw(getattr(D, 'O' + op + 'l' + _UX[e]), Kq, v))
                out.append(em.raw(getattr(D, 'O' + op + 'h' + _UX[e]), Kq, v))
            vs, e = out, 2 * e
        return vs

    def narrow(vs, e, to):
        """vectors of e-byte lanes as vectors of their low `to` bytes"""
        while e > to:
            name = {2: 'Oqnrwh', 4: 'Oqnrww', 8: 'Oqnrwl'}[e]
            vs = [em.raw(getattr(D, name), Kq, vs[j], vs[j + 1]) for j in range(0, len(vs), 2)]
            e //= 2
        return vs

    reds = {id(r.i): r for r in fast.vreds}
    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc) or id(i) in reds:
            continue
        mo = _memop(i)
        if mo is not None:
            e, signed, k, st = mo
            if st:
                vk = _stclass(fn, cls, i.arg[0], e, k)
                vs = narrow(vec(i.arg[0], vk), _SIZE[vk], e)
                for j, v in enumerate(vs):
                    em.ins(Ins(D.Oqstore, Kq, R, v, fast.opmap(i.arg[1], _at(fast, k0, j, q // e), env, state, None)))
                continue
            vs = [em.raw(D.Oqload, Kq, fast.opmap(i.arg[0], _at(fast, k0, j, q // e), env, state, None)) for j in range(e // w)]
            env[rval(i.to)] = widen(vs, e, signed, _SIZE[k])
            continue
        k = i.cls
        if i.op in (D.Oextsw, D.Oextuw):
            env[rval(i.to)] = widen(vec(i.arg[0], Kw), 4, i.op == D.Oextsw, 8)
        elif i.op in _SUBEXT:
            bits, signed = _SUBEXT[i.op]
            src = cls.get(rval(i.arg[0]), k) if ir.istmp(i.arg[0]) else k
            x = _LANE[src]
            top = 8 * _SIZE[src] - bits
            vs = vec(i.arg[0], src)
            if signed:
                vs = [em.raw(getattr(D, 'Oqsar' + x), Kq, em.raw(getattr(D, 'Oqshl' + x), Kq, v, em.con(top)), em.con(top))
                      for v in vs]
            else:
                m = vec(fast.fp.con((1 << bits) - 1), src)[0]
                vs = [em.raw(D.Oqand, Kq, v, m) for v in vs]
            if src != k:  # w to l: the value's sign, or zeros
                vs = widen(vs, 4, signed, 8)
            env[rval(i.to)] = vs
        elif i.op == D.Omul and k == Kl and _pow2(ir.conval(fn, i.arg[1])) is not None:
            env[rval(i.to)] = [em.raw(D.Oqshll, Kq, v, em.con(_pow2(ir.conval(fn, i.arg[1])))) for v in vec(i.arg[0], Kl)]
        elif i.op == D.Ocopy:
            env[rval(i.to)] = vec(i.arg[0], k)
        elif i.op in _CONV:
            name = _CONV[i.op][0]
            if name == 'quitofs' and D.T.name.startswith('amd64'):
                name = 'qitofs'  # plan: never negative here
            env[rval(i.to)] = [em.raw(getattr(D, 'O' + name), Kq, v) for v in vec(i.arg[0], _CONV[i.op][1])]
        else:
            name = _opname(i)
            op = getattr(D, 'O' + name)
            a = vec(i.arg[0], k)
            if i.op in (D.Oshl, D.Oshr, D.Osar):
                env[rval(i.to)] = [em.raw(op, Kq, v, i.arg[1]) for v in a]
            elif i.arg[1] == R:
                env[rval(i.to)] = [em.raw(op, Kq, v) for v in a]
            else:
                b = vec(i.arg[1], k)
                env[rval(i.to)] = [em.raw(op, Kq, x, y) for x, y in zip(a, b)]
    new = dict(state)
    for r in fast.vreds:  # each copy an accumulator, all of a value's vectors into it
        x = _LANE[r.cls]
        r.acc0 = fast.fp.raw(getattr(D, 'Oqdup' + x), Kq, fast.fp.con(-1 if r.op == D.Oand else 0))
        r.in0 = newtmp('v', Kq, fn)
        r.in1 = r.ext = None
        op = getattr(D, 'O' + _REDOPS[r.op] + ('' if r.op in (D.Oand, D.Oor, D.Oxor) else x))
        acc = r.in0
        for v in vec(r.v, r.cls):
            if r.op == D.Osub:
                acc = em.raw(getattr(D, 'Oqsub' + x), Kq, acc, v)
            else:
                acc = em.raw(op, Kq, acc, v)
        r.out0 = acc
        r.out1 = None
        _keep(r, copy)
    reds_t = {r.t for r in fast.vreds}
    for p in ir.phis(fast.sh.H):
        t = rval(p.to)
        if t not in fast.A.biv and t not in reds_t:
            v = ir.phiargof(p, fast.sh.latch)
            new[('phi', t)] = state[('phi', t)] if v == p.to else v
    return new


def _narrow(fast, rts):
    """Loops over bytes (rgb2gray): unsigned byte loads, one after another or the three bytes of each pixel, arithmetic
    C does in int, and byte stores, in lanes of 16 bits, 8 a vector (16 with AVX2). Exact when no value the loop
    computes can leave [0, 65535]: then its low 16 bits are all of it, and a store keeps the low 8 of those as C's does.
    Ranges: a byte [0, 255], constants themselves, and the operation's on its operands'. The vf iterations a vector
    does read and write exactly the bytes the scalar ones did (all three of a pixel's). Returns vf, or 0."""
    if _ext.qbytes() == 32 or D.T.name.startswith('amd64'):
        if 'avx' not in _ext.x86:  # vpshufb and vpmovzxbw: VEX
            return 0
    elif not D.T.name.startswith('arm64'):
        return 0
    if fast.vreds or any(rval(p.to) not in fast.A.biv for p in ir.phis(fast.sh.H)):
        return 0
    A = fast.A
    fn = fast.fn
    rng = {}
    groups = {}   # base ident -> {offset: load}
    loads, stores = [], []

    def r(a):
        c = ir.conval(fn, a)
        if c is not None:
            return (c, c)
        return rng.get(rval(a)) if ir.istmp(a) else None

    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc):
            continue
        if i.op == D.Oloadub and i.cls == Kw:
            f = A.form(i.arg[0], 64)
            if f is None or not f.S.isconst() or f.S.c not in (1, 3) or A.avlevel(f) == 'X':
                return 0
            if f.S.c == 3:
                groups.setdefault(f.L.noconst().ident(), {})[f.L.c] = i
            loads.append((i, f))
            rng[rval(i.to)] = (0, 255)
            continue
        if i.op == D.Ostoreb:
            f = A.form(i.arg[1], 64)
            if f is None or not f.S.isconst() or f.S.c != 1 or A.avlevel(f) == 'X' or r(i.arg[0]) is None:
                return 0
            stores.append((i, f))
            continue
        if i.cls != Kw or not ir.istmp(i.to):
            return 0
        a, b = r(i.arg[0]), (r(i.arg[1]) if i.arg[1] != R else None)
        if a is None or (i.arg[1] != R and b is None):
            return 0
        op = i.op
        if op == D.Oadd:
            v = (a[0] + b[0], a[1] + b[1])
        elif op == D.Osub:
            v = (a[0] - b[1], a[1] - b[0])
        elif op == D.Omul:
            ps = [x * y for x in a for y in b]
            v = (min(ps), max(ps))
        elif op in (D.Oshr, D.Osar, D.Oshl) and b[0] == b[1] and 0 <= b[0] < 16 and ir.conval(fn, i.arg[1]) is not None:
            v = (a[0] << b[0], a[1] << b[0]) if op == D.Oshl else (a[0] >> b[0], a[1] >> b[0])
        elif op == D.Oand and b[0] == b[1] and b[0] >= 0:
            v = (0, min(a[1], b[0]))
        elif op in (D.Oor, D.Oxor) and min(a[0], b[0]) >= 0:
            v = (0, (1 << max(a[1], b[1]).bit_length()) - 1)
        else:
            return 0
        if v[0] < 0 or v[1] > 0xffff:
            return 0
        rng[rval(i.to)] = v
    if not stores:
        return 0
    if not loads and all(ir.conval(fn, st.arg[0]) is not None for st, _ in stores):
        # a fill (sieve's comp[i] = 0): lanes of bytes, a whole vector stored at a time
        for st, f in stores:
            for x, g in stores:
                if x is not st and rts.may_alias(st.arg[1], x.arg[1]):
                    return 0
        fast.vnarrow = 'fill'
        fast.vmem = ()
        return _ext.qbytes()
    for offs in groups.values():  # all three bytes of a pixel: the vector reads no byte the loop didn't
        c0 = min(offs)
        if sorted(offs) != [c0, c0 + 1, c0 + 2]:
            return 0
    # stores and loads of one array apart, other arrays not overlapping (no run-time checks here)
    for st, f in stores:
        for x, g in loads + stores:
            a1, a2 = st.arg[1], (x.arg[1] if ir.isstore(x.op) else x.arg[0])
            if x is st or not rts.may_alias(a1, a2):
                continue
            return 0
    fast.vnarrow = groups
    fast.vmem = ()
    return _ext.qbytes() // 2


def plan(fast, rts):
    """The number of lanes to vectorize this loop with, or 0. Needs fast.plan() done."""
    A = fast.A
    sh = fast.sh
    fast.vnarrow = None
    fast.vdown = None
    vf = _plan(fast, rts)
    if vf and fast.vdown and any(r.ordered or (r.op == D.Osel1 and r.index is not None) for r in fast.vreds):
        return 0  # stepping down the lanes are in the other order: only reductions whose order doesn't matter
    if vf and fast.vdown and (fast.vnarrow is not None and not isinstance(fast.vnarrow, tuple)):
        return 0  # _narrow's pixels: up only
    return vf


def _plan(fast, rts):
    A = fast.A
    sh = fast.sh
    mos = [m for m in map(_memop, fast.cloned) if m is not None]
    if any(i.op in (D.Oloadub, D.Oloadsb, D.Ostoreb, D.Oloaduh, D.Oloadsh, D.Ostoreh) for i in fast.cloned) or \
            len({m[0] for m in mos}) > 1 or any(not m[3] and m[0] < _SIZE[m[2]] for m in mos):
        vf = _bytes(fast, rts)
        if vf:
            return vf
        fast.vnarrow = None
        vf = _mixed(fast, rts)
        if vf:
            return vf
        fast.vnarrow = None
        fast.vreds = []
        if any(i.op in (D.Oloadub, D.Ostoreb) for i in fast.cloned):
            return _narrow(fast, rts)
        # shorts that _bytes didn't take: _Float16 arrays (_halves) go on below
    fast.vreds = []
    skip = set()
    rest = []
    for p in ir.phis(sh.H):
        if rval(p.to) not in A.biv:
            r = _reduction(fast, p, fast.cloned) or _select(fast, p, fast.cloned) or _ordered(fast, p, fast.cloned)
            if r is None and _last(fast, p, fast.cloned):
                continue
            if r is None:
                rest.append(p)
                continue
            if r.op == D.Osel1:
                skip.add(id(r.cmp))
                skip.add(id(r.sel0))
            x = _widening(fast, r, fast.cloned) if not r.ordered else None
            if x is not None:
                r.ext = x
                skip.add(id(x))
            fast.vreds.append(r)
            skip.add(id(r.i))
            if r.third is not None:
                skip.add(id(r.third))
    sels = [r for r in fast.vreds if r.op == D.Osel1]
    for p in rest:  # positions of select reductions' values (argmax)
        ix = _index(fast, p, fast.cloned, sels)
        if ix is None:
            return 0
        next(r for r in sels if r.cmp.to == ix.sel0.arg[0] and r.index is None).index = ix
        skip.add(id(ix.sel0))
        skip.add(id(ix.sel1))
    pure = not any(ir.isstore(i.op) or i.op == D.Ocall for b in fast.lp.blocks for i in ir.ins(b))
    for r in sels:
        r.restart = False
        if r.flt and r.index is None and pure:
            # equal floats aren't the same (two zeros): the lanes join through the condition, and where that gives a
            # zero, which zero the loop would have kept depends on the order: the scalar loop runs again from the start
            r.restart = True
            continue
        if r.flt and r.index is None and r.cls == Ks:  # an iteration count orders equal floats
            ix = Index()
            ix.q, ix.b, ix.step, ix.sel0, ix.sel1, ix.bfirst, ix.rel = None, None, 1, None, None, True, True
            r.index = ix
        if r.flt and r.index is None:  # equal floats aren't the same: an index to order them
            c = _counter(fast)
            if c is None:
                return 0
            ix = Index()
            ix.q, ix.b, ix.step, ix.sel0, ix.sel1, ix.bfirst, ix.rel = None, c[0], c[1], None, None, True, False
            r.index = ix
        if any(y.op == D.Osel0 and y.arg[0] == r.cmp.to and y is not r.sel0 and (r.index is None or y is not r.index.sel0)
               for y in fast.cloned):
            return 0
    widths = set()
    mem = []
    made = set()
    third = None
    hloads, hstores, hinner = _halves(fast)
    masks = set()  # comparisons' results: read by sel0 only, which _operand sees to (they aren't in made)
    mask = None  # the sel0 before a sel1
    for n, i in enumerate(fast.cloned):
        if i.op in (D.Onop, D.Odbgloc) or id(i) in skip or id(i) in hinner:
            continue
        if id(i) in hloads or id(i) in hstores:  # 4 halves as 4 float lanes, stepping by 2 bytes
            st = id(i) in hstores
            if st and not _operand(fast, hstores[id(i)].arg[0], made):
                return 0
            f = A.form(i.arg[1 if st else 0], 64)
            if f is None or not f.S.isconst() or f.S.c != 2 or A.avlevel(f) == 'X' or not (
                    D.T.name.startswith('arm64') or 'f16c' in _ext.x86):
                return 0
            widths.add(4)
            mem.append((i, f, n, st))
            if not st:
                made.add(rval(hloads[id(i)].to))
            continue
        c = _compare(i)
        if c is not None:
            widths.add(_SIZE[c[3]])
            if not (_operand(fast, i.arg[0], made) and _operand(fast, i.arg[1], made)):
                return 0
            masks.add(rval(i.to))
            continue
        if i.op == D.Osel0:
            if not ir.istmp(i.arg[0]) or rval(i.arg[0]) not in masks:
                return 0
            mask = i.arg[0]
            continue
        if i.op == D.Osel1:
            if mask is None or i.cls not in _LANE:
                return 0
            mask = None
            widths.add(_SIZE[i.cls])
            if not (_operand(fast, i.arg[0], made) and _operand(fast, i.arg[1], made)):
                return 0
            made.add(rval(i.to))
            continue
        if i.op == D.Ofma0:
            if i.cls not in (Ks, Kd):
                return 0
            third = i
            if not _operand(fast, i.arg[0], made):
                return 0
            continue
        if i.op == D.Ofma1:
            if third is None or i.cls not in (Ks, Kd):
                return 0
            third = None
            widths.add(_SIZE[i.cls])
            if not (_operand(fast, i.arg[0], made) and _operand(fast, i.arg[1], made)):
                return 0
            made.add(rval(i.to))
            continue
        size = _loadsize(i)
        k = _storeclass(i)
        if size is not None or k is not None:
            if k is not None:
                size = _SIZE[k]
                if not _operand(fast, i.arg[0], made):
                    return 0
            f = A.form(i.arg[1 if k is not None else 0], 64)
            if f is None or not f.S.isconst() or abs(f.S.c) != size or A.avlevel(f) == 'X' or not _way(fast, f.S.c):
                return 0
            widths.add(size)
            mem.append((i, f, n, k is not None))
            if k is None:
                made.add(rval(i.to))
            continue
        name = _opname(i)
        if name is None:
            return 0
        widths.add(_SIZE[i.cls])
        args = [i.arg[0]] if i.arg[1] == R or i.op in (D.Oshl, D.Oshr, D.Osar) else i.arg[:2]
        if i.op in (D.Oshl, D.Oshr, D.Osar):
            c = ir.conval(fast.fn, i.arg[1])
            if c is None or not 0 < c < 8 * _SIZE[i.cls]:
                return 0
        for a in args:
            if not _operand(fast, a, made):
                return 0
        made.add(rval(i.to))
    if third is not None or mask is not None or not mem:
        return 0
    for r in fast.vreds:
        if r.ordered:
            if not all(_operand(fast, a, made) for a in (r.v if isinstance(r.v, tuple) else (r.v,))):
                return 0
            widths.add(_SIZE[r.cls])
        elif r.ext is not None:
            if not _operand(fast, r.ext.arg[0], made):
                return 0
            widths.add(4)
        else:
            if not all(_operand(fast, a, made) for a in (r.v if isinstance(r.v, tuple) else (r.v,))):
                return 0
            widths.add(_SIZE[r.cls])
    if len(widths) != 1:
        return 0
    vf = _ext.qbytes() // widths.pop()
    if any(r.ordered for r in fast.vreds) and not _worth(fast, vf, skip):
        return 0
    far = set()
    for i, f, n, isstore in mem:
        if not isstore:
            continue
        for j, g, m, other in mem:
            if j is i or (other and m < n):
                continue  # each pair of stores once
            if not rts.may_alias(i.arg[1], j.arg[1] if other else j.arg[0]):
                continue
            if f.L.noconst().ident() != g.L.noconst().ident():
                far.add(tuple(sorted((f.L.noconst().ident(), g.L.noconst().ident()), key=str)))
            elif f.S.c != g.S.c:
                return 0
            elif other:
                d = f.L.c - g.L.c
                if d != 0 and abs(d) < vf * f.S.c:
                    return 0
            elif not _distance(f, g, vf, n, m):
                return 0
    if len(far) > MAXCHECKS:
        return 0
    fast.vchecks = _checks(mem, far)
    fast.vmem = mem
    return vf


def _checks(mem, far):
    """For each pair of bases that may be one array: (base, lowest offset, past the highest at t = 0) of each."""
    ranges = {}
    for i, f, n, isstore in mem:
        key = f.L.noconst().ident()
        L, lo, hi, st = ranges.get(key, (f.L.noconst(), f.L.c, f.L.c + abs(f.S.c), abs(f.S.c)))
        ranges[key] = (L, min(lo, f.L.c), max(hi, f.L.c + abs(f.S.c)), max(st, abs(f.S.c)))
    return [(ranges[a], ranges[b]) for a, b in sorted(far, key=str)]


def overlapchecks(fast, em, T, step):
    """Conditions, in em, that hold when no two of those ranges can overlap over T iterations: ends compared unsigned, so
    a range that wraps round fails. LLVM's LoopVersioning; if they fail the scalar loop runs, in the original order."""
    out = []
    for (la, alo, ahi, ast), (lb, blo, bhi, bst) in fast.vchecks:
        a, b = fast.lin(la, 64), fast.lin(lb, 64)
        # each range steps by its own elements' size (an int array beside a byte one: four times the bytes)
        asp, bsp = em.mulc(Kl, T, ast), em.mulc(Kl, T, bst)
        if fast.vdown:  # the ranges reach down from the offsets at t = 0
            a0, a1 = em.sub(Kl, em.addc(Kl, a, alo + ast), asp), em.addc(Kl, a, ahi)
            b0, b1 = em.sub(Kl, em.addc(Kl, b, blo + bst), bsp), em.addc(Kl, b, bhi)
        else:
            a0, a1 = em.addc(Kl, a, alo), em.add(Kl, em.addc(Kl, a, ahi - ast), asp)
            b0, b1 = em.addc(Kl, b, blo), em.add(Kl, em.addc(Kl, b, bhi - bst), bsp)
        out.append(em.raw(D.Oor, Kw, em.cmp(D.Ocugel, b0, a1), em.cmp(D.Ocugel, a0, b1)))
    return out


def _operand(fast, a, made):
    """a can be had as a vector: made in the loop, or the same on every iteration."""
    if not ir.istmp(a):
        return True  # a constant
    if not fast.A.inloop(a):
        return True
    t = rval(a)
    if t in made:
        return True
    f = fast.A.forms.get(t)
    return f is not None and f.inv()


INTERLEAVE = 4  # vectors an iteration, each with its own accumulators, where that's allowed: LLVM's interleave count


def interleave(fast):
    """How many vectors the loop does an iteration: more than one only for a loop that collects reductions whose
    order C doesn't fix and stores nothing, so each copy keeps its own accumulators (a chain of compares and selects,
    or of adds, is latency-bound in one), and stores never come closer than _distance allowed for."""
    stores = any(ir.isstore(i.op) for b in fast.lp.blocks for i in ir.ins(b))
    if stores and not fast.vreds:
        # a loop that stores, as gemm's c[j] += a * b[j]: two vectors an iteration, fewer of the loop's own instructions;
        # four when an iteration is a long chain of dependent float operations (conv3x3's nine fmas in C's order,
        # jacobi2d's four adds), so more of them are in flight than the core's window would otherwise hold
        return 4 if _chain(fast) >= 4 and D.T.name.startswith('arm64') else 2  # x86-64's windows are larger
    if not fast.vreds or any(r.ordered for r in fast.vreds):
        return 1
    if stores:
        return 1
    if any(r.op == D.Osel1 for r in fast.vreds) and D.T.name.startswith('arm64'):
        # a compare and a select on each accumulator's chain, about 5 cycles: 8 of them in flight (32 registers; x86-64
        # has 16, and an index doubles them). Snapdragon X, 1M doubles: max 196 to 121 us (numpy 114)
        return 2 * INTERLEAVE
    return INTERLEAVE


def _chain(fast):
    """The longest run of dependent float operations in an iteration."""
    depth = {}
    best = 0
    for i in [x for b in fast.sh.order for x in ir.ins(b)]:
        if not ir.istmp(i.to) or i.cls not in (Ks, Kd) or ir.isload(i.op):
            continue
        d = 1 + max([depth.get(rval(a), 0) for a in i.arg if ir.istmp(a)] + [0])
        depth[rval(i.to)] = d
        best = max(best, d)
    return best


def _apart(fast, lanes):
    """Whether every store and any access to the same base plan() checked are the same element or lanes apart: then
    copies of the body, one after the other, still see each element in the scalar loop's order."""
    for i, f, n, isstore in fast.vmem:
        if not isstore:
            continue
        for j, g, m, other in fast.vmem:
            if j is i or f.L.noconst().ident() != g.L.noconst().ident():
                continue  # other bases: told apart by the alias analysis or the run-time checks, over every iteration
            d = f.L.c - g.L.c
            if d != 0 and abs(d) < lanes * abs(f.S.c):
                return False
    return True


_NARROWOPS = {D.Oadd: 'qaddh', D.Osub: 'qsubh', D.Omul: 'qmulh', D.Oshl: 'qshlh', D.Oshr: 'qshrh', D.Osar: 'qshrh',
              D.Oand: 'qand', D.Oor: 'qor', D.Oxor: 'qxor'}  # sar of what _narrow knows is never negative: shr


def _narrow_body(fast, em, state, copy):
    """body() for a loop over bytes (_narrow): 16-bit lanes; a pixel's three loads together. On arm64, with one pixel
    group, one ld3 into v28-v30 serves them and a channel times a byte constant is umull, or umlal into what it's
    added to (gcc's widening multiply-accumulate): rgb2gray's three weights are three instructions."""
    fn = fast.fn
    A = fast.A
    k0 = copy * fast.vf
    env = {}
    pixels = {}
    pending = {}  # tmp -> ('ch', k): channel k of the ld3; ('mul', c, k): channel k times c, not made yet
    arm = D.T.name.startswith('arm64') and len(fast.vnarrow) == 1
    # with two copies an iteration, the first loads both copies' 16 pixels with one ld3 and the second reads the upper
    # halves (umull2, umlal2): gcc's 16 pixels an ld3
    hi = arm and fast.ic == 2 and copy == 1
    wide = arm and fast.ic == 2
    uses = {}
    for i in fast.cloned:
        for a in i.arg:
            if ir.istmp(a):
                uses[rval(a)] = uses.get(rval(a), 0) + 1

    def dupb(c):
        if ('dupb', c) not in fast.memo:
            fast.memo[('dupb', c)] = fast.fp.raw(D.Oqdupb, Kq, fast.fp.con(c))
        return fast.memo[('dupb', c)]

    def vec(a):
        if ir.istmp(a) and rval(a) in pending:
            p = pending.pop(rval(a))
            if p[0] == 'ch':
                env[rval(a)] = em.raw(D.Oqwide3h if hi else D.Oqwide3, Kq, em.con(p[1]))
            elif p[0] == 'shr':
                env[rval(a)] = em.raw(D.Oqshrh, Kq, vec(p[1]), em.con(p[2]))
            else:
                env[rval(a)] = em.raw(D.Oqwmul3h if hi else D.Oqwmul3, Kq, dupb(p[1]), em.con(p[2]))
        if ir.istmp(a) and rval(a) in env:
            return env[rval(a)]
        c = ir.conval(fn, a)
        key = ('duph', c)
        if key not in fast.memo:
            fast.memo[key] = fast.fp.raw(D.Oqduph, Kq, fast.fp.con(c & 0xffff))
        return fast.memo[key]

    def channel(a):
        """k when a is an ld3 channel nothing else reads, else None."""
        p = pending.get(rval(a)) if ir.istmp(a) else None
        return p[1] if p is not None and p[0] == 'ch' and uses.get(rval(a)) == 1 else None

    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc):
            continue
        if i.op == D.Oloadub:
            f = A.form(i.arg[0], 64)
            if f.S.c == 3:
                key = f.L.noconst().ident()
                offs = fast.vnarrow[key]
                c0 = min(offs)
                if key not in pixels:
                    base = fast.opmap(offs[c0].arg[0], k0, env, state, None)
                    if arm:
                        if not hi:
                            em.ins(Ins(D.Oqld3x16 if wide else D.Oqld3x, Kq, R, base, R))
                        pixels[key] = None
                    else:  # x86-64: the 24 bytes as two windows, each channel two vpshufb and an or
                        wa, wb = em.raw(D.Oqxwin, Kq, base, em.con(0)), em.raw(D.Oqxwin, Kq, base, em.con(8))
                        pixels[key] = {c: em.raw(D.Oqor, Kq, em.raw(D.Oqxshufb, Kq, wa, em.con(2 * (c - c0))),
                                                 em.raw(D.Oqxshufb, Kq, wb, em.con(2 * (c - c0) + 1)))
                                       for c in sorted(offs)}
                if arm:
                    pending[rval(i.to)] = ('ch', f.L.c - c0)
                else:
                    env[rval(i.to)] = pixels[key][f.L.c]
            else:
                env[rval(i.to)] = em.raw(D.Oqldbh, Kq, fast.opmap(i.arg[0], k0, env, state, None))
            continue
        if i.op == D.Ostoreb:
            p = pending.get(rval(i.arg[0])) if ir.istmp(i.arg[0]) else None
            if p is not None and p[0] == 'shr':  # shrn: the shift and the narrowing in one
                pending.pop(rval(i.arg[0]))
                v = vec(p[1])
                em.ins(Ins(D.Oqshrn0, Kq, R, v, R))
                em.ins(Ins(D.Oqstshrn, Kq, R, fast.opmap(i.arg[1], k0, env, state, None), em.con(p[2])))
                continue
            em.ins(Ins(D.Oqstbh, Kq, R, vec(i.arg[0]), fast.opmap(i.arg[1], k0, env, state, None)))
            continue
        if arm and i.op in (D.Oshr, D.Osar) and uses.get(rval(i.to)) == 1 and \
                1 <= (ir.conval(fn, i.arg[1]) or 0) <= 8 and ir.istmp(i.arg[0]):
            pending[rval(i.to)] = ('shr', i.arg[0], ir.conval(fn, i.arg[1]))
            continue
        if arm and i.op == D.Omul:
            for x, y in ((i.arg[0], i.arg[1]), (i.arg[1], i.arg[0])):
                k, c = channel(x), ir.conval(fn, y)
                if k is not None and c is not None and 0 <= c <= 255:
                    pending.pop(rval(x))
                    pending[rval(i.to)] = ('mul', c, k)
                    break
            if rval(i.to) in pending:
                continue
        if arm and i.op == D.Oadd:
            done = False
            for x, y in ((i.arg[0], i.arg[1]), (i.arg[1], i.arg[0])):
                p = pending.get(rval(x)) if ir.istmp(x) else None
                if p is not None and p[0] == 'mul' and uses.get(rval(x)) == 1:
                    acc = vec(y)
                    pending.pop(rval(x))
                    em.ins(Ins(D.Oqwmla0, Kq, R, acc, R))
                    env[rval(i.to)] = em.raw(D.Oqwmla3h if hi else D.Oqwmla3, Kq, dupb(p[1]), em.con(p[2]))
                    done = True
                    break
            if done:
                continue
        op = getattr(D, 'O' + _NARROWOPS[i.op])
        if i.op in (D.Oshl, D.Oshr, D.Osar):
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0]), i.arg[1])
        else:
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0]), vec(i.arg[1]))
    return dict(state)


def body(fast, em, state, copy=0):
    """One vector's worth of an iteration of the vector loop, in em, the copy-th of interleave(fast); state only has
    induction variables, which it leaves for advance()."""
    if fast.vnarrow == 'fill':
        for i in fast.cloned:
            if i.op == D.Ostoreb:
                c = ir.conval(fast.fn, i.arg[0]) & 0xff
                if ('dupb', c) not in fast.memo:
                    fast.memo[('dupb', c)] = fast.fp.raw(D.Oqdupb, Kq, fast.fp.con(c))
                em.ins(Ins(D.Oqstore, Kq, R, fast.memo[('dupb', c)],
                           fast.opmap(i.arg[1], copy * fast.vf, {}, state, None)))
        return dict(state)
    if isinstance(fast.vnarrow, tuple) and fast.vnarrow[0] == 'bytes':
        return _bytes_body(fast, em, state, copy)
    if isinstance(fast.vnarrow, tuple) and fast.vnarrow[0] == 'mixed':
        return _mixed_body(fast, em, state, copy)
    if fast.vnarrow is not None:
        return _narrow_body(fast, em, state, copy)
    fn = fast.fn
    env = {}
    k0 = copy * fast.vf  # the scalar iteration this copy starts at
    third = None
    lane = None
    # within the iteration, by address form: plan() kept every address on one base and step (others the run-time checks
    # keep apart), so the same form is the same vector and other offsets of a base overlap only when less than a
    # vector apart
    avail = {}    # key -> vector there
    pending = {}  # key -> store nothing has read yet
    span = _ext.qbytes()

    def mkey(a):
        f = fast.A.form(a, 64)
        return (f.L.noconst().ident(), f.S.c, f.L.c)

    def overlaps(x, key):
        return x != key and x[:2] == key[:2] and abs(x[2] - key[2]) < span

    def vec(a, k):
        if ir.istmp(a) and rval(a) in env:
            return env[rval(a)]
        key = ('dup', a, k)
        if key not in fast.memo:
            em0, fast.em = fast.em, fast.fp
            s = fast.opmap(a, 0, {}, state, None) if ir.istmp(a) else a
            fast.em = em0
            fast.memo[key] = fast.fp.raw(getattr(D, 'Oqdup' + _LANE[k]), Kq, s)
        return fast.memo[key]

    skip = set()
    for r in fast.vreds:
        skip.add(id(r.i))
        if r.op == D.Osel1:
            skip.add(id(r.cmp))
            skip.add(id(r.sel0))
            if r.index is not None and r.index.sel0 is not None:
                skip.add(id(r.index.sel0))
                skip.add(id(r.index.sel1))
        if r.ext is not None:
            skip.add(id(r.ext))
        if r.third is not None:
            skip.add(id(r.third))
    hloads, hstores, hinner = _halves(fast)  # this Fast's instructions (plan() looked at a probe's)
    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc) or id(i) in skip or id(i) in hinner:
            continue
        if id(i) in hloads or id(i) in hstores:  # half arrays: kept out of avail (their vectors are floats)
            st = id(i) in hstores
            key = mkey(i.arg[1] if st else i.arg[0])
            for x in [x for x in avail if overlaps(x, key) or x == key]:
                del avail[x]
            for p in [p for p in pending if overlaps(p, key) or p == key]:
                del pending[p]
            if st:
                em.ins(Ins(D.Oqsth, Kq, R, vec(hstores[id(i)].arg[0], Ks), fast.opmap(i.arg[1], k0, env, state, None)))
            else:
                env[rval(hloads[id(i)].to)] = em.raw(D.Oqldh, Kq, fast.opmap(i.arg[0], k0, env, state, None))
            continue
        c = _compare(i)
        if c is not None:
            name, swap, negate, k = c
            a, b = vec(i.arg[0], k), vec(i.arg[1], k)
            m = em.raw(getattr(D, 'O' + name + _LANE[k]), Kq, *((b, a) if swap else (a, b)))
            env[rval(i.to)] = em.raw(D.Oqnot, Kq, m) if negate else m
            continue
        if i.op == D.Osel0:
            em.ins(Ins(D.Oqsel0, Kq, R, env[rval(i.arg[0])], R))
            continue
        if i.op == D.Osel1:
            env[rval(i.to)] = em.raw(D.Oqsel1, Kq, vec(i.arg[0], i.cls), vec(i.arg[1], i.cls))
            continue
        if i.op == D.Ofma0:
            third = vec(i.arg[0], i.cls)
            continue
        if i.op == D.Ofma1:
            a, b = vec(i.arg[0], i.cls), vec(i.arg[1], i.cls)
            em.ins(Ins(D.Oqfma0, Kq, R, third, R))
            env[rval(i.to)] = em.raw(getattr(D, 'Oqfma1' + _LANE[i.cls]), Kq, a, b)
            continue
        if _loadsize(i) is not None:
            key = mkey(i.arg[0])
            if key in avail:
                env[rval(i.to)] = avail[key]  # what this iteration stored there (unroll-and-jam's c[j], row k then k')
                continue
            for p in [p for p in pending if overlaps(p, key)]:
                del pending[p]
            env[rval(i.to)] = avail[key] = em.raw(D.Oqload, Kq, fast.opmap(i.arg[0], _at(fast, k0), env, state, None))
            continue
        k = _storeclass(i)
        if k is not None:
            v = vec(i.arg[0], k)
            key = mkey(i.arg[1])
            if key in pending:
                em.out.remove(pending[key])  # overwritten before anything read it
            for x in [x for x in avail if overlaps(x, key)]:
                del avail[x]
            for p in [p for p in pending if overlaps(p, key)]:
                del pending[p]
            st = Ins(D.Oqstore, Kq, R, v, fast.opmap(i.arg[1], _at(fast, k0), env, state, None))
            em.ins(st)
            avail[key] = v
            pending[key] = st
            continue
        name = _opname(i)
        op = getattr(D, 'O' + name)
        if i.op in (D.Oshl, D.Oshr, D.Osar):
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0], i.cls), i.arg[1])
        elif i.arg[1] == R:
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0], i.cls))
        else:
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0], i.cls), vec(i.arg[1], i.cls))
    new = dict(state)
    for r in fast.vreds:
        if r.ordered:  # lane by lane, in the order the scalar loop adds them
            x = _LANE[r.cls]
            get = getattr(D, 'Oqget' + x)
            acc = state[('phi', r.t)]
            vs = [vec(a, r.cls) for a in (r.v if isinstance(r.v, tuple) else (r.v,))]
            for n in range(_ext.qbytes() // _SIZE[r.cls]):
                lanes = [em.raw(get, r.cls, w, em.con(n)) for w in vs]
                if r.op == D.Ofma1:
                    em.ins(Ins(D.Ofma0, r.cls, R, acc, R))
                    acc = em.raw(D.Ofma1, r.cls, lanes[0], lanes[1], contract=r.i.contract)
                else:
                    acc = em.raw(r.op, r.cls, acc, lanes[0])
            new[('phi', r.t)] = acc
            continue
        if r.op == D.Osel1:  # through a condition: the comparison and the select on each lane
            r.acc0 = fast.fp.raw(getattr(D, 'Oqdup' + _LANE[r.cls]), Kq, fast.fp.con(r.identity))
            r.in0 = newtmp('v', Kq, fn)
            r.in1 = None
            name, swap, negate, k = _compare(r.cmp)
            phi = r.i.arg[0] if r.i.arg[1] == r.v else r.i.arg[1]
            pick = lambda a, phi=phi, k=k: r.in0 if a == phi else vec(a, k)
            a, b = pick(r.cmp.arg[0]), pick(r.cmp.arg[1])
            m = em.raw(getattr(D, 'O' + name + _LANE[k]), Kq, *((b, a) if swap else (a, b)))
            if negate:
                m = em.raw(D.Oqnot, Kq, m)
            em.ins(Ins(D.Oqsel0, Kq, R, m, R))
            r.out0 = em.raw(D.Oqsel1, Kq, pick(r.i.arg[0]), pick(r.i.arg[1]))
            r.out1 = None
            ix = r.index
            if ix is not None and ix.rel:  # 32-bit lanes: 1 + the iteration count, the same on every lane (0: none)
                ix.acc0 = fast.fp.raw(D.Oqdupw, Kq, fast.fp.con(0))
                ix.in0 = newtmp('v', Kq, fn)
                bv = _count(fast, em, copy, k0)
                em.ins(Ins(D.Oqsel0, Kq, R, m, R))
                ix.out0 = em.raw(D.Oqsel1, Kq, *((bv, ix.in0) if ix.bfirst else (ix.in0, bv)))
                if copy == 0:
                    ix.copies = []
                ix.copies.append((ix.acc0, ix.in0, ix.out0))
            elif ix is not None:  # the position beside each lane: b's value this copy, on every lane (closeloop adds n)
                up = ix.step > 0  # positions grow with the iterations
                ix.acc0 = fast.fp.raw(D.Oqdupl, Kq, fast.fp.con(_SENTINEL if r.later != up else -_SENTINEL))
                ix.in0 = newtmp('v', Kq, fn)
                bk = fast.opmap(ix.b, k0, env, state, None)
                bv = em.raw(D.Oqdupl, Kq, bk)
                em.ins(Ins(D.Oqsel0, Kq, R, m, R))
                ix.out0 = em.raw(D.Oqsel1, Kq, *((bv, ix.in0) if ix.bfirst else (ix.in0, bv)))
                if copy == 0:
                    ix.copies = []
                ix.copies.append((ix.acc0, ix.in0, ix.out0))
            _keep(r, copy)
            continue
        if r.cls in (Ks, Kd):
            identity = fast.fp.raw(getattr(D, 'Oqdup' + _LANE[r.cls]), Kq, fast.fp.con(r.identity))
        else:
            identity = fast.fp.raw(getattr(D, 'Oqdup' + _LANE[r.cls]), Kq, fast.fp.con(-1 if r.op == D.Oand else 0))
        r.acc0 = identity
        r.in0 = newtmp('v', Kq, fn)
        if r.ext is not None:
            signed = r.ext.op == D.Oextsw
            r.in1 = newtmp('v', Kq, fn)
            w = vec(r.ext.arg[0], Kw)
            low, high = (D.Oqsaddl, D.Oqsaddh) if signed else (D.Oquaddl, D.Oquaddh)
            r.out0 = em.raw(low, Kq, r.in0, w)
            r.out1 = em.raw(high, Kq, r.in1, w)
            _keep(r, copy)
            continue
        r.in1 = None
        if r.op == D.Ofma1:
            em.ins(Ins(D.Oqfma0, Kq, R, r.in0, R))
            r.out0 = em.raw(getattr(D, 'Oqfma1' + _LANE[r.cls]), Kq, vec(r.v[0], r.cls), vec(r.v[1], r.cls))
            r.out1 = None
            _keep(r, copy)
            continue
        op = getattr(D, 'O' + _FREDOPS[r.op] + ('' if r.op in (D.Oand, D.Oor, D.Oxor) else _LANE[r.cls]))
        r.out0 = em.raw(op, Kq, r.in0, vec(r.v, r.cls))
        r.out1 = None
        _keep(r, copy)
    reds = {r.t for r in fast.vreds} | {rval(r.index.q.to) for r in fast.vreds
                                        if r.op == D.Osel1 and r.index is not None and r.index.q is not None}
    for p in ir.phis(fast.sh.H):  # the rest set to one value each iteration (_last)
        t = rval(p.to)
        if t not in fast.A.biv and t not in reds:
            v = ir.phiargof(p, fast.sh.latch)
            new[('phi', t)] = state[('phi', t)] if v == p.to else v  # phi(x, itself): x throughout
    return new


def _count(fast, em, copy, k0):
    """1 + the iteration count from the vector loop's start of this copy's first lane, on every lane (Index.rel):
    a vector counter, its phi in closeloop, stepping by an iteration's lanes after the last copy."""
    c = fast.vcount
    if copy == 0:
        if c is None:
            c = fast.vcount = {}
        if 'in0' not in c:
            c['acc0'] = fast.fp.raw(D.Oqdupw, Kq, fast.fp.con(1))
            c['in0'] = newtmp('v', Kq, fast.fn)
        c['cur'] = {}
    if copy not in c['cur']:
        v = c['in0']
        if k0:
            v = em.raw(D.Oqaddw, Kq, v, fast.fp.raw(D.Oqdupw, Kq, fast.fp.con(k0)))
        c['cur'][copy] = v
        if copy == fast.ic - 1:
            c['out0'] = em.raw(D.Oqaddw, Kq, c['in0'], fast.fp.raw(D.Oqdupw, Kq, fast.fp.con(fast.vf * fast.ic)))
    return c['cur'][copy]


_SENTINEL = 1 << 62  # a lane's index before it takes a value: loses every tie (minus it where the later one wins)


def _closerel(fast, em, MB, FP, r, totals, state, out):
    """_closeindexed for 32-bit lanes (Index.rel): each lane's 1 + count (0: never took one), the position of
    count c b's value then: b now - (iterations done - c) * step."""
    ix = r.index
    cnt = fast.vcount
    if 'phi' not in cnt:
        ir.addphi(MB, Kq, cnt['in0'], [(FP, cnt['acc0']), (MB, cnt['out0'])])
        cnt['phi'] = True
    done = em.raw(D.Oextuw, Kl, em.raw(D.Oqgetw, Kw, cnt['out0'], em.con(0)), R)  # 1 + iterations done
    idxs = []
    for acc0, in0, out0 in ix.copies:
        ir.addphi(MB, Kq, in0, [(FP, acc0), (MB, out0)])
        idxs.append(out0)
    k = r.cls
    x = _LANE[k]
    lanes = _ext.qbytes() // _SIZE[k]
    v = state[('phi', r.t)]
    # in counts: the scalar value's comes before every lane's (0)
    idx = em.con(0)
    phi = r.i.arg[0] if r.i.arg[1] == r.v else r.i.arg[1]
    eqop = {Ks: D.Oceqs, Kw: D.Oceqw}[k]
    for total, itotal in zip(totals, idxs):
        for n in range(lanes):
            lv = em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n))
            li = em.raw(D.Oextuw, Kl, em.raw(D.Oqgetw, Kw, itotal, em.con(n)), R)
            some = em.raw(D.Ocnel, Kw, li, em.con(0))
            if n:
                li = em.raw(D.Oadd, Kl, li, em.con(n))
            at = lambda a: v if a == phi else lv
            cond = em.raw(r.cmp.op, Kw, at(r.cmp.arg[0]), at(r.cmp.arg[1]))
            wins = cond if r.i.arg[0] == r.v else em.raw(D.Oxor, Kw, cond, em.con(1))
            eq = em.raw(eqop, Kw, lv, v)
            tie = em.raw(D.Ocugtl if r.later else D.Ocultl, Kw, li, idx)
            if not r.later:  # the scalar value's count 0 is first: a lane only ties earlier than another lane
                tie = em.raw(D.Oand, Kw, tie, em.raw(D.Ocnel, Kw, idx, em.con(0)))
            take = em.raw(D.Oor, Kw, em.raw(D.Oand, Kw, wins, em.raw(D.Oxor, Kw, eq, em.con(1))),
                          em.raw(D.Oand, Kw, eq, tie))
            take = em.raw(D.Oand, Kw, take, some)
            em.ins(Ins(D.Osel0, Kw, R, take, R))
            v = em.raw(D.Osel1, k, lv, v)
            em.ins(Ins(D.Osel0, Kw, R, take, R))
            idx = em.raw(D.Osel1, Kl, li, idx)
    out[('phi', r.t)] = v
    if ix.q is not None:  # count c: b now - (done - c) * step; none taken: the scalar index
        em0, fast.em = fast.em, em
        try:
            b = fast.opmap(ix.b, 0, {}, state, None)
        finally:
            fast.em = em0
        pos = em.raw(D.Osub, Kl, b, em.raw(D.Omul, Kl, em.raw(D.Osub, Kl, done, idx), em.con(ix.step)))
        anyl = em.raw(D.Ocnel, Kw, idx, em.con(0))
        em.ins(Ins(D.Osel0, Kw, R, anyl, R))
        out[('phi', rval(ix.q.to))] = em.raw(D.Osel1, Kl, pos, state[('phi', rval(ix.q.to))])


def _closeindexed(fast, em, MB, FP, r, totals, state, out):
    """A select reduction with positions (argmax, or a float maximum): each lane (value, index) into the scalar pair:
    taken where the loop's condition picks it over the value so far, or, equal, where its index comes first (later, if
    equal ones win: >=). The scalar pair is the loop's before the vector loop, whose positions all come before the lanes'
    (a hidden index: below every lane's)."""
    ix = r.index
    idxs = []
    for acc0, in0, out0 in ix.copies:
        ir.addphi(MB, Kq, in0, [(FP, acc0), (MB, out0)])
        idxs.append(out0)
    k = r.cls
    x = _LANE[k]
    lanes = _ext.qbytes() // _SIZE[k]
    v = state[('phi', r.t)]
    up = ix.step > 0
    idx = state[('phi', rval(ix.q.to))] if ix.q is not None else em.con(-(_SENTINEL >> 1) if up else _SENTINEL >> 1)
    phi = r.i.arg[0] if r.i.arg[1] == r.v else r.i.arg[1]
    eqop = D.Oceqd if k == Kd else D.Oceql
    for total, itotal in zip(totals, idxs):
        for n in range(lanes):
            lv = em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n))
            li = em.raw(D.Oqgetl, Kl, itotal, em.con(n))
            if n:
                li = em.raw(D.Oadd, Kl, li, em.con(n * ix.step))
            at = lambda a: v if a == phi else lv
            cond = em.raw(r.cmp.op, Kw, at(r.cmp.arg[0]), at(r.cmp.arg[1]))
            wins = cond if r.i.arg[0] == r.v else em.raw(D.Oxor, Kw, cond, em.con(1))
            eq = em.raw(eqop, Kw, lv, v)
            tie = em.raw(D.Ocsgtl if r.later == up else D.Ocsltl, Kw, li, idx)  # after (>=) or before, in the loop's order
            take = em.raw(D.Oor, Kw, em.raw(D.Oand, Kw, wins, em.raw(D.Oxor, Kw, eq, em.con(1))),
                          em.raw(D.Oand, Kw, eq, tie))
            em.ins(Ins(D.Osel0, Kw, R, take, R))
            v = em.raw(D.Osel1, k, lv, v)
            em.ins(Ins(D.Osel0, Kw, R, take, R))
            idx = em.raw(D.Osel1, Kl, li, idx)
    out[('phi', r.t)] = v
    if ix.q is not None:
        out[('phi', rval(ix.q.to))] = idx


def _keep(r, copy):
    """Each copy's accumulators, for closeloop."""
    if copy == 0:
        r.copies = []
    r.copies.append((r.acc0, r.in0, r.in1, r.out0, r.out1))


def closeloop(fast, em, MB, FP, state):
    """The accumulators' phis in MB, and, in em, a block after it, the scalar values with what they've collected."""
    out = dict(state)
    fast.vrestart = None
    for r in fast.vreds:
        if r.ordered:
            continue  # already in the scalar value
        totals = []
        for acc0, in0, in1, out0, out1 in r.copies:
            ir.addphi(MB, Kq, in0, [(FP, acc0), (MB, out0)])
            if r.ext is not None:
                ir.addphi(MB, Kq, in1, [(FP, acc0), (MB, out1)])
                totals.append(em.raw(D.Oqaddl, Kq, out0, out1))
            else:
                totals.append(out0)
        k = r.cls
        x = _LANE[k]
        lanes = _ext.qbytes() // _SIZE[k]
        v = state[('phi', r.t)]
        if r.op == D.Osel1 and r.index is not None:
            (_closerel if r.index.rel else _closeindexed)(fast, em, MB, FP, r, totals, state, out)
            continue
        if r.op == D.Osel1:  # each lane into the scalar value as the loop would: s = cond(lane, s) ? ... : ...
            phi = r.i.arg[0] if r.i.arg[1] == r.v else r.i.arg[1]
            for total, n in [(t, n) for t in totals for n in range(lanes)]:
                lane = em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n))
                at = lambda a: v if a == phi else lane
                c = em.raw(r.cmp.op, Kw, at(r.cmp.arg[0]), at(r.cmp.arg[1]))
                em.ins(Ins(D.Osel0, Kw, R, c, R))
                v = em.raw(D.Osel1, k, at(r.i.arg[0]), at(r.i.arg[1]))
            out[('phi', r.t)] = v
            if r.restart:  # a zero: the scalar loop again from the start (inner.emit_versioned)
                z = em.raw(D.Oceqd if k == Kd else D.Oceqs, Kw, v, em.con(0))
                fast.vrestart = z if fast.vrestart is None else em.raw(D.Oor, Kw, fast.vrestart, z)
            continue
        op = D.Oadd if r.op in (D.Oadd, D.Osub, D.Ofma1) else r.op
        # the copies' accumulators into one, lane by lane (a subtraction's each hold 0 - their part: added)
        vop = getattr(D, 'O' + _FREDOPS[op] + ('' if op in (D.Oand, D.Oor, D.Oxor) else ('l' if r.ext is not None else x)))
        total = totals[0]
        for t in totals[1:]:
            total = em.raw(vop, Kq, total, t)
        for n in range(lanes):
            v = em.raw(op, k, v, em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n)))
        out[('phi', r.t)] = v
    return out
