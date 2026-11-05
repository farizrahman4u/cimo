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
  gcc's fold-left reductions, when what else the loop does is worth it (_worth);
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


def _loadsize(i):
    if i.op == D.Oload and i.cls in _SIZE:
        return _SIZE[i.cls]
    if i.op in (D.Oloadsw, D.Oloaduw) and i.cls == Kw:
        return 4
    return None


def _storeclass(i):
    return {D.Ostorew: Kw, D.Ostorel: Kl, D.Ostores: Ks, D.Ostored: Kd}.get(i.op)


def _distance(store, load, vf, storeat, loadat):
    """store and load are forms with one base and step. d = store's address - load's. Nothing changes if they're the
    same element, or a vector's worth apart. A load ahead of a store sees what it saw before only if the vector load
    comes first, since the vector store would write what the scalar loop hasn't yet."""
    d = store.L.c - load.L.c
    size = abs(store.S.c)
    if d == 0 or abs(d) >= vf * size:
        return True
    return d < 0 and loadat < storeat


class Reduction:
    """s = s op v, round the loop: t is s's phi, op the instruction, v the operand, ext how v comes from 32-bit lanes."""
    __slots__ = ('t', 'i', 'op', 'v', 'ext', 'acc0', 'in0', 'in1', 'out0', 'out1', 'cls', 'ordered', 'third',
                 'cmp', 'sel0', 'identity', 'copies')


_REDOPS = {D.Oadd: 'qadd', D.Osub: 'qsub', D.Oand: 'qand', D.Oor: 'qor', D.Oxor: 'qxor'}


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
    integer, so the order doesn't matter; floats aren't taken (two zeros, NaNs)."""
    if p.cls not in (Kw, Kl):
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
    if any(cmp.to in y.arg[:2] for y in cloned if y is not sel0):
        return None
    for q in ir.phis(fast.sh.H):
        if q is not p and (p.to in q.arg[:q.narg] or nxt in q.arg[:q.narg]):
            return None
    bits = 8 * _SIZE[p.cls]
    unsigned = c[0].startswith('qcu')
    if greater:
        identity = 0 if unsigned else -(1 << (bits - 1))
    else:
        identity = -1 if unsigned else (1 << (bits - 1)) - 1
    r = Reduction()
    r.t, r.i, r.op, r.v, r.cls = rval(p.to), sel1, D.Osel1, x, p.cls
    r.ext = r.third = None
    r.ordered = False
    r.cmp, r.sel0, r.identity = cmp, sel0, identity
    return r


def _ordered(fast, p, cloned):
    """p as a float Reduction done in order, or None: s = s + v, s - v or fma(a, b, s), nothing else reading s or the
    new value but the phi."""
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
    r.ordered = True
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


def plan(fast, rts):
    """The number of lanes to vectorize this loop with, or 0. Needs fast.plan() done."""
    A = fast.A
    sh = fast.sh
    fast.vreds = []
    skip = set()
    for p in ir.phis(sh.H):
        if rval(p.to) not in A.biv:
            r = _reduction(fast, p, fast.cloned) or _select(fast, p, fast.cloned) or _ordered(fast, p, fast.cloned)
            if r is None:
                return 0
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
    widths = set()
    mem = []
    made = set()
    third = None
    masks = set()  # comparisons' results: read by sel0 only, which _operand sees to (they aren't in made)
    mask = None  # the sel0 before a sel1
    for n, i in enumerate(fast.cloned):
        if i.op in (D.Onop, D.Odbgloc) or id(i) in skip:
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
            if f is None or not f.S.isconst() or f.S.c != size or A.avlevel(f) == 'X':
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
            if not _operand(fast, r.v, made):
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
    return vf


def _checks(mem, far):
    """For each pair of bases that may be one array: (base, lowest offset, past the highest at t = 0) of each."""
    ranges = {}
    for i, f, n, isstore in mem:
        key = f.L.noconst().ident()
        L, lo, hi = ranges.get(key, (f.L.noconst(), f.L.c, f.L.c + f.S.c))
        ranges[key] = (L, min(lo, f.L.c), max(hi, f.L.c + f.S.c))
    return [(ranges[a], ranges[b]) for a, b in sorted(far, key=str)]


def overlapchecks(fast, em, T, step):
    """Conditions, in em, that hold when no two of those ranges can overlap over T iterations: ends compared unsigned, so
    a range that wraps round fails. LLVM's LoopVersioning; if they fail the scalar loop runs, in the original order."""
    out = []
    span = em.mulc(Kl, T, step)
    for (la, alo, ahi), (lb, blo, bhi) in fast.vchecks:
        a, b = fast.lin(la, 64), fast.lin(lb, 64)
        a0, a1 = em.addc(Kl, a, alo), em.add(Kl, em.addc(Kl, a, ahi - step), span)
        b0, b1 = em.addc(Kl, b, blo), em.add(Kl, em.addc(Kl, b, bhi - step), span)
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
    if not fast.vreds or any(r.ordered for r in fast.vreds):
        return 1
    if any(_storeclass(i) is not None for b in fast.lp.blocks for i in ir.ins(b)):
        return 1
    return INTERLEAVE


def body(fast, em, state, copy=0):
    """One vector's worth of an iteration of the vector loop, in em, the copy-th of interleave(fast); state only has
    induction variables, which it leaves for advance()."""
    fn = fast.fn
    env = {}
    k0 = copy * fast.vf  # the scalar iteration this copy starts at
    third = None
    lane = None

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
        if r.ext is not None:
            skip.add(id(r.ext))
        if r.third is not None:
            skip.add(id(r.third))
    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc) or id(i) in skip:
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
            env[rval(i.to)] = em.raw(D.Oqload, Kq, fast.opmap(i.arg[0], k0, env, state, None))
            continue
        k = _storeclass(i)
        if k is not None:
            v = vec(i.arg[0], k)
            em.ins(Ins(D.Oqstore, Kq, R, v, fast.opmap(i.arg[1], k0, env, state, None)))
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
                    acc = em.raw(D.Ofma1, r.cls, lanes[0], lanes[1])
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
            _keep(r, copy)
            continue
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
        op = getattr(D, 'O' + _REDOPS[r.op] + ('' if r.op in (D.Oand, D.Oor, D.Oxor) else _LANE[r.cls]))
        r.out0 = em.raw(op, Kq, r.in0, vec(r.v, r.cls))
        r.out1 = None
        _keep(r, copy)
    return new


def _keep(r, copy):
    """Each copy's accumulators, for closeloop."""
    if copy == 0:
        r.copies = []
    r.copies.append((r.acc0, r.in0, r.in1, r.out0, r.out1))


def closeloop(fast, em, MB, FP, state):
    """The accumulators' phis in MB, and, in em, a block after it, the scalar values with what they've collected."""
    out = dict(state)
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
        if r.op == D.Osel1:  # each lane into the scalar value as the loop would: s = cond(lane, s) ? ... : ...
            phi = r.i.arg[0] if r.i.arg[1] == r.v else r.i.arg[1]
            for total, n in [(t, n) for t in totals for n in range(lanes)]:
                lane = em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n))
                at = lambda a: v if a == phi else lane
                c = em.raw(r.cmp.op, Kw, at(r.cmp.arg[0]), at(r.cmp.arg[1]))
                em.ins(Ins(D.Osel0, Kw, R, c, R))
                v = em.raw(D.Osel1, k, at(r.i.arg[0]), at(r.i.arg[1]))
            out[('phi', r.t)] = v
            continue
        op = D.Oadd if r.op in (D.Oadd, D.Osub) else r.op
        # the copies' accumulators into one, lane by lane (a subtraction's each hold 0 - their part: added)
        vop = getattr(D, 'O' + _REDOPS[op] + ('' if op in (D.Oand, D.Oor, D.Oxor) else ('l' if r.ext is not None else x)))
        total = totals[0]
        for t in totals[1:]:
            total = em.raw(vop, Kq, total, t)
        for n in range(lanes):
            v = em.raw(op, k, v, em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n)))
        out[('phi', r.t)] = v
    return out
