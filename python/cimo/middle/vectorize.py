"""LLVM's LoopVectorize, on what inner.py has already proved: a counted loop of one straight run of instructions,
whose accesses are consecutive, becomes one vector operation per instruction, and the scalar loop stays as the
remainder and as the way out when the guards fail. Lanes are 128 bits' worth, so 4 floats or ints, or 2 doubles or longs.

Legal when, as LoopVectorizationLegality and LoopAccessAnalysis ask:
- every instruction has a vector form, with lanes of one width;
- each load and store steps by its own size, upwards;
- two accesses of one base and step are far enough apart that a vector's worth of iterations can't change what
  either sees: see _distance;
- nothing is carried round the loop but the induction variables and integer reductions (_reduction), whose order C
  doesn't fix; a float's sum depends on its order, so those stay scalar;
- operands are computed in the loop by a vectorized instruction, or are the same on every iteration.

Each lane does what the scalar instruction did, in the same rounding, so results are bit for bit the scalar loop's.
Arrays that may overlap are compared before the loop (overlapchecks), up to MAXCHECKS pairs.
Not yet: other widening and narrowing, conditions, downward steps, SLP.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ks, Kd, Kq, Ins, optab, rval
from ..backend.util import newtmp
from . import ir

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
    __slots__ = ('t', 'i', 'op', 'v', 'ext', 'acc0', 'in0', 'in1', 'out0', 'out1', 'cls')


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
    r.ext = None
    return r


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
            r = _reduction(fast, p, fast.cloned)
            if r is None:
                return 0
            x = _widening(fast, r, fast.cloned)
            if x is not None:
                r.ext = x
                skip.add(id(x))
            fast.vreds.append(r)
            skip.add(id(r.i))
    widths = set()
    mem = []
    made = set()
    third = None
    for n, i in enumerate(fast.cloned):
        if i.op in (D.Onop, D.Odbgloc) or id(i) in skip:
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
    if third is not None or not mem:
        return 0
    for r in fast.vreds:
        if r.ext is not None:
            if not _operand(fast, r.ext.arg[0], made):
                return 0
            widths.add(4)
        else:
            if not _operand(fast, r.v, made):
                return 0
            widths.add(_SIZE[r.cls])
    if len(widths) != 1:
        return 0
    vf = 16 // widths.pop()
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


def body(fast, em, state):
    """One iteration of the vector loop, in em; state only has induction variables, which it leaves for advance()."""
    fn = fast.fn
    env = {}
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
        if r.ext is not None:
            skip.add(id(r.ext))
    for i in fast.cloned:
        if i.op in (D.Onop, D.Odbgloc) or id(i) in skip:
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
            env[rval(i.to)] = em.raw(D.Oqload, Kq, fast.opmap(i.arg[0], 0, env, state, None))
            continue
        k = _storeclass(i)
        if k is not None:
            v = vec(i.arg[0], k)
            em.ins(Ins(D.Oqstore, Kq, R, v, fast.opmap(i.arg[1], 0, env, state, None)))
            continue
        name = _opname(i)
        op = getattr(D, 'O' + name)
        if i.op in (D.Oshl, D.Oshr, D.Osar):
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0], i.cls), i.arg[1])
        elif i.arg[1] == R:
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0], i.cls))
        else:
            env[rval(i.to)] = em.raw(op, Kq, vec(i.arg[0], i.cls), vec(i.arg[1], i.cls))
    for r in fast.vreds:
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
            continue
        r.in1 = None
        op = getattr(D, 'O' + _REDOPS[r.op] + ('' if r.op in (D.Oand, D.Oor, D.Oxor) else _LANE[r.cls]))
        r.out0 = em.raw(op, Kq, r.in0, vec(r.v, r.cls))
        r.out1 = None
    return dict(state)


def closeloop(fast, em, MB, FP, state):
    """The accumulators' phis in MB, and, in em, a block after it, the scalar values with what they've collected."""
    out = dict(state)
    for r in fast.vreds:
        ir.addphi(MB, Kq, r.in0, [(FP, r.acc0), (MB, r.out0)])
        total = r.out0
        k = r.cls
        if r.ext is not None:
            ir.addphi(MB, Kq, r.in1, [(FP, r.acc0), (MB, r.out1)])
            total = em.raw(D.Oqaddl, Kq, r.out0, r.out1)
        x = _LANE[k]
        lanes = 16 // _SIZE[k]
        op = D.Oadd if r.op in (D.Oadd, D.Osub) else r.op
        v = state[('phi', r.t)]
        for n in range(lanes):
            v = em.raw(op, k, v, em.raw(getattr(D, 'Oqget' + x), k, total, em.con(n)))
        out[('phi', r.t)] = v
    return out
