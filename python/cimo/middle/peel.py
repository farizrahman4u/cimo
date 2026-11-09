"""Full unrolling of small constant-trip loops whose body has branches (cimo.gpu): the loop is peeled once per
iteration, each copy of all its blocks entered from the one before, and the loop itself is left behind its last copy,
where its exit test is false on entry. The cleanup's constant propagation then folds every copy's exit test and drops
the loop. inner.full_unroll does one- and two-block bodies more cheaply; this is for the rest (an inlined expf in an
epilogue loop), so the arrays they index by the loop counter become registers on the GPU."""

from ..backend import defs as D
from ..backend.defs import RTmp, RCon, CBits, Kw, Kl, rtype, rval, s32, s64
from . import ir
from . import unswitch
import cimo.middle as M

_CMP = {}
for _k, _base in ((Kw, D.Ocmpw), (Kl, D.Ocmpl)):
    for _n, _f in enumerate((lambda a, b: a == b, lambda a, b: a != b, lambda a, b: a >= b, lambda a, b: a > b,
                             lambda a, b: a <= b, lambda a, b: a < b, 'uge', 'ugt', 'ule', 'ult')):
        _CMP[_base + _n] = (_k, _n, _f)


def _const(fn, r):
    if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
        return fn.con[rval(r)].i
    return None


def trips(fn, lp, limit):
    """The trip count of lp if its header's exit test compares a counter (a phi of the header: a constant on entry,
    plus a constant each iteration) with a constant, and it is at most limit; else None."""
    h = lp.header
    if len(lp.latches) != 1 or h.jmp.type != D.Jjnz or rtype(h.jmp.arg) != RTmp:
        return None
    inside1 = h.s1 in lp.blocks
    if inside1 == (h.s2 in lp.blocks):
        return None
    c = fn.tmp[rval(h.jmp.arg)].def_
    if c is None or c.op not in _CMP or fn.tmp[rval(h.jmp.arg)].bid != h.id:
        return None
    k, n, f = _CMP[c.op]
    a, b = c.arg
    swap = False
    if _const(fn, a) is not None:
        a, b, swap = b, a, True
    bound = _const(fn, b)
    if bound is None or rtype(a) != RTmp:
        return None
    phi = next((p for p in ir.phis(h) if p.to == a), None)
    outs = lp.outside_preds()
    if phi is None or phi.narg != 2 or len(outs) != 1:
        return None
    x0 = _const(fn, ir.phiargof(phi, outs[0]))
    nxt = ir.phiargof(phi, lp.latches[0])
    d = fn.tmp[rval(nxt)].def_ if rtype(nxt) == RTmp else None
    if x0 is None or d is None or d.op not in (D.Oadd, D.Osub) or d.arg[0] != a or _const(fn, d.arg[1]) is None:
        return None
    step = _const(fn, d.arg[1]) * (1 if d.op == D.Oadd else -1)
    bits = 32 if k == Kw else 64
    mask = (1 << bits) - 1

    def sv(v):
        return s32(v & mask) if k == Kw else s64(v & mask)

    def holds(x):
        p, q = (bound, x) if swap else (x, bound)
        if isinstance(f, str):
            p, q = p & mask, q & mask
            return {'uge': p >= q, 'ugt': p > q, 'ule': p <= q, 'ult': p < q}[f]
        return f(sv(p), sv(q))
    x, t = x0, 0
    while holds(x) == inside1:
        t += 1
        if t > limit:
            return None
        x = sv(x + step)
    return t


def size(lp):
    return sum(len(ir.ins(b)) + 1 for b in lp.blocks)


def peel(fn, lp, first=True):
    """One iteration of lp copied in front of it. Returns False if lp's shape doesn't allow it."""
    exits = lp.exits()
    targets = {e for _, e in exits}
    outs = lp.outside_preds()
    if len(targets) != 1 or len(outs) != 1 or outs[0].jmp.type != D.Jjmp:
        return False
    pre, E = outs[0], targets.pop()
    # E may have other predecessors only when they are copies peeled before, and then every use of the loop's
    # values after it must be an argument of a phi of E, which clone() extends to the new copy's edges
    if any(p not in lp.blocks for p in E.pred[:E.npred]):
        if first:
            return False
        defined = set()
        for b in lp.blocks:
            defined.update(rval(p.to) for p in ir.phis(b))
            defined.update(rval(i.to) for i in ir.ins(b) if ir.istmp(i.to))
        if unswitch._uses(fn, lp, defined, {E}):
            return False
    h, latch = lp.header, lp.latches[0]
    if latch.jmp.type != D.Jjmp or any(not ir.copyable(i) for b in lp.blocks for i in ir.ins(b)):
        return False
    env = {}
    copy = unswitch.clone(fn, lp, E, exits, envout=env)
    ch, cl = copy[h], copy[latch]
    ir.retarget(pre, h, ch)
    ir.retarget(cl, ch, h)
    for p in ir.phis(ch):  # the copy is entered from the preheader only
        keep = [(x, a) for x, a in zip(p.blk[:p.narg], p.arg[:p.narg]) if x is pre]
        p.blk, p.arg, p.narg = [x for x, _ in keep], [a for _, a in keep], len(keep)
    for p in ir.phis(h):  # and the loop from the copy's latch, with the copy's values
        a = ir.phiargof(p, latch)
        a = D.TMP(env[rval(a)]) if rtype(a) == RTmp and rval(a) in env else a
        for n in range(p.narg):
            if p.blk[n] is pre:
                p.blk[n], p.arg[n] = cl, a
    return True


def transform(fn, lp, limit, maxins):
    """Peel lp completely if it has a constant trip count of at most limit and the copies stay under maxins."""
    if lp.children:
        return False
    t = trips(fn, lp, limit)
    if t is None or t * size(lp) > maxins:
        return False
    for n in range(t):
        if not peel(fn, lp, n == 0):
            return False
        ir.refresh(fn)
        from . import loops
        lp = next((l for l in loops.find_loops(fn) if l.header is lp.header), None)
        if lp is None:
            return True
    M.count('peel')
    return True
