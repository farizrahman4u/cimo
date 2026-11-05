"""Versioning on a unit stride (LLVM's symbolic stride speculation, in LoopAccessAnalysis and LoopVersioning): an
innermost loop that steps through memory by a value it doesn't change, `a[i * st]`, gets a copy in which that value
is 1, chosen once before the loop when it is:

    for (...) s += a[i * st];    ->    if (st == 1) for (...) s += a[i]; else for (...) s += a[i * st];

In the copy the accesses are consecutive, so the vectorizer can take it; callers that pass a stride for generality
(scramjit's reductions over any axis) mostly pass 1. Code doubles, so only loops that unswitch could copy, of at
most unswitch.MAXINS instructions, MAXPERFN times in a function, and only when the target vectorizes.
"""

from ..backend import defs as D
from ..backend.defs import Kw, Kl, Ins, rval
from ..backend import ext as _ext
from . import ir, unswitch
import cimo.middle as M

MAXPERFN = 4
_count = {}
_versioned = {}  # function -> headers of loops kept as the general case: never versioned again
_THROUGH = None  # what an address is made with, back from a load or store to the multiply by the stride


def _through():
    global _THROUGH
    if _THROUGH is None:
        _THROUGH = frozenset((D.Oadd, D.Osub, D.Omul, D.Oshl, D.Ocopy, D.Oextsw, D.Oextuw))
    return _THROUGH


def _stride(fn, lp):
    """A value lp doesn't change that multiplies something lp does change on the way to an address, or None."""
    ids = {b.id for b in lp.blocks}
    defs = {}
    for b in lp.blocks:
        for i in ir.ins(b):
            if ir.istmp(i.to):
                defs[rval(i.to)] = i
    inside = lambda r: ir.istmp(r) and (rval(r) in defs or fn.tmp[rval(r)].bid in ids)
    work = []
    for b in lp.blocks:
        for i in ir.ins(b):
            if D.isload(i.op):
                work.append(i.arg[0])
            elif D.isstore(i.op):
                work.append(i.arg[1])
    seen = set()
    while work:
        r = work.pop()
        if not ir.istmp(r) or rval(r) in seen or rval(r) not in defs:
            continue
        seen.add(rval(r))
        i = defs[rval(r)]
        if i.op not in _through():
            continue
        if i.op == D.Omul:
            for a, other in ((i.arg[0], i.arg[1]), (i.arg[1], i.arg[0])):
                if ir.istmp(a) and not inside(a) and inside(other) and fn.tmp[rval(a)].cls in (Kw, Kl):
                    return a
        work += [a for a in i.arg[:2] if ir.istmp(a)]
    return None


def transform(fn, lp):
    """Version lp on its stride being 1 (needs fillcfg, filluse, a preheader); returns True if it changed."""
    n = _count.get(fn.name, 0)
    if n >= MAXPERFN or lp.children or not (M.vectorizing and _ext.vector):
        return False
    if sum(len(ir.ins(b)) for b in lp.blocks) > unswitch.MAXINS:
        return False
    if any(not ir.copyable(i) for b in lp.blocks for i in ir.ins(b)):
        return False
    seen = _versioned.setdefault(fn.name, set())
    if lp.header.id in seen:
        return False
    st = _stride(fn, lp)
    if st is None:
        return False
    g = unswitch.shape(lp, many=True, fn=fn)
    if g is None:
        return False
    P, E, exits = g
    k = fn.tmp[rval(st)].cls
    one = ir.getcon(1, fn)
    copy = unswitch.clone(fn, lp, E, exits, {st: one})
    cond = ir.newtmp('st', Kw, fn)
    ir.setins(P, ir.ins(P) + [Ins(D.Oceql if k == Kl else D.Oceqw, Kw, cond, st, one)])
    ir.setjnz(P, cond, copy[lp.header], lp.header)
    seen.add(lp.header.id)
    _count[fn.name] = n + 1
    M.count('stride')
    return True
