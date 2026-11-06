"""Port of QBE ifopt.c."""

import sys

from . import defs as D
from .defs import R, RTmp, Ins, KBASE, Kw, Jjmp, Odbgloc, Onop, Osel0, Osel1, rtype, rval
from . import ext as _ext
from .util import addins, phiarg
from .cfg import ifgraph, fillloop
from .gcm import pinned

MaxIns = 2
MaxPhis = 2
# cimo (ext.isel): what LLVM's SimplifyCFG charges TCC_Expensive for: not worth running on the path
# that doesn't need it
_COSTLY = (D.Odiv, D.Orem, D.Oudiv, D.Ourem, D.Ofsqrt)
# cimo: the ordered comparisons of floats (<, <=, >, >=), which raise invalid on a NaN: not run on the path that
# doesn't ask for them (math.h's isgreater is !isunordered(x, y) && x > y)
_SIGNALING = frozenset(getattr(D, 'Oc%s%s' % (c, k)) for c in ('gt', 'ge', 'lt', 'le') for k in 'sd')


def okbranch(b):
    n = 0
    for i in b.ins[:b.nins]:
        if i.op != Odbgloc:
            if pinned(i) or (_ext.isel and i.op in _COSTLY) or i.op in _SIGNALING:
                return 0
            if i.op != Onop:
                n += 1
    return int(n <= MaxIns)


def okjoin(b):
    n = 0
    p = b.phi
    while p:
        if KBASE(p.cls) != 0 and not _ext.isel:  # cimo: arm64 has fcsel
            return 0
        n += 1
        p = p.link
    return int(n <= MaxPhis)


def predictable(fn, ifb):
    """cimo (ext.isel): a condition that doesn't change in the loop predicts perfectly, so a select
    only costs (LLVM's EarlyIfConverter::shouldConvertIf)."""
    if not ifb.loop or rtype(ifb.jmp.arg) != RTmp:
        return False
    return fn.rpo[fn.tmp[rval(ifb.jmp.arg)].bid].loop < ifb.loop


def okgraph(ifb, thenb, elseb, joinb):
    if joinb.npred != 2 or not okjoin(joinb):
        return 0
    assert thenb is not elseb
    if thenb is not ifb and not okbranch(thenb):
        return 0
    if elseb is not ifb and not okbranch(elseb):
        return 0
    return 1


def _addins(ins, i):
    """addins(&ins, &nins, i) on a plain list"""
    if i.op != Onop:
        ins.append(i.copy())


def _addbins(ins, b):
    for k in range(b.nins):
        _addins(ins, b.ins[k])


def convert(ifb, thenb, elseb, joinb):
    ins = []
    _addbins(ins, ifb)
    if thenb is not ifb:
        _addbins(ins, thenb)
    if elseb is not ifb:
        _addbins(ins, elseb)
    assert joinb.npred == 2
    if joinb.phi:
        sel = Ins(Osel0, Kw, R, ifb.jmp.arg, R)
        _addins(ins, sel)
    sel = Ins(Osel1)
    p = joinb.phi
    while p:
        sel.to = p.to
        sel.cls = p.cls
        sel.arg[0] = phiarg(p, thenb)
        sel.arg[1] = phiarg(p, elseb)
        _addins(ins, sel)
        p = p.link
    ifb.ins = [i.copy() for i in ins]
    ifb.nins = len(ins)
    ifb.jmp.type = Jjmp
    ifb.jmp.arg = R
    ifb.s1 = joinb
    ifb.s2 = None
    joinb.npred = 1
    joinb.pred[0] = ifb
    joinb.phi = None


def ifconvert(fn):
    """eliminate if-then[-else] graphlets
    using sel instructions
    needs rpo pred use; breaks cfg use"""
    from .parse import printfn
    if D.debug['K']:
        sys.stderr.write("\n> If-conversion:\n")

    if _ext.isel:
        fillloop(fn)
    ifb = fn.start
    while ifb:
        g = ifgraph(ifb)
        if g:
            thenb, elseb, joinb = g
            if okgraph(ifb, thenb, elseb, joinb) and not (_ext.isel and predictable(fn, ifb)):
                if D.debug['K']:
                    sys.stderr.write("    @%s -> @%s, @%s -> @%s\n"
                                     % (ifb.name, thenb.name, elseb.name, joinb.name))
                convert(ifb, thenb, elseb, joinb)
        ifb = ifb.link

    if D.debug['K']:
        sys.stderr.write("\n> After if-conversion:\n")
        printfn(fn, sys.stderr)
