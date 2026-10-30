"""Port of QBE ifopt.c."""

import sys

from . import defs as D
from .defs import R, Ins, KBASE, Kw, Jjmp, Odbgloc, Onop, Osel0, Osel1
from .util import addins, phiarg
from .cfg import ifgraph
from .gcm import pinned

MaxIns = 2
MaxPhis = 2


def okbranch(b):
    n = 0
    for i in b.ins[:b.nins]:
        if i.op != Odbgloc:
            if pinned(i):
                return 0
            if i.op != Onop:
                n += 1
    return int(n <= MaxIns)


def okjoin(b):
    n = 0
    p = b.phi
    while p:
        if KBASE(p.cls) != 0:
            return 0
        n += 1
        p = p.link
    return int(n <= MaxPhis)


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

    ifb = fn.start
    while ifb:
        g = ifgraph(ifb)
        if g:
            thenb, elseb, joinb = g
            if okgraph(ifb, thenb, elseb, joinb):
                if D.debug['K']:
                    sys.stderr.write("    @%s -> @%s, @%s -> @%s\n"
                                     % (ifb.name, thenb.name, elseb.name, joinb.name))
                convert(ifb, thenb, elseb, joinb)
        ifb = ifb.link

    if D.debug['K']:
        sys.stderr.write("\n> After if-conversion:\n")
        printfn(fn, sys.stderr)
