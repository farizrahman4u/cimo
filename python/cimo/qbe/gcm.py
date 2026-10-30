"""Port of QBE gcm.c."""

import sys

from . import defs as D
from .defs import (R, RTmp, KBASE, U32, NIns, Ins, UXXX, UPhi, UIns, UJmp, Odiv, Orem,
                   Oudiv, Ourem, Oneg, Oadd, Osub, Omul, Oand, Oor, Oxor, Osar, Oshr,
                   Oshl, Onop, isload, isstore, isalloc, optab, rtype, rval)
from .util import die, iscmp, igroup, addins, newtmp, emiti
from .cfg import dom, lca, filldepth, fillloop

NOBID = U32


def isdivwl(i):
    if i.op in (Odiv, Orem, Oudiv, Ourem):
        return int(KBASE(i.cls) == 0)
    return 0


def pinned(i):
    return optab[i.op].pinned or isdivwl(i)


def canelim(i):
    """pinned ins that can be eliminated if unused"""
    return isload(i.op) or isalloc(i.op) or isdivwl(i)


def schedearly(fn, r):
    if rtype(r) != RTmp:
        return 0

    t = fn.tmp[rval(r)]
    if t.gcmbid != NOBID:
        return t.gcmbid

    b = fn.rpo[t.bid]
    if t.def_:
        t.gcmbid = 0  # mark as visiting
        t.gcmbid = earlyins(fn, b, t.def_)
    else:
        # phis do not move
        t.gcmbid = t.bid

    return t.gcmbid


def earlyins(fn, b, i):
    b0 = schedearly(fn, i.arg[0])
    assert b0 != NOBID
    b1 = schedearly(fn, i.arg[1])
    assert b1 != NOBID
    if fn.rpo[b0].depth < fn.rpo[b1].depth:
        assert dom(fn.rpo[b0], fn.rpo[b1])
        b0 = b1
    return b.id if pinned(i) else b0


def earlyblk(fn, bid):
    b = fn.rpo[bid]
    p = b.phi
    while p:
        for n in range(p.narg):
            schedearly(fn, p.arg[n])
        p = p.link
    for i in b.ins[:b.nins]:
        if pinned(i):
            schedearly(fn, i.arg[0])
            schedearly(fn, i.arg[1])
    schedearly(fn, b.jmp.arg)


def lcabid(fn, bid1, bid2):
    """least common ancestor in dom tree"""
    if bid1 == NOBID:
        return bid2
    if bid2 == NOBID:
        return bid1

    b = lca(fn.rpo[bid1], fn.rpo[bid2])
    assert b
    return b.id


def bestbid(fn, earlybid, latebid):
    if latebid == NOBID:
        return NOBID  # unused

    assert earlybid != NOBID

    earlyb = fn.rpo[earlybid]
    bestb = curb = fn.rpo[latebid]
    assert dom(earlyb, curb)

    while curb is not earlyb:
        curb = curb.idom
        if curb.loop < bestb.loop:
            bestb = curb
    return bestb.id


def schedlate(fn, r):
    """return lca bid of ref uses"""
    if rtype(r) != RTmp:
        return NOBID

    t = fn.tmp[rval(r)]
    if t.visit:
        return t.gcmbid

    t.visit = 1
    earlybid = t.gcmbid
    if earlybid == NOBID:
        return NOBID  # not used

    # reuse gcmbid for late bid
    t.gcmbid = t.bid
    latebid = NOBID
    for k in range(t.nuse):
        u = t.use[k]
        assert u.bid < fn.nblk
        b = fn.rpo[u.bid]
        if u.type == UPhi:
            uselatebid = latephi(fn, u.u, r)
        elif u.type == UIns:
            uselatebid = lateins(fn, b, u.u, r)
        elif u.type == UJmp:
            uselatebid = latejmp(b, r)
        else:
            die("unreachable")
        latebid = lcabid(fn, latebid, uselatebid)
    # latebid may be NOBID if the temp is used
    # in fixed instructions that may be eliminated
    # and are themselves unused transitively

    if t.def_ and not pinned(t.def_):
        t.gcmbid = bestbid(fn, earlybid, latebid)
    # else, keep the early one

    # now, gcmbid is the best bid
    return t.gcmbid


def lateins(fn, b, i, r):
    """returns lca bid of uses or NOBID if
    the definition can be eliminated"""
    assert i.arg[0] == r or i.arg[1] == r

    latebid = schedlate(fn, i.to)
    if pinned(i):
        if latebid == NOBID:
            if canelim(i):
                return NOBID
        return b.id

    return latebid


def latephi(fn, p, r):
    if not p.narg:
        return NOBID  # marked as unused

    latebid = NOBID
    for n in range(p.narg):
        if p.arg[n] == r:
            latebid = lcabid(fn, latebid, p.blk[n].id)

    assert latebid != NOBID
    return latebid


def latejmp(b, r):
    if b.jmp.arg == R:
        return NOBID
    else:
        assert b.jmp.arg == r
        return b.id


def lateblk(fn, bid):
    b = fn.rpo[bid]
    prev = None
    p = b.phi
    while p:
        if schedlate(fn, p.to) == NOBID:
            p.narg = 0  # mark unused
            # remove phi
            if prev is None:
                b.phi = p.link
            else:
                prev.link = p.link
        else:
            prev = p
        p = p.link

    for i in b.ins[:b.nins]:
        if pinned(i):
            schedlate(fn, i.to)


def addgcmins(fn, vins):
    for i in vins:
        assert rtype(i.to) == RTmp
        t = fn.tmp[rval(i.to)]
        b = fn.rpo[t.gcmbid]
        addins(b, i)


def gcmmove(fn):
    """move live instructions to the
    end of their target block; use-
    before-def errors are fixed by
    schedblk"""
    vins = []

    for t in fn.tmp[:fn.ntmp]:
        if not t.def_:
            continue
        if t.bid == t.gcmbid:
            continue
        i = t.def_
        if pinned(i) and not canelim(i):
            continue
        assert rtype(i.to) == RTmp
        assert t is fn.tmp[rval(i.to)]
        if t.gcmbid != NOBID:
            if i.op != Onop:
                vins.append(i.copy())
        i.set(Ins(Onop))
    addgcmins(fn, vins)


def schedins(fn, b, k, pos, vins):
    """dfs ordering; k is an index in b.ins, pos maps id(ins) to its index;
    returns the index after the group"""
    i0, i1 = igroup(b, k)
    for k in range(i0, i1):
        i = b.ins[k]
        for n in range(2):
            if rtype(i.arg[n]) != RTmp:
                continue
            t = fn.tmp[rval(i.arg[n])]
            if t.bid != b.id or not t.def_:
                continue
            schedins(fn, b, pos[id(t.def_)], pos, vins)
    for k in range(i0, i1):
        i = b.ins[k]
        if i.op != Onop:
            vins.append(i.copy())
        i.set(Ins(Onop))
    return i1


def schedblk(fn):
    """order ins within a block"""
    b = fn.start
    while b:
        vins = []
        pos = {id(b.ins[k]): k for k in range(b.nins)}
        k = 0
        while k < b.nins:
            k = schedins(fn, b, k, pos, vins)
        b.ins = [i.copy() for i in vins]
        b.nins = len(vins)
        b = b.link


def cheap(i):
    if KBASE(i.cls) != 0:
        return 0
    if i.op in (Oneg, Oadd, Osub, Omul, Oand, Oor, Oxor, Osar, Oshr, Oshl):
        return 1
    return int(iscmp(i.op) is not None)


def sinkref(fn, b, r):
    """sinkref(fn, b, &ref): returns the new value of ref."""
    if rtype(r) != RTmp:
        return r
    t = fn.tmp[rval(r)]
    if (not t.def_
            or t.bid == b.id
            or pinned(t.def_)
            or not cheap(t.def_)):
        return r

    # sink t->def to b
    i = t.def_.copy()
    r = newtmp("snk", t.cls, fn)
    t = None  # invalidated
    i.to = r
    fn.tmp[rval(r)].gcmbid = b.id
    emiti(i)
    i.arg[0] = sinkref(fn, b, i.arg[0])
    i.arg[1] = sinkref(fn, b, i.arg[1])
    return r


def sink(fn):
    """redistribute trivial ops to point of
    use to reduce register pressure
    requires rpo, use; breaks use"""
    b = fn.start
    while b:
        for i in b.ins[:b.nins]:
            if isload(i.op):
                i.arg[0] = sinkref(fn, b, i.arg[0])
            elif isstore(i.op):
                i.arg[1] = sinkref(fn, b, i.arg[1])
        b.jmp.arg = sinkref(fn, b, b.jmp.arg)
        b = b.link
    addgcmins(fn, D.insb[D.curi:NIns])


def gcm(fn):
    """requires use dom
    maintains rpo pred dom
    breaks use"""
    from .ssa import filluse
    from .parse import printfn

    filldepth(fn)
    fillloop(fn)

    for t in fn.tmp[:fn.ntmp]:
        t.visit = 0
        t.gcmbid = NOBID
    for bid in range(fn.nblk):
        earlyblk(fn, bid)
    for bid in range(fn.nblk):
        lateblk(fn, bid)

    gcmmove(fn)
    filluse(fn)
    D.curi = NIns
    sink(fn)
    filluse(fn)
    schedblk(fn)

    if D.debug['G']:
        sys.stderr.write("\n> After GCM:\n")
        printfn(fn, sys.stderr)
