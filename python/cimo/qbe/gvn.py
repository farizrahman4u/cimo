"""Port of QBE gvn.c."""

import sys

from . import defs as D
from . import copy as C
from .defs import (R, RTmp, CON_Z, CBits, Con, Ins, KBASE, KWIDE, U32, Kx, Kw, Kl, UXXX,
                   UPhi, UIns, UJmp, Jjmp, Jjnz, Onop, Osub, Oadd, Osel0, optab, rtype,
                   rval, rkind, s32)
from .util import die, getcon, newcon, isconbits, argcls, addins
from .cfg import dom, reachesnotvia, fillcfg, fillloop
from .fold import foldint, foldref
from .gcm import pinned

con01 = [R, R]


def mix(x0, x1):
    return (x0 + 17 * x1) & U32


def rhash(r):
    return mix(rkind(r), rval(r))


def ihash(i):
    h = mix(i.op, i.cls & U32)
    h = mix(h, rhash(i.arg[0]))
    h = mix(h, rhash(i.arg[1]))
    return h


def ieq(ia, ib):
    if ia.op == ib.op:
        if ia.cls == ib.cls:
            if ia.arg[0] == ib.arg[0]:
                if ia.arg[1] == ib.arg[1]:
                    return 1
    return 0


gvntbl = None
gvntbln = 0


def gvndup(i, insert):
    idx = ihash(i) % gvntbln
    while True:
        ii = gvntbl[idx]
        if ii is None:
            break
        if ieq(i, ii):
            return ii

        idx += 1
        if gvntbln <= idx:
            idx = 0
    if insert:
        gvntbl[idx] = i
    return None


def replaceuse(fn, u, r1, r2):
    from .ssa import adduse
    t2 = None
    if rtype(r2) == RTmp:
        t2 = fn.tmp[rval(r2)]
    b = fn.rpo[u.bid]
    if u.type == UPhi:
        p = u.u
        for n in range(p.narg):
            if p.arg[n] == r1:
                p.arg[n] = r2
        if t2:
            adduse(t2, UPhi, b, p)
    elif u.type == UIns:
        i = u.u
        for n in range(2):
            if i.arg[n] == r1:
                i.arg[n] = r2
        if t2:
            adduse(t2, UIns, b, i)
    elif u.type == UJmp:
        if b.jmp.arg == r1:
            b.jmp.arg = r2
        if t2:
            adduse(t2, UJmp, b)
    else:
        die("unreachable")


def replaceuses(fn, r1, r2):
    assert rtype(r1) == RTmp
    t1 = fn.tmp[rval(r1)]
    k = 0
    while k < t1.nuse:
        replaceuse(fn, t1.use[k], r1, r2)
        k += 1
    t1.nuse = 0


def dedupphi(fn, b):
    prev = None
    p = b.phi
    while p:
        r = C.phicopyref(fn, b, p)
        if r != R:
            replaceuses(fn, p.to, r)
            p.to = R
            if prev is None:
                b.phi = p.link
            else:
                prev.link = p.link
            p = p.link
        else:
            prev = p
            p = p.link


def rcmp(a, b):
    if rtype(a) != rtype(b):
        return rtype(a) - rtype(b)
    return rval(a) - rval(b)


def normins(fn, i):
    # truncate constant bits to
    # 32 bits for s/w uses
    for n in range(2):
        if not KWIDE(argcls(i, n)):
            v = isconbits(fn, i.arg[n])
            if v is not None:
                if (v & 0xffffffff) != v:
                    i.arg[n] = getcon(v & 0xffffffff, fn)
    # order arg[0] <= arg[1] for
    # commutative ops, preferring
    # RTmp in arg[0]
    if optab[i.op].commutes:
        if rcmp(i.arg[0], i.arg[1]) > 0:
            r = i.arg[1]
            i.arg[1] = i.arg[0]
            i.arg[0] = r


_z = Con(CBits, 0)


def negcon(cls, c):
    return foldint(c, Osub, cls, _z, c)


def assoccon(fn, b, i1):
    from .ssa import adduse
    op = i1.op
    if op == Osub:
        op = Oadd

    if (not optab[op].assoc
            or KBASE(i1.cls) != 0
            or rtype(i1.arg[0]) != RTmp
            or rtype(i1.arg[1]) != D.RCon):
        return
    c1 = fn.con[rval(i1.arg[1])].copy()

    t2 = fn.tmp[rval(i1.arg[0])]
    if not t2.def_:
        return
    i2 = t2.def_

    if (op != (Oadd if i2.op == Osub else i2.op)
            or rtype(i2.arg[1]) != D.RCon):
        return
    c2 = fn.con[rval(i2.arg[1])].copy()

    assert KBASE(i2.cls) == 0
    assert KWIDE(i2.cls) >= KWIDE(i1.cls)

    if i1.op == Osub and negcon(i1.cls, c1):
        return
    if i2.op == Osub and negcon(i2.cls, c2):
        return
    c = Con()
    if foldint(c, op, i1.cls, c1, c2):
        return

    if op == Oadd and c.type == CBits:
        if ((i1.cls == Kl and c.i < 0)
                or (i1.cls == Kw and s32(c.i) < 0)):
            fail = negcon(i1.cls, c)
            assert fail == 0
            op = Osub

    i1.op = op
    i1.arg[0] = i2.arg[0]
    i1.arg[1] = newcon(c, fn)
    adduse(fn.tmp[rval(i1.arg[0])], UIns, b, i1)


def killins(fn, i, r):
    replaceuses(fn, i.to, r)
    i.set(Ins(Onop))


def dedupins(fn, b, i):
    normins(fn, i)
    if i.op == Onop or pinned(i):
        return

    # when sel instructions are inserted
    # before gvn, we may want to optimize
    # them here
    assert i.op != Osel0
    assert i.to != R
    assoccon(fn, b, i)

    r = C.copyref(fn, b, i)
    if r != R:
        killins(fn, i, r)
        return
    r = foldref(fn, i)
    if r != R:
        killins(fn, i, r)
        return
    i1 = gvndup(i, 1)
    if i1:
        killins(fn, i, i1.to)
        return


def cmpeqz(fn, r):
    """Returns (arg, cls, eqval) or None."""
    if rtype(r) != RTmp:
        return None
    i = fn.tmp[rval(r)].def_
    if i:
        if optab[i.op].cmpeqwl:
            if i.arg[1] == CON_Z:
                return i.arg[0], argcls(i, 0), optab[i.op].eqval
    return None


def branchdom(fn, bif, bbr1, bbr2, b):
    assert bif.jmp.type == Jjnz

    if (b is not bif
            and dom(bbr1, b)
            and not reachesnotvia(fn, bbr2, b, bif)):
        return 1

    return 0


def domzero(fn, d, b):
    """Returns z or None."""
    if branchdom(fn, d, d.s1, d.s2, b):
        return 0
    if branchdom(fn, d, d.s2, d.s1, b):
        return 1
    return None


def zeroval(fn, b, r, cls):
    """infer 0/non-0 value from dominating jnz; returns z or None"""
    d = b.idom
    while d:
        if d.jmp.type != Jjnz:
            d = d.idom
            continue
        if r == d.jmp.arg and cls == Kw:
            z = domzero(fn, d, b)
            if z is not None:
                return z
        ce = cmpeqz(fn, d.jmp.arg)
        if ce is not None and r == ce[0] and cls == ce[1]:
            z = domzero(fn, d, b)
            if z is not None:
                return z ^ ce[2]
        d = d.idom
    return None


def usecls(u, r, cls):
    if u.type == UIns:
        k = Kx  # widest use
        if u.u.arg[0] == r:
            k = argcls(u.u, 0)
        if u.u.arg[1] == r:
            if k == Kx or not KWIDE(k):
                k = argcls(u.u, 1)
        return cls if k == Kx else k
    elif u.type == UPhi:
        if u.u.to == R:
            return cls  # eliminated
        return u.u.cls
    elif u.type == UJmp:
        return Kw
    die("unreachable")


def propjnz0(fn, bif, s0, snon0, r, cls):
    if s0.npred != 1 or rtype(r) != RTmp:
        return
    t = fn.tmp[rval(r)]
    k = 0
    while k < t.nuse:
        u = t.use[k]
        b = fn.rpo[u.bid]
        # we may compare an l temp with a w
        # comparison; so check that the use
        # does not involve high bits
        if usecls(u, r, cls) == cls:
            if branchdom(fn, bif, s0, snon0, b):
                replaceuse(fn, u, r, CON_Z)
        k += 1


def dedupjmp(fn, b):
    if b.jmp.type != Jjnz:
        return

    # propagate jmp arg as 0 through s2
    propjnz0(fn, b, b.s2, b.s1, b.jmp.arg, Kw)
    # propagate cmp eq/ne 0 def of jmp arg as 0
    ce = cmpeqz(fn, b.jmp.arg)
    if ce is not None:
        arg, cls, eqval = ce
        ps = [b.s1, b.s2]
        propjnz0(fn, b, ps[eqval ^ 1], ps[eqval], arg, cls)

    # collapse trivial/constant jnz to jmp
    v = 1
    z = 0
    hit = b.s1 is b.s2
    if not hit:
        cv = isconbits(fn, b.jmp.arg)
        if cv is not None:
            v = cv
            hit = True
    if not hit:
        zz = zeroval(fn, b, b.jmp.arg, Kw)
        if zz is not None:
            z = zz
            hit = True
    if hit:
        if v == 0 or z:
            b.s1 = b.s2
        # we later move active ins out of dead blks
        b.s2 = None
        b.jmp.type = Jjmp
        b.jmp.arg = R


def rebuildcfg(fn):
    nblk = fn.nblk
    rpo = list(fn.rpo[:nblk])

    fillcfg(fn)

    # move instructions that were in
    # killed blocks and may be active
    # in the computation in the start
    # block
    s = fn.start
    for n in range(nblk):
        b = rpo[n]
        if b.id != U32:
            continue
        # blk unreachable after GVN
        assert b is not s
        for i in b.ins[:b.nins]:
            if not optab[i.op].pinned:
                if gvndup(i, 0) is i:
                    addins(s, i)


def gvn(fn):
    """requires rpo pred ssa use
    recreates rpo preds
    breaks pred use dom ssa (GCM fixes ssa)"""
    global gvntbl, gvntbln
    from .ssa import filluse, ssacheck
    from .parse import printfn

    con01[0] = getcon(0, fn)
    con01[1] = getcon(1, fn)

    # copy.c uses the visit bit
    b = fn.start
    while b:
        p = b.phi
        while p:
            p.visit = 0
            p = p.link
        b = b.link

    fillloop(fn)
    C.narrowpars(fn)
    filluse(fn)
    ssacheck(fn)

    nins = 0
    b = fn.start
    while b:
        b.visit = 0
        nins += b.nins
        b = b.link

    gvntbln = nins + nins // 2
    gvntbl = [None] * gvntbln
    for n in range(fn.nblk):
        b = fn.rpo[n]
        dedupphi(fn, b)
        for i in b.ins[:b.nins]:
            dedupins(fn, b, i)
        dedupjmp(fn, b)
    rebuildcfg(fn)
    gvntbl = None

    if D.debug['G']:
        sys.stderr.write("\n> After GVN:\n")
        printfn(fn, sys.stderr)
