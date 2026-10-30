"""Port of QBE load.c."""

import sys
from functools import cmp_to_key

from . import defs as D
from .defs import (R, RTmp, RCon, RInt, M64, CAddr, Con, Ins, Phi, Kw, Kl, Ks, Kd,
                   ALoc, AEsc, AUnk, ACon, ASym, NoAlias, MayAlias, MustAlias,
                   Oloadsb, Oloadub, Oloadsh, Oloaduh, Oloadsw, Oloaduw, Oload,
                   Ostoreb, Ostoreh, Ostorew, Ostorel, Ostores, Ostored, Ocast, Oextuw,
                   Oand, Oadd, Oor, Oshl, Oshr, Ocall, Oblit0, Oblit1, Ocopy, Oextsb,
                   BIT, KWIDE, TMP, isload, isstore, rtype, rval, rsval)
from .util import die, newtmp, getcon, newcon, idup
from .alias import alias, escapes
from .cfg import dom
from .parse import printfn


def MASK(w):
    """must work when w==8"""
    return (BIT(8 * w - 1) * 2 - 1) & M64


LRoot, LLoad, LNoLoad = range(3)


class Loc:
    __slots__ = ('type', 'off', 'blk')

    def __init__(self, type=LRoot, off=0, blk=None):
        self.type = type
        self.off = off
        self.blk = blk

    def copy(self):
        return Loc(self.type, self.off, self.blk)


def _s16(x):
    x &= 0xffff
    return x - 0x10000 if x >> 15 else x


class Slice:
    __slots__ = ('ref', 'off', 'sz', 'cls')

    def __init__(self, ref=R, off=0, sz=0, cls=0):
        self.ref = ref
        self.off = off
        self.sz = sz
        self.cls = cls

    def copy(self):
        return Slice(self.ref, self.off, self.sz, self.cls)


class Insert:
    """new.ins is ins; new.phi.{m,p} are phi_m, phi_p."""
    __slots__ = ('isphi', 'num', 'bid', 'off', 'ins', 'phi_m', 'phi_p')

    def __init__(self):
        self.isphi = 0
        self.num = 0
        self.bid = 0
        self.off = 0
        self.ins = None
        self.phi_m = None
        self.phi_p = None


curf = None
inum = 0     # current insertion number
ilog = []    # global insertion log
nlog = 0     # number of entries in the log


def _newlog():
    global nlog
    nlog += 1
    del ilog[nlog - 1:]
    ist = Insert()
    ilog.append(ist)
    return ist


def loadsz(l):
    op = l.op
    if op == Oloadsb or op == Oloadub:
        return 1
    if op == Oloadsh or op == Oloaduh:
        return 2
    if op == Oloadsw or op == Oloaduw:
        return 4
    if op == Oload:
        return 8 if KWIDE(l.cls) else 4
    die("unreachable")


def storesz(s):
    op = s.op
    if op == Ostoreb:
        return 1
    if op == Ostoreh:
        return 2
    if op == Ostorew or op == Ostores:
        return 4
    if op == Ostorel or op == Ostored:
        return 8
    die("unreachable")


def iins(cls, op, a0, a1, l):
    global inum
    ist = _newlog()
    ist.isphi = 0
    ist.num = inum
    inum += 1
    ist.bid = l.blk.id
    ist.off = l.off
    ist.ins = Ins(op, cls, R, a0, a1)
    ist.ins.to = newtmp("ld", cls, curf)
    return ist.ins.to


def cast(r, cls, l):
    """cast(&r, cls, l): returns the new value of r."""
    if rtype(r) == RCon:
        return r
    assert rtype(r) == RTmp
    cls0 = curf.tmp[rval(r)].cls
    if cls0 == cls or (cls == Kw and cls0 == Kl):
        return r
    if KWIDE(cls0) < KWIDE(cls):
        if cls0 == Ks:
            r = iins(Kw, Ocast, r, R, l)
        r = iins(Kl, Oextuw, r, R, l)
        if cls == Kd:
            r = iins(Kd, Ocast, r, R, l)
    else:
        if cls0 == Kd and cls != Kl:
            r = iins(Kl, Ocast, r, R, l)
        if cls0 != Kd or cls != Kw:
            r = iins(cls, Ocast, r, R, l)
    return r


def mask(cls, r, msk, l):
    """mask(cls, &r, msk, l): returns the new value of r."""
    r = cast(r, cls, l)
    return iins(cls, Oand, r, getcon(msk, curf), l)


_ldop = {1: Oloadub, 2: Oloaduh, 4: Oloaduw, 8: Oload}


def load(sl, msk, l):
    ld = _ldop[sl.sz]
    all_ = msk == MASK(sl.sz)
    if all_:
        cls = sl.cls
    else:
        cls = Kl if sl.sz > 4 else Kw
    r = sl.ref
    # sl.ref might not be live here,
    # but its alias base ref will be
    # (see killsl() below)
    if rtype(r) == RTmp:
        a = curf.tmp[rval(r)].alias
        if a.type in (ALoc, AEsc, AUnk):
            r = TMP(a.base)
            if a.offset:
                r1 = getcon(a.offset, curf)
                r = iins(Kl, Oadd, r, r1, l)
        elif a.type in (ACon, ASym):
            c = Con()
            c.type = CAddr
            c.sym = a.sym.copy()
            c.i = a.offset
            r = newcon(c, curf)
        else:
            die("unreachable")
    r = iins(cls, ld, r, R, l)
    if not all_:
        r = mask(cls, r, msk, l)
    return r


def rebase(sl):
    if rtype(sl.ref) != RTmp:
        return
    a = curf.tmp[rval(sl.ref)].alias
    if a.offset == _s16(a.offset):
        if a.type == ALoc or a.type == AEsc or a.type == AUnk:
            sl.ref = TMP(a.base)
            sl.off = a.offset


def killsl(r, sl):
    if rtype(sl.ref) != RTmp:
        return 0
    a = curf.tmp[rval(sl.ref)].alias
    if a.type in (ALoc, AEsc, AUnk):
        return int(TMP(a.base) == r)
    if a.type in (ACon, ASym):
        return 0
    die("unreachable")


def def_(sl, msk, b, i, il):
    """def(sl, msk, b, i, il): i is an index in b.ins, or None.

    returns a ref containing the contents of the slice
    passed as argument, all the bits set to 0 in the
    mask argument are zeroed in the result;
    the returned ref has an integer class when the
    mask does not cover all the bits of the slice,
    otherwise, it has class sl.cls
    the procedure returns R when it fails"""
    global nlog
    # invariants:
    # -1- b dominates il->blk; so we can use
    #     temporaries of b in il->blk
    # -2- if il->type != LNoLoad, then il->blk
    #     postdominates the original load; so it
    #     is safe to load in il->blk
    # -3- if il->type != LNoLoad, then b
    #     postdominates il->blk (and by 2, the
    #     original load)
    assert dom(b, il.blk)
    oldl = nlog
    oldt = curf.ntmp

    def Load():
        global nlog
        curf.ntmp = oldt
        nlog = oldl
        if il.type != LLoad:
            return R
        return load(sl, msk, il)

    if i is None:
        i = b.nins
    cls = Kl if sl.sz > 4 else Kw
    msks = MASK(sl.sz)

    while i > 0:
        i -= 1
        ins = b.ins[i]
        if killsl(ins.to, sl) or (ins.op == Ocall and escapes(sl.ref, curf)):
            return Load()
        ld = isload(ins.op)
        if ld:
            sz = loadsz(ins)
            r1 = ins.arg[0]
            r = ins.to
        elif isstore(ins.op):
            sz = storesz(ins)
            r1 = ins.arg[1]
            r = ins.arg[0]
        elif ins.op == Oblit1:
            assert rtype(ins.arg[0]) == RInt
            sz = abs(rsval(ins.arg[0]))
            assert i > 0
            i -= 1
            ins = b.ins[i]
            assert ins.op == Oblit0
            r1 = ins.arg[1]
        else:
            continue
        res, off = alias(sl.ref, sl.off, sl.sz, r1, sz, curf)
        if res == MustAlias:
            if ins.op == Oblit0:
                sl1 = sl.copy()
                sl1.ref = ins.arg[0]
                if off >= 0:
                    assert off < sz
                    sl1.off = off
                    sz -= off
                    off = 0
                else:
                    sl1.off = 0
                    sl1.sz = _s16(sl1.sz + off)
                if sz > sl1.sz:
                    sz = sl1.sz
                assert sz <= 8
                sl1.sz = sz
            if off < 0:
                off = -off
                msk1 = ((MASK(sz) << (8 * off)) & M64) & msks
                op = Oshl
            else:
                msk1 = (MASK(sz) >> (8 * off)) & msks
                op = Oshr
            if (msk1 & msk) == 0:
                continue
            if ins.op == Oblit0:
                r = def_(sl1, MASK(sz), b, i, il)
                if r == R:
                    return Load()
            if off:
                cls1 = cls
                if op == Oshr and off + sl.sz > 4:
                    cls1 = Kl
                r = cast(r, cls1, il)
                r1 = getcon(8 * off, curf)
                r = iins(cls1, op, r, r1, il)
            if (msk1 & msk) != msk1 or off + sz < sl.sz:
                r = mask(cls, r, msk1 & msk, il)
            if (msk & ~msk1) != 0:
                r1 = def_(sl, msk & ~msk1, b, i, il)
                if r1 == R:
                    return Load()
                r = iins(cls, Oor, r, r1, il)
            if msk == msks:
                r = cast(r, sl.cls, il)
            return r
        elif res == MayAlias:
            if ld:
                continue
            else:
                return Load()
        elif res == NoAlias:
            continue
        else:
            die("unreachable")

    for k in range(nlog):
        ist = ilog[k]
        if ist.isphi and ist.bid == b.id:
            if ist.phi_m.ref == sl.ref:
                if ist.phi_m.off == sl.off:
                    if ist.phi_m.sz == sl.sz:
                        r = ist.phi_p.to
                        if msk != msks:
                            r = mask(cls, r, msk, il)
                        else:
                            r = cast(r, sl.cls, il)
                        return r

    p = b.phi
    while p:
        if killsl(p.to, sl):
            # scanning predecessors in that
            # case would be unsafe
            return Load()
        p = p.link

    if b.npred == 0:
        return Load()
    if b.npred == 1:
        bp = b.pred[0]
        assert bp.loop >= il.blk.loop
        l = il.copy()
        if bp.s2:
            l.type = LNoLoad
        r1 = def_(sl, msk, bp, None, l)
        if r1 == R:
            return Load()
        return r1

    r = newtmp("ld", sl.cls, curf)
    p = Phi()
    ist = _newlog()
    ist.isphi = 1
    ist.bid = b.id
    ist.phi_m = sl.copy()
    ist.phi_p = p
    p.to = r
    p.cls = sl.cls
    p.narg = b.npred
    p.arg = [R] * p.narg
    p.blk = [None] * p.narg
    l = Loc()
    for np in range(b.npred):
        bp = b.pred[np]
        if (not bp.s2
                and il.type != LNoLoad
                and bp.loop < il.blk.loop):
            l.type = LLoad
        else:
            l.type = LNoLoad
        l.blk = bp
        l.off = bp.nins
        r1 = def_(sl, msks, bp, None, l)
        if r1 == R:
            return Load()
        p.arg[np] = r1
        p.blk[np] = bp
        # XXX - multiplicity in predecessors!!!
    if msk != msks:
        r = mask(cls, r, msk, il)
    return r


def icmp(a, b):
    c = a.bid - b.bid
    if c:
        return c
    if a.isphi and b.isphi:
        return 0
    if a.isphi:
        return -1
    if b.isphi:
        return +1
    c = a.off - b.off
    if c:
        return c
    return a.num - b.num


def loadopt(fn):
    """require rpo ssa alias"""
    global curf, ilog, nlog, inum
    curf = fn
    ilog = []
    nlog = 0
    inum = 0
    lim = sys.getrecursionlimit()
    if lim < 50000:
        sys.setrecursionlimit(50000)
    b = fn.start
    while b:
        for ii in range(b.nins):
            i = b.ins[ii]
            if not isload(i.op):
                continue
            sz = loadsz(i)
            sl = Slice(i.arg[0], 0, sz, i.cls)
            l = Loc(LRoot, ii, b)
            rebase(sl)
            i.arg[1] = def_(sl, MASK(sz), b, ii, l)
        b = b.link
    # qsort (glibc: stable merge sort)
    del ilog[nlog:]
    ilog.sort(key=cmp_to_key(icmp))
    sent = Insert()
    sent.bid = fn.nblk  # add a sentinel
    ilog.append(sent)
    k = 0
    for n in range(fn.nblk):
        b = fn.rpo[n]
        while ilog[k].bid == n and ilog[k].isphi:
            ist = ilog[k]
            ist.phi_p.link = b.phi
            b.phi = ist.phi_p
            k += 1
        ni = 0
        ib = []
        while True:
            if ilog[k].bid == n and ilog[k].off == ni:
                i = ilog[k].ins
                k += 1
            else:
                if ni == b.nins:
                    break
                i = b.ins[ni]
                ni += 1
                if isload(i.op) and i.arg[1] != R:
                    ext = Oextsb + i.op - Oloadsb
                    if i.op in (Oloadsb, Oloadub, Oloadsh, Oloaduh):
                        i.op = ext
                    elif i.op in (Oloadsw, Oloaduw) and i.cls == Kl:
                        i.op = ext
                    elif i.op in (Oloadsw, Oloaduw, Oload):
                        i.op = Ocopy
                    else:
                        die("unreachable")
                    i.arg[0] = i.arg[1]
                    i.arg[1] = R
            ib.append(i)
        b.ins = [x.copy() for x in ib]
        b.nins = len(ib)
    ilog = []
    if lim < 50000:
        sys.setrecursionlimit(lim)
    if D.debug['M']:
        sys.stderr.write("\n> After load elimination:\n")
        printfn(fn, sys.stderr)
