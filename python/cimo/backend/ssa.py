"""Port of QBE ssa.c."""

import sys

from . import defs as D
from .defs import (BSet, R, RTmp, Tmp0, U32, UNDEF, Kx, WFull, Wsb, Wub, Wsw, Wuw, Kw,
                   UPhi, UIns, UJmp, Use, Phi, Oparsb, Oloadsb, Oextsb, Oload, TMP,
                   isparbh, isload, isext, rtype, rval)
from .util import (die, err, iscmp, phicls, newtmp, clsmerge, bsinit, bszero, bshas,
                   bsset, bsclr, bscopy)
from .cfg import filldom, fillfron, dom, sdom
from .live import filllive
from .parse import printfn


def adduse(tmp, ty, b, u=None):
    """adduse(tmp, ty, b, ...): u is the Phi (UPhi) or Ins (UIns)."""
    if tmp.use is None:
        return
    n = tmp.nuse
    tmp.nuse += 1
    del tmp.use[n:]
    if ty == UPhi or ty == UIns:
        tmp.use.append(Use(ty, b.id, u))
    elif ty == UJmp:
        tmp.use.append(Use(ty, b.id, None))
    else:
        die("unreachable")


def filluse(fn):
    """fill usage, width, phi, and class information
    must not change .visit fields"""
    tmp = fn.tmp
    for t in range(Tmp0, fn.ntmp):
        tt = tmp[t]
        tt.def_ = None
        tt.bid = U32
        tt.ndef = 0
        tt.nuse = 0
        tt.cls = 0
        tt.phi = 0
        tt.width = WFull
        if tt.use is None:
            tt.use = []
    b = fn.start
    while b:
        p = b.phi
        while p:
            assert rtype(p.to) == RTmp
            tp = rval(p.to)
            tmp[tp].bid = b.id
            tmp[tp].ndef += 1
            tmp[tp].cls = p.cls
            tp = phicls(tp, fn.tmp)
            for a in range(p.narg):
                if rtype(p.arg[a]) == RTmp:
                    t = rval(p.arg[a])
                    adduse(tmp[t], UPhi, b, p)
                    t = phicls(t, fn.tmp)
                    if t != tp:
                        tmp[t].phi = tp
            p = p.link
        for i in b.ins[:b.nins]:
            if i.to != R:
                assert rtype(i.to) == RTmp
                w = WFull
                if isparbh(i.op):
                    w = Wsb + (i.op - Oparsb)
                if isload(i.op) and i.op != Oload:
                    w = Wsb + (i.op - Oloadsb)
                if isext(i.op):
                    w = Wsb + (i.op - Oextsb)
                if iscmp(i.op) is not None:
                    w = Wub
                if w == Wsw or w == Wuw:
                    if i.cls == Kw:
                        w = WFull
                t = rval(i.to)
                tmp[t].width = w
                tmp[t].def_ = i
                tmp[t].bid = b.id
                tmp[t].ndef += 1
                tmp[t].cls = i.cls
            for m in range(2):
                if rtype(i.arg[m]) == RTmp:
                    t = rval(i.arg[m])
                    adduse(tmp[t], UIns, b, i)
        if rtype(b.jmp.arg) == RTmp:
            adduse(tmp[rval(b.jmp.arg)], UJmp, b)
        b = b.link


def refindex(t, fn):
    return newtmp(fn.tmp[t].name, fn.tmp[t].cls, fn)


def phiins(fn):
    u = BSet()
    defs = BSet()
    bsinit(u, fn.nblk)
    bsinit(defs, fn.nblk)
    nt = fn.ntmp
    for t in range(Tmp0, nt):
        fn.tmp[t].visit = 0
        if fn.tmp[t].phi != 0:
            continue
        if fn.tmp[t].ndef == 1:
            ok = 1
            defb = fn.tmp[t].bid
            for use in fn.tmp[t].use[:fn.tmp[t].nuse]:
                ok &= int(use.bid == defb)
            if ok or defb == fn.start.id:
                continue
        bszero(u)
        k = Kx
        blist = []  # stack: *--bp = b is append, *bp++ is pop
        tt = TMP(t)
        b = fn.start
        while b:
            b.visit = 0
            r = R
            for i in b.ins[:b.nins]:
                if r != R:
                    if i.arg[0] == tt:
                        i.arg[0] = r
                    if i.arg[1] == tt:
                        i.arg[1] = r
                if i.to == tt:
                    if not bshas(b.out, t):
                        r = refindex(t, fn)
                        i.to = r
                    else:
                        if not bshas(u, b.id):
                            bsset(u, b.id)
                            blist.append(b)
                        k, c = clsmerge(k, i.cls)
                        if c:
                            die("invalid input")
            if r != R and b.jmp.arg == tt:
                b.jmp.arg = r
            b = b.link
        bscopy(defs, u)
        while blist:
            fn.tmp[t].visit = t
            b = blist.pop()
            bsclr(u, b.id)
            for n in range(b.nfron):
                a = b.fron[n]
                v = a.visit
                a.visit += 1
                if v == 0:
                    if bshas(a.in_, t):
                        p = Phi()
                        p.cls = k
                        p.to = TMP(t)
                        p.link = a.phi
                        p.arg = []
                        p.blk = []
                        a.phi = p
                        if not bshas(defs, a.id):
                            if not bshas(u, a.id):
                                bsset(u, a.id)
                                blist.append(a)


class Name:
    __slots__ = ('r', 'b', 'up')

    def __init__(self, r, b, up):
        self.r = r
        self.b = b
        self.up = up


def rendef(r, b, stk, fn):
    """rendef(&r, ...): returns the new value of r."""
    t = rval(r)
    if r == R or not fn.tmp[t].visit:
        return r
    r1 = refindex(t, fn)
    fn.tmp[rval(r1)].visit = t
    stk[t] = Name(r1, b, stk[t])
    return r1


def getstk(t, b, stk):
    n = stk[t]
    while n and not dom(n.b, b):
        n = n.up
    stk[t] = n
    if not n:
        # uh, oh, warn
        return UNDEF
    return n.r


def renblk(b, stk, fn):
    # iterative version of the recursive C renblk(); all the work is done
    # before the recursive calls, so a preorder walk visits in the same order
    work = [b]
    while work:
        b = work.pop()
        _renblk1(b, stk, fn)
        kids = []
        s = b.dom
        while s:
            kids.append(s)
            s = s.dlink
        work.extend(reversed(kids))


def _renblk1(b, stk, fn):
    tmp = fn.tmp
    p = b.phi
    while p:
        p.to = rendef(p.to, b, stk, fn)
        p = p.link
    for i in b.ins[:b.nins]:
        for m in range(2):
            t = rval(i.arg[m])
            if rtype(i.arg[m]) == RTmp:
                if tmp[t].visit:
                    i.arg[m] = getstk(t, b, stk)
        i.to = rendef(i.to, b, stk, fn)
    t = rval(b.jmp.arg)
    if rtype(b.jmp.arg) == RTmp:
        if tmp[t].visit:
            b.jmp.arg = getstk(t, b, stk)
    succ = [b.s1, None if b.s2 is b.s1 else b.s2]
    for s in succ:
        if not s:
            break
        p = s.phi
        while p:
            t = rval(p.to)
            t = tmp[t].visit
            if t:
                p.narg += 1
                p.arg.append(getstk(t, b, stk))
                p.blk.append(b)
            p = p.link


def ssa(fn):
    """require rpo and use"""
    nt = fn.ntmp
    stk = [None] * nt
    d = D.debug['L']
    D.debug['L'] = 0
    filldom(fn)
    if D.debug['N']:
        f = sys.stderr
        f.write("\n> Dominators:\n")
        b1 = fn.start
        while b1:
            if b1.dom:
                f.write("%10s:" % b1.name)
                b = b1.dom
                while b:
                    f.write(" %s" % b.name)
                    b = b.dlink
                f.write("\n")
            b1 = b1.link
    fillfron(fn)
    filllive(fn)
    phiins(fn)
    renblk(fn.start, stk, fn)
    D.debug['L'] = d
    if D.debug['N']:
        sys.stderr.write("\n> After SSA construction:\n")
        printfn(fn, sys.stderr)


def phicheck(p, b, t):
    for n in range(p.narg):
        if p.arg[n] == t:
            b1 = p.blk[n]
            if b1 is not b and not sdom(b, b1):
                return 1
    return 0


def ssacheck(fn):
    """require use and ssa"""
    t = None
    bu = None

    def fail():
        if t.visit:
            die("%%%s violates ssa invariant", t.name)
        else:
            err("ssa temporary %%%s is used undefined in @%s", t.name, bu.name)

    for n in range(Tmp0, fn.ntmp):
        t = fn.tmp[n]
        if t.ndef > 1:
            err("ssa temporary %%%s defined more than once", t.name)
        if t.nuse > 0 and t.ndef == 0:
            bu = fn.rpo[t.use[0].bid]
            fail()
    b = fn.start
    while b:
        p = b.phi
        while p:
            r = p.to
            t = fn.tmp[rval(r)]
            for u in t.use[:t.nuse]:
                bu = fn.rpo[u.bid]
                if u.type == UPhi:
                    if phicheck(u.u, b, r):
                        fail()
                else:
                    if bu is not b and not sdom(b, bu):
                        fail()
            p = p.link
        pos = None
        for ii in range(b.nins):
            i = b.ins[ii]
            if rtype(i.to) != RTmp:
                continue
            r = i.to
            t = fn.tmp[rval(r)]
            for u in t.use[:t.nuse]:
                bu = fn.rpo[u.bid]
                if u.type == UPhi:
                    if phicheck(u.u, b, r):
                        fail()
                else:
                    if bu is b:
                        if u.type == UIns:
                            # u->u.ins <= i (pointer comparison)
                            if pos is None:
                                pos = {id(x): k for k, x in enumerate(b.ins[:b.nins])}
                            if pos[id(u.u)] <= ii:
                                fail()
                    else:
                        if not sdom(b, bu):
                            fail()
        b = b.link
