"""Port of QBE cfg.c."""

from . import defs as D
from .defs import Blk, Ins, U32, Jxxx, Jjmp, Jjnz, Jret0, Ocopy, Onop, Odbgloc, R, req
from .util import addins, addbins, phiarg, phiargn


def newblk():
    return Blk()


def fixphis(f):
    b = f.start
    while b:
        assert b.id < f.nblk
        p = b.phi
        while p:
            n0 = 0
            for n in range(p.narg):
                bp = p.blk[n]
                if bp.id != U32:
                    if bp.s1 is b or bp.s2 is b:
                        p.blk[n0] = bp
                        p.arg[n0] = p.arg[n]
                        n0 += 1
            assert n0 > 0
            p.narg = n0
            del p.blk[n0:], p.arg[n0:]
            p = p.link
        b = b.link


def addpred(bp, b):
    b.npred += 1
    b.pred[b.npred - 1:] = [bp]


def fillpreds(f):
    b = f.start
    while b:
        b.npred = 0
        b.pred = []
        b = b.link
    b = f.start
    while b:
        if b.s1:
            addpred(b, b.s1)
        if b.s2 and b.s2 is not b.s1:
            addpred(b, b.s2)
        b = b.link


def porec(b, npo):
    # iterative version of the recursive C porec(); same visiting order
    stack = []
    if not b or b.id != U32:
        return npo
    b.id = 0  # marker
    stack.append([b, 0])
    while stack:
        top = stack[-1]
        b = top[0]
        s1 = b.s1
        s2 = b.s2
        if s1 and s2 and s1.loop > s2.loop:
            s1, s2 = b.s2, b.s1
        if top[1] == 0:
            top[1] = 1
            c = s1
        elif top[1] == 1:
            top[1] = 2
            c = s2
        else:
            b.id = npo
            npo += 1
            stack.pop()
            continue
        if c and c.id == U32:
            c.id = 0
            stack.append([c, 0])
    return npo


def fillrpo(f):
    b = f.start
    while b:
        b.id = U32
        b = b.link
    f.nblk = porec(f.start, 0)
    f.rpo = [None] * f.nblk
    prev = None
    b = f.start
    while b:
        if b.id == U32:
            if prev is None:
                f.start = b.link
            else:
                prev.link = b.link
        else:
            b.id = f.nblk - b.id - 1
            f.rpo[b.id] = b
            prev = b
        b = b.link


def fillcfg(f):
    """fill rpo, preds; prune dead blks"""
    fillrpo(f)
    fillpreds(f)
    fixphis(f)


def inter(b1, b2):
    if b1 is None:
        return b2
    while b1 is not b2:
        if b1.id < b2.id:
            b1, b2 = b2, b1
        while b1.id > b2.id:
            b1 = b1.idom
            assert b1
    return b1


def filldom(fn):
    b = fn.start
    while b:
        b.idom = None
        b.dom = None
        b.dlink = None
        b = b.link
    while True:
        ch = 0
        for n in range(1, fn.nblk):
            b = fn.rpo[n]
            d = None
            for p in range(b.npred):
                if b.pred[p].idom or b.pred[p] is fn.start:
                    d = inter(d, b.pred[p])
            if d is not b.idom:
                ch += 1
                b.idom = d
        if not ch:
            break
    b = fn.start
    while b:
        d = b.idom
        if d:
            assert d is not b
            b.dlink = d.dom
            d.dom = b
        b = b.link


def sdom(b1, b2):
    assert b1 and b2
    if b1 is b2:
        return 0
    while b2.id > b1.id:
        b2 = b2.idom
    return int(b1 is b2)


def dom(b1, b2):
    return b1 is b2 or sdom(b1, b2)


def addfron(a, b):
    for n in range(a.nfron):
        if a.fron[n] is b:
            return
    if not a.nfron:
        a.fron = []
    a.nfron += 1
    a.fron[a.nfron - 1:] = [b]


def fillfron(fn):
    b = fn.start
    while b:
        b.nfron = 0
        b = b.link
    b = fn.start
    while b:
        if b.s1:
            a = b
            while not sdom(a, b.s1):
                addfron(a, b.s1)
                a = a.idom
        if b.s2:
            a = b
            while not sdom(a, b.s2):
                addfron(a, b.s2)
                a = a.idom
        b = b.link


def loopmark(hd, b, f):
    # iterative version of the recursive C loopmark(); same visiting order
    stack = [b]
    while stack:
        b = stack.pop()
        if b.id < hd.id or b.visit == hd.id:
            continue
        b.visit = hd.id
        f(hd, b)
        for p in range(b.npred - 1, -1, -1):
            stack.append(b.pred[p])


def loopiter(fn, f):
    b = fn.start
    while b:
        b.visit = U32
        b = b.link
    for n in range(fn.nblk):
        b = fn.rpo[n]
        for p in range(b.npred):
            if b.pred[p].id >= n:
                loopmark(b, b.pred[p], f)


def filldepth(fn):
    b = fn.start
    while b:
        b.depth = -1
        b = b.link
    fn.start.depth = 0
    b = fn.start
    while b:
        if b.depth != -1:
            b = b.link
            continue
        depth = 1
        d = b.idom
        while d.depth == -1:
            depth += 1
            d = d.idom
        depth += d.depth
        b.depth = depth
        d = b.idom
        while d.depth == -1:
            depth -= 1
            d.depth = depth
            d = d.idom
        b = b.link


def lca(b1, b2):
    if not b1:
        return b2
    if not b2:
        return b1
    while b1.depth > b2.depth:
        b1 = b1.idom
    while b2.depth > b1.depth:
        b2 = b2.idom
    while b1 is not b2:
        b1 = b1.idom
        b2 = b2.idom
    return b1


def multloop(hd, b):
    b.loop *= 10


def fillloop(fn):
    b = fn.start
    while b:
        b.loop = 1
        b = b.link
    loopiter(fn, multloop)


def _uffind(b, uf):
    """uffind(&b, uf): returns the new value of b."""
    b1 = uf[b.id]
    if b1:
        b1 = _uffind(b1, uf)
        uf[b.id] = b1
        return b1
    return b


def simpljmp(fn):
    """requires rpo and no phis, breaks cfg"""
    ret = newblk()
    ret.id = fn.nblk
    fn.nblk += 1
    ret.jmp.type = Jret0
    uf = [None] * fn.nblk
    b = fn.start
    while b:
        assert not b.phi
        if b.jmp.type == Jret0:
            b.jmp.type = Jjmp
            b.s1 = ret
        if b.nins == 0:
            if b.jmp.type == Jjmp:
                b.s1 = _uffind(b.s1, uf)
                if b.s1 is not b:
                    uf[b.id] = b.s1
        b = b.link
    last = None
    b = fn.start
    while b:
        if b.s1:
            b.s1 = _uffind(b.s1, uf)
        if b.s2:
            b.s2 = _uffind(b.s2, uf)
        if b.s1 and b.s1 is b.s2:
            b.jmp.type = Jjmp
            b.s2 = None
        last = b
        b = b.link
    if last is None:
        fn.start = ret
    else:
        last.link = ret


def reachrec(b, to):
    if b is to:
        return 1
    if not b or b.visit:
        return 0
    b.visit = 1
    if reachrec(b.s1, to):
        return 1
    if reachrec(b.s2, to):
        return 1
    return 0


def reaches(fn, b, to):
    """Blk.visit needs to be clear at entry"""
    assert to
    r = reachrec(b, to)
    b = fn.start
    while b:
        b.visit = 0
        b = b.link
    return r


def reachesnotvia(fn, b, to, excl):
    excl.visit = 1
    return reaches(fn, b, to)


def ifgraph(ifb):
    """Returns (thenb, elseb, joinb) or None."""
    if ifb.jmp.type != Jjnz:
        return None
    s1 = ifb.s1
    s2 = ifb.s2
    swap = False
    if s1.id > s2.id:
        s1 = ifb.s2
        s2 = ifb.s1
        swap = True

    def ret(thenb, elseb, joinb):
        if swap:
            # the C code swaps the output pointers
            return elseb, thenb, joinb
        return thenb, elseb, joinb
    if s1 is s2:
        return None
    if s1.jmp.type != Jjmp or s1.npred != 1:
        return None
    if s1.s1 is s2:
        # if-then / if-else
        if s2.npred != 2:
            return None
        return ret(s1, ifb, s2)
    if s2.jmp.type != Jjmp or s2.npred != 1:
        return None
    if s1.s1 is not s2.s1 or s1.s1.npred != 2:
        return None
    assert s1.s1 is not ifb
    return ret(s1, s2, s1.s1)


class _Jmp:
    __slots__ = ('type', 'arg', 's1', 's2')

    def __init__(self):
        self.type = 0
        self.arg = R
        self.s1 = None
        self.s2 = None

    def set(self, o):
        self.type = o.type
        self.arg = o.arg
        self.s1 = o.s1
        self.s2 = o.s2


def jmpeq(a, b):
    return a.type == b.type and req(a.arg, b.arg) and a.s1 is b.s1 and a.s2 is b.s2


def jmpnophi(j):
    if j.s1 and j.s1.phi:
        return 0
    if j.s2 and j.s2.phi:
        return 0
    return 1


def simplcfg(fn):
    """require cfg rpo, breaks use"""
    import sys
    from .parse import printfn
    if D.debug['C']:
        sys.stderr.write("\n> Before CFG simplification:\n")
        printfn(fn, sys.stderr)

    cpy = Ins(Ocopy)
    b = fn.start
    while b:
        if b.npred == 1:
            bb = b.pred[0]
            p = b.phi
            while p:
                cpy.cls = p.cls
                cpy.to = p.to
                cpy.arg[0] = phiarg(p, bb)
                addins(bb, cpy)
                p = p.link
            b.phi = None
        b = b.link

    jmp = [_Jmp() for _ in range(fn.nblk)]
    empty = [0] * fn.nblk
    b = fn.start
    while b:
        j = jmp[b.id]
        j.type = b.jmp.type
        j.arg = b.jmp.arg
        j.s1 = b.s1
        j.s2 = b.s2
        empty[b.id] = int(not b.phi)
        for i in b.ins[:b.nins]:
            if i.op != Onop and i.op != Odbgloc:
                empty[b.id] = 0
                break
        b = b.link

    while True:
        done = 1
        b = fn.start
        while b:
            if b.id == U32:
                b = b.link
                continue
            j = jmp[b.id]
            if j.type == Jjmp and j.s1.npred == 1:
                assert not j.s1.phi
                addbins(b, j.s1)
                empty[b.id] &= empty[j.s1.id]
                jj = jmp[j.s1.id]
                for bb in (jj.s1, jj.s2):
                    if not bb:
                        break
                    p = bb.phi
                    while p:
                        n = phiargn(p, j.s1)
                        p.blk[n] = b
                        p = p.link
                j.s1.id = U32
                j.set(jj)
                done = 0
            elif (j.type == Jjnz
                  and empty[j.s1.id] and empty[j.s2.id]
                  and jmpeq(jmp[j.s1.id], jmp[j.s2.id])
                  and jmpnophi(jmp[j.s1.id])):
                j.set(jmp[j.s1.id])
                done = 0
            b = b.link
        if done:
            break

    b = fn.start
    while b:
        if b.id != U32:
            j = jmp[b.id]
            b.jmp.type = j.type
            b.jmp.arg = j.arg
            b.s1 = j.s1
            b.s2 = j.s2
            assert not j.s1 or j.s1.id != U32
            assert not j.s2 or j.s2.id != U32
        b = b.link

    fillcfg(fn)

    if D.debug['C']:
        sys.stderr.write("\n> After CFG simplification:\n")
        printfn(fn, sys.stderr)
