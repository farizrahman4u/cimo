"""Port of QBE spill.c."""

import sys

from . import defs as D
from .defs import (R, RTmp, RMem, RCall, Tmp0, NIns, U32, M64, BSet, Ocopy, Ocall, Oload,
                   Ostorew, KWIDE, KBASE, BIT, TMP, SLOT, rtype, rval, s32)
from .util import (bsinit, bszero, bscount, bsset, bsclr, bscopy, bsunion, bsinter, bsdiff,
                   bshas, bsiter, bselems, dumpts, emit, emiti, idup, isreg, phicls)
from .cfg import loopiter
from .parse import printfn


def aggreg(hd, b):
    # aggregate looping information at
    # loop headers
    bsunion(hd.gen, b.gen)
    for k in range(2):
        if b.nlive[k] > hd.nlive[k]:
            hd.nlive[k] = b.nlive[k]


def tmpuse(r, use, loop, fn):
    if rtype(r) == RMem:
        m = fn.mem[rval(r)]
        tmpuse(m.base, 1, loop, fn)
        tmpuse(m.index, 1, loop, fn)
    elif rtype(r) == RTmp and rval(r) >= Tmp0:
        t = fn.tmp[rval(r)]
        t.nuse += use
        t.ndef += int(not use)
        t.cost = (t.cost + loop) & U32


def fillcost(fn):
    """evaluate spill costs of temporaries,
    this also fills usage information
    requires rpo, preds"""
    err = sys.stderr
    loopiter(fn, aggreg)
    if D.debug['S']:
        err.write("\n> Loop information:\n")
        b = fn.start
        while b:
            a = 0
            while a < b.npred:
                if b.id <= b.pred[a].id:
                    break
                a += 1
            if a != b.npred:
                err.write("\t%-10s" % b.name)
                err.write(" (% 3d " % b.nlive[0])
                err.write("% 3d) " % b.nlive[1])
                dumpts(b.gen, fn.tmp, err)
            b = b.link
    for n in range(fn.ntmp):
        t = fn.tmp[n]
        t.cost = U32 if n < Tmp0 else 0
        t.nuse = 0
        t.ndef = 0
    b = fn.start
    while b:
        p = b.phi
        while p:
            t = fn.tmp[rval(p.to)]
            tmpuse(p.to, 0, 0, fn)
            for a in range(p.narg):
                n = p.blk[a].loop
                t.cost = (t.cost + n) & U32
                tmpuse(p.arg[a], 1, n, fn)
            p = p.link
        n = b.loop
        for i in b.ins[:b.nins]:
            tmpuse(i.to, 0, n, fn)
            tmpuse(i.arg[0], 1, n, fn)
            tmpuse(i.arg[1], 1, n, fn)
        tmpuse(b.jmp.arg, 1, n, fn)
        b = b.link
    if D.debug['S']:
        err.write("\n> Spill costs:\n")
        for n in range(Tmp0, fn.ntmp):
            err.write("\t%-10s %d\n" % (fn.tmp[n].name, s32(fn.tmp[n].cost)))
        err.write("\n")


fst = None    # temps to prioritize in registers (for tcmp1)
tmp = None    # current temporaries (for tcmpX)
ntmp = 0      # current # of temps (for limit)
locs = 0      # stack size used by locals
slot4 = 0     # next slot of 4 bytes
slot8 = 0     # ditto, 8 bytes
mask = [BSet(), BSet()]  # class masks

# qsort() in glibc 2.39 is a stable merge sort; the comparators below are
# consistent, so a stable sort on the equivalent key gives the same order.


def _tcmp0_key(t):
    return -tmp[t].cost


def _tcmp1_key(t):
    return (-bshas(fst, t), -tmp[t].cost)


def slot(t):
    global slot4, slot8
    assert t >= Tmp0, "cannot spill register"
    s = tmp[t].slot
    if s == -1:
        # specific to NAlign == 3
        # nice logic to pack stack slots
        # on demand, there can be only
        # one hole and slot4 points to it
        #
        # invariant: slot4 <= slot8
        if KWIDE(tmp[t].cls):
            s = slot8
            if slot4 == slot8:
                slot4 += 2
            slot8 += 2
        else:
            s = slot4
            if slot4 == slot8:
                slot8 += 2
                slot4 += 1
            else:
                slot4 = slot8
        s += locs
        tmp[t].slot = s
    return SLOT(s)


def limit(b, k, f):
    """restricts b to hold at most k
    temporaries, preferring those
    present in f (if given), then
    those with the largest spill
    cost"""
    global fst
    nt = bscount(b)
    if nt <= k:
        return
    tarr = list(bselems(b))
    b.t = 0
    if nt > 1:
        if not f:
            tarr.sort(key=_tcmp0_key)
        else:
            fst = f
            tarr.sort(key=_tcmp1_key)
    i = 0
    while i < k and i < nt:
        bsset(b, tarr[i])
        i += 1
    while i < nt:
        slot(tarr[i])
        i += 1


def limit2(b1, k1, k2, f):
    """spills temporaries to fit the
    target limits using the same
    preferences as limit(); assumes
    that k1 gprs and k2 fprs are
    currently in use"""
    b2 = BSet()
    bsinit(b2, ntmp)
    bscopy(b2, b1)
    bsinter(b1, mask[0])
    bsinter(b2, mask[1])
    limit(b1, D.T.ngpr - k1, f)
    limit(b2, D.T.nfpr - k2, f)
    bsunion(b1, b2)


def sethint(u, r):
    for t in bselems(u, Tmp0):
        tmp[phicls(t, tmp)].hint_m |= r


def reloads(u, v):
    """reloads temporaries in u that are
    not in v from their slots"""
    for t in bselems(u, Tmp0):
        if not bshas(v, t):
            emit(Oload, tmp[t].cls, TMP(t), slot(t), R)


def store(r, s):
    if s != -1:
        emit(Ostorew + tmp[rval(r)].cls, 0, R, r, SLOT(s))


def regcpy(i):
    return i.op == Ocopy and isreg(i.arg[0])


def dopm(b, i, v):
    """i is an index in b.ins; returns an index."""
    T = D.T
    ins = b.ins
    u = BSet()
    bsinit(u, ntmp)
    # consecutive copies from
    # registers need to be handled
    # as one large instruction
    #
    # fixme: there is an assumption
    # that calls are always followed
    # by copy instructions here, this
    # might not be true if previous
    # passes change
    i += 1
    i1 = i
    while True:
        i -= 1
        t = rval(ins[i].to)
        if ins[i].to != R:
            if bshas(v, t):
                bsclr(v, t)
                store(ins[i].to, tmp[t].slot)
        bsset(v, rval(ins[i].arg[0]))
        if not (i != 0 and regcpy(ins[i - 1])):
            break
    bscopy(u, v)
    if i != 0 and ins[i - 1].op == Ocall:
        v.t &= ~T.retregs(ins[i - 1].arg[1], None)
        limit2(v, T.nrsave[0], T.nrsave[1], None)
        r = 0
        n = 0
        while T.rsave[n] >= 0:
            r |= BIT(T.rsave[n])
            n += 1
        v.t |= T.argregs(ins[i - 1].arg[1], None)
    else:
        limit2(v, 0, 0, None)
        r = v.t & M64
    sethint(v, r)
    reloads(u, v)
    while True:
        i1 -= 1
        emiti(ins[i1])
        if i1 == i:
            break
    return i


def merge(u, bu, v, bv):
    if bu.loop <= bv.loop:
        bsunion(u, v)
    else:
        for t in bselems(v):
            if tmp[t].slot == -1:
                bsset(u, t)


def spill(fn):
    """spill code insertion
    requires spill costs, rpo, liveness

    Note: this will replace liveness
    information (in, out) with temporaries
    that must be in registers at block
    borders

    Be careful with:
    - Ocopy instructions to ensure register
      constraints"""
    global tmp, ntmp, locs, slot4, slot8
    from .live import liveon
    T = D.T
    tmp = fn.tmp
    ntmp = fn.ntmp
    u = BSet()
    v = BSet()
    w = BSet()
    bsinit(u, ntmp)
    bsinit(v, ntmp)
    bsinit(w, ntmp)
    bsinit(mask[0], ntmp)
    bsinit(mask[1], ntmp)
    locs = fn.slot
    slot4 = 0
    slot8 = 0
    lvarg = [0, 0]
    for t in range(ntmp):
        k = 0
        if T.fpr0 <= t < T.fpr0 + T.nfpr:
            k = 1
        if t >= Tmp0:
            k = KBASE(tmp[t].cls)
        bsset(mask[k], t)

    for bi in range(fn.nblk - 1, -1, -1):
        b = fn.rpo[bi]
        # invariant: all blocks with bigger rpo got
        # their in,out updated.

        # 1. find temporaries in registers at
        # the end of the block (put them in v)
        D.curi = 0
        s1 = b.s1
        s2 = b.s2
        hd = None
        if s1 and s1.id <= b.id:
            hd = s1
        if s2 and s2.id <= b.id:
            if not hd or s2.id >= hd.id:
                hd = s2
        if hd:
            # back-edge
            bszero(v)
            hd.gen.t |= T.rglob  # don't spill registers
            for k in range(2):
                n = T.ngpr if k == 0 else T.nfpr
                bscopy(u, b.out)
                bsinter(u, mask[k])
                bscopy(w, u)
                bsinter(u, hd.gen)
                bsdiff(w, hd.gen)
                if bscount(u) < n:
                    j = bscount(w)  # live through
                    l = hd.nlive[k]
                    limit(w, s32(n - (l - j)), None)
                    bsunion(u, w)
                else:
                    limit(u, n, None)
                bsunion(v, u)
        elif s1:
            # avoid reloading temporaries
            # in the middle of loops
            bszero(v)
            liveon(w, b, s1)
            merge(v, b, w, s1)
            if s2:
                liveon(u, b, s2)
                merge(v, b, u, s2)
                bsinter(w, u)
            limit2(v, 0, 0, w)
        else:
            bscopy(v, b.out)
            if rtype(b.jmp.arg) == RCall:
                v.t |= T.retregs(b.jmp.arg, None)
        if rtype(b.jmp.arg) == RTmp:
            t = rval(b.jmp.arg)
            assert KBASE(tmp[t].cls) == 0
            bsset(v, t)
            limit2(v, 0, 0, None)
            if not bshas(v, t):
                b.jmp.arg = slot(t)
        for t in bselems(b.out, Tmp0):
            if not bshas(v, t):
                slot(t)
        bscopy(b.out, v)

        # 2. process the block instructions
        D.curi = NIns
        i = b.nins
        while i != 0:
            i -= 1
            ins = b.ins[i]
            if regcpy(ins):
                i = dopm(b, i, v)
                continue
            bszero(w)
            if ins.to != R:
                assert rtype(ins.to) == RTmp
                t = rval(ins.to)
                if bshas(v, t):
                    bsclr(v, t)
                else:
                    # make sure we have a reg
                    # for the result
                    assert t >= Tmp0, "dead reg"
                    bsset(v, t)
                    bsset(w, t)
            j = T.memargs(ins.op)
            for n in range(2):
                if rtype(ins.arg[n]) == RMem:
                    j -= 1
            for n in range(2):
                rt = rtype(ins.arg[n])
                if rt == RMem:
                    t = rval(ins.arg[n])
                    m = fn.mem[t]
                    if rtype(m.base) == RTmp:
                        bsset(v, rval(m.base))
                        bsset(w, rval(m.base))
                    if rtype(m.index) == RTmp:
                        bsset(v, rval(m.index))
                        bsset(w, rval(m.index))
                elif rt == RTmp:
                    t = rval(ins.arg[n])
                    lvarg[n] = bshas(v, t)
                    bsset(v, t)
                    jj = j
                    j -= 1
                    if jj <= 0:
                        bsset(w, t)
            bscopy(u, v)
            limit2(v, 0, 0, w)
            for n in range(2):
                if rtype(ins.arg[n]) == RTmp:
                    t = rval(ins.arg[n])
                    if not bshas(v, t):
                        # do not reload if the
                        # argument is dead
                        if not lvarg[n]:
                            bsclr(u, t)
                        ins.arg[n] = slot(t)
            reloads(u, v)
            if ins.to != R:
                t = rval(ins.to)
                store(ins.to, tmp[t].slot)
                if t >= Tmp0:
                    # in case i->to was a
                    # dead temporary
                    bsclr(v, t)
            emiti(ins)
            r = v.t & M64  # Tmp0 is NBit
            if r:
                sethint(v, r)
        if b is fn.start:
            assert (v.t & M64) == (T.rglob | fn.reg)
        else:
            assert (v.t & M64) == T.rglob

        p = b.phi
        while p:
            assert rtype(p.to) == RTmp
            t = rval(p.to)
            if bshas(v, t):
                bsclr(v, t)
                store(p.to, tmp[t].slot)
            elif bshas(b.in_, t):
                # only if the phi is live
                p.to = slot(rval(p.to))
            p = p.link
        bscopy(b.in_, v)
        idup(b, D.curi, NIns - D.curi)

    # align the locals to a 16 byte boundary
    # specific to NAlign == 3
    slot8 += slot8 & 3
    fn.slot += slot8

    if D.debug['S']:
        err = sys.stderr
        err.write("\n> Block information:\n")
        b = fn.start
        while b:
            err.write("\t%-10s (% 5d) " % (b.name, b.loop))
            dumpts(b.out, fn.tmp, err)
            b = b.link
        err.write("\n> After spilling:\n")
        printfn(fn, err)
