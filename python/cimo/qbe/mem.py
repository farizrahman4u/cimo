"""Port of QBE mem.c."""

import sys

from . import defs as D
from .defs import (R, RTmp, RInt, Tmp0, NBit, M64, UNDEF, CON_Z, ALoc, Ins, Kl, KBASE,
                   UIns, UJmp, Onop, Ocopy, Ocast, Oalloc, Oalloc1, Oloadsw, Oloaduw,
                   Oload, Oloadsb, Oextsb, Oargc, Oblit0, Oblit1, Jretc, Jret0, BIT,
                   TMP, INT, optab, isload, isstore, isarg, isret, rtype, rval, rsval)
from .util import err
from .alias import getalias
from .load import loadsz, storesz
from .cfg import loopiter
from .parse import printfn

INT_MAX = 0x7fffffff


def _nop(i):
    """*i = (Ins){.op = Onop}"""
    i.set(Ins(Onop))


def _insidx(b, i):
    """index of the Ins object i in b.ins (C pointer arithmetic)"""
    for k in range(b.nins):
        if b.ins[k] is i:
            return k
    raise AssertionError("instruction not in block")


def promote(fn):
    """require use, maintains use counts"""
    # promote uniform stack slots to temporaries
    b = fn.start
    for i in b.ins[:b.nins]:
        if Oalloc > i.op or i.op > Oalloc1:
            continue
        # specific to NAlign == 3
        assert rtype(i.to) == RTmp
        t = fn.tmp[rval(i.to)]
        if t.ndef != 1:
            continue
        k = -1
        s = -1
        skip = False
        for u in t.use[:t.nuse]:
            if u.type != UIns:
                skip = True
                break
            l = u.u
            if isload(l.op):
                if s == -1 or s == loadsz(l):
                    s = loadsz(l)
                    continue
            if isstore(l.op):
                if i.to == l.arg[1] and i.to != l.arg[0]:
                    if s == -1 or s == storesz(l):
                        if k == -1 or k == optab[l.op].argcls[0][0]:
                            s = storesz(l)
                            k = optab[l.op].argcls[0][0]
                            continue
            skip = True
            break
        if skip:
            continue
        # get rid of the alloc and replace uses
        _nop(i)
        t.ndef -= 1
        for u in t.use[:t.nuse]:
            l = u.u
            if isstore(l.op):
                l.cls = k
                l.op = Ocopy
                l.to = l.arg[1]
                l.arg[1] = R
                t.nuse -= 1
                t.ndef += 1
            else:
                if k == -1:
                    err("slot %%%s is read but never stored to",
                        fn.tmp[rval(l.arg[0])].name)
                # try to turn loads into copies so we
                # can eliminate them later
                if (l.op == Oloadsw or l.op == Oloaduw) and k == Kl:
                    l.op = Oextsb + (l.op - Oloadsb)
                elif l.op in (Oloadsw, Oloaduw, Oload):
                    if KBASE(k) != KBASE(l.cls):
                        l.op = Ocast
                    else:
                        l.op = Ocopy
                else:
                    l.op = Oextsb + (l.op - Oloadsb)
    if D.debug['M']:
        sys.stderr.write("\n> After slot promotion:\n")
        printfn(fn, sys.stderr)


class Range:
    """[a, b) with 0 <= a"""
    __slots__ = ('a', 'b')

    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    def copy(self):
        return Range(self.a, self.b)


class Store:
    """i is the Ins; blk, idx locate it (for i+1)"""
    __slots__ = ('ip', 'i', 'blk', 'idx')

    def __init__(self, ip, i, blk, idx):
        self.ip = ip
        self.i = i
        self.blk = blk
        self.idx = idx


class Slot:
    __slots__ = ('t', 'sz', 'm', 'l', 'r', 's', 'st', 'nst')

    def __init__(self):
        self.t = 0
        self.sz = 0
        self.m = 0
        self.l = 0
        self.r = Range()
        self.s = None
        self.st = []
        self.nst = 0


def rin(r, n):
    return r.a <= n < r.b


def rovlap(r0, r1):
    return r0.b and r1.b and r0.a < r1.b and r1.a < r0.b


def radd(r, n):
    if not r.b:
        r.a = n
        r.b = n + 1
    elif n < r.a:
        r.a = n
    elif n >= r.b:
        r.b = n + 1


def slot(r, fn, sl):
    """slot(&s, &off, r, fn, sl): returns (s, off), or None."""
    a = D.Alias()
    getalias(a, r, fn)
    if a.type != ALoc:
        return None
    t = fn.tmp[a.base]
    if t.visit < 0:
        return None
    return sl[t.visit], a.offset


def _shl(x, off):
    # (bits)x << off; C shifts by off mod 64 on the host
    return (x << (off & 63)) & M64


def load(r, x, ip, fn, sl):
    so = slot(r, fn, sl)
    if so:
        s, off = so
        s.l |= _shl(x, off)
        s.l &= s.m
        if s.l:
            radd(s.r, ip)


def store(r, x, ip, i, blk, idx, fn, sl):
    so = slot(r, fn, sl)
    if so:
        s, off = so
        if s.l:
            radd(s.r, ip)
            s.l &= ~_shl(x, off) & M64
        else:
            s.nst += 1
            del s.st[s.nst - 1:]
            s.st.append(Store(ip, i, blk, idx))


def _scmp_key(s):
    # scmp: by decreasing size, then increasing r.a (qsort is a stable
    # merge sort in glibc)
    return (-s.sz, s.r.a)


def maxrpo(hd, b):
    if hd.loop < b.id:
        hd.loop = b.id


def coalesce(fn):
    # minimize the stack usage
    # by coalescing slots
    sl = []
    for n in range(Tmp0, fn.ntmp):
        t = fn.tmp[n]
        t.visit = -1
        if t.alias.type == ALoc:
            if t.alias.slot is t.alias:
                if t.bid == fn.start.id:
                    if t.alias.loc_sz != -1:
                        t.visit = len(sl)
                        s = Slot()
                        sl.append(s)
                        s.t = n
                        s.sz = t.alias.loc_sz
                        s.m = t.alias.loc_m
                        s.s = None
                        s.st = []
                        s.nst = 0

    # one-pass liveness analysis
    b = fn.start
    while b:
        b.loop = -1
        b = b.link
    loopiter(fn, maxrpo)
    bl = []  # (blk, idx) of blit0 instructions
    br = [Range() for _ in range(fn.nblk)]
    ip = INT_MAX - 1
    for n in range(fn.nblk - 1, -1, -1):
        b = fn.rpo[n]
        succ = [b.s1, b.s2]
        br[n].b = ip
        ip -= 1
        for s in sl:
            s.l = 0
            for ps in succ:
                if not ps:
                    break
                m = ps.id
                if m > n and rin(s.r, br[m].a):
                    s.l = s.m
                    radd(s.r, ip)
        if b.jmp.type == Jretc:
            ip -= 1
            load(b.jmp.arg, M64, ip, fn, sl)
        for ii in range(b.nins - 1, -1, -1):
            i = b.ins[ii]
            arg = i.arg
            if i.op == Oargc:
                ip -= 1
                load(arg[1], M64, ip, fn, sl)
            if isload(i.op):
                x = BIT(loadsz(i)) - 1
                ip -= 1
                load(arg[0], x, ip, fn, sl)
            if isstore(i.op):
                x = BIT(storesz(i)) - 1
                store(arg[1], x, ip, i, b, ii, fn, sl)
                ip -= 1
            if i.op == Oblit0:
                i1 = b.ins[ii + 1]
                assert i1.op == Oblit1
                assert rtype(i1.arg[0]) == RInt
                sz = abs(rsval(i1.arg[0]))
                x = M64 if sz >= NBit else BIT(sz) - 1
                store(arg[1], x, ip, i, b, ii, fn, sl)
                ip -= 1
                load(arg[0], x, ip, fn, sl)
                bl.append((b, ii))
        for s in sl:
            if s.l:
                radd(s.r, ip)
                if b.loop != -1:
                    assert b.loop >= n
                    radd(s.r, br[b.loop].b - 1)
        br[n].a = ip

    # kill dead stores
    for s in sl:
        for n in range(s.nst):
            if not rin(s.r, s.st[n].ip):
                st = s.st[n]
                i = st.i
                if i.op == Oblit0:
                    _nop(st.blk.ins[st.idx + 1])
                _nop(i)

    # kill slots with an empty live range
    total = 0
    freed = 0
    stk = []
    sl0 = []
    for s in sl:
        total += s.sz
        if not s.r.b:
            stk.append(s.t)
            freed += s.sz
        else:
            sl0.append(s)
    sl = sl0
    if D.debug['M']:
        err_ = sys.stderr
        err_.write("\n> Slot coalescing:\n")
        if stk:
            err_.write("\tkill [")
            for m in stk:
                err_.write(" %%%s" % fn.tmp[m].name)
            err_.write(" ]\n")
    while stk:
        t = fn.tmp[stk.pop()]
        assert t.ndef == 1 and t.def_
        i = t.def_
        if isload(i.op):
            i.op = Ocopy
            i.arg[0] = UNDEF
            continue
        _nop(i)
        for u in t.use[:t.nuse]:
            if u.type == UJmp:
                b = fn.rpo[u.bid]
                assert isret(b.jmp.type)
                b.jmp.type = Jret0
                b.jmp.arg = R
                continue
            assert u.type == UIns
            i = u.u
            if i.to != R:
                assert rtype(i.to) == RTmp
                stk.append(rval(i.to))
            elif isarg(i.op):
                assert i.op == Oargc
                i.arg[1] = CON_Z  # crash
            else:
                if i.op == Oblit0:
                    b = fn.rpo[u.bid]
                    _nop(b.ins[_insidx(b, i) + 1])
                _nop(i)

    # fuse slots by decreasing size
    sl.sort(key=_scmp_key)
    nsl = len(sl)
    fused = 0
    for n in range(nsl):
        s0 = sl[n]
        if s0.s:
            continue
        s0.s = s0
        r = s0.r.copy()
        for si in range(n + 1, nsl):
            s = sl[si]
            if s.s or not s.r.b:
                continue
            if rovlap(r, s.r):
                # O(n); can be approximated
                # by 'goto Skip;' if need be
                skip = False
                for m in range(n, si):
                    if sl[m].s is s0:
                        if rovlap(sl[m].r, s.r):
                            skip = True
                            break
                if skip:
                    continue
            radd(r, s.r.a)
            radd(r, s.r.b - 1)
            s.s = s0
            fused += s.sz

    # substitute fused slots
    for si in range(nsl):
        s = sl[si]
        t = fn.tmp[s.t]
        # the visit link is stale,
        # reset it before the slot()
        # calls below
        t.visit = si
        assert t.ndef == 1 and t.def_
        if s.s is s:
            continue
        _nop(t.def_)
        ts = fn.tmp[s.s.t]
        assert t.bid == ts.bid
        bb = fn.rpo[t.bid]
        if _insidx(bb, t.def_) < _insidx(bb, ts.def_):
            # make sure the slot we
            # selected has a def that
            # dominates its new uses
            t.def_.set(ts.def_)
            _nop(ts.def_)
            ts.def_ = t.def_
        for u in t.use[:t.nuse]:
            if u.type == UJmp:
                b = fn.rpo[u.bid]
                b.jmp.arg = TMP(s.s.t)
                continue
            assert u.type == UIns
            arg = u.u.arg
            for n in range(2):
                if arg[n] == TMP(s.t):
                    arg[n] = TMP(s.s.t)

    # fix newly overlapping blits
    for (b, ii) in bl:
        i = b.ins[ii]
        if i.op == Oblit0:
            so = slot(i.arg[0], fn, sl)
            if so:
                s, off0 = so
                so0 = slot(i.arg[1], fn, sl)
                if so0:
                    s0, off1 = so0
                    if s.s is s0.s:
                        i1 = b.ins[ii + 1]
                        if off0 < off1:
                            sz = rsval(i1.arg[0])
                            assert sz >= 0
                            i1.arg[0] = INT(-sz)
                        elif off0 == off1:
                            _nop(i)
                            _nop(i1)

    if D.debug['M']:
        err_ = sys.stderr
        for n0 in range(nsl):
            s0 = sl[n0]
            if s0.s is not s0:
                continue
            err_.write("\tfuse (% 3db) [" % s0.sz)
            for s in sl[n0:]:
                if s.s is not s0:
                    continue
                err_.write(" %%%s" % fn.tmp[s.t].name)
                if s.r.b:
                    err_.write("[%d,%d)" % (s.r.a - ip, s.r.b - ip))
                else:
                    err_.write("{}")
            err_.write(" ]\n")
        err_.write("\tsums %u/%u/%u (killed/fused/total)\n\n" % (freed, fused, total))
        printfn(fn, sys.stderr)
