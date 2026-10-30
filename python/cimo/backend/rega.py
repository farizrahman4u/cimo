"""Port of QBE rega.c."""

import sys

from . import defs as D
from .defs import (R, RTmp, RMem, RSlot, Tmp0, NIns, U32, M64, BSet, Ocopy, Ocall, Oswap,
                   Jjmp, KBASE, BIT, TMP, SLOT, req, rtype, rval, s32)
from .util import (bsinit, bszero, bsset, bsclr, bscopy, bshas, bselems, die, emit, emiti,
                   idup, isreg, phicls, strf)
from .cfg import newblk
from .parse import printfn


class RMap:
    __slots__ = ('t', 'r', 'w', 'b', 'n')

    def __init__(self):
        self.t = [0] * Tmp0
        self.r = [0] * Tmp0
        self.w = [0] * Tmp0  # wait list, for unmatched hints
        self.b = BSet()
        self.n = 0


NPm = 64  # max copies in a parallel move

regu = 0       # registers used
tmp = None     # function temporaries
mem = None     # function mem references
pm_src = [R] * NPm   # parallel move constructed
pm_dst = [R] * NPm
pm_cls = [0] * NPm
npm = 0        # size of pm
loop = 0       # current loop level

stmov = 0      # stats: added moves
stblk = 0      # stats: added blocks

INT_MAX = 0x7fffffff


class _Ptr:
    """A Ref *: an element of a list (Ins.arg) or an attribute (Mem.base/index)."""
    __slots__ = ('o', 'k', 'isattr')

    def __init__(self, o, k, isattr):
        self.o = o
        self.k = k
        self.isattr = isattr

    def get(self):
        return getattr(self.o, self.k) if self.isattr else self.o[self.k]

    def set(self, v):
        if self.isattr:
            setattr(self.o, self.k, v)
        else:
            self.o[self.k] = v


def hint(t):
    """*hint(t)"""
    return tmp[phicls(t, tmp)].hint_r


def sethint(t, r):
    p = tmp[phicls(t, tmp)]
    if p.hint_r == -1 or p.hint_w > loop:
        p.hint_r = r
        p.hint_w = loop
        tmp[t].visit = -1


def rcopy(ma, mb):
    ma.t[:] = mb.t
    ma.r[:] = mb.r
    ma.w[:] = mb.w
    bscopy(ma.b, mb.b)
    ma.n = mb.n


def rfind(m, t):
    for i in range(m.n):
        if m.t[i] == t:
            return m.r[i]
    return -1


def rref(m, t):
    r = rfind(m, t)
    if r == -1:
        s = tmp[t].slot
        assert s != -1, "should have spilled"
        return SLOT(s)
    else:
        return TMP(r)


def radd(m, t, r):
    global regu
    T = D.T
    assert t >= Tmp0 or t == r, "invalid temporary"
    assert ((T.gpr0 <= r < T.gpr0 + T.ngpr)
            or (T.fpr0 <= r < T.fpr0 + T.nfpr)), "invalid register"
    assert not bshas(m.b, t), "temporary has mapping"
    assert not bshas(m.b, r), "register already allocated"
    assert m.n <= T.ngpr + T.nfpr, "too many mappings"
    bsset(m.b, t)
    bsset(m.b, r)
    m.t[m.n] = t
    m.r[m.n] = r
    m.n += 1
    regu |= BIT(r)


def ralloctry(m, t, try_):
    T = D.T
    if t < Tmp0:
        assert bshas(m.b, t)
        return TMP(t)
    if bshas(m.b, t):
        r = rfind(m, t)
        assert r != -1
        return TMP(r)
    r = tmp[t].visit
    if r == -1 or bshas(m.b, r):
        r = hint(t)
    if r == -1 or bshas(m.b, r):
        if try_:
            return R
        regs = tmp[phicls(t, tmp)].hint_m
        regs |= m.b.t & M64
        if KBASE(tmp[t].cls) == 0:
            r0 = T.gpr0
            r1 = r0 + T.ngpr
        else:
            r0 = T.fpr0
            r1 = r0 + T.nfpr
        for r in range(r0, r1):
            if not (regs & BIT(r)):
                break
        else:
            for r in range(r0, r1):
                if not bshas(m.b, r):
                    break
            else:
                die("no more regs")
    # Found:
    radd(m, t, r)
    sethint(t, r)
    tmp[t].visit = r
    h = hint(t)
    if h != -1 and h != r:
        m.w[h] = t
    return TMP(r)


def ralloc(m, t):
    return ralloctry(m, t, 0)


def rfree(m, t):
    assert t >= Tmp0 or not (BIT(t) & D.T.rglob)
    if not bshas(m.b, t):
        return -1
    i = 0
    while m.t[i] != t:
        assert i + 1 < m.n
        i += 1
    r = m.r[i]
    bsclr(m.b, t)
    bsclr(m.b, r)
    m.n -= 1
    m.t[i:m.n] = m.t[i + 1:m.n + 1]
    m.r[i:m.n] = m.r[i + 1:m.n + 1]
    assert t >= Tmp0 or t == r
    return r


def mdump(m):
    err = sys.stderr
    for i in range(m.n):
        if m.t[i] >= Tmp0:
            err.write(" (%s, R%d)" % (tmp[m.t[i]].name, m.r[i]))
    err.write("\n")


def pmadd(src, dst, k):
    global npm
    if npm == NPm:
        die("no more pm slots")
    pm_src[npm] = src
    pm_dst[npm] = dst
    pm_cls[npm] = k
    npm += 1


ToMove, Moving, Moved = range(3)


def pmrec(status, i, k):
    """k is a one-element list (int *k)."""
    # note, this routine might emit
    # too many large instructions
    if req(pm_src[i], pm_dst[i]):
        status[i] = Moved
        return -1
    assert KBASE(pm_cls[i]) == KBASE(k[0])
    k[0] |= pm_cls[i]
    j = 0
    while j < npm:
        if req(pm_dst[j], pm_src[i]):
            break
        j += 1
    st = Moved if j == npm else status[j]
    if st == Moving:
        c = j  # start of cycle
        emit(Oswap, k[0], R, pm_src[i], pm_dst[i])
    elif st == ToMove:
        status[i] = Moving
        c = pmrec(status, j, k)
        if c == i:
            c = -1  # end of cycle
        elif c != -1:
            emit(Oswap, k[0], R, pm_src[i], pm_dst[i])
        else:
            # fall through
            c = -1
            emit(Ocopy, pm_cls[i], pm_dst[i], pm_src[i], R)
    elif st == Moved:
        c = -1
        emit(Ocopy, pm_cls[i], pm_dst[i], pm_src[i], R)
    else:
        die("unreachable")
    status[i] = Moved
    return c


def pmgen():
    status = [ToMove] * npm
    assert not npm or status[npm - 1] == ToMove
    for i in range(npm):
        if status[i] == ToMove:
            pmrec(status, i, [pm_cls[i]])


def move(r, to, m):
    r1 = -1 if req(to, R) else rfree(m, rval(to))
    if bshas(m.b, r):
        # r is used and not by to
        assert r1 != r
        n = 0
        while m.r[n] != r:
            assert n + 1 < m.n
            n += 1
        t = m.t[n]
        rfree(m, t)
        bsset(m.b, r)
        ralloc(m, t)
        bsclr(m.b, r)
    t = r if req(to, R) else rval(to)
    radd(m, t, r)


def regcpy(i):
    return i.op == Ocopy and isreg(i.arg[0])


def dopm(b, i, m):
    """i is an index in b.ins; returns an index."""
    global npm
    T = D.T
    ins = b.ins
    m0 = RMap()  # okay since we don't use m0.b
    m0.t[:] = m.t
    m0.r[:] = m.r
    m0.w[:] = m.w
    m0.n = m.n
    i += 1
    i1 = i
    while True:
        i -= 1
        move(rval(ins[i].arg[0]), ins[i].to, m)
        if not (i != 0 and regcpy(ins[i - 1])):
            break
    assert m0.n <= m.n
    if i != 0 and ins[i - 1].op == Ocall:
        def_ = T.retregs(ins[i - 1].arg[1], None) | T.rglob
        r = 0
        while T.rsave[r] >= 0:
            if not (BIT(T.rsave[r]) & def_):
                move(T.rsave[r], R, m)
            r += 1
    npm = 0
    for n in range(m.n):
        t = m.t[n]
        s = tmp[t].slot
        r1 = m.r[n]
        r = rfind(m0, t)
        if r != -1:
            pmadd(TMP(r1), TMP(r), tmp[t].cls)
        elif s != -1:
            pmadd(TMP(r1), SLOT(s), tmp[t].cls)
    for ip in range(i, i1):
        if ins[ip].to != R:
            rfree(m, rval(ins[ip].to))
        r = rval(ins[ip].arg[0])
        if rfind(m, r) == -1:
            radd(m, r, r)
    pmgen()
    return i


def prio1(r1, r2):
    # trivial heuristic to begin with,
    # later we can use the distance to
    # the definition instruction
    return hint(rval(r1)) != -1


def insert(r, rs, p):
    i = p
    rs[i] = r
    while True:
        c = i > 0
        i -= 1
        if not (c and prio1(r.get(), rs[i].get())):
            break
        rs[i + 1] = rs[i]
        rs[i] = r


def doblk(b, cur):
    global stmov
    T = D.T
    insb = D.insb
    ra = [None] * 4
    if rtype(b.jmp.arg) == RTmp:
        b.jmp.arg = ralloc(cur, rval(b.jmp.arg))
    D.curi = NIns
    i1 = b.nins
    while i1 != 0:
        i1 -= 1
        emiti(b.ins[i1])
        ii = D.curi
        i = insb[ii]
        rf = -1
        skip = False
        if i.op == Ocall:
            rs = T.argregs(i.arg[1], None) | T.rglob
            r = 0
            while T.rsave[r] >= 0:
                if not (BIT(T.rsave[r]) & rs):
                    rfree(cur, T.rsave[r])
                r += 1
        else:
            dflt = True
            if i.op == Ocopy:
                if regcpy(i):
                    D.curi += 1
                    i1 = dopm(b, i1, cur)
                    stmov = (stmov + ii + 1 - D.curi) & U32
                    continue
                if isreg(i.to):
                    if rtype(i.arg[0]) == RTmp:
                        sethint(rval(i.arg[0]), rval(i.to))
                # fall through
            if dflt:
                if i.to != R:
                    assert rtype(i.to) == RTmp
                    r = rval(i.to)
                    if r < Tmp0 and (BIT(r) & T.rglob):
                        pass
                    else:
                        rf = rfree(cur, r)
                        if rf == -1:
                            assert not isreg(i.to)
                            D.curi += 1
                            skip = True
                        else:
                            i.to = TMP(rf)
        if skip:
            continue
        nr = 0
        for x in range(2):
            rt = rtype(i.arg[x])
            if rt == RMem:
                m = mem[rval(i.arg[x])]
                if rtype(m.base) == RTmp:
                    insert(_Ptr(m, 'base', True), ra, nr)
                    nr += 1
                if rtype(m.index) == RTmp:
                    insert(_Ptr(m, 'index', True), ra, nr)
                    nr += 1
            elif rt == RTmp:
                insert(_Ptr(i.arg, x, False), ra, nr)
                nr += 1
        for r in range(nr):
            ra[r].set(ralloc(cur, rval(ra[r].get())))
        if i.op == Ocopy and req(i.to, i.arg[0]):
            D.curi += 1

        # try to change the register of a hinted
        # temporary if rf is available
        if rf != -1:
            t = cur.w[rf]
            if t != 0:
                if not bshas(cur.b, rf) and hint(t) == rf:
                    rt = rfree(cur, t)
                    if rt != -1:
                        tmp[t].visit = -1
                        ralloc(cur, t)
                        assert bshas(cur.b, rf)
                        emit(Ocopy, tmp[t].cls, TMP(rt), TMP(rf), R)
                        stmov = (stmov + 1) & U32
                        cur.w[rf] = 0
                        for r in range(nr):
                            if req(ra[r].get(), TMP(rt)):
                                ra[r].set(TMP(rf))
                        # one could iterate this logic with
                        # the newly freed rt, but in this case
                        # the above loop must be changed
    idup(b, D.curi, NIns - D.curi)


def _carve_key(b):
    """qsort() comparison function to peel
    loop nests from inside out (a total order)"""
    # todo, evaluate if this order is really
    # better than the simple postorder
    return (-b.loop, -b.id)


def prio2(t1, t2):
    """comparison function to order temporaries
    for allocation at the end of blocks"""
    if (tmp[t1].visit ^ tmp[t2].visit) < 0:  # != signs
        return +1 if tmp[t1].visit != -1 else -1
    if (hint(t1) ^ hint(t2)) < 0:
        return +1 if hint(t1) != -1 else -1
    return s32(tmp[t1].cost - tmp[t2].cost)


def rega(fn):
    """register allocation
    depends on rpo, phi, cost, (and obviously spill)"""
    global stmov, stblk, regu, tmp, mem, loop, npm
    T = D.T
    insb = D.insb

    # 1. setup
    stmov = 0
    stblk = 0
    regu = 0
    tmp = fn.tmp
    mem = fn.mem
    end = [RMap() for _ in range(fn.nblk)]
    beg = [RMap() for _ in range(fn.nblk)]
    for n in range(fn.nblk):
        bsinit(end[n].b, fn.ntmp)
        bsinit(beg[n].b, fn.ntmp)
    cur = RMap()
    bsinit(cur.b, fn.ntmp)

    loop = INT_MAX
    for t in range(fn.ntmp):
        tmp[t].hint_r = t if t < Tmp0 else -1
        tmp[t].hint_w = loop
        tmp[t].visit = -1
    blk = []
    b = fn.start
    while b:
        blk.append(b)
        b = b.link
    blk.sort(key=_carve_key)
    b = fn.start
    for i in b.ins[:b.nins]:
        if i.op != Ocopy or not isreg(i.arg[0]):
            break
        else:
            assert rtype(i.to) == RTmp
            sethint(rval(i.to), rval(i.arg[0]))

    # 2. assign registers
    for b in blk:
        n = b.id
        loop = b.loop
        cur.n = 0
        bszero(cur.b)
        cur.w[:] = [0] * Tmp0
        rl = []
        for t in bselems(b.out, Tmp0):
            j = len(rl)
            rl.append(t)
            while j > 0 and prio2(t, rl[j - 1]) > 0:
                j -= 1
                rl[j + 1] = rl[j]
                rl[j] = t
        x = len(rl)
        for r in bselems(b.out):
            if r >= Tmp0:
                break
            radd(cur, r, r)
        for j in range(x):
            ralloctry(cur, rl[j], 1)
        for j in range(x):
            ralloc(cur, rl[j])
        rcopy(end[n], cur)
        doblk(b, cur)
        bscopy(b.in_, cur.b)
        p = b.phi
        while p:
            if rtype(p.to) == RTmp:
                bsclr(b.in_, rval(p.to))
            p = p.link
        rcopy(beg[n], cur)

    # 3. emit copies shared by multiple edges
    # to the same block
    s = fn.start
    while s:
        if s.npred <= 1:
            s = s.link
            continue
        m = beg[s.id]

        # rl maps a register that is live at the
        # beginning of s to the one used in all
        # predecessors (if any, -1 otherwise)
        rl = [0] * Tmp0

        # to find the register of a phi in a
        # predecessor, we have to find the
        # corresponding argument
        p = s.phi
        while p:
            if rtype(p.to) != RTmp:
                p = p.link
                continue
            r = rfind(m, rval(p.to))
            if r == -1:
                p = p.link
                continue
            for u in range(p.narg):
                b = p.blk[u]
                src = p.arg[u]
                if rtype(src) != RTmp:
                    continue
                x = rfind(end[b.id], rval(src))
                if x == -1:  # spilled
                    continue
                rl[r] = x if (not rl[r] or rl[r] == x) else -1
            if rl[r] == 0:
                rl[r] = -1
            p = p.link

        # process non-phis temporaries
        for j in range(m.n):
            t = m.t[j]
            r = m.r[j]
            if rl[r] or t < Tmp0:  # todo, remove this
                continue
            for bp in s.pred[:s.npred]:
                x = rfind(end[bp.id], t)
                if x == -1:  # spilled
                    continue
                rl[r] = x if (not rl[r] or rl[r] == x) else -1
            if rl[r] == 0:
                rl[r] = -1

        npm = 0
        for j in range(m.n):
            t = m.t[j]
            r = m.r[j]
            x = rl[r]
            assert x != 0 or t < Tmp0  # todo, ditto
            if x > 0 and not bshas(m.b, x):
                pmadd(TMP(x), TMP(r), tmp[t].cls)
                m.r[j] = x
                bsset(m.b, x)
        D.curi = NIns
        pmgen()
        j = NIns - D.curi
        if j == 0:
            s = s.link
            continue
        stmov = (stmov + j) & U32
        s.nins += j
        s.ins = ([insb[k].copy() for k in range(D.curi, D.curi + j)]
                 + [s.ins[k].copy() for k in range(s.nins - j)])
        s = s.link

    if D.debug['R']:
        err = sys.stderr
        err.write("\n> Register mappings:\n")
        for n in range(fn.nblk):
            b = fn.rpo[n]
            err.write("\t%-10s beg" % b.name)
            mdump(beg[n])
            err.write("\t           end")
            mdump(end[n])
        err.write("\n")

    # 4. emit remaining copies in new blocks
    blist = None
    b = fn.start
    while True:
        for ps in ('s1', 's2'):
            s = getattr(b, ps)
            if not s:
                break
            npm = 0
            p = s.phi
            while p:
                dst = p.to
                assert rtype(dst) == RSlot or rtype(dst) == RTmp
                if rtype(dst) == RTmp:
                    r = rfind(beg[s.id], rval(dst))
                    if r == -1:
                        p = p.link
                        continue
                    dst = TMP(r)
                u = 0
                while p.blk[u] is not b:
                    assert u + 1 < p.narg
                    u += 1
                src = p.arg[u]
                if rtype(src) == RTmp:
                    src = rref(end[b.id], rval(src))
                pmadd(src, dst, p.cls)
                p = p.link
            for t in bselems(s.in_, Tmp0):
                src = rref(end[b.id], t)
                dst = rref(beg[s.id], t)
                pmadd(src, dst, tmp[t].cls)
            D.curi = NIns
            pmgen()
            if D.curi == NIns:
                continue
            b1 = newblk()
            b1.loop = (b.loop + s.loop) // 2
            b1.link = blist
            blist = b1
            fn.nblk += 1
            b1.name = strf("%s_%s", b.name, s.name)
            stmov = (stmov + NIns - D.curi) & U32
            stblk = (stblk + 1) & U32
            idup(b1, D.curi, NIns - D.curi)
            b1.jmp.type = Jjmp
            b1.s1 = s
            setattr(b, ps, b1)
        if not b.link:
            b.link = blist
            break
        b = b.link
    b = fn.start
    while b:
        b.phi = None
        b = b.link
    fn.reg = regu

    if D.debug['R']:
        err = sys.stderr
        err.write("\n> Register allocation statistics:\n")
        err.write("\tnew moves:  %d\n" % s32(stmov))
        err.write("\tnew blocks: %d\n" % s32(stblk))
        err.write("\n> After register allocation:\n")
        printfn(fn, err)
