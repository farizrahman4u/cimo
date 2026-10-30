"""Port of QBE arm64/isel.c (with the local fcon/foldaddr/immediate patch)."""

import sys

from .. import defs as D
from ..defs import (R, RTmp, RCon, NIns, M64, Ins, Con, Mem, CBits, CAddr, SExt, SThr,
                    Kw, Kl, KWIDE, KBASE, TMP, CON, MEM, CALL, SLOT, CON_Z, rtype, rval,
                    s64, s32, INRANGE, Oalloc, Oalloc1, Oflag, Oafcmp, Oacmp, Oacmn,
                    Ocall, Onop, Oadd, Osub, Oand, Oor, Oxor, Omul, Oshl, Ocopy, Oload,
                    Oaddr, Oloadsb, Oloadub, Oloadsh, Oloaduh, Oloadsw, Oloaduw, Ostoreb,
                    Ostoreh, Ostorew, Ostores, Ostorel, Ostored, Jret0, Jjmp, Jhlt, Jjnz,
                    Jjf, Jjfine, isload, isstore, Oneg, Ofma0, Ofma1, Oafmadd, Oafmsub,
                    Oafnmadd, Oafnmsub)
from ..util import (err, newtmp, newcon, getcon, chuse, emit, emiti, idup, intern, iscmp,
                    cmpop, salloc, argcls, vgrow)
from ..parse import printfn
from .regs import R0
from .. import ext as _ext

(Iother, Iplo12, Iphi12, Iplo24, Inlo12, Inhi12, Inlo24) = range(7)


def _stashbits(n, size):
    from ..emit import stashbits
    return stashbits(n, size)


def imm(c, k):
    """Returns (kind, n) where n is *pn (None when not set)."""
    if c.type != CBits:
        return Iother, None
    n = c.i
    if k == Kw:
        n = s32(n)
    i = Iplo12
    if n < 0:
        i = Inlo12
        n = s64(-n)
    pn = n
    if (n & 0x000fff) == n:
        return i, pn
    if (n & 0xfff000) == n:
        return i + 1, pn
    if (n & 0xffffff) == n:
        return i + 2, pn
    return Iother, pn


def arm64_logimm(x, k):
    x &= M64
    if k == Kw:
        x = ((x & 0xffffffff) | x << 32) & M64
    if x & 1:
        x = ~x & M64
    if x == 0:
        return 0
    if x == 0xaaaaaaaaaaaaaaaa:
        return 1
    while True:
        n = x & 0xf
        if (0x1111111111111111 * n) & M64 == x:
            break
        n = x & 0xff
        if (0x0101010101010101 * n) & M64 == x:
            break
        n = x & 0xffff
        if (0x0001000100010001 * n) & M64 == x:
            break
        n = x & 0xffffffff
        if (0x0000000100000001 * n) & M64 == x:
            break
        n = x
        break
    # Check:
    return int((n & ((n + (n & -n)) & M64)) == 0)


# floating-point constants are loaded once in the start block
# and reused, instead of being reloaded at every use
class FCon:
    __slots__ = ('bits', 'k', 'tmp')

    def __init__(self, bits, k, tmp):
        self.bits = bits
        self.k = k
        self.tmp = tmp


fcon = []
nfcon = 0


def fconref(bits, k, fn):
    global nfcon
    for n in range(nfcon):
        if fcon[n].bits == bits and fcon[n].k == k:
            return fcon[n].tmp
    n = nfcon
    nfcon += 1
    vgrow(fcon, nfcon)
    fcon[n] = FCon(bits, k, newtmp("fcon", k, fn))
    return fcon[n].tmp


def _newfpcon(s, fn):
    """vgrow(&fn->con, ++fn->ncon); c = ...; *c = (Con){.type = CAddr};
    c->sym.id = intern("\"<asloc>fp<s>\""); returns its index."""
    fn.ncon += 1
    vgrow(fn.con, fn.ncon, Con)
    c = Con(CAddr)
    fn.con[fn.ncon - 1] = c
    c.sym.id = intern("\"%sfp%d\"" % (D.T.asloc, s))
    return fn.ncon - 1


def emitfcon(fn):
    for n in range(nfcon):
        s = _stashbits(fcon[n].bits, 8 if KWIDE(fcon[n].k) else 4)
        ci = _newfpcon(s, fn)
        r = newtmp("isel", Kl, fn)
        emit(Oload, fcon[n].k, fcon[n].tmp, r, R)
        emit(Ocopy, Kl, r, CON(ci), R)


def fixarg(pa, pi, k, phi, fn):
    """fixarg(&pa[pi], ...)"""
    r0 = pa[pi]
    rt = rtype(r0)
    if rt == RCon:
        c = fn.con[rval(r0)]
        if c.type == CAddr and ((c.sym.type & SExt)
                                or (D.T.apple and (c.sym.type & SThr))):
            r1 = newtmp("isel", Kl, fn)
            pa[pi] = r1
            if c.i:
                r2 = newtmp("isel", Kl, fn)
                cc = Con(CBits, c.i)
                r3 = newcon(cc, fn)
                emit(Oadd, Kl, r1, r2, r3)
                r1 = r2
            if D.T.apple and (c.sym.type & SThr):
                emit(Ocopy, Kl, r1, TMP(R0), R)
                r1 = newtmp("isel", Kl, fn)
                r2 = newtmp("isel", Kl, fn)
                emit(Ocall, 0, R, r1, CALL(33))
                emit(Ocopy, Kl, TMP(R0), r2, R)
                emit(Oload, Kl, r1, r2, R)
                r1 = r2
            cc = c.copy()
            cc.i = 0
            r3 = newcon(cc, fn)
            emit(Ocopy, Kl, r1, r3, R)
            return
        if KBASE(k) == 0 and phi:
            return
        if KBASE(k) == 1 and c.type == CBits:
            pa[pi] = fconref(c.i, k, fn)
            return
        r1 = newtmp("isel", k, fn)
        if KBASE(k) == 0:
            emit(Ocopy, k, r1, r0, R)
        else:
            n = _stashbits(c.i, 8 if KWIDE(k) else 4)
            ci = _newfpcon(n, fn)
            r2 = newtmp("isel", Kl, fn)
            emit(Oload, k, r1, r2, R)
            emit(Ocopy, Kl, r2, CON(ci), R)
        pa[pi] = r1
    elif rt == RTmp:
        s = fn.tmp[rval(r0)].slot
        if s == -1:
            return
        r1 = newtmp("isel", Kl, fn)
        emit(Oaddr, Kl, r1, SLOT(s), R)
        pa[pi] = r1


def selcmp(arg, k, fn):
    """arg is a two-element list, modified in place (swap)."""
    if KBASE(k) == 1:
        emit(Oafcmp, k, R, arg[0], arg[1])
        iarg = D.insb[D.curi].arg
        fixarg(iarg, 0, k, 0, fn)
        fixarg(iarg, 1, k, 0, fn)
        return 0
    swap = rtype(arg[0]) == RCon
    if swap:
        r = arg[1]
        arg[1] = arg[0]
        arg[0] = r
    fix = 1
    cmp = Oacmp
    r = arg[1]
    if rtype(r) == RCon:
        c = fn.con[rval(r)]
        im, n = imm(c, k)
        if im in (Iplo12, Iphi12):
            fix = 0
        elif im in (Inlo12, Inhi12):
            cmp = Oacmn
            r = getcon(n, fn)
            fix = 0
    emit(cmp, k, R, arg[0], r)
    iarg = D.insb[D.curi].arg
    fixarg(iarg, 0, k, 0, fn)
    if fix:
        fixarg(iarg, 1, k, 0, fn)
    return int(swap)


# original instructions of the block being selected; used to find
# same-block definitions when folding address arithmetic
selblk = None
selpos = 0  # index in selblk.ins


def accsize(i):
    op = i.op
    if op in (Oloadsb, Oloadub, Ostoreb):
        return 1
    if op in (Oloadsh, Oloaduh, Ostoreh):
        return 2
    if op in (Oloadsw, Oloaduw, Ostorew, Ostores):
        return 4
    if op in (Ostorel, Ostored):
        return 8
    if op == Oload:
        return 8 if KWIDE(i.cls) else 4
    return 0


def foldaddr(pa, pi, sz, fn):
    """fold "%a = add %base, off" into [base, #off] for a load/store
    with off a valid scaled uimm12; the add is deleted once all its
    uses have been folded (blocks are selected bottom-up)"""
    pr = pa[pi]
    if rtype(pr) != RTmp or fn.tmp[rval(pr)].slot != -1:
        return
    ins = selblk.ins
    d = selpos
    while d > 0:
        d -= 1
        if ins[d].to == pr:
            break
    else:
        return
    # Found:
    di = ins[d]
    if (di.op != Oadd or di.cls != Kl
            or rtype(di.arg[0]) != RTmp or rtype(di.arg[1]) != RCon
            or fn.tmp[rval(di.arg[0])].slot != -1):
        return
    c = fn.con[rval(di.arg[1])]
    if c.type != CBits:
        return
    off = c.i
    if off < 0 or off % sz or off // sz > 4095:
        return
    m = Mem()
    m.base = di.arg[0]
    m.index = R
    m.offset.type = CBits
    m.offset.i = off
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = m
    chuse(di.arg[0], +1, fn)
    chuse(pr, -1, fn)
    if fn.tmp[rval(pr)].nuse == 0:
        chuse(di.arg[0], -1, fn)
        di.set(Ins(Onop))
    pa[pi] = MEM(fn.nmem - 1)


def log2p(n):
    if n <= 0 or (n & (n - 1)):
        return -1
    l = 0
    while n > 1:
        n >>= 1
        l += 1
    return l


def negof(r, k, fn):
    """cimo extension: if r is "neg" of class k defined earlier in the block being
    selected (before its fma0), return the negated operand and drop the use of r (the neg
    is deleted once unused, as foldaddr does); else None."""
    if rtype(r) != RTmp:
        return None
    ins = selblk.ins
    d = selpos - 1
    while d > 0:
        d -= 1
        if ins[d].to == r:
            break
    else:
        return None
    di = ins[d]
    if di.op != Oneg or di.cls != k:
        return None
    x = di.arg[0]
    chuse(x, +1, fn)
    chuse(r, -1, fn)
    if fn.tmp[rval(r)].nuse == 0:
        chuse(x, -1, fn)
        di.set(Ins(Onop))
    return x


# (negated product, negated addend) -> op
_FMAOP = {(0, 0): Oafmadd, (1, 0): Oafmsub, (0, 1): Oafnmsub, (1, 1): Oafnmadd}


def selfma(i, fn):
    """cimo extension: "fma0 c; r = fma1 a, b" (r = a*b + c) becomes one machine op,
    r = afmadd a, MEM{base=b, index=c}, with negations of a, b or c folded in
    (afmsub: c - a*b, afnmsub: a*b - c, afnmadd: -(a*b) - c). The fma0 is at selpos-1."""
    k = i.cls
    i0 = selblk.ins[selpos - 1]
    assert i0.op == Ofma0 and i0.cls == k
    a, b, c = i.arg[0], i.arg[1], i0.arg[0]
    neg = [0, 0]
    x = negof(a, k, fn)
    if x is not None:
        a = x
        neg[0] ^= 1
    x = negof(b, k, fn)
    if x is not None:
        b = x
        neg[0] ^= 1
    x = negof(c, k, fn)
    if x is not None:
        c = x
        neg[1] = 1
    m = Mem()
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = m
    emit(_FMAOP[tuple(neg)], k, i.to, a, MEM(fn.nmem - 1))
    iarg = D.insb[D.curi].arg
    fixarg(iarg, 0, k, 0, fn)
    bc = [b, c]
    fixarg(bc, 0, k, 0, fn)
    fixarg(bc, 1, k, 0, fn)
    m.base, m.index = bc


def callable(r, fn):
    if rtype(r) == RTmp:
        return 1
    if rtype(r) == RCon:
        c = fn.con[rval(r)]
        if c.type == CAddr:
            if c.i == 0:
                return 1
    return 0


def sel(i, fn):
    """i is a copy (C passes Ins by value)."""
    if INRANGE(i.op, Oalloc, Oalloc1):
        i0 = D.curi - 1
        salloc(i.to, i.arg[0], fn)
        fixarg(D.insb[i0].arg, 0, Kl, 0, fn)
        return
    cm = iscmp(i.op)
    if cm:
        ck, cc = cm
        emit(Oflag, i.cls, i.to, R, R)
        i0 = D.insb[D.curi]
        if selcmp(i.arg, ck, fn):
            i0.op += cmpop(cc)
        else:
            i0.op += cc
        return
    if i.op == Ocall:
        if callable(i.arg[0], fn):
            emiti(i)
            return
    if i.op == Onop:
        return
    if i.op == Ofma1:
        selfma(i, fn)
        return
    if i.op == Ofma0:
        return  # selected with its fma1
    immarg = False
    if KBASE(i.cls) == 0 and rtype(i.arg[1]) == RCon \
            and fn.con[rval(i.arg[1])].type == CBits:
        c = fn.con[rval(i.arg[1])]
        if i.op in (Oadd, Osub):
            im, n = imm(c, i.cls)
            if im in (Inlo12, Inhi12):
                i.op = Osub if i.op == Oadd else Oadd
                i.arg[1] = getcon(n, fn)
                immarg = True
            elif im in (Iplo12, Iphi12):
                immarg = True
        elif i.op in (Oand, Oor, Oxor):
            n = c.i
            if i.cls == Kl and ((n >> 24) or (n & ~0xfff) == 0) \
                    and arm64_logimm(n, Kl):
                immarg = True
        elif i.op == Omul:
            l = log2p(c.i)
            if l >= 0:
                i.op = Oshl
                i.arg[1] = getcon(l, fn)
                immarg = True
    if immarg:
        # Immarg:
        emiti(i)
        iarg = D.insb[D.curi].arg
        fixarg(iarg, 0, argcls(i, 0), 0, fn)
        return
    if isload(i.op):
        foldaddr(i.arg, 0, accsize(i), fn)
    if isstore(i.op):
        foldaddr(i.arg, 1, accsize(i), fn)
    emiti(i)
    iarg = D.insb[D.curi].arg  # fixarg() can change curi
    fixarg(iarg, 0, argcls(i, 0), 0, fn)
    fixarg(iarg, 1, argcls(i, 1), 0, fn)


def seljmp(b, fn):
    if b.jmp.type in (Jret0, Jjmp, Jhlt):
        return
    assert b.jmp.type == Jjnz
    r = b.jmp.arg
    use = -1
    b.jmp.arg = R
    ir = None
    i = b.nins
    while i > 0:
        i -= 1
        if b.ins[i].to == r:
            use = fn.tmp[rval(r)].nuse
            ir = b.ins[i]
            break
    cm = iscmp(ir.op) if ir is not None and use == 1 else None
    if cm:
        ck, cc = cm
        if selcmp(ir.arg, ck, fn):
            cc = cmpop(cc)
        b.jmp.type = Jjf + cc
        ir.set(Ins(Onop))
    else:
        selcmp([r, CON_Z], Kw, fn)
        b.jmp.type = Jjfine


def arm64_isel(fn):
    global nfcon, selblk, selpos

    # assign slots to fast allocs
    b = fn.start
    # specific to NAlign == 3; or change n=4 and sz /= 4 below
    al = Oalloc
    n = 4
    while al <= Oalloc1:
        for i in b.ins[:b.nins]:
            if i.op == al:
                if rtype(i.arg[0]) != RCon:
                    break
                sz = fn.con[rval(i.arg[0])].i
                if sz < 0 or sz >= 0x7fffffff - 15:
                    err("invalid alloc size %d", sz)
                sz = (sz + n - 1) & -n
                sz //= 4
                fn.tmp[rval(i.to)].slot = fn.slot
                fn.slot += sz
                i.set(Ins(Onop))
        al += 1
        n *= 2

    nfcon = 0
    b = fn.start.link
    while True:
        if not b:
            b = fn.start
        D.curi = NIns
        for sb in (b.s1, b.s2):
            if not sb:
                break
            p = sb.phi
            while p:
                n = 0
                while p.blk[n] is not b:
                    assert n + 1 < p.narg
                    n += 1
                fixarg(p.arg, n, p.cls, 1, fn)
                p = p.link
        seljmp(b, fn)
        selblk = b
        i = b.nins
        while i != 0:
            i -= 1
            selpos = i
            sel(b.ins[i].copy(), fn)
        if b is fn.start:
            emitfcon(fn)
        idup(b, D.curi, NIns - D.curi)
        if b is fn.start:
            break
        b = b.link

    if _ext.ldp:
        from .pair import schedule
        schedule(fn)

    if D.debug['I']:
        sys.stderr.write("\n> After instruction selection:\n")
        printfn(fn, sys.stderr)
