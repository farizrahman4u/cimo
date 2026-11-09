"""Port of QBE amd64/sysv.c."""

import sys

from .. import defs as D
from ..defs import (R, RTmp, RType, NIns, Ins, Phi, Jmp_, Kx, Kw, Kl, Kd, KBASE, BIT, TMP, INT,
                    CALL, SLOT, CON_Z, rtype, rval, FEnd, FPad, Fs, Fd, Fb, Fh, Fw, Fl, FTyp,
                    Ciult, Ocmpw, Ostorew, Ostorel, Oadd, Oload, Oloadsw, Ocopy, Ocall,
                    Oblit0, Oblit1, Oalloc, Osalloc, Opar, Oarg, Oargc, Oarge, Oargv, Oparc,
                    Opare, Ovastart, Ovaarg, Jret0, Jretc, Jretw, Jjmp, Jjnz, isret, isarg,
                    ispar)
from ..util import err, die, newtmp, getcon, emit, emiti, idup, icpy, vgrow
from ..cfg import newblk
from ..parse import printfn
from .regs import (RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, RBX, R12, R13, R14, R15, RBP,
                   XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11,
                   XMM12, XMM13, XMM14, NGPS_SYSV, NCLR_SYSV, NFPS)


class AClass:
    __slots__ = ('type', 'inmem', 'align', 'size', 'cls', 'ref')

    def __init__(self):  # QBE's alloc() zeroes
        self.type = None
        self.inmem = 0
        self.align = 0
        self.size = 0
        self.cls = [0, 0]
        self.ref = [R, R]


class RAlloc:
    __slots__ = ('i', 'link')

    def __init__(self, i, link):
        self.i = i
        self.link = link


def classify(a, t, s):
    s1 = s
    for n in range(t.nunion):
        for f in t.fields[n]:
            if f.type == FEnd:
                break
            assert s <= 16
            ci = s // 8
            if f.type == FPad:
                # don't change anything
                s += f.len
            elif f.type in (Fs, Fd):
                if a.cls[ci] == Kx:
                    a.cls[ci] = Kd
                s += f.len
            elif f.type in (Fb, Fh, Fw, Fl):
                a.cls[ci] = Kl
                s += f.len
            elif f.type == FTyp:
                classify(a, D.typ[f.len], s)
                s += D.typ[f.len].size
            else:
                die("unreachable")
        s = s1


def typclass(a, t, novec=0):
    """novec: a vector type as the composite of its lanes (variadic arguments: va_arg reads them so)."""
    if t.isvec and not novec and t.size in (8, 16):
        # cimo: a GNU vector of 8 or 16 bytes in one xmm (SSE, SSEUP), as gcc passes it
        a.type = t
        a.size = t.size
        a.align = t.align
        a.inmem = 0
        a.cls[0] = Kd if t.size == 8 else D.Kq
        a.cls[1] = Kx
        return
    sz = t.size
    al = 1 << t.align

    # the ABI requires sizes to be rounded
    # up to the nearest multiple of 8, moreover
    # it makes it easy load and store structures
    # in registers
    if al < 8:
        al = 8
    sz = (sz + al - 1) & -al & 0xffffffff

    a.type = t
    a.size = sz
    a.align = t.align

    if t.isdark or sz > 16 or sz == 0:
        # large or unaligned structures are
        # required to be passed in memory
        a.inmem = 1
        return

    a.cls[0] = Kx
    a.cls[1] = Kx
    a.inmem = 0
    classify(a, t, 0)


_retreg = [[RAX, RDX], [XMM0, XMM0 + 1]]


def _second(a):
    """Whether the second eightbyte of a (classified) composite is passed: it exists and is not
    all padding. Fix over C QBE: an eightbyte with no field (Kx, e.g. the second half of
    `struct { _Alignas(16) char c; }`) took a register of an undefined class (a negative shift
    in retr, a class assertion in rega); the ABI's NO_CLASS, it takes no register (as gcc)."""
    return a.size > 8 and a.cls[1] != Kx


def retr(reg, aret):
    """reg: a 2-element list, filled."""
    nr = [0, 0]
    ca = 0
    n = 0
    while n * 8 < aret.size:
        if aret.cls[n] != Kx:  # fix over C QBE: see _second
            k = 1 if aret.cls[n] == D.Kq else KBASE(aret.cls[n])
            reg[n] = TMP(_retreg[k][nr[k]])
            nr[k] += 1
            ca += 1 << (2 * k)
        n += 1
    return ca


def _load(k, to, addr, cls):
    """to = the eightbyte (or a vector's 16 bytes, class q) at addr; cls: the load's class otherwise."""
    if k == D.Kq:  # 16 bytes: with AVX2's 32-byte q, into both halves (qload16)
        from .. import ext as _ext
        emit(D.Oqload16 if _ext.qbytes() >= 32 else Oload, D.Kq, to, addr, R)
    else:
        emit(Oload, cls, to, addr, R)


def _store(k, v, addr):
    if k == D.Kq:
        from .. import ext as _ext
        emit(D.Oqstore16 if _ext.qbytes() >= 32 else D.Oqstore, 0, R, v, addr)
    else:
        emit(Ostorel, 0, R, v, addr)


def _bigcopy(size):
    from .. import ext as _ext
    return _ext.bigcopy is not None and size >= _ext.bigcopy


def _memcpycall(fn, dst, src, size):
    """cimo: memcpy(dst, src, size) as a lowered call (emitted backwards), where no argument register is set yet:
    an unrolled blit of a big struct passed or returned by value took minutes (gcc's pr23135: 128 KB, 8 calls)."""
    from ..arm64.abi import _memcpy
    fn.leaf = 0
    emit(Ocopy, Kl, R, TMP(RAX), R)  # spill and rega expect a copy after a call
    emit(Ocall, 0, R, _memcpy(fn), CALL(3 << 4 | 1))
    emit(Ocopy, Kl, TMP(RDX), getcon(size, fn), R)
    emit(Ocopy, Kl, TMP(RSI), src, R)
    emit(Ocopy, Kl, TMP(RDI), dst, R)


def selret(b, fn):
    j = b.jmp.type

    if not isret(j) or j == Jret0:
        return

    r0 = b.jmp.arg
    b.jmp.type = Jret0

    if j == Jretc:
        aret = AClass()
        typclass(aret, D.typ[fn.retty])
        if aret.inmem:
            assert rtype(fn.retr) == RTmp
            emit(Ocopy, Kl, TMP(RAX), fn.retr, R)
            if _bigcopy(aret.type.size):
                _memcpycall(fn, fn.retr, r0, aret.type.size)
            else:
                emit(Oblit1, 0, R, INT(aret.type.size), R)
                emit(Oblit0, 0, R, r0, fn.retr)
            ca = 1
        else:
            reg = [R, R]
            ca = retr(reg, aret)
            if _second(aret):
                r = newtmp("abi", Kl, fn)
                emit(Oload, Kl, reg[1], r, R)
                emit(Oadd, Kl, r, r0, getcon(8, fn))
            _load(aret.cls[0], reg[0], r0, Kl)
    else:
        k = j - Jretw
        if KBASE(k) == 0:
            emit(Ocopy, k, TMP(RAX), r0, R)
            ca = 1
        else:
            emit(Ocopy, k, TMP(XMM0), r0, R)
            ca = 1 << 2

    b.jmp.arg = CALL(ca)


def argsclass(ins, i0, i1, ac, op, aret, env):
    """ins[i0:i1] are the arguments; env: a one-element list (Ref *env)."""
    if aret and aret.inmem:
        nint = 5  # hidden argument
    else:
        nint = 6
    nsse = 8
    varc = 0
    envc = 0
    for k in range(i0, i1):
        i = ins[k]
        a = ac[k - i0]
        o = i.op - op + Oarg
        if o == Oarg:
            if KBASE(i.cls) == 0:
                if nint > 0:
                    nint -= 1
                    a.inmem = 0
                else:
                    a.inmem = 2
            else:
                if nsse > 0:
                    nsse -= 1
                    a.inmem = 0
                else:
                    a.inmem = 2
            a.align = 3
            a.size = 8
            a.cls[0] = i.cls
        elif o == Oargc:
            n = rval(i.arg[0])
            typclass(a, D.typ[n], varc)
            if a.inmem:
                continue
            ni = ns = 0
            n = 0
            while n * 8 < a.size:
                if a.cls[n] == Kx:  # fix over C QBE: see _second
                    pass
                elif KBASE(a.cls[n]) == 0:
                    ni += 1
                else:
                    ns += 1
                n += 1
            if nint >= ni and nsse >= ns:
                nint -= ni
                nsse -= ns
            else:
                a.inmem = 1
        elif o == Oarge:
            envc = 1
            if op == Opar:
                env[0] = i.to
            else:
                env[0] = i.arg[0]
        elif o == Oargv:
            varc = 1
        else:
            die("unreachable")

    if varc and envc:
        err("sysv abi does not support variadic env calls")

    return ((varc | envc) << 12) | ((6 - nint) << 4) | ((8 - nsse) << 8)


amd64_sysv_rsave = [
    RDI, RSI, RDX, RCX, R8, R9, R10, R11, RAX,
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, -1
]
amd64_sysv_rclob = [RBX, R12, R13, R14, R15, -1]

assert (len(amd64_sysv_rsave) == NGPS_SYSV + NFPS + 1
        and len(amd64_sysv_rclob) == NCLR_SYSV + 1)

# layout of call's second argument (RCall)
#
#  29     12    8    4  3  0
#  |0...00|x|xxxx|xxxx|xx|xx|                  range
#          |    |    |  |  ` gp regs returned (0..2)
#          |    |    |  ` sse regs returned   (0..2)
#          |    |    ` gp regs passed         (0..6)
#          |    ` sse regs passed             (0..8)
#          ` 1 if rax is used to pass data    (0..1)


def amd64_sysv_retregs(r, p):
    assert rtype(r) == D.RCall
    b = 0
    ni = rval(r) & 3
    nf = (rval(r) >> 2) & 3
    if ni >= 1:
        b |= BIT(RAX)
    if ni >= 2:
        b |= BIT(RDX)
    if nf >= 1:
        b |= BIT(XMM0)
    if nf >= 2:
        b |= BIT(XMM1)
    if p is not None:
        p[0] = ni
        p[1] = nf
    return b


def amd64_sysv_argregs(r, p):
    assert rtype(r) == D.RCall
    b = 0
    ni = (rval(r) >> 4) & 15
    nf = (rval(r) >> 8) & 15
    ra = (rval(r) >> 12) & 1
    for j in range(ni):
        b |= BIT(amd64_sysv_rsave[j])
    for j in range(nf):
        b |= BIT(XMM0 + j)
    if p is not None:
        p[0] = ni + ra
        p[1] = nf
    return b | (BIT(RAX) if ra else 0)


def rarg(ty, n):
    """n: [ni, ns], updated."""
    if KBASE(ty) == 0:
        r = TMP(amd64_sysv_rsave[n[0]])
        n[0] += 1
    else:
        r = TMP(XMM0 + n[1])
        n[1] += 1
    return r


def selcall(fn, ins, i0, i1, rap):
    """ins[i0:i1] are the arguments of the call ins[i1]; rap: a one-element list
    (RAlloc **rap)."""
    env = [R]
    ac = [AClass() for _ in range(i1 - i0)]
    ic = ins[i1]

    aret = None
    if ic.arg[1] != R:
        assert rtype(ic.arg[1]) == RType
        aret = AClass()
        typclass(aret, D.typ[rval(ic.arg[1])])
        ca = argsclass(ins, i0, i1, ac, Oarg, aret, env)
    else:
        ca = argsclass(ins, i0, i1, ac, Oarg, None, env)

    stk = 0
    for a in reversed(ac):
        if a.inmem:
            if a.align > 4:
                err("sysv abi requires alignments of 16 or less")
            stk += a.size
            if a.align == 4:
                stk += stk & 15
    stk += stk & 15
    if stk:
        r = getcon(-stk, fn)
        emit(Osalloc, Kl, R, r, R)

    if ic.arg[1] != R:
        if aret.inmem:
            # get the return location from eax
            # it saves one callee-save reg
            r1 = newtmp("abi", Kl, fn)
            emit(Ocopy, Kl, ic.to, TMP(RAX), R)
            ca += 1
        else:
            # todo, may read out of bounds.
            # gcc did this up until 5.2, but
            # this should still be fixed.
            if _second(aret):
                r = newtmp("abi", Kl, fn)
                aret.ref[1] = newtmp("abi", aret.cls[1], fn)
                emit(Ostorel, 0, R, aret.ref[1], r)
                emit(Oadd, Kl, r, ic.to, getcon(8, fn))
            aret.ref[0] = newtmp("abi", aret.cls[0], fn)
            _store(aret.cls[0], aret.ref[0], ic.to)
            reg = [R, R]
            ca += retr(reg, aret)
            if _second(aret):
                emit(Ocopy, aret.cls[1], aret.ref[1], reg[1], R)
            emit(Ocopy, aret.cls[0], aret.ref[0], reg[0], R)
            r1 = ic.to
        # allocate return pad
        # specific to NAlign == 3
        al = aret.align - 2 if aret.align >= 2 else 0
        ra = RAlloc(Ins(Oalloc + al, Kl, r1, getcon(aret.size, fn)), rap[0])
        rap[0] = ra
    else:
        ra = None
        if KBASE(ic.cls) == 0:
            emit(Ocopy, ic.cls, ic.to, TMP(RAX), R)
            ca += 1
        else:
            emit(Ocopy, ic.cls, ic.to, TMP(XMM0), R)
            ca += 1 << 2

    emit(Ocall, ic.cls, R, ic.arg[0], CALL(ca))

    if env[0] != R:
        emit(Ocopy, Kl, TMP(RAX), env[0], R)
    elif (ca >> 12) & 1:  # vararg call
        emit(Ocopy, Kw, TMP(RAX), getcon((ca >> 8) & 15, fn), R)

    n = [0, 0]
    if ra and aret.inmem:
        emit(Ocopy, Kl, rarg(Kl, n), ra.i.to, R)  # pass hidden argument

    for k in range(i0, i1):
        i = ins[k]
        a = ac[k - i0]
        if i.op >= Oarge or a.inmem:
            continue
        r1 = rarg(a.cls[0], n)
        if i.op == Oargc:
            if _second(a):
                r2 = rarg(a.cls[1], n)
                r = newtmp("abi", Kl, fn)
                emit(Oload, a.cls[1], r2, r, R)
                emit(Oadd, Kl, r, i.arg[1], getcon(8, fn))
            _load(a.cls[0], r1, i.arg[1], a.cls[0])
        else:
            emit(Ocopy, i.cls, r1, i.arg[0], R)

    if not stk:
        return

    r = newtmp("abi", Kl, fn)
    off = 0
    for k in range(i0, i1):
        i = ins[k]
        a = ac[k - i0]
        if i.op >= Oarge or not a.inmem:
            continue
        r1 = newtmp("abi", Kl, fn)
        if i.op == Oargc:
            if a.align == 4:
                off += off & 15
            if _bigcopy(a.type.size):  # no argument register is set yet
                _memcpycall(fn, r1, i.arg[1], a.type.size)
            else:
                emit(Oblit1, 0, R, INT(a.type.size), R)
                emit(Oblit0, 0, R, i.arg[1], r1)
        else:
            emit(Ostorel, 0, R, i.arg[0], r1)
        emit(Oadd, Kl, r1, r, getcon(off, fn))
        off += a.size
    emit(Osalloc, Kl, r, getcon(stk, fn), R)


def selpar(fn, i0, i1):
    ins = fn.start.ins
    env = [R]
    ac = [AClass() for _ in range(i1 - i0)]
    D.curi = NIns
    n = [0, 0]

    aret = None
    if fn.retty >= 0:
        aret = AClass()
        typclass(aret, D.typ[fn.retty])
        fa = argsclass(ins, i0, i1, ac, Opar, aret, env)
    else:
        fa = argsclass(ins, i0, i1, ac, Opar, None, env)
    fn.reg = amd64_sysv_argregs(CALL(fa), None)

    for k in range(i0, i1):
        i = ins[k]
        a = ac[k - i0]
        if i.op != Oparc or a.inmem:
            continue
        if _second(a):
            r = newtmp("abi", Kl, fn)
            a.ref[1] = newtmp("abi", Kl, fn)
            emit(Ostorel, 0, R, a.ref[1], r)
            emit(Oadd, Kl, r, i.to, getcon(8, fn))
        a.ref[0] = newtmp("abi", D.Kq if a.cls[0] == D.Kq else Kl, fn)
        _store(a.cls[0], a.ref[0], i.to)
        # specific to NAlign == 3
        al = a.align - 2 if a.align >= 2 else 0
        emit(Oalloc + al, Kl, i.to, getcon(a.size, fn), R)

    if fn.retty >= 0 and aret.inmem:
        r = newtmp("abi", Kl, fn)
        emit(Ocopy, Kl, r, rarg(Kl, n), R)
        fn.retr = r

    s = 4
    for k in range(i0, i1):
        i = ins[k]
        a = ac[k - i0]
        if a.inmem == 1:
            if a.align > 4:
                err("sysv abi requires alignments of 16 or less")
            if a.align == 4:
                s = (s + 3) & -4
            fn.tmp[rval(i.to)].slot = -s
            s += a.size // 4
            continue
        elif a.inmem == 2:
            emit(Oload, i.cls, i.to, SLOT(-s), R)
            s += 2
            continue
        if i.op == Opare:
            continue
        r = rarg(a.cls[0], n)
        if i.op == Oparc:
            emit(Ocopy, a.cls[0], a.ref[0], r, R)
            if _second(a):
                r = rarg(a.cls[1], n)
                emit(Ocopy, a.cls[1], a.ref[1], r, R)
        else:
            emit(Ocopy, i.cls, i.to, r, R)

    if env[0] != R:
        emit(Ocopy, Kl, env[0], TMP(RAX), R)

    return fa | (s * 4) << 12


def split(fn, b):
    fn.nblk += 1
    bn = newblk()
    idup(bn, D.curi, NIns - D.curi)
    D.curi = NIns
    b.visit += 1
    bn.visit = b.visit
    bn.name = "%s.%d" % (b.name, b.visit)
    bn.loop = b.loop
    bn.link = b.link
    b.link = bn
    return bn


def chpred(b, bp, bp1):
    p = b.phi
    while p:
        a = 0
        while p.blk[a] is not bp:
            assert a + 1 < p.narg
            a += 1
        p.blk[a] = bp1
        p = p.link


def selvaarg(fn, b, i):
    c4 = getcon(4, fn)
    c8 = getcon(8, fn)
    c16 = getcon(16, fn)
    ap = i.arg[0]
    isint = KBASE(i.cls) == 0

    # @b [...]
    #     r0 =l add ap, (0 or 4)
    #     nr =l loadsw r0
    #     r1 =w cultw nr, (48 or 176)
    #     jnz r1, @breg, @bstk
    # @breg
    #     r0 =l add ap, 16
    #     r1 =l loadl r0
    #     lreg =l add r1, nr
    #     r0 =w add nr, (8 or 16)
    #     r1 =l add ap, (0 or 4)
    #     storew r0, r1
    # @bstk
    #     r0 =l add ap, 8
    #     lstk =l loadl r0
    #     r1 =l add lstk, 8
    #     storel r1, r0
    # @b0
    #     %loc =l phi @breg %lreg, @bstk %lstk
    #     i->to =(i->cls) load %loc

    loc = newtmp("abi", Kl, fn)
    emit(Oload, i.cls, i.to, loc, R)
    b0 = split(fn, b)
    b0.jmp = Jmp_()
    b0.jmp.type = b.jmp.type
    b0.jmp.arg = b.jmp.arg
    b0.s1 = b.s1
    b0.s2 = b.s2
    if b.s1:
        chpred(b.s1, b, b0)
    if b.s2 and b.s2 is not b.s1:
        chpred(b.s2, b, b0)

    lreg = newtmp("abi", Kl, fn)
    nr = newtmp("abi", Kl, fn)
    r0 = newtmp("abi", Kw, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Ostorew, Kw, R, r0, r1)
    emit(Oadd, Kl, r1, ap, CON_Z if isint else c4)
    emit(Oadd, Kw, r0, nr, c8 if isint else c16)
    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Oadd, Kl, lreg, r1, nr)
    emit(Oload, Kl, r1, r0, R)
    emit(Oadd, Kl, r0, ap, c16)
    breg = split(fn, b)
    breg.jmp.type = Jjmp
    breg.s1 = b0

    lstk = newtmp("abi", Kl, fn)
    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r1, r0)
    emit(Oadd, Kl, r1, lstk, c8)
    emit(Oload, Kl, lstk, r0, R)
    emit(Oadd, Kl, r0, ap, c8)
    bstk = split(fn, b)
    bstk.jmp.type = Jjmp
    bstk.s1 = b0

    p = Phi()
    p.cls = Kl
    p.to = loc
    p.narg = 2
    p.blk = [bstk, breg]
    p.arg = [lstk, lreg]
    b0.phi = p
    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kw, fn)
    b.jmp.type = Jjnz
    b.jmp.arg = r1
    b.s1 = breg
    b.s2 = bstk
    c = getcon(48 if isint else 176, fn)
    emit(Ocmpw + Ciult, Kw, r1, nr, c)
    emit(Oloadsw, Kl, nr, r0, R)
    emit(Oadd, Kl, r0, ap, CON_Z if isint else c4)


def selvastart(fn, fa, ap):
    gp = ((fa >> 4) & 15) * 8
    fp = 48 + ((fa >> 8) & 15) * 16
    sp = fa >> 12
    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r1, r0)
    emit(Oadd, Kl, r1, TMP(RBP), getcon(-176, fn))
    emit(Oadd, Kl, r0, ap, getcon(16, fn))
    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r1, r0)
    emit(Oadd, Kl, r1, TMP(RBP), getcon(sp, fn))
    emit(Oadd, Kl, r0, ap, getcon(8, fn))
    r0 = newtmp("abi", Kl, fn)
    emit(Ostorew, Kw, R, getcon(fp, fn), r0)
    emit(Oadd, Kl, r0, ap, getcon(4, fn))
    emit(Ostorew, Kw, R, getcon(gp, fn), ap)


def amd64_sysv_abi(fn):
    b = fn.start
    while b:
        b.visit = 0
        b = b.link

    # lower parameters
    b = fn.start
    i = 0
    while i < b.nins:
        if not ispar(b.ins[i].op):
            break
        i += 1
    fa = selpar(fn, 0, i)
    n0 = NIns - D.curi
    ioff = i
    n1 = b.nins - ioff
    vgrow(b.ins, n0 + n1)
    icpy(b.ins, n0, b.ins, ioff, n1)
    icpy(b.ins, 0, D.insb, D.curi, n0)
    b.nins = n0 + n1

    # lower calls, returns, and vararg instructions
    ral = [None]
    b = fn.start
    while True:
        b = b.link
        if not b:
            b = fn.start  # do it last
        if not b.visit:
            D.curi = NIns
            selret(b, fn)
            ins = b.ins
            i = b.nins
            while i != 0:
                i -= 1
                op = ins[i].op
                if op == Ocall:
                    i0 = i
                    while i0 > 0:
                        if not isarg(ins[i0 - 1].op):
                            break
                        i0 -= 1
                    selcall(fn, ins, i0, i, ral)
                    i = i0
                elif op == Ovastart:
                    selvastart(fn, fa, ins[i].arg[0])
                elif op == Ovaarg:
                    selvaarg(fn, b, ins[i])
                elif op == Oarg or op == Oargc:
                    die("unreachable")
                else:
                    emiti(ins[i])
            if b is fn.start:
                ra = ral[0]
                while ra:
                    emiti(ra.i)
                    ra = ra.link
                ral[0] = None
            idup(b, D.curi, NIns - D.curi)
        if b is fn.start:
            break

    if D.debug['A']:
        sys.stderr.write("\n> After ABI lowering:\n")
        printfn(fn, sys.stderr)
