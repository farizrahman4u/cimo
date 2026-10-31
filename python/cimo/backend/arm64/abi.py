"""Port of QBE arm64/abi.c.

Extension (not in QBE): the Windows arm64 variant (T.windows). Non-variadic calls follow
AAPCS64 as on Linux. Calls to and definitions of variadic functions use Windows' rules for
*every* argument, named ones included: as if all were laid out on a stack whose first 64
bytes travel in x0-x7, so floats go in integer registers (as their bits, Cgp below) and HFAs
are ordinary composites. A variadic function saves x0-x7 right below its stack arguments,
and va_list is a `char *` walking that area and then the stack (like Apple's va_arg).
Not supported: a composite split between x7 and the stack (MSVC allows it).
"""

import sys

from .. import defs as D
from ..defs import (R, RTmp, RCall, NIns, U32, Ins, Phi, Jmp_, Kx, Kw, Kl, Ks, Kd,
                    KWIDE, KBASE, BIT, TMP, INT, CALL, SLOT, CON_Z, rtype, rval,
                    FEnd, Fs, Fd, FTyp, Cislt, Ocmpw, Ostorew, Ostorel, Ostores, Ostored,
                    Ostoreb, Ostoreh, Oadd, Osub, Oload, Oloadsb, Oloadsw, Ocopy, Ocall,
                    Oblit0, Oblit1, Oalloc, Oaddr, Oargsb, Oargub, Oargsh, Oarguh,
                    Oparsb, Oparub, Oparsh, Oparuh, Opar, Oarg, Oparc, Oargc, Opare,
                    Oarge, Oargv, Ovastart, Ovaarg, Oextsb, Jret0, Jretc, Jretw, Jretsb,
                    Jjmp, Jjnz, isret, isarg, ispar, isargbh, isparbh, isretbh)
from ..defs import Ocast
from ..util import (err, die, newtmp, getcon, emit, emiti, idup, icpy, vgrow)
from ..cfg import newblk
from ..parse import printfn
from .regs import SP, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, V0, V1, V2, V3, V4, V5, V6, V7

Cstk = 1  # pass on the stack
Cptr = 2  # replaced by a pointer
Cgp = 4  # Windows variadic: a float passed in an integer register (cls holds Kw/Kl)


class Class:
    __slots__ = ('class_', 'ishfa', 'hfa_base', 'hfa_size', 'size', 'align', 't', 'nreg',
                 'ngp', 'nfp', 'reg', 'cls')

    def __init__(self):
        self.class_ = 0
        self.ishfa = 0
        self.hfa_base = 0
        self.hfa_size = 0
        self.size = 0
        self.align = 0
        self.t = None
        self.nreg = 0
        self.ngp = 0
        self.nfp = 0
        self.reg = [0] * 4
        self.cls = [0] * 4


class Params:
    __slots__ = ('ngp', 'nfp', 'stk')

    def __init__(self, ngp=0, nfp=0, stk=0):
        self.ngp = ngp
        self.nfp = nfp
        self.stk = stk


# gp/fp "pointers" into these arrays are indices
gpreg = [R0, R1, R2, R3, R4, R5, R6, R7, 0, 0, 0, 0]
fpreg = [V0, V1, V2, V3, V4, V5, V6, V7, 0, 0, 0, 0]
store = {Kw: Ostorew, Kl: Ostorel, Ks: Ostores, Kd: Ostored}

# layout of call's second argument (RCall)
#
#         13
#  29   14 |    9    5   2  0
#  |0.00|x|x|xxxx|xxxx|xxx|xx|                  range
#        | |    |    |   |  ` gp regs returned (0..2)
#        | |    |    |   ` fp regs returned    (0..4)
#        | |    |    ` gp regs passed          (0..8)
#        | |     ` fp regs passed              (0..8)
#        | ` indirect result register x8 used  (0..1)
#        ` env pointer passed in x9            (0..1)


def isfloatv(t, cls):
    """cls is a one-element list (char *cls)."""
    for n in range(t.nunion):
        for f in t.fields[n]:
            if f.type == FEnd:
                break
            if f.type == Fs:
                if cls[0] == Kd:
                    return 0
                cls[0] = Ks
            elif f.type == Fd:
                if cls[0] == Ks:
                    return 0
                cls[0] = Kd
            elif f.type == FTyp and isfloatv(D.typ[f.len], cls):
                pass
            else:
                return 0
    return 1


def typclass(c, t, gp, fp, nohfa=0):
    """gp, fp are indices into gpreg, fpreg. nohfa: treat HFAs as other composites (Windows
    variadic calls)."""
    sz = (t.size + 7) & -8
    c.t = t
    c.class_ = 0
    c.ngp = 0
    c.nfp = 0
    c.align = 8

    if t.align > 3:
        err("alignments larger than 8 are not supported")

    c.size = sz & U32
    base = [Kx]
    c.ishfa = isfloatv(t, base)
    c.hfa_base = base[0]
    hfasz = t.size // (8 if KWIDE(c.hfa_base) else 4)
    c.ishfa &= int(not t.isdark and hfasz <= 4 and not nohfa)
    c.hfa_size = hfasz & 0xff

    if c.ishfa:
        n = 0
        while n < hfasz:
            c.reg[n] = fpreg[fp]
            fp += 1
            c.cls[n] = c.hfa_base
            n += 1
            c.nfp += 1
        c.nreg = n
    elif t.isdark or sz > 16 or sz == 0:
        # large structs are replaced by a
        # pointer to some caller-allocated
        # memory
        c.class_ |= Cptr
        c.size = 8
        c.ngp = 1
        c.reg[0] = gpreg[gp]
        c.cls[0] = Kl
    else:
        n = 0
        while n < sz // 8:
            c.reg[n] = gpreg[gp]
            gp += 1
            c.cls[n] = Kl
            n += 1
            c.ngp += 1
        c.nreg = n


def sttmps(tmp, ti, cls, nreg, mem, fn):
    """sttmps(&tmp[ti], ...)"""
    assert nreg <= 4
    off = 0
    for n in range(nreg):
        tmp[ti + n] = newtmp("abi", cls[n], fn)
        r = newtmp("abi", Kl, fn)
        emit(store[cls[n]], 0, R, tmp[ti + n], r)
        emit(Oadd, Kl, r, mem, getcon(off, fn))
        off += 8 if KWIDE(cls[n]) else 4


# todo, may read out of bounds
def ldregs(reg, cls, n, mem, fn):
    off = 0
    for i in range(n):
        r = newtmp("abi", Kl, fn)
        emit(Oload, cls[i], TMP(reg[i]), r, R)
        emit(Oadd, Kl, r, mem, getcon(off, fn))
        off += 8 if KWIDE(cls[i]) else 4


def selret(b, fn):
    j = b.jmp.type

    if not isret(j) or j == Jret0:
        return

    r = b.jmp.arg
    b.jmp.type = Jret0

    if j == Jretc:
        cr = Class()
        typclass(cr, D.typ[fn.retty], 0, 0)
        if cr.class_ & Cptr:
            assert rtype(fn.retr) == RTmp
            emit(Oblit1, 0, R, INT(cr.t.size), R)
            emit(Oblit0, 0, R, r, fn.retr)
            cty = 0
        else:
            ldregs(cr.reg, cr.cls, cr.nreg, r, fn)
            cty = (cr.nfp << 2) | cr.ngp
    else:
        k = j - Jretw
        if KBASE(k) == 0:
            emit(Ocopy, k, TMP(R0), r, R)
            cty = 1
        else:
            emit(Ocopy, k, TMP(V0), r, R)
            cty = 1 << 2

    b.jmp.arg = CALL(cty)


def argsclass(ins, i0, i1, carg, winva=0):
    """ins[i0:i1] classified into carg[0:i1-i0]. winva: Windows variadic rules."""
    va = 0
    envc = 0
    gp = 0
    fp = 0
    ngp = 8
    nfp = 8
    for k in range(i0, i1):
        i = ins[k]
        c = carg[k - i0]
        op = i.op
        if op in (Oargsb, Oargub, Oparsb, Oparub, Oargsh, Oarguh, Oparsh, Oparuh,
                  Opar, Oarg):
            if op in (Oargsb, Oargub, Oparsb, Oparub):
                c.size = 1
            elif op in (Oargsh, Oarguh, Oparsh, Oparuh):
                c.size = 2
            else:
                c.size = 8
                if D.T.apple and not KWIDE(i.cls):
                    c.size = 4
            # Scalar:
            c.align = c.size
            c.cls[0] = i.cls
            if va:
                c.class_ |= Cstk
                c.size = 8
                c.align = 8
                continue
            if winva:
                c.size = c.align = 8
                if ngp > 0:
                    ngp -= 1
                    c.reg[0] = gpreg[gp]
                    gp += 1
                    if KBASE(i.cls) == 1:
                        c.class_ |= Cgp
                        c.cls[0] = Kl if i.cls == Kd else Kw
                    continue
                c.class_ |= Cstk
                continue
            if KBASE(i.cls) == 0 and ngp > 0:
                ngp -= 1
                c.reg[0] = gpreg[gp]
                gp += 1
                continue
            if KBASE(i.cls) == 1 and nfp > 0:
                nfp -= 1
                c.reg[0] = fpreg[fp]
                fp += 1
                continue
            c.class_ |= Cstk
        elif op in (Oparc, Oargc):
            typclass(c, D.typ[rval(i.arg[0])], gp, fp, winva)
            if winva and 0 < ngp < c.ngp:
                err("a %d-byte struct split between x7 and the stack in a variadic call "
                    "is not supported on Windows" % c.t.size)
            if c.ngp <= ngp:
                if c.nfp <= nfp:
                    ngp -= c.ngp
                    nfp -= c.nfp
                    gp += c.ngp
                    fp += c.nfp
                    continue
                else:
                    nfp = 0
            else:
                ngp = 0
            c.class_ |= Cstk
        elif op in (Opare, Oarge):
            c.reg[0] = R9
            c.cls[0] = Kl
            envc = 1
        elif op == Oargv:
            va = int(D.T.apple != 0)
        else:
            die("unreachable")

    return envc << 14 | gp << 5 | fp << 9


def arm64_retregs(r, p):
    assert rtype(r) == RCall
    v = rval(r)
    ngp = v & 3
    nfp = (v >> 2) & 7
    if p is not None:
        p[0] = ngp
        p[1] = nfp
    b = 0
    while ngp:
        ngp -= 1
        b |= BIT(R0 + ngp)
    while nfp:
        nfp -= 1
        b |= BIT(V0 + nfp)
    return b


def arm64_argregs(r, p):
    assert rtype(r) == RCall
    v = rval(r)
    ngp = (v >> 5) & 15
    nfp = (v >> 9) & 15
    x8 = (v >> 13) & 1
    x9 = (v >> 14) & 1
    if p is not None:
        p[0] = ngp + x8 + x9
        p[1] = nfp
    b = 0
    while ngp:
        ngp -= 1
        b |= BIT(R0 + ngp)
    while nfp:
        nfp -= 1
        b |= BIT(V0 + nfp)
    return b | (x8 << R8) | (x9 << R9)


def stkblob(r, c, fn, il):
    """il is a list used as a stack (Insl *ilp): the C list head is il[-1]."""
    al = c.t.align - 2  # NAlign == 3
    if al < 0:
        al = 0
    sz = c.t.size if c.class_ & Cptr else c.size
    il.append(Ins(Oalloc + al, Kl, r, getcon(sz, fn), R))


def align(x, al):
    return (x + al - 1) & -al & U32


def _emitil(il):
    """for (; il; il=il->link) emiti(il->i);"""
    while il:
        emiti(il.pop())


def selcall(fn, ins, i0, i1, il):
    ca = [Class() for _ in range(i1 - i0)]
    winva = D.T.windows and any(ins[k].op == Oargv for k in range(i0, i1))
    cty = argsclass(ins, i0, i1, ca, winva)

    stk = 0
    for k in range(i0, i1):
        i = ins[k]
        c = ca[k - i0]
        if c.class_ & Cptr:
            i.arg[0] = newtmp("abi", Kl, fn)
            stkblob(i.arg[0], c, fn, il)
            i.op = Oarg
        if c.class_ & Cstk:
            stk = align(stk, c.align)
            stk += c.size
    stk = align(stk, 16)
    rstk = getcon(stk, fn)
    if stk:
        emit(Oadd, Kl, TMP(SP), TMP(SP), rstk)

    ic = ins[i1]
    if ic.arg[1] != R:
        cr = Class()
        typclass(cr, D.typ[rval(ic.arg[1])], 0, 0)
        stkblob(ic.to, cr, fn, il)
        cty |= (cr.nfp << 2) | cr.ngp
        if cr.class_ & Cptr:
            # spill & rega expect calls to be
            # followed by copies from regs,
            # so we emit a dummy
            cty |= 1 << 13 | 1
            emit(Ocopy, Kw, R, TMP(R0), R)
        else:
            tmp = [R] * 4
            sttmps(tmp, 0, cr.cls, cr.nreg, ic.to, fn)
            for n in range(cr.nreg):
                r = TMP(cr.reg[n])
                emit(Ocopy, cr.cls[n], tmp[n], r, R)
    else:
        if KBASE(ic.cls) == 0:
            emit(Ocopy, ic.cls, ic.to, TMP(R0), R)
            cty |= 1
        else:
            emit(Ocopy, ic.cls, ic.to, TMP(V0), R)
            cty |= 1 << 2

    emit(Ocall, 0, R, ic.arg[0], CALL(cty))

    if cty & (1 << 13):
        # struct return argument
        emit(Ocopy, Kl, TMP(R8), ic.to, R)

    casts = []
    for k in range(i0, i1):
        i = ins[k]
        c = ca[k - i0]
        if (c.class_ & Cstk) != 0:
            continue
        if c.class_ & Cgp:
            r = newtmp("abi", c.cls[0], fn)
            emit(Ocopy, c.cls[0], TMP(c.reg[0]), r, R)
            casts.append((c.cls[0], r, i.arg[0]))
        elif i.op == Oarg or i.op == Oarge or isargbh(i.op):
            emit(Ocopy, c.cls[0], TMP(c.reg[0]), i.arg[0], R)
        if i.op == Oargc:
            ldregs(c.reg, c.cls, c.nreg, i.arg[1], fn)
    for k, r, a in casts:  # before every copy into an argument register
        emit(Ocast, k, r, a, R)

    # populate the stack
    off = 0
    for k in range(i0, i1):
        i = ins[k]
        c = ca[k - i0]
        if (c.class_ & Cstk) == 0:
            continue
        off = align(off, c.align)
        r = newtmp("abi", Kl, fn)
        if i.op == Oarg or isargbh(i.op):
            if c.size == 1:
                op = Ostoreb
            elif c.size == 2:
                op = Ostoreh
            elif c.size in (4, 8):
                op = store[c.cls[0]]
            else:
                die("unreachable")
            emit(op, 0, R, i.arg[0], r)
        else:
            assert i.op == Oargc
            emit(Oblit1, 0, R, INT(c.size), R)
            emit(Oblit0, 0, R, i.arg[1], r)
        emit(Oadd, Kl, r, TMP(SP), getcon(off, fn))
        off += c.size
    if stk:
        emit(Osub, Kl, TMP(SP), TMP(SP), rstk)

    for k in range(i0, i1):
        i = ins[k]
        c = ca[k - i0]
        if c.class_ & Cptr:
            emit(Oblit1, 0, R, INT(c.t.size), R)
            emit(Oblit0, 0, R, i.arg[1], i.arg[0])


def selpar(fn, i0, i1):
    ins = fn.start.ins
    ca = [Class() for _ in range(i1 - i0)]
    D.curi = NIns

    winva = D.T.windows and fn.vararg
    cty = argsclass(ins, i0, i1, ca, winva)
    fn.reg = arm64_argregs(CALL(cty), None)

    il = []
    tmp = [R] * 16
    t = 0
    for k in range(i0, i1):
        i = ins[k]
        c = ca[k - i0]
        if i.op != Oparc or (c.class_ & (Cptr | Cstk)):
            continue
        sttmps(tmp, t, c.cls, c.nreg, i.to, fn)
        stkblob(i.to, c, fn, il)
        t += c.nreg
    _emitil(il)

    if fn.retty >= 0:
        cr = Class()
        typclass(cr, D.typ[fn.retty], 0, 0)
        if cr.class_ & Cptr:
            fn.retr = newtmp("abi", Kl, fn)
            emit(Ocopy, Kl, fn.retr, TMP(R8), R)
            fn.reg |= BIT(R8)

    casts = []
    for k in range(i0, i1):  # after every copy out of an argument register
        i = ins[k]
        c = ca[k - i0]
        if c.class_ & Cgp:
            r = newtmp("abi", c.cls[0], fn)
            emit(Ocast, i.cls, i.to, r, R)
            casts.append(r)
    t = 0
    off = 0
    for k in range(i0, i1):
        i = ins[k]
        c = ca[k - i0]
        if c.class_ & Cgp:
            emit(Ocopy, c.cls[0], casts.pop(0), TMP(c.reg[0]), R)
        elif i.op == Oparc and not (c.class_ & Cptr):
            if c.class_ & Cstk:
                off = align(off, c.align)
                fn.tmp[rval(i.to)].slot = -(off + 2)
                off += c.size
            else:
                for n in range(c.nreg):
                    r = TMP(c.reg[n])
                    emit(Ocopy, c.cls[n], tmp[t], r, R)
                    t += 1
        elif c.class_ & Cstk:
            off = align(off, c.align)
            if isparbh(i.op):
                op = Oloadsb + (i.op - Oparsb)
            else:
                op = Oload
            emit(op, c.cls[0], i.to, SLOT(-(off + 2)), R)
            off += c.size
        else:
            emit(Ocopy, c.cls[0], i.to, TMP(c.reg[0]), R)

    return Params(stk=align(off, 8), ngp=(cty >> 5) & 15, nfp=(cty >> 9) & 15)


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


def apple_selvaarg(fn, b, i):
    c8 = getcon(8, fn)
    ap = i.arg[0]
    stk8 = newtmp("abi", Kl, fn)
    stk = newtmp("abi", Kl, fn)

    emit(Ostorel, 0, R, stk8, ap)
    emit(Oadd, Kl, stk8, stk, c8)
    emit(Oload, i.cls, i.to, stk, R)
    emit(Oload, Kl, stk, ap, R)


def arm64_selvaarg(fn, b, i):
    c8 = getcon(8, fn)
    c16 = getcon(16, fn)
    c24 = getcon(24, fn)
    c28 = getcon(28, fn)
    ap = i.arg[0]
    isgp = KBASE(i.cls) == 0

    # @b [...]
    #     r0 =l add ap, (24 or 28)
    #     nr =l loadsw r0
    #     r1 =w csltw nr, 0
    #     jnz r1, @breg, @bstk
    # @breg
    #     r0 =l add ap, (8 or 16)
    #     r1 =l loadl r0
    #     lreg =l add r1, nr
    #     r0 =w add nr, (8 or 16)
    #     r1 =l add ap, (24 or 28)
    #     storew r0, r1
    # @bstk
    #     lstk =l loadl ap
    #     r0 =l add lstk, 8
    #     storel r0, ap
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
    emit(Oadd, Kl, r1, ap, c24 if isgp else c28)
    emit(Oadd, Kw, r0, nr, c8 if isgp else c16)
    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Oadd, Kl, lreg, r1, nr)
    emit(Oload, Kl, r1, r0, R)
    emit(Oadd, Kl, r0, ap, c8 if isgp else c16)
    breg = split(fn, b)
    breg.jmp.type = Jjmp
    breg.s1 = b0

    lstk = newtmp("abi", Kl, fn)
    r0 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r0, ap)
    emit(Oadd, Kl, r0, lstk, c8)
    emit(Oload, Kl, lstk, ap, R)
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
    emit(Ocmpw + Cislt, Kw, r1, nr, CON_Z)
    emit(Oloadsw, Kl, nr, r0, R)
    emit(Oadd, Kl, r0, ap, c24 if isgp else c28)


def apple_selvastart(fn, p, ap):
    off = getcon(p.stk, fn)
    stk = newtmp("abi", Kl, fn)
    arg = newtmp("abi", Kl, fn)

    emit(Ostorel, 0, R, arg, ap)
    emit(Oadd, Kl, arg, stk, off)
    emit(Oaddr, Kl, stk, SLOT(-1), R)


def win_selvastart(fn, p, ap):
    """Windows: x0-x7 are saved right below the stack arguments (SLOT(-1), see emit), so
    the first variadic argument is at 8 * (named gp registers) + (named stack bytes)."""
    off = getcon((8 * p.ngp + p.stk) & U32, fn)
    stk = newtmp("abi", Kl, fn)
    arg = newtmp("abi", Kl, fn)

    emit(Ostorel, 0, R, arg, ap)
    emit(Oadd, Kl, arg, stk, off)
    emit(Oaddr, Kl, stk, SLOT(-1), R)


def arm64_selvastart(fn, p, ap):
    rsave = newtmp("abi", Kl, fn)

    r0 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r0, ap)
    emit(Oadd, Kl, r0, rsave, getcon((p.stk + 192) & U32, fn))

    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r1, r0)
    emit(Oadd, Kl, r1, rsave, getcon(64, fn))
    emit(Oadd, Kl, r0, ap, getcon(8, fn))

    r0 = newtmp("abi", Kl, fn)
    r1 = newtmp("abi", Kl, fn)
    emit(Ostorel, Kw, R, r1, r0)
    emit(Oadd, Kl, r1, rsave, getcon(192, fn))
    emit(Oaddr, Kl, rsave, SLOT(-1), R)
    emit(Oadd, Kl, r0, ap, getcon(16, fn))

    # p.ngp, p.nfp are uint: the products wrap to 32 bits
    r0 = newtmp("abi", Kl, fn)
    emit(Ostorew, Kw, R, getcon(((p.ngp - 8) * 8) & U32, fn), r0)
    emit(Oadd, Kl, r0, ap, getcon(24, fn))

    r0 = newtmp("abi", Kl, fn)
    emit(Ostorew, Kw, R, getcon(((p.nfp - 8) * 16) & U32, fn), r0)
    emit(Oadd, Kl, r0, ap, getcon(28, fn))


def arm64_abi(fn):
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
    p = selpar(fn, 0, i)
    n0 = NIns - D.curi
    ioff = i
    n1 = b.nins - ioff
    vgrow(b.ins, n0 + n1)
    icpy(b.ins, n0, b.ins, ioff, n1)
    icpy(b.ins, 0, D.insb, D.curi, n0)
    b.nins = n0 + n1

    # lower calls, returns, and vararg instructions
    il = []
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
                    selcall(fn, ins, i0, i, il)
                    i = i0
                elif op == Ovastart:
                    if D.T.windows:
                        win_selvastart(fn, p, ins[i].arg[0])
                    elif D.T.apple:
                        apple_selvastart(fn, p, ins[i].arg[0])
                    else:
                        arm64_selvastart(fn, p, ins[i].arg[0])
                elif op == Ovaarg:
                    if D.T.apple or D.T.windows:
                        apple_selvaarg(fn, b, ins[i])
                    else:
                        arm64_selvaarg(fn, b, ins[i])
                elif op == Oarg or op == Oargc:
                    die("unreachable")
                else:
                    emiti(ins[i])
            if b is fn.start:
                _emitil(il)
            idup(b, D.curi, NIns - D.curi)
        if b is fn.start:
            break

    if D.debug['A']:
        sys.stderr.write("\n> After ABI lowering:\n")
        printfn(fn, sys.stderr)


def apple_extsb(fn):
    """abi0 for apple target; introduces
    necessary sign extensions in calls
    and returns"""
    b = fn.start
    while b:
        D.curi = NIns
        j = b.jmp.type
        if isretbh(j):
            r = newtmp("abi", Kw, fn)
            op = Oextsb + (j - Jretsb)
            emit(op, Kw, r, b.jmp.arg, R)
            b.jmp.arg = r
            b.jmp.type = Jretw
        ins = b.ins
        i = b.nins
        while i > 0:
            i -= 1
            emiti(ins[i])
            if ins[i].op != Ocall:
                continue
            i0 = i1 = i
            while i0 > 0:
                if not isarg(ins[i0 - 1].op):
                    break
                i0 -= 1
            i = i1
            while i > i0:
                i -= 1
                emiti(ins[i])
                if isargbh(ins[i].op):
                    ins[i].to = newtmp("abi", Kl, fn)
                    D.insb[D.curi].arg[0] = ins[i].to
            i = i1
            while i > i0:
                i -= 1
                if isargbh(ins[i].op):
                    op = Oextsb + (ins[i].op - Oargsb)
                    emit(op, Kw, ins[i].to, ins[i].arg[0], R)
        idup(b, D.curi, NIns - D.curi)
        b = b.link

    if D.debug['A']:
        sys.stderr.write("\n> After Apple pre-ABI:\n")
        printfn(fn, sys.stderr)
