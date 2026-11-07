"""Port of QBE amd64/isel.c (and runmatch from util.c).

For x86_64, do the following:

- check that constants are used only in
  places allowed
- ensure immediates always fit in 32b
- expose machine register contraints
  on instructions like division.
- implement fast locals (the streak of
  constant allocX in the first basic block)
- recognize complex addressing modes

Invariant: the use counts that are used
           in sel() must be sound.  This
           is not so trivial, maybe the
           dce should be moved out...

Pointers to a Ref (`Ref *r`) are (list, index) pairs; `Ins *` into insb or b.ins are indices.
"""

import sys

from .. import defs as D
from ..defs import (R, RTmp, RCon, RMem, RSlot, NIns, Ins, Addr, Con, Num, Kw, Kl, Ks, Kd,
                    KWIDE, KBASE, BIT, TMP, MEM, SLOT, CON_Z, CALL, NCmpI, CUndef, CBits,
                    CAddr, SExt, SThr, Cine, Cfeq, Cfne, Cflt, Cfle, Cfgt, Cfge, rtype, rval,
                    s64, isstore, isload, isext, isxsel, optab, Ocopy, Ocall, Oaddr, Oadd,
                    Oload, Odiv, Orem, Oudiv, Ourem, Oxidiv, Oxdiv, Osign, Osar, Oshr, Oshl,
                    Ouwtof, Oultof, Osltof, Oextuw, Ocast, Oor, Oand, Ostoui, Odtoui,
                    Ostosi, Odtosi, Onop, Ostored, Ostores, Ostorel, Ostorew, Ostoreh,
                    Ostoreb, Odbgloc, Osalloc, Osub, Oneg, Omul, Oxor, Oxtest, Oswtof, Oexts,
                    Otruncd, Oalloc4, Oalloc8, Oalloc16, Oalloc, Oalloc1, Oflag, Oflagfo,
                    Oflagfuo, Oxcmp, Oxsel, Osel0, Osel1, Jret0, Jjmp, Jhlt, Jjnz, Jjf)
from ..util import (err, die, newtmp, getcon, newcon, addcon, chuse, emit, emiti, idup,
                    vgrow, intern, iscmp, cmpop, argcls, salloc, isreg)
from ..emit import stashbits
from .. import ext as _ext
from ..parse import printfn
from .regs import RAX, RCX, RDX, RDI
from .targ import amd64_op

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT_MAX = INT32_MAX
INT_MIN = INT32_MIN


def noimm(r, fn):
    if rtype(r) != RCon:
        return 0
    c = fn.con[rval(r)]
    if c.type == CAddr:
        # we only support the 'small'
        # code model of the ABI, this
        # means that we can always
        # address data with 32bits
        return 0
    elif c.type == CBits:
        val = c.i
        return val < INT32_MIN or val > INT32_MAX
    else:
        die("invalid constant")


def rslot(r, fn):
    if rtype(r) != RTmp:
        return -1
    return fn.tmp[rval(r)].slot


def hascon(r, fn):
    """The Con (an object to update in place) or None."""
    t = rtype(r)
    if t == RCon:
        return fn.con[rval(r)]
    elif t == RMem:
        return fn.mem[rval(r)].offset
    return None


def fixarg(lst, idx, k, i, fn):
    """fixarg(&lst[idx], k, i, fn); i: the instruction holding the ref, or None."""
    r1 = r0 = lst[idx]
    s = rslot(r0, fn)
    op = i.op if i is not None else Ocopy
    c = None
    if KBASE(k) == 1 and rtype(r0) == RCon:
        # load floating points from memory
        # slots, they can't be used as
        # immediates
        r1 = MEM(fn.nmem)
        fn.nmem += 1
        vgrow(fn.mem, fn.nmem, Addr)
        a = Addr()
        a.offset.type = CAddr
        n = stashbits(fn.con[rval(r0)].i, 8 if KWIDE(k) else 4)
        # quote the name so that we do not
        # add symbol prefixes on the apple
        # target variant
        buf = "\"%sfp%d\"" % (D.T.asloc, n)
        a.offset.sym.id = intern(buf)
        fn.mem[fn.nmem - 1] = a
    elif (op == Ocall and i is not None and lst is i.arg and idx == 0
          and rtype(r0) == RCon and fn.con[rval(r0)].type != CAddr):
        # use a temporary register so that we
        # produce an indirect call
        r1 = newtmp("isel", Kl, fn)
        emit(Ocopy, Kl, r1, r0, R)
    elif op != Ocopy and k == Kl and noimm(r0, fn):
        # load constants that do not fit in
        # a 32bit signed integer into a
        # long temporary
        r1 = newtmp("isel", Kl, fn)
        emit(Ocopy, Kl, r1, r0, R)
    elif s != -1:
        # load fast locals' addresses into
        # temporaries right before the
        # instruction
        r1 = newtmp("isel", Kl, fn)
        emit(Oaddr, Kl, r1, SLOT(s), R)
    elif (op != Ocall and _hascon_ext(r0, fn)):
        c = hascon(r0, fn)
        r1 = newtmp("isel", Kl, fn)
        # cimo: the offset of a memory access stays its displacement, not an add of its own
        disp = (c.i and INT32_MIN <= c.i <= INT32_MAX and not _bigaddend(c) and i is not None
                and lst is i.arg and (rtype(r0) == RMem or (isload(op) and idx == 0)
                                      or (isstore(op) and idx == 1)))
        if c.i and not disp:
            r2 = newtmp("isel", Kl, fn)
            cc = Con(CBits, c.i)
            r3 = newcon(cc, fn)
            # fix over C QBE: lea, not add. For a phi's argument this comes after the block's last instruction, between
            # a compare and the jump that reads its flags (YARPGen: extern_array + 4 went the wrong way, then crashed)
            a = Addr()
            a.base = r2
            if noimm(r3, fn):  # an address's offset is 32 bits
                r4 = newtmp("isel", Kl, fn)
                a.index = r4
                a.scale = 1
            else:
                a.offset = cc
            fn.nmem += 1
            vgrow(fn.mem, fn.nmem, Addr)
            fn.mem[fn.nmem - 1] = a
            emit(Oaddr, Kl, r1, MEM(fn.nmem - 1), R)
            if noimm(r3, fn):
                emit(Ocopy, Kl, r4, r3, R)
        else:
            r2 = r1
        if D.T.apple and (c.sym.type & SThr):
            emit(Ocopy, Kl, r2, TMP(RAX), R)
            r2 = newtmp("isel", Kl, fn)
            r3 = newtmp("isel", Kl, fn)
            emit(Ocall, 0, R, r3, CALL(17))
            emit(Ocopy, Kl, TMP(RDI), r2, R)
            emit(Oload, Kl, r3, r2, R)
        cc = c.copy()
        cc.i = 0
        r3 = newcon(cc, fn)
        emit(Oaddr, Kl, r2, r3, R)
        if rtype(r0) == RMem:
            m = fn.mem[rval(r0)]
            m.offset = Con(CBits, c.i) if disp else Con(CUndef, 0)
            m.base = r1
            r1 = r0
        elif disp:
            a = Addr()
            a.offset = Con(CBits, c.i)
            a.base = r1
            fn.nmem += 1
            vgrow(fn.mem, fn.nmem, Addr)
            fn.mem[fn.nmem - 1] = a
            r1 = MEM(fn.nmem - 1)
    elif (not (isstore(op) and i is not None and lst is i.arg and idx == 1)
          and not isload(op) and op != Ocall and rtype(r0) == RCon
          and fn.con[rval(r0)].type == CAddr):
        # turn address operands into
        # lea/mov instructions
        r1 = newtmp("isel", Kl, fn)
        emit(Oaddr, Kl, r1, r0, R)
    elif rtype(r0) == RMem:
        # eliminate memory operands of
        # the form $foo(%rip, ...)
        m = fn.mem[rval(r0)]
        if m.base == R:
            if m.offset.type == CAddr:
                r0 = newtmp("isel", Kl, fn)
                emit(Oaddr, Kl, r0, newcon(m.offset, fn), R)
                m.offset.type = CUndef
                m.base = r0
    elif isxsel(op) and rtype(lst[idx]) == RCon:
        r1 = newtmp("isel", i.cls, fn)
        emit(Ocopy, i.cls, r1, lst[idx], R)
    lst[idx] = r1


def _bigaddend(c):
    """Whether a symbol's addend can't be relocated: rip-relative reaches 2 GB, of which the image
    takes some. Fix over C QBE (gcc's pr94734.c with a symbol as the base): the addend goes in a
    register instead, as for a constant base."""
    return c.type == CAddr and not -(1 << 30) <= c.i < 1 << 30


def _hascon_ext(r0, fn):
    c = hascon(r0, fn)
    return (c is not None and c.type == CAddr
            and ((c.sym.type & SExt) or (D.T.apple and c.sym.type == SThr) or _bigaddend(c)))


def seladdr(lst, idx, tn, fn):
    r0 = lst[idx]
    if rtype(r0) == RTmp:
        a = Addr()
        if not amatch(a, tn, r0, fn):
            return
        if ((a.offset.type == CBits and not INT32_MIN <= a.offset.i <= INT32_MAX)
                or _bigaddend(a.offset)):
            # fix over C QBE: a displacement is a signed 32-bit field (gcc's pr94734.c,
            # arr[0xdeadbeefbeefdead] once inlined); the address stays an add, whose constant
            # fixarg loads into a register
            return
        if a.base != R:
            if a.offset.type == CAddr:
                # apple as does not support
                # $foo(%r0, %r1, M); try to
                # rewrite it or bail out if
                # impossible
                if a.index != R or rtype(a.base) != RTmp:
                    return
                else:
                    a.index = a.base
                    a.scale = 1
                    a.base = R
        chuse(r0, -1, fn)
        fn.nmem += 1
        vgrow(fn.mem, fn.nmem, Addr)
        fn.mem[fn.nmem - 1] = a
        chuse(a.base, +1, fn)
        chuse(a.index, +1, fn)
        lst[idx] = MEM(fn.nmem - 1)


def cmpswap(arg, op):
    if op in (NCmpI + Cflt, NCmpI + Cfle):
        return 1
    if op in (NCmpI + Cfgt, NCmpI + Cfge):
        return 0
    return int(rtype(arg[0]) == RCon)


_ORDERED = frozenset((NCmpI + Cflt, NCmpI + Cfle, NCmpI + Cfgt, NCmpI + Cfge))


def selcmp(arg, k, swap, fn, c=None):
    """arg: a 2-element list, swapped in place. c: the condition; cimo: the ordered relations (<, <=, >, >=) of
    floats compare with comis, which raises invalid on a quiet NaN as IEEE 754 and C's Annex F ask; ==, != and
    the rest stay quiet (ucomis)"""
    if swap:
        r = arg[1]
        arg[1] = arg[0]
        arg[0] = r
    emit(D.Oxcmpe if c in _ORDERED and KBASE(k) == 1 else Oxcmp, k, R, arg[1], arg[0])
    icmp = D.insb[D.curi]
    if rtype(arg[0]) == RCon:
        assert k != Kw
        icmp.arg[1] = newtmp("isel", k, fn)
        emit(Ocopy, k, icmp.arg[1], arg[0], R)
        ci = D.insb[D.curi]
        fixarg(ci.arg, 0, k, ci, fn)
    fixarg(icmp.arg, 0, k, icmp, fn)
    fixarg(icmp.arg, 1, k, icmp, fn)


def _fix2(i, fn):
    """fixarg on both arguments of the instruction just emitted (i1 = curi)."""
    i1 = D.insb[D.curi]
    fixarg(i1.arg, 0, argcls(i, 0), i1, fn)
    fixarg(i1.arg, 1, argcls(i, 1), i1, fn)


def sel(i, tn, fn):
    """i: a copy of the instruction (passed by value in C)."""
    tmp = [R] * 7
    if rtype(i.to) == RTmp:
        if not isreg(i.to) and not isreg(i.arg[0]) and not isreg(i.arg[1]):
            if fn.tmp[rval(i.to)].nuse == 0:
                chuse(i.arg[0], -1, fn)
                chuse(i.arg[1], -1, fn)
                return
    i0 = D.curi
    k = i.cls
    op = i.op
    if D.isqop(op):
        _qsel(i, tn, fn)
    elif op in (Odiv, Orem, Oudiv, Ourem):
        if KBASE(k) == 1:
            _emit_fix(i, fn)
        else:
            if op == Odiv or op == Oudiv:
                r0, r1 = TMP(RAX), TMP(RDX)
            else:
                r0, r1 = TMP(RDX), TMP(RAX)
            emit(Ocopy, k, i.to, r0, R)
            emit(Ocopy, k, R, r1, R)
            if rtype(i.arg[1]) == RCon:
                # immediates not allowed for
                # divisions in x86
                r0 = newtmp("isel", k, fn)
            else:
                r0 = i.arg[1]
            if fn.tmp[rval(r0)].slot != -1:
                err("unlikely argument %%%s in %s",
                    fn.tmp[rval(r0)].name, optab[op].name)
            if op == Odiv or op == Orem:
                emit(Oxidiv, k, R, r0, R)
                emit(Osign, k, TMP(RDX), TMP(RAX), R)
            else:
                emit(Oxdiv, k, R, r0, R)
                emit(Ocopy, k, TMP(RDX), CON_Z, R)
            emit(Ocopy, k, TMP(RAX), i.arg[0], R)
            ci = D.insb[D.curi]
            fixarg(ci.arg, 0, k, ci, fn)
            if rtype(i.arg[1]) == RCon:
                emit(Ocopy, k, r0, i.arg[1], R)
    elif op in (Osar, Oshr, Oshl):
        r0 = i.arg[1]
        if rtype(r0) == RCon:
            _emit_fix(i, fn)
        else:
            if fn.tmp[rval(r0)].slot != -1:
                err("unlikely argument %%%s in %s",
                    fn.tmp[rval(r0)].name, optab[op].name)
            i.arg[1] = TMP(RCX)
            emit(Ocopy, Kw, R, TMP(RCX), R)
            emiti(i)
            i1 = D.insb[D.curi]
            emit(Ocopy, Kw, TMP(RCX), r0, R)
            fixarg(i1.arg, 0, argcls(i, 0), i1, fn)
    elif op == Ouwtof:
        r0 = newtmp("utof", Kl, fn)
        emit(Osltof, k, i.to, r0, R)
        emit(Oextuw, Kl, r0, i.arg[0], R)
        ci = D.insb[D.curi]
        fixarg(ci.arg, 0, k, ci, fn)
    elif op == Oultof:
        # %mask =l and %arg.0, 1
        # %isbig =l shr %arg.0, 63
        # %divided =l shr %arg.0, %isbig
        # %or =l or %mask, %divided
        # %float =d sltof %or
        # %cast =l cast %float
        # %addend =l shl %isbig, 52
        # %sum =l add %cast, %addend
        # %result =d cast %sum
        r0 = newtmp("utof", k, fn)
        if k == Ks:
            kc, sh = Kw, 23
        else:
            kc, sh = Kl, 52
        for j in range(4):
            tmp[j] = newtmp("utof", Kl, fn)
        for j in range(4, 7):
            tmp[j] = newtmp("utof", kc, fn)
        emit(Ocast, k, i.to, tmp[6], R)
        emit(Oadd, kc, tmp[6], tmp[4], tmp[5])
        emit(Oshl, kc, tmp[5], tmp[1], getcon(sh, fn))
        emit(Ocast, kc, tmp[4], r0, R)
        emit(Osltof, k, r0, tmp[3], R)
        emit(Oor, Kl, tmp[3], tmp[0], tmp[2])
        emit(Oshr, Kl, tmp[2], i.arg[0], tmp[1])
        ii = D.insb[D.curi].copy()
        D.curi += 1
        sel(ii, None, fn)
        emit(Oshr, Kl, tmp[1], i.arg[0], getcon(63, fn))
        ci = D.insb[D.curi]
        fixarg(ci.arg, 0, Kl, ci, fn)
        emit(Oand, Kl, tmp[0], i.arg[0], getcon(1, fn))
        ci = D.insb[D.curi]
        fixarg(ci.arg, 0, Kl, ci, fn)
    elif op in (Ostoui, Odtoui):
        if op == Ostoui:
            i.op = Ostosi
            kc = Ks
            tmp[4] = getcon(0xdf000000, fn)
        else:
            i.op = Odtosi
            kc = Kd
            tmp[4] = getcon(s64(0xc3e0000000000000), fn)
        if k == Kw:
            r0 = newtmp("ftou", Kl, fn)
            emit(Ocopy, Kw, i.to, r0, R)
            i.cls = Kl
            i.to = r0
            _emit_fix(i, fn)
        else:
            # %try0 =l {s,d}tosi %fp
            # %mask =l sar %try0, 63
            #
            #    mask is all ones if the first
            #    try was oob, all zeroes o.w.
            #
            # %fps ={s,d} sub %fp, (1<<63)
            # %try1 =l {s,d}tosi %fps
            #
            # %tmp =l and %mask, %try1
            # %res =l or %tmp, %try0
            r0 = newtmp("ftou", kc, fn)
            for j in range(4):
                tmp[j] = newtmp("ftou", Kl, fn)
            emit(Oor, Kl, i.to, tmp[0], tmp[3])
            emit(Oand, Kl, tmp[3], tmp[2], tmp[1])
            emit(i.op, Kl, tmp[2], r0, R)
            emit(Oadd, kc, r0, tmp[4], i.arg[0])
            i1 = D.insb[D.curi]  # fixarg() can change curi
            fixarg(i1.arg, 0, kc, i1, fn)
            fixarg(i1.arg, 1, kc, i1, fn)
            emit(Osar, Kl, tmp[1], tmp[0], getcon(63, fn))
            emit(i.op, Kl, tmp[0], i.arg[0], R)
            ci = D.insb[D.curi]
            fixarg(ci.arg, 0, Kl, ci, fn)
    elif op == Onop:
        pass
    elif op in _RMW.values():
        seladdr(i.arg, 1, tn, fn)
        _emit_fix(i, fn)
    elif op in (Ostored, Ostores, Ostorel, Ostorew, Ostoreh, Ostoreb):
        if rtype(i.arg[0]) == RCon:
            if i.op == Ostored:
                i.op = Ostorel
            if i.op == Ostores:
                i.op = Ostorew
        seladdr(i.arg, 1, tn, fn)
        _emit_fix(i, fn)
    elif isload(op):
        seladdr(i.arg, 0, tn, fn)
        _emit_fix(i, fn)
    elif ((op in (Oswtof, Osltof) or op == Ocast and KBASE(k) == 1)
          and rtype(i.arg[0]) == RCon and KBASE(argcls(i, 0)) == 0):
        # fix over C QBE: an integer constant converted to a float (cvtsi2sd, movq to xmm)
        # takes no immediate; cimo's simplifier leaves such conversions (C QBE's fold does not)
        emiti(i)
        r = newtmp("isel", argcls(i, 0), fn)
        D.insb[D.curi].arg[0] = r
        emit(Ocopy, argcls(i, 0), r, i.arg[0], R)
        ci = D.insb[D.curi]
        fixarg(ci.arg, 0, argcls(i, 0), ci, fn)
    elif op in _EMIT or isext(op) or isxsel(op):
        _emit_fix(i, fn)
    elif op in (D.Oh2f, D.Of2h):  # ext.x86 f16c: the operand in a register (vmovd and vcvtps2ph take no immediate)
        emiti(i)
        i1 = D.insb[D.curi]
        ka = argcls(i, 0)
        if rtype(i1.arg[0]) == RCon:
            r = newtmp("isel", ka, fn)
            i1.arg[0] = r
            emit(Ocopy, ka, r, i.arg[0], R)
            fixarg(D.insb[D.curi].arg, 0, ka, D.insb[D.curi], fn)
        else:
            fixarg(i1.arg, 0, ka, i1, fn)
    elif op in _EXTOPS:
        # cimo extensions (ext.vla, ext.math): getsp, setsp, fsqrt, fabs and the bit
        # builtins; integer operands must be in registers (bsr, bsf and bswap take no
        # immediate)
        emiti(i)
        i1 = D.insb[D.curi]
        if op in _EXTINT and rtype(i1.arg[0]) == RCon:
            r = newtmp("isel", k, fn)
            i1.arg[0] = r
            emit(Ocopy, k, r, i.arg[0], R)
        else:
            fixarg(i1.arg, 0, argcls(i, 0), i1, fn)
    elif op in (Oalloc4, Oalloc8, Oalloc16):
        salloc(i.to, i.arg[0], fn)
    else:
        kx = iscmp(op)
        if kx is not None:
            kc, x = kx
            if x == NCmpI + Cfeq:
                # zf is set when operands are
                # unordered, so we may have to
                # check pf
                r0 = newtmp("isel", Kw, fn)
                r1 = newtmp("isel", Kw, fn)
                emit(Oand, Kw, i.to, r0, r1)
                emit(Oflagfo, k, r1, R, R)
                i.to = r0
            elif x == NCmpI + Cfne:
                r0 = newtmp("isel", Kw, fn)
                r1 = newtmp("isel", Kw, fn)
                emit(Oor, Kw, i.to, r0, r1)
                emit(Oflagfuo, k, r1, R, R)
                i.to = r0
            swap = cmpswap(i.arg, x)
            if swap:
                x = cmpop(x)
            emit(Oflag + x, k, i.to, R, R)
            selcmp(i.arg, kc, swap, fn, x)
        else:
            die("unknown instruction %s", optab[op].name)

    while i0 > D.curi:
        i0 -= 1
        assert rslot(D.insb[i0].arg[0], fn) == -1
        assert rslot(D.insb[i0].arg[1], fn) == -1


_EMIT = frozenset((Odbgloc, Ocall, Osalloc, Ocopy, Oadd, Osub, Oneg, Omul, Oand, Oor, Oxor,
                   Oxtest, Ostosi, Odtosi, Oswtof, Osltof, Oexts, Otruncd, Ocast))


_EXTINT = frozenset((D.Obclz, D.Obctz, D.Obpopcnt, D.Obswap, D.Obswap16, D.Owrfpcr, D.Owrfpsr))
_EXTOPS = _EXTINT | frozenset((D.Ogetsp, D.Osetsp, D.Ofsqrt, D.Ofabs, D.Ordfpcr, D.Ordfpsr))


def _inreg(r, k, fn):
    """r, or a temporary holding it if it's a constant (FMA takes registers)."""
    if rtype(r) != RCon:
        return r
    t = newtmp("isel", k, fn)
    emit(Ocopy, k, t, r, R)
    ci = D.insb[D.curi]
    fixarg(ci.arg, 0, k, ci, fn)
    return t


def _negof(r, k, blk, at, fn):
    """If r is a neg of class k earlier in blk than at, the negated operand, and r's use dropped (the neg goes
    once unused), as arm64's negof; else None."""
    if rtype(r) != RTmp:
        return None
    for d in range(at - 1, -1, -1):
        di = blk.ins[d]
        if di.to == r:
            if di.op != Oneg or di.cls != k:
                return None
            x = di.arg[0]
            chuse(x, +1, fn)
            chuse(r, -1, fn)
            return x
    return None


# (negated product, negated addend): r = a*b + c, c - a*b, a*b - c, -(a*b) - c (arm64's names)
_FMAOP = {(0, 0): D.Oafmadd, (1, 0): D.Oafmsub, (0, 1): D.Oafnmsub, (1, 1): D.Oafnmadd}


def _selfma(i, blk, k1, fn):
    """cimo (ext.x86 fma): "fma0 c; r = fma1 a, b" is r = afmadd a, MEM{base=b, index=c}, as on arm64, so
    register allocation sees all three, a negated factor or addend folded in (vfnmadd, vfmsub, vfnmsub); emit
    picks the 231 or the 213 form by which one r shares."""
    i0 = blk.ins[k1 - 1]
    assert i0.op == D.Ofma0 and i0.cls == i.cls
    k = i.cls
    a, b, c = i.arg[0], i.arg[1], i0.arg[0]
    neg = [0, 0]
    x = _negof(a, k, blk, k1 - 1, fn)
    if x is not None:
        a, neg[0] = x, neg[0] ^ 1
    x = _negof(b, k, blk, k1 - 1, fn)
    if x is not None:
        b, neg[0] = x, neg[0] ^ 1
    x = _negof(c, k, blk, k1 - 1, fn)
    if x is not None:
        c, neg[1] = x, 1
    m = Addr()
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem, Addr)
    fn.mem[fn.nmem - 1] = m
    emit(_FMAOP[tuple(neg)], k, i.to, R, MEM(fn.nmem - 1))
    at = D.curi
    a = _inreg(a, k, fn)
    b = _inreg(b, k, fn)
    c = _inreg(c, k, fn)
    D.insb[at].arg[0] = a
    m.base, m.index = b, c


def _selqfma(i, i0, fn, tn=None):
    """cimo (ext.x86 fma): vfmadd231ps/pd keep the addend in the destination, as arm64's fmla: the addend is
    copied there first, and register allocation keeps the two in one register (rega's radd). qxfmam's second operand
    is an address, its whole addressing mode selected."""
    assert i0.op == D.Oqfma0
    sym = None
    if i.op in (D.Oqxfmams, D.Oqxfmamd):
        seladdr(i.arg, 1, tn, fn)
        if rtype(i.arg[1]) == RMem:
            m = fn.mem[rval(i.arg[1])]
            if m.base == R and m.offset.type == CAddr:  # no sym(%rip, %idx, s): the symbol's address in a
                sym = newtmp("isel", Kl, fn)            # register first, as fixarg does for other loads
                c = newcon(m.offset, fn)
                m.offset.type = CUndef
                m.base = sym
    emit(i.op, D.Kq, i.to, i.arg[0], i.arg[1])
    emit(D.Oqfma0, D.Kq, R, i.to, R)
    emit(Ocopy, D.Kq, i.to, i0.arg[0], R)
    if sym is not None:
        emit(Oaddr, Kl, sym, c, R)


def _qsel(i, tn, fn):
    """cimo (ext.vector): registers, a whole addressing mode, and a constant for shifts, shuffles and lanes."""
    op = i.op
    if op in (D.Oqload, D.Oqldbh, D.Oqload16, D.Oqldh):
        seladdr(i.arg, 0, tn, fn)
    elif op in (D.Oqstore, D.Oqstbh, D.Oqstore16, D.Oqsth):
        seladdr(i.arg, 1, tn, fn)
    elif op in (D.Oqld3bh, D.Oqxwin):  # a base register: the emitter loads at offsets from it
        emiti(i)
        fixarg(D.insb[D.curi].arg, 0, Kl, D.insb[D.curi], fn)
        return
    if D.Oqdupb <= op <= D.Oqdupd and rtype(i.arg[0]) == RCon:
        c = fn.con[rval(i.arg[0])]
        if c.type == CBits and c.i in (0, -1) and op not in (D.Oqdups, D.Oqdupd):
            emiti(i)  # pxor, pcmpeqd
            return
        k = argcls(i, 0)
        r = newtmp("isel", k, fn)
        a = i.arg[0]
        i.arg[0] = r
        emiti(i)
        emit(Ocopy, k, r, a, R)
        ci = D.insb[D.curi]
        fixarg(ci.arg, 0, k, ci, fn)
        return
    emiti(i)
    i1 = D.insb[D.curi]
    keep1 = D.Oqshlb <= op <= D.Oqsarl or D.Oqgetb <= op <= D.Oqgetd or op in (D.Oqxshufd, D.Oqxshufb)
    fixarg(i1.arg, 0, argcls(i, 0), i1, fn)
    if i1.arg[1] != R and argcls(i, 1) != D.Kx and not keep1:
        fixarg(i1.arg, 1, argcls(i, 1), i1, fn)


def _emit_fix(i, fn):
    emiti(i)
    _fix2(i, fn)


def flagi(ins, i0, i, fn):
    """Index of the flag-setting instruction before ins[i] (scanning back to i0), or None."""
    while i > i0:
        i -= 1
        o = amd64_op[ins[i].op]
        if o.zflag:
            if ins[i].op in (Osar, Oshr, Oshl) and not _shiftsflags(ins[i], fn):
                return None
            return i
        if o.lflag:
            continue
        return None
    return None


def _shiftsflags(i, fn):
    """fix over C QBE: a shift sets the flags only if its count (masked to 5 or 6 bits) is not 0:
    sarl %cl, %edx leaves them as the instruction before set them when %cl is 0 (gcc's
    builtin-bitops-1.c once cimo's simplifier dropped the `& 1` between the shift and the branch)."""
    r = i.arg[1]
    if rtype(r) != RCon or fn.con[rval(r)].type != CBits:
        return False
    return fn.con[rval(r)].i & (63 if KWIDE(i.cls) else 31) != 0


def selsel(fn, b, i, tn):
    """i: index in b.ins of an Osel1; returns the index of its Osel0."""
    ins = b.ins
    assert ins[i].op == Osel1
    isel0 = i
    while 0 < isel0:
        if ins[isel0].op == Osel0:
            break
        assert ins[isel0].op == Osel1
        isel0 -= 1
    assert ins[isel0].op == Osel0
    r = ins[isel0].arg[0]
    if rtype(r) == RCon:  # cimo: the middle end's if-conversion, its condition folded since: copies
        yes = fn.con[rval(r)].type != CBits or fn.con[rval(r)].i != 0
        isel1 = i
        while isel0 < isel1:
            x = ins[isel1]
            sel(Ins(Ocopy, x.cls, x.to, x.arg[0] if yes else x.arg[1], R), tn, fn)
            isel1 -= 1
        ins[isel0] = Ins(Onop)
        return isel0
    assert rtype(r) == RTmp
    t = fn.tmp[rval(r)]
    fi = flagi(ins, 0, isel0, fn)
    if fi is not None and ins[fi].to == r and iscmp(ins[fi].op) is None and ins[fi].cls != Kw:
        fi = None  # fix over C QBE: as in seljmp, a 64-bit instruction's flags are not the condition's
    if (fi is None or ins[fi].to != r) and t.nuse == 1:
        # cimo: the integer comparison earlier in the block, something that sets the flags (an add) since: compared
        # again here, its operands being temporaries that still hold what they held (edit_distance's v <= cur, whose
        # cur + 1 the scheduler put in between: setle, movzbl, test, cmov otherwise)
        for n in range(isel0 - 1, -1, -1):
            x = ins[n]
            if x.to == r:
                kc = iscmp(x.op)
                if kc is not None and kc[1] < NCmpI:
                    fi = n
                break
    cr = [R, R]
    gencmp = gencpy = swap = 0
    k = Kw
    c = Cine
    other = False
    if fi is None or ins[fi].to != r:
        gencmp = 1
        cr[0] = r
        cr[1] = CON_Z
    else:
        kc = iscmp(ins[fi].op)
        if kc is not None:
            k, c = kc
            if c == NCmpI + Cfeq or c == NCmpI + Cfne:
                # these are selected as 'and'
                # or 'or', so we check their
                # result with Cine
                c = Cine
                other = True
            else:
                swap = cmpswap(ins[fi].arg, c)
                if swap:
                    c = cmpop(c)
                if t.nuse == 1:
                    gencmp = 1
                    cr[0] = ins[fi].arg[0]
                    cr[1] = ins[fi].arg[1]
                    ins[fi] = Ins(Onop)
        elif (ins[fi].op == Oand and t.nuse == 1
              and (rtype(ins[fi].arg[0]) == RTmp or rtype(ins[fi].arg[1]) == RTmp)):
            f = ins[fi]
            f.op = Oxtest
            f.to = R
            if rtype(f.arg[1]) == RCon:
                r = f.arg[1]
                f.arg[1] = f.arg[0]
                f.arg[0] = r
        else:
            other = True
    if other:
        # since flags are not tracked in liveness,
        # the result of the flag-setting instruction
        # has to be marked as live
        if t.nuse == 1:
            gencpy = 1
    # generate conditional moves
    isel1 = i
    while isel0 < isel1:
        ins[isel1].op = Oxsel + c
        sel(ins[isel1].copy(), tn, fn)
        isel1 -= 1
    assert not gencmp or not gencpy
    if gencmp:
        selcmp(cr, k, swap, fn, c)
    if gencpy:
        emit(Ocopy, Kw, R, r, R)
    ins[isel0] = Ins(Onop)
    return isel0


def seljmp(b, fn):
    if b.jmp.type in (Jret0, Jjmp, Jhlt):
        return
    assert b.jmp.type == Jjnz
    r = b.jmp.arg
    b.jmp.arg = R
    assert rtype(r) == RTmp
    t = fn.tmp[rval(r)]
    if b.s1 is b.s2:
        chuse(r, -1, fn)
        b.jmp.type = Jjmp
        b.s2 = None
        return
    ins = b.ins
    fi = flagi(ins, 0, b.nins, fn)
    kc = iscmp(ins[fi].op) if fi is not None and ins[fi].to == r else None
    if fi is not None and kc is None and ins[fi].cls != Kw:
        # fix over C QBE: jnz tests the low 32 bits; a 64-bit instruction's flags are of all 64 (pr94412: a
        # lane of a vector, loaded as w from the l `or` that stored both lanes, tested whole)
        fi = None
    if fi is None or ins[fi].to != r:
        selcmp([r, CON_Z], Kw, 0, fn)
        b.jmp.type = Jjf + Cine
    elif (kc is not None and kc[1] != NCmpI + Cfeq  # see sel(), selsel()
          and kc[1] != NCmpI + Cfne):
        k, c = kc
        swap = cmpswap(ins[fi].arg, c)
        if swap:
            c = cmpop(c)
        if t.nuse == 1:
            selcmp(ins[fi].arg, k, swap, fn, c)
            ins[fi] = Ins(Onop)
        b.jmp.type = Jjf + c
    elif (ins[fi].op == Oand and t.nuse == 1
          and (rtype(ins[fi].arg[0]) == RTmp or rtype(ins[fi].arg[1]) == RTmp)):
        f = ins[fi]
        f.op = Oxtest
        f.to = R
        b.jmp.type = Jjf + Cine
        if rtype(f.arg[1]) == RCon:
            r = f.arg[1]
            f.arg[1] = f.arg[0]
            f.arg[0] = r
    else:
        # since flags are not tracked in liveness,
        # the result of the flag-setting instruction
        # has to be marked as live
        if t.nuse == 1:
            emit(Ocopy, Kw, R, r, R)
        b.jmp.type = Jjf + Cine


Pob, Pbis, Pois, Pobis, Pbi1, Pobi1 = range(6)

# mgen generated code
#
# (with-vars (o b i s)
#   (patterns
#     (ob   (add (con o) (tmp b)))
#     (bis  (add (tmp b) (mul (tmp i) (con s 1 2 4 8))))
#     (ois  (add (con o) (mul (tmp i) (con s 1 2 4 8))))
#     (obis (add (con o) (tmp b) (mul (tmp i) (con s 1 2 4 8))))
#     (bi1  (add (tmp b) (tmp i)))
#     (obi1 (add (con o) (tmp b) (tmp i)))
# ))

_Oaddtbl = [
    2,
    2, 2,
    4, 4, 5,
    6, 6, 8, 8,
    4, 4, 9, 10, 9,
    7, 7, 5, 8, 9, 5,
    4, 4, 12, 10, 12, 12, 12,
    4, 4, 9, 10, 9, 9, 12, 9,
    11, 11, 5, 8, 9, 5, 12, 9, 5,
    7, 7, 5, 8, 9, 5, 12, 9, 5, 5,
    11, 11, 5, 8, 9, 5, 12, 9, 5, 5, 5,
    4, 4, 9, 10, 9, 9, 12, 9, 9, 9, 9, 9,
    7, 7, 5, 8, 9, 5, 12, 9, 5, 5, 5, 9, 5,
]
assert len(_Oaddtbl) == 91


def opn(op, l, r):
    if l < r:
        l, r = r, l
    if op == Omul:
        if 2 <= l:
            if r == 0:
                return 3
        return 2
    elif op == Oadd:
        return _Oaddtbl[(l + l * l) // 2 + r]
    return 2


def refn(r, tn, con):
    t = rtype(r)
    if t == RTmp:
        n = tn[rval(r)]
        if not n.n:
            n.n = 2
        return n.n
    elif t == RCon:
        c = con[rval(r)]
        if c.type != CBits:
            return 1
        n = c.i
        if n == 8 or n == 4 or n == 2 or n == 1:
            return 0
        return 1
    return INT_MIN


match = [0] * 13
match[4] = BIT(Pob)
match[5] = BIT(Pbi1)
match[6] = BIT(Pob) | BIT(Pois)
match[7] = BIT(Pob) | BIT(Pobi1)
match[8] = BIT(Pbi1) | BIT(Pbis)
match[9] = BIT(Pbi1) | BIT(Pobi1)
match[10] = BIT(Pbi1) | BIT(Pbis) | BIT(Pobi1) | BIT(Pobis)
match[11] = BIT(Pob) | BIT(Pobi1) | BIT(Pobis)
match[12] = BIT(Pbi1) | BIT(Pobi1) | BIT(Pobis)

matcher = [None] * 6
matcher[Pbi1] = [1, 3, 1, 3, 2, 0]
matcher[Pbis] = [5, 1, 8, 5, 27, 1, 5, 1, 2, 5, 13, 3, 1, 1, 3, 3, 3, 2, 0, 1,
                 3, 3, 3, 2, 3, 1, 0, 1, 29]
matcher[Pob] = [1, 3, 0, 3, 1, 0]
matcher[Pobi1] = [5, 3, 9, 9, 10, 33, 12, 35, 45, 1, 5, 3, 11, 9, 7, 9, 4, 9,
                  17, 1, 3, 0, 3, 1, 3, 2, 0, 3, 1, 1, 3, 0, 34, 1, 37, 1, 5, 2,
                  5, 7, 2, 7, 8, 37, 29, 1, 3, 0, 1, 32]
matcher[Pobis] = [5, 2, 10, 7, 11, 19, 49, 1, 1, 3, 3, 3, 2, 1, 3, 0, 3, 1, 0,
                  1, 3, 0, 5, 1, 8, 5, 25, 1, 5, 1, 2, 5, 13, 3, 1, 1, 3, 3, 3,
                  2, 0, 1, 3, 3, 3, 2, 26, 1, 51, 1, 5, 1, 6, 5, 9, 1, 3, 0, 51,
                  3, 1, 1, 3, 0, 45]
matcher[Pois] = [1, 3, 0, 1, 3, 3, 3, 2, 0]

# end of generated code


def runmatch(code, tn, ref, var):
    """Port of util.c runmatch: the mgen matcher's bytecode interpreter."""
    assert rtype(ref) == RTmp
    stk = []
    pc = 0
    while True:
        bc = code[pc]
        if not bc:
            return
        if bc == 1 or bc == 2:  # pushsym, push
            assert len(stk) < 20
            assert rtype(ref) == RTmp
            n = tn[rval(ref)]
            if bc == 1 and n.nl > n.nr:
                stk.append(n.l)
                ref = n.r
            else:
                stk.append(n.r)
                ref = n.l
            pc += 1
        elif bc == 3 or bc == 4:
            if bc == 3:  # set
                pc += 1
                var[code[pc]] = ref
                if code[pc + 1] == 0:
                    return
            # pop
            assert stk
            ref = stk.pop()
            pc += 1
        elif bc == 5:  # switch
            assert rtype(ref) == RTmp
            n = tn[rval(ref)].n
            s = pc + 1
            i = code[s]
            s += 1
            while i > 0:
                if n == code[s]:
                    s += 1
                    break
                s += 2
                i -= 1
            pc += code[s]
        else:  # jump
            assert bc >= 10
            pc = bc - 10


class _NumTab(dict):
    """tn: Num per temporary, zeroed (created on first use; C clears the whole array per
    block)."""

    def __missing__(self, k):
        v = self[k] = Num()
        return v


def anumber(tn, b, con):
    for i in b.ins[:b.nins]:
        if rtype(i.to) != RTmp:
            continue
        n = tn[rval(i.to)]
        n.l = i.arg[0]
        n.r = i.arg[1]
        n.nl = refn(n.l, tn, con) & 0xff  # uchar
        n.nr = refn(n.r, tn, con) & 0xff
        n.n = opn(i.op, n.nl, n.nr) & 0xff


def adisp(c, tn, r, fn, s):
    v = [R, R]
    while r != R:
        assert rtype(r) == RTmp
        n = refn(r, tn, fn.con)
        if not (match[n] & BIT(Pob)):
            break
        runmatch(matcher[Pob], tn, r, v)
        assert rtype(v[0]) == RCon
        addcon(c, fn.con[rval(v[0])], s)
        r = v[1]
    return r


_pat = [Pobis, Pobi1, Pbis, Pois, Pbi1]


def amatch(a, tn, r, fn):
    if rtype(r) != RTmp:
        return 0

    n = refn(r, tn, fn.con)
    v = [R, R, R, R]
    p = -1
    for q in _pat:
        if match[n] & BIT(q):
            runmatch(matcher[q], tn, r, v)
            p = q
            break
    if p < 0:
        v[1] = r

    co = Con()
    ro = v[0]
    rb = adisp(co, tn, v[1], fn, 1)
    ri = v[2]
    rs = v[3]
    s = 1

    if p < 0 and co.type != CUndef:
        if amatch(a, tn, rb, fn):
            return addcon(a.offset, co, 1)
    if ro != R:
        assert rtype(ro) == RCon
        c = fn.con[rval(ro)]
        if not addcon(co, c, 1):
            return 0
    if rs != R:
        assert rtype(rs) == RCon
        c = fn.con[rval(rs)]
        assert c.type == CBits
        s = c.i
    ri = adisp(co, tn, ri, fn, s)
    a.offset = co
    a.base = rb
    a.index = ri
    a.scale = s

    if rtype(ri) == RTmp:
        if fn.tmp[rval(ri)].slot != -1:
            if a.scale != 1 or fn.tmp[rval(rb)].slot != -1:
                return 0
            a.base = ri
            a.index = rb
    if a.base != R:
        assert rtype(a.base) == RTmp
        s = fn.tmp[rval(a.base)].slot
        if s != -1:
            a.base = SLOT(s)
    return 1


_RMW = {Oadd: D.Oxaddm, Osub: D.Oxsubm, Oand: D.Oxandm, Oor: D.Oxorm, Oxor: D.Oxxorm}
# an extension to 64 bits: the loads of a word it can be, and the load that does both. A 32-bit load is any of three, its
# extension says which half it means
_W = (Oload, D.Oloadsw, D.Oloaduw)
_ZEXT = {D.Oextub: ((D.Oloadub,), D.Oloadub), D.Oextuh: ((D.Oloaduh,), D.Oloaduh), D.Oextuw: (_W, D.Oloaduw),
         D.Oextsb: ((D.Oloadsb,), D.Oloadsb), D.Oextsh: ((D.Oloadsh,), D.Oloadsh), D.Oextsw: (_W, D.Oloadsw)}


def _writes(op):
    """Whether op may write memory."""
    return isstore(op) or op in _RMW.values() or op in (Ocall, D.Oblit0, D.Oblit1, D.Ovastart, D.Oqstore, D.Oqstbh)


_FMAM = {D.Oqfma1s: D.Oqxfmams, D.Oqfma1d: D.Oqxfmamd}


def _fmafold(fn, ins, defs):
    """A vector load that only a vector fma reads, nothing that may write memory in between, is the fma's memory
    operand (vfmadd231ps (%rax), %ymm1, %ymm2): gemm's row of b, one instruction fewer per fma."""
    tmp = fn.tmp
    for k, i in enumerate(ins):
        if i.op not in _FMAM or k == 0 or ins[k - 1].op != D.Oqfma0 or fn.hasvolatile:
            continue
        for n in (1, 0):  # either factor: the product commutes
            a = i.arg[n]
            if rtype(a) != RTmp or tmp[rval(a)].nuse != 1 or i.arg[1 - n] == a:
                continue
            d = defs.get(rval(a))
            if d is None or d >= k - 1 or ins[d].op != D.Oqload:
                continue
            if any(_writes(x.op) for x in ins[d + 1:k]):
                continue
            if rtype(ins[k - 1].arg[0]) == RTmp and ins[k - 1].arg[0] == a:
                continue
            ins[k] = Ins(_FMAM[i.op], D.Kq, i.to, i.arg[1 - n], ins[d].arg[0])
            ins[d] = Ins(Onop)
            break


def _loadsfold(fn):
    """cimo: what x86-64 does in the load or the store itself, as gcc. A load extended to 64 bits extends to 64 bits
    (movzbl and then movzbq of its byte: 8.8 to 7.2 ms on Zen 4's histogram). A load, an operation on it and a store
    of the result to the same address, nothing in between that may write memory, are one instruction (addl $1, (mem):
    5.9 ms, gcc's 6.3). Needs use counts."""
    tmp = fn.tmp
    b = fn.start
    while b:
        ins = b.ins[:b.nins]
        defs = {}
        for k, i in enumerate(ins):
            if rtype(i.to) == RTmp:
                defs[rval(i.to)] = k
        for k, i in enumerate(ins):
            if i.op in _ZEXT and i.cls == Kl and rtype(i.arg[0]) == RTmp:
                d = defs.get(rval(i.arg[0]))
                if d is not None and d < k and ins[d].op in _ZEXT[i.op][0] and ins[d].cls == Kw \
                        and tmp[rval(i.arg[0])].nuse == 1:
                    ld = ins[d]
                    ld.op = _ZEXT[i.op][1]
                    ld.cls = Kl
                    ld.to = i.to
                    ins[k] = Ins(Onop)
        if 'fma' in _ext.x86:
            _fmafold(fn, ins, defs)
        for k, st in enumerate(ins):
            k2 = {Ostorew: Kw, Ostorel: Kl}.get(st.op)
            if k2 is None or rtype(st.arg[0]) != RTmp or fn.hasvolatile:  # a volatile access stays as written
                continue
            o = defs.get(rval(st.arg[0]))
            if o is None or o >= k or ins[o].op not in _RMW or ins[o].cls != k2 or tmp[rval(st.arg[0])].nuse != 1:
                continue
            op = ins[o]
            for n in (0, 1):
                a = op.arg[n]
                if rtype(a) != RTmp:
                    continue
                d = defs.get(rval(a))
                if d is None or d >= o or tmp[rval(a)].nuse != 1:
                    continue
                ld = ins[d]
                if not ((ld.op == Oload or (k2 == Kw and ld.op in (D.Oloadsw, D.Oloaduw))) and ld.cls == k2
                        and ld.arg[0] == st.arg[1]):
                    continue
                if n == 1 and op.op == Osub:
                    continue
                if any(_writes(x.op) for x in ins[d + 1:k] if x is not op):
                    continue
                v = op.arg[1 - n]
                ins[k] = Ins(_RMW[op.op], k2, R, v, st.arg[1])
                ins[o] = Ins(Onop)
                ins[d] = Ins(Onop)
                tmp[rval(st.arg[1])].nuse -= 1 if rtype(st.arg[1]) == RTmp else 0
                break
        b.ins = ins
        b.nins = len(ins)
        b = b.link


def amd64_isel(fn):
    """instruction selection
    requires use counts (as given by parsing)"""
    # assign slots to fast allocs
    b = fn.start
    # specific to NAlign == 3  (or change n=4 and sz /= 4 below)
    n = 4
    for al in range(Oalloc, Oalloc1 + 1):
        for k in range(b.nins):
            i = b.ins[k]
            if i.op == al:
                if rtype(i.arg[0]) != RCon:
                    break
                sz = fn.con[rval(i.arg[0])].i
                if sz < 0 or sz >= INT_MAX - 15:
                    err("invalid alloc size %d", sz)
                sz = (sz + n - 1) & -n
                sz //= 4
                if sz > INT_MAX - fn.slot:
                    die("alloc too large")
                fn.tmp[rval(i.to)].slot = fn.slot
                fn.slot += sz
                fn.salign = 2 + al - Oalloc
                b.ins[k] = Ins(Onop)
        n *= 2

    _loadsfold(fn)

    # process basic blocks
    b = fn.start
    while b:
        D.curi = NIns
        for sb in (b.s1, b.s2):
            if not sb:
                break
            p = sb.phi
            while p:
                a = 0
                while p.blk[a] is not b:
                    assert a + 1 < p.narg
                    a += 1
                fixarg(p.arg, a, p.cls, None, fn)
                p = p.link
        num = _NumTab()
        anumber(num, b, fn.con)
        seljmp(b, fn)
        k = b.nins
        while k != 0:
            k -= 1
            i = b.ins[k]
            assert i.op != Osel0
            if i.op == Osel1:
                k = selsel(fn, b, k, num)
            elif i.op == D.Ofma1:
                _selfma(i.copy(), b, k, fn)
                k -= 1  # its fma0
            elif i.op in (D.Oqfma1s, D.Oqfma1d, D.Oqxfmams, D.Oqxfmamd):
                _selqfma(i.copy(), b.ins[k - 1], fn, num)
                k -= 1
            else:
                sel(i.copy(), num, fn)
        idup(b, D.curi, NIns - D.curi)
        b = b.link

    if D.debug['I']:
        sys.stderr.write("\n> After instruction selection:\n")
        printfn(fn, sys.stderr)
