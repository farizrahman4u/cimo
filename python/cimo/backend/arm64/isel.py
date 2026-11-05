"""Port of QBE arm64/isel.c (with the local fcon/foldaddr/immediate patch)."""

import struct
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
from ..defs import Tmp0, Oextsw, Oextuw, Oextub, Oextuh, Osar, Oshr, Cieq, Cine, Kd, req
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


# the bit patterns fmov encodes (ext.isel)
_FPIMM = {Kd: {}, D.Ks: {}}
for _i in range(256):
    _e = (_i >> 4) & 7
    _v = (-1.0 if _i & 0x80 else 1.0) * (16 + (_i & 15)) / 16 * 2.0 ** (_e + 1 if _e < 4 else _e - 7)
    _FPIMM[Kd][struct.unpack('<q', struct.pack('<d', _v))[0]] = _v
    _FPIMM[D.Ks][struct.unpack('<I', struct.pack('<f', _v))[0]] = _v
del _i, _e, _v


def emitfcon(fn):
    for n in range(nfcon):
        if _ext.isel:
            bits = fcon[n].bits
            if bits == 0 or bits in _FPIMM[fcon[n].k]:
                emit(D.Oafcon, fcon[n].k, fcon[n].tmp, getcon(bits, fn), R)
                continue
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
                if imm(cc, Kl)[0] in (Iplo12, Iphi12):
                    emit(Oadd, Kl, r1, r2, r3)
                else:  # fix over C QBE: any offset was an immediate (no encoding)
                    r4 = newtmp("isel", Kl, fn)
                    emit(Oadd, Kl, r1, r2, r4)
                    emit(Ocopy, Kl, r4, r3, R)
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


def _defin(r, before):
    """The instruction of the block being selected, before index `before`, that defines r."""
    ins = selblk.ins
    d = before
    while d > 0:
        d -= 1
        if ins[d].to == r:
            return d, ins[d]
    return None


def _plain(r, fn):
    """r is a temporary, not a register or a stack slot's address."""
    return rtype(r) == RTmp and rval(r) >= Tmp0 and fn.tmp[rval(r)].slot == -1


def _sweep(r, d, fn):
    """Delete instruction d, which defines r, if r has no uses left."""
    if fn.tmp[rval(r)].nuse == 0:
        di = selblk.ins[d]
        for a in di.arg:
            chuse(a, -1, fn)
        di.set(Ins(Onop))


def _scaled(r, before, sz, fn):
    """(index, shift, defining position) if r is "shl x, log2(sz)" or "mul x, sz"."""
    if sz == 1:
        return None
    found = _defin(r, before)
    if found is None:
        return None
    d, di = found
    if di.cls != Kl or not _plain(di.arg[0], fn) or rtype(di.arg[1]) != RCon:
        return None
    c = fn.con[rval(di.arg[1])]
    if c.type != CBits:
        return None
    l = log2p(sz)
    if (di.op == Oshl and c.i == l) or (di.op == Omul and c.i == sz):
        return di.arg[0], l, d
    return None


def foldindex(pa, pi, sz, fn):
    """cimo (ext.isel): fold "%a = add %base, %index" into [base, index], with the index's
    "shl x, log2(sz)" and "extsw"/"extuw" if it has them, as foldaddr does for an offset. The
    instructions go once all their uses are folded."""
    pr = pa[pi]
    if not _plain(pr, fn):
        return
    found = _defin(pr, selpos)
    if found is None:
        return
    d, di = found
    a0, a1 = di.arg
    if di.op != Oadd or di.cls != Kl or not _plain(a0, fn) or not _plain(a1, fn):
        return
    base, index, shift, sd, shl = a0, a1, 0, None, None
    x = _scaled(a1, d, sz, fn)
    if x is not None:
        index, shift, sd = x
        shl = a1
    else:
        x = _scaled(a0, d, sz, fn)
        if x is not None:
            base, index, shift, sd = a1, x[0], x[1], x[2]
            shl = a0
    ext = 0
    ed = None
    wide = index
    found = _defin(index, sd if sd is not None else d)
    if found is not None and found[1].op in (Oextsw, Oextuw) and found[1].cls == Kl \
            and _plain(found[1].arg[0], fn):
        ed = found[0]
        ext = 1 if found[1].op == Oextsw else 2
        index = found[1].arg[0]
    m = Mem()
    m.base = base
    m.index = index
    m.scale = shift | ext << 4
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = m
    chuse(base, +1, fn)
    chuse(index, +1, fn)
    chuse(pr, -1, fn)
    _sweep(pr, d, fn)
    if sd is not None:
        _sweep(shl, sd, fn)
    if ed is not None:
        _sweep(wide, ed, fn)
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


def _widening(di, fn):
    """(kind, x, y) if the product di, in class l, is of two extended words only used here: 1 for both
    signed, 2 for both unsigned, with the words; else None."""
    if di.cls != Kl:
        return None
    e0, e1 = _only(di.arg[0], fn), _only(di.arg[1], fn)
    if e0 is None or e1 is None or e0[1].op != e1[1].op or e0[1].op not in (Oextsw, Oextuw) \
            or not _plain(e0[1].arg[0], fn) or not _plain(e1[1].arg[0], fn):
        return None
    x, y = e0[1].arg[0], e1[1].arg[0]
    kind = 1 if e0[1].op == Oextsw else 2
    e0[1].set(Ins(Onop))
    e1[1].set(Ins(Onop))
    return kind, x, y


def selmadd(i, fn):
    """cimo (ext.isel): "r = add c, m" or "r = sub c, m", with m a product only used here and
    defined in this block, becomes one instruction (see Oamadd in defs). Returns True if so."""
    for n in ((1, 0) if i.op == Oadd else (1,)):
        m = i.arg[n]
        c = i.arg[n ^ 1]
        if not _plain(m, fn) or not _plain(c, fn) or fn.tmp[rval(m)].nuse != 1:
            continue
        found = _defin(m, selpos)
        if found is None or found[1].op != Omul or found[1].cls != i.cls:
            continue
        di = found[1]
        if any(rtype(a) == RCon and log2p(fn.con[rval(a)].i) >= 0 for a in di.arg):
            continue  # a shift
        wide = _widening(di, fn)
        ab = list(di.arg) if wide is None else [wide[1], wide[2]]
        di.set(Ins(Onop))
        x = Mem()
        fn.nmem += 1
        vgrow(fn.mem, fn.nmem)
        fn.mem[fn.nmem - 1] = x
        x.scale = 0 if wide is None else wide[0]
        emit(D.Oamadd if i.op == Oadd else D.Oamsub, i.cls, i.to, c, MEM(fn.nmem - 1))
        iarg = D.insb[D.curi].arg
        fixarg(iarg, 0, i.cls, 0, fn)
        fixarg(ab, 0, i.cls, 0, fn)
        fixarg(ab, 1, i.cls, 0, fn)
        x.base, x.index = ab
        return True
    return False


# aalu: the operation and the kind of shift, in Mem.scale; emit prints them
ALU = {Oadd: 0, Osub: 1, Oand: 2, Oor: 3, Oxor: 4}
ALU_NOT = {Oand: 5, Oor: 6, Oxor: 7}  # bic, orn, eon: the second operand complemented
SH_LSL, SH_LSR, SH_ASR, SH_ROR, SH_SXTW, SH_UXTW = range(6)


def _only(r, fn):
    """(position, instruction) defining r in this block, if this is r's only use."""
    if not _plain(r, fn) or fn.tmp[rval(r)].nuse != 1:
        return None
    return _defin(r, selpos)


def _conbits(r, fn):
    """The value of a constant ref, or None."""
    if rtype(r) != RCon or fn.con[rval(r)].type != CBits:
        return None
    return fn.con[rval(r)].i


def _loaded(r, fn):
    """r is a 32-bit load that an extension of it can join."""
    d = _only(r, fn)
    return d is not None and D.isload(d[1].op)


_phifn = _phis = None


def _accumulator(a, to, fn):
    """a comes from a phi that gets its value back from `to`, through arithmetic: to is on the loop's chain."""
    global _phifn, _phis
    if _phifn is not fn:
        _phifn = fn
        _phis = {}
        for b in fn.rpo[:fn.nblk]:
            p = b.phi
            while p:
                _phis[p.to] = p
                p = p.link

    def walk(r, found):
        """Calls found(r) for what r comes from, a phi being as far as it goes; true if one call is."""
        todo = [r]
        seen = set()
        while todo and len(seen) < 32:
            r = todo.pop()
            if r in seen or rtype(r) != RTmp:
                continue
            seen.add(r)
            if found(r):
                return True
            if r not in _phis:
                d = fn.tmp[rval(r)].def_
                if d is not None and d.op in _CHAINOPS:
                    todo += d.arg[:2]
        return False

    def carried(r):
        p = _phis.get(r)
        return p is not None and any(walk(p.arg[n], lambda x: x == to) for n in range(p.narg))
    return walk(a, carried)


def _shifted(r, k, wantext, fn):
    """(value, kind, amount, instruction) if r is a shift that can be a shifted operand."""
    found = _only(r, fn)
    if found is None:
        return None
    di = found[1]
    if di.cls != k or not _plain(di.arg[0], fn):
        return None
    w = 64 if k == Kl else 32
    if wantext and k == Kl and di.op in (Oextsw, Oextuw):
        if _loaded(di.arg[0], fn):  # ldrsw does the extension for free: s += a[i] 2x slower on Neoverse otherwise
            return None
        return di.arg[0], SH_SXTW if di.op == Oextsw else SH_UXTW, 0, di
    n = _conbits(di.arg[1], fn)
    if n is None:
        return None
    if di.op == Omul:
        l = log2p(n)
        return (di.arg[0], SH_LSL, l, di) if l > 0 else None
    how = {Oshl: SH_LSL, Oshr: SH_LSR, Osar: SH_ASR}.get(di.op)
    if how is None or not 0 < n & (w - 1):
        return None
    if wantext and k == Kl and how == SH_LSL and n <= 4:
        ex = _only(di.arg[0], fn)
        if ex is not None and ex[1].op in (Oextsw, Oextuw) and _plain(ex[1].arg[0], fn) \
                and not _loaded(ex[1].arg[0], fn):
            e = ex[1]
            kind = SH_SXTW if e.op == Oextsw else SH_UXTW
            x = e.arg[0]
            e.set(Ins(Onop))
            chuse(x, 0, fn)
            return x, kind, n, di
    return di.arg[0], how, n & (w - 1), di


def _aalu(i, a, alu, how, n, x, fn):
    """Emit i.to = a <alu> (x <how> n)."""
    m = Mem()
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = m
    m.base = x
    m.scale = alu | how << 4
    m.offset.type = CBits
    m.offset.i = n
    emit(D.Oaalu, i.cls, i.to, a, MEM(fn.nmem - 1))


def selalu(i, fn):
    """cimo (ext.isel): an arithmetic or logic instruction whose operand is a shift or an extension
    only used here (`add x0, x1, x2, lsl #3`, `add x0, x1, w2, sxtw`), or the complement of a value
    (bic, orn, eon); "xor x, -1" is mvn. Returns True if selected."""
    k = i.cls
    if i.op == Oxor and _conbits(i.arg[1], fn) in (-1, 0xffffffff if k == Kw else -1) \
            and _plain(i.arg[0], fn):
        _aalu(i, CON_Z, 6, SH_LSL, 0, i.arg[0], fn)
        return True
    order = ((1, 0),) if i.op == Osub else ((1, 0), (0, 1))
    for bn, an in order:
        a, b = i.arg[an], i.arg[bn]
        if not _plain(a, fn):
            continue
        if i.op in ALU_NOT:
            found = _only(b, fn)
            if found is not None and found[1].op == Oxor and found[1].cls == k \
                    and _conbits(found[1].arg[1], fn) in (-1, 0xffffffff if k == Kw else -1) \
                    and _plain(found[1].arg[0], fn):
                x = found[1].arg[0]
                found[1].set(Ins(Onop))
                _aalu(i, a, ALU_NOT[i.op], SH_LSL, 0, x, fn)
                return True
        # not onto the loop's accumulator: there the other input waits for a folded shift's 2 cycles.
        # Measured, folded / not: s += x << 3 1.82 on Snapdragon X and 0.84 to 0.96 on Neoverse N1, V1 and
        # V2 (LLVM's FeatureALULSLFast), s -= x >> 3 1.82 and 1.19 to 1.69. Anywhere else it never loses:
        # through the shifted value 1.00 and 0.55 to 0.71, off any chain 1.03 and 0.77 to 0.87
        s = _shifted(b, k, i.op in (Oadd, Osub), fn) if selblk.loop <= 1 or not _accumulator(a, i.to, fn) else None
        if s is not None:
            x, how, n, di = s
            di.set(Ins(Onop))
            _aalu(i, a, ALU[i.op], how, n, x, fn)
            return True
    return False


_CHAINOPS = frozenset((Oadd, Osub, Oand, D.Oor, Oxor, D.Ocopy))
_DEF32 = frozenset((Oadd, Osub, Oneg, Omul, Oand, Oor, Oxor, Oshl, Oshr, Osar, Oextub, Oextuh, D.Oextsb, D.Oextsh,
                    D.Oloadub, D.Oloadsb, D.Oloaduh, D.Oloadsh, D.Oloadsw, D.Oloaduw, Oload))


def _insertfield(r, lsb, width, k, w, fn):
    """How r, whose bits are all in the field at lsb, comes from a register: (b, b's lsb, its defining
    instructions, to go), or None. Only the field's bits of b are read, so masks are dropped."""
    d = _only(r, fn)
    low = (1 << width) - 1
    d = d[1] if d is not None else None
    if d is None or (d.cls != k and d.op not in (Oextub, Oextuh)):
        return None
    full = (1 << w) - 1
    if d.op == Oshl and _conbits(d.arg[1], fn) == lsb and lsb and _plain(d.arg[0], fn):
        if lsb + width == w:
            return d.arg[0], 0, [d]
        inner = _only(d.arg[0], fn)
        inner = inner[1] if inner is not None else None
        if inner is None:
            return None
        if inner.op == Oand and inner.cls == k and (_conbits(inner.arg[1], fn) or 0) & full == low \
                and _plain(inner.arg[0], fn):
            return inner.arg[0], 0, [d, inner]
        if (inner.op == Oextub and width == 8 or inner.op == Oextuh and width == 16) and _plain(inner.arg[0], fn):
            return inner.arg[0], 0, [d, inner]
        return None
    if d.op == Oand and (_conbits(d.arg[1], fn) or 0) & full == low << lsb and _plain(d.arg[0], fn):
        inner = _only(d.arg[0], fn)
        inner = inner[1] if inner is not None else None
        if inner is not None and inner.cls == k and _plain(inner.arg[0], fn):
            if inner.op == Oshl and _conbits(inner.arg[1], fn) == lsb and lsb:
                return inner.arg[0], 0, [d, inner]
            s = _conbits(inner.arg[1], fn)
            if inner.op in (Oshr, Osar) and not lsb and s is not None and 0 < s and s + width <= w:
                return inner.arg[0], s, [d, inner]
        return (d.arg[0], 0, [d]) if not lsb else None
    if not lsb and (d.op == Oextub and width == 8 or d.op == Oextuh and width == 16) and _plain(d.arg[0], fn):
        return d.arg[0], 0, [d]
    if not lsb and d.op == Oshr and _conbits(d.arg[1], fn) == w - width and _plain(d.arg[0], fn):
        return d.arg[0], w - width, [d]
    return None


def selbfi(i, fn):
    """(a & ~field) | bits for the field, as LLVM's tryBitfieldInsertOp: one bfi or bfxil."""
    k = i.cls
    w = 64 if k == Kl else 32
    for l, r in ((i.arg[0], i.arg[1]), (i.arg[1], i.arg[0])):
        found = _only(l, fn)
        if found is None or found[1].op != Oand or found[1].cls != k or not _plain(found[1].arg[0], fn):
            continue
        keep = _conbits(found[1].arg[1], fn)
        if keep is None:
            continue
        field = ~keep & ((1 << w) - 1)
        if not field:
            continue
        lsb = 0
        while not field >> lsb & 1:
            lsb += 1
        field >>= lsb
        if field & (field + 1):
            continue
        width = 0
        while field >> width:
            width += 1
        got = _insertfield(r, lsb, width, k, w, fn)
        if got is None:
            continue
        b, blsb, dead = got
        a = found[1].arg[0]
        for d in dead + [found[1]]:
            d.set(Ins(Onop))
        m = Mem()
        fn.nmem += 1
        vgrow(fn.mem, fn.nmem)
        fn.mem[fn.nmem - 1] = m
        m.base = b
        m.scale = blsb
        m.offset.type = CBits
        m.offset.i = lsb | width << 8
        emit(D.Oabfi, k, i.to, a, MEM(fn.nmem - 1))
        return True
    return False


def selbits(i, fn):
    """cimo (ext.isel): bit fields and rotations. Returns True if selected."""
    k = i.cls
    w = 64 if k == Kl else 32
    if i.op == Oand:
        m = _conbits(i.arg[1], fn)
        if m is None:
            return False
        m &= (1 << w) - 1
        if m == 0 or m & (m + 1):
            return False
        width = 0
        while m >> width:
            width += 1
        found = _only(i.arg[0], fn)
        if found is None or found[1].op not in (Oshr, Osar) or found[1].cls != k \
                or not _plain(found[1].arg[0], fn):
            return False
        lsb = _conbits(found[1].arg[1], fn)
        if lsb is None or not 0 < lsb < w or lsb + width > w:
            return False
        x = found[1].arg[0]
        found[1].set(Ins(Onop))
        emit(D.Oaubfx, k, i.to, x, getcon(lsb | width << 8, fn))
        return True
    if i.op == Oshr and k == Kw:  # (unsigned char)x >> n
        b = _conbits(i.arg[1], fn)
        found = _only(i.arg[0], fn)
        if b is not None and found is not None and found[1].op in (Oextub, Oextuh) and _plain(found[1].arg[0], fn):
            bits = 8 if found[1].op == Oextub else 16
            if 0 < b < bits:
                x = found[1].arg[0]
                found[1].set(Ins(Onop))
                emit(D.Oaubfx, k, i.to, x, getcon(b | (bits - b) << 8, fn))
                return True
    if i.op in (Osar, Oshr):
        b = _conbits(i.arg[1], fn)
        found = _only(i.arg[0], fn)
        if b is None or found is None or found[1].op != Oshl or found[1].cls != k \
                or not _plain(found[1].arg[0], fn):
            return False
        a = _conbits(found[1].arg[1], fn)
        if a is None or not 0 < a <= b < w:
            return False
        x = found[1].arg[0]
        found[1].set(Ins(Onop))
        emit(D.Oasbfx if i.op == Osar else D.Oaubfx, k, i.to, x, getcon((b - a) | (w - b) << 8, fn))
        return True
    if i.op == Oor:
        if selbfi(i, fn):
            return True
        for l, r in ((i.arg[0], i.arg[1]), (i.arg[1], i.arg[0])):
            fl, fr = _only(l, fn), _only(r, fn)
            if fl is None or fr is None or fl[1].op != Oshl or fr[1].op != Oshr \
                    or fl[1].cls != k or fr[1].cls != k or fl[1].arg[0] != fr[1].arg[0] \
                    or not _plain(fl[1].arg[0], fn):
                continue
            a, b = _conbits(fl[1].arg[1], fn), _conbits(fr[1].arg[1], fn)
            if a is None or b is None or not 0 < a < w or a + b != w:
                continue
            x = fl[1].arg[0]
            fl[1].set(Ins(Onop))
            fr[1].set(Ins(Onop))
            chuse(x, -1, fn)
            emit(D.Oaror, k, i.to, x, getcon(b, fn))
            return True
    return False


# which load gives a 64-bit extension of a 32-bit load's result by itself
_EXTLOAD = {Oextsw: {Oloadsw: Oloadsw, Oloaduw: Oloadsw, Oloadsb: Oloadsb, Oloadsh: Oloadsh,
                     Oloadub: Oloadub, Oloaduh: Oloaduh},
            Oextuw: {Oloadsw: Oloaduw, Oloaduw: Oloaduw, Oloadub: Oloadub, Oloaduh: Oloaduh},
            D.Oextsb: {Oloadsb: Oloadsb}, D.Oextub: {Oloadub: Oloadub},
            D.Oextsh: {Oloadsh: Oloadsh}, D.Oextuh: {Oloaduh: Oloaduh}}


def selextload(i, fn):
    """cimo (ext.isel), as gcc and LLVM: the extension of a load only used here is the load: ldrsw,
    ldrsb, no uxtb after ldrb... The load stays where it is, so nothing crosses another access; it
    now defines the extension's result."""
    found = _only(i.arg[0], fn)
    if found is None or found[1].cls != Kw:
        return False
    op = _EXTLOAD[i.op].get(found[1].op)
    if op is None:
        return False
    di = found[1]
    di.op = op
    di.cls = i.cls
    di.to = i.to
    return True


def selmulcon(i, fn):
    """cimo (ext.isel), as LLVM: a product with a constant near a power of two, in shifts: 9 * x is
    add x, x, lsl #3, 7 * x is (x << 3) - x, -8 * x is neg lsl #3, 10 * x is 5 * x << 1. A mul takes 3-4
    cycles and a register for the constant. Returns True if selected."""
    x = i.arg[0]
    n = _conbits(i.arg[1], fn)
    if n is None or not _plain(x, fn):
        return False
    k = i.cls
    w = 64 if k == Kl else 32
    n &= (1 << w) - 1
    if n >> (w - 1):
        n -= 1 << w
        if n < 0 and log2p(-n) > 0:  # sub d, xzr, x, lsl #k
            _aalu(i, CON_Z, 1, SH_LSL, log2p(-n), x, fn)
            return True
        return False
    if n < 3:
        return False
    tz = 0
    while not (n >> tz) & 1:
        tz += 1
    m = n >> tz
    if m == 1:  # a power of two: a shift, as sel does
        return False
    up, down = log2p(m - 1), log2p(m + 1)
    if up <= 0 and down <= 0:
        return False
    to = i.to
    if tz:
        t = newtmp('isel', k, fn)
        emit(Oshl, k, to, t, getcon(tz, fn))
        to = t
    i1 = i.copy()
    i1.to = to
    if up > 0:
        chuse(x, +1, fn)
        _aalu(i1, x, 0, SH_LSL, up, x, fn)
    else:
        chuse(x, +1, fn)
        t = newtmp('isel', k, fn)
        emit(Osub, k, to, t, x)
        emit(Oshl, k, t, x, getcon(down, fn))
    return True


def selubfiz(i, fn):
    """cimo (ext.isel): (x & mask) << n as one instruction."""
    n = _conbits(i.arg[1], fn)
    found = _only(i.arg[0], fn)
    if n is None or found is None or found[1].op != Oand or found[1].cls != i.cls \
            or not _plain(found[1].arg[0], fn):
        return False
    w = 64 if i.cls == Kl else 32
    m = _conbits(found[1].arg[1], fn)
    if m is None:
        return False
    m &= (1 << w) - 1
    if m == 0 or m & (m + 1):
        return False
    width = 0
    while m >> width:
        width += 1
    n &= w - 1
    if not 0 < n or n + width > w:
        return False
    x = found[1].arg[0]
    found[1].set(Ins(Onop))
    emit(D.Oaubfiz, i.cls, i.to, x, getcon(n | width << 8, fn))
    return True


def selcincadd(i, fn):
    """cimo (ext.isel): a + (b < c), as cinc. The comparison, or its 64-bit extension, is only used
    here. Returns True if selected."""
    k = i.cls
    for an, bn in ((0, 1), (1, 0)):
        a, c = i.arg[an], i.arg[bn]
        if not _plain(a, fn):
            continue
        found = _only(c, fn)
        if found is None:
            continue
        dc = found[1]
        if dc.op in (Oextuw, Oextsw) and k == Kl:  # of a 0 or 1, so either extension
            inner = _only(dc.arg[0], fn)
            if inner is None:
                continue
            outer, dc = dc, inner[1]
        else:
            outer = None
        cm = iscmp(dc.op)
        if not cm or (outer is None and dc.cls != k):
            continue
        ck, cc = cm
        args = list(dc.arg)
        if outer is not None:
            outer.set(Ins(Onop))
        dc.set(Ins(Onop))
        x = Mem()
        fn.nmem += 1
        vgrow(fn.mem, fn.nmem)
        fn.mem[fn.nmem - 1] = x
        x.offset.type = CBits
        x.base = a
        x.scale = CSINC | 1 << 4
        chuse(a, +1, fn)
        emit(D.Oacsel, k, i.to, a, MEM(fn.nmem - 1))
        if selcmp(args, ck, fn):
            cc = cmpop(cc)
        x.offset.i = cc
        return True
    return False


def selneg(i, fn):
    """cimo (ext.isel): the negation of a product only used here: mneg, fnmul."""
    found = _only(i.arg[0], fn)
    if found is None or found[1].op != Omul or found[1].cls != i.cls:
        return False
    di = found[1]
    if KBASE(i.cls) == 1:
        a, b = list(di.arg)
        di.set(Ins(Onop))
        emit(D.Oafnmul, i.cls, i.to, a, b)
        iarg = D.insb[D.curi].arg
        fixarg(iarg, 0, i.cls, 0, fn)
        fixarg(iarg, 1, i.cls, 0, fn)
        return True
    if any(rtype(a) == RCon and log2p(fn.con[rval(a)].i) >= 0 for a in di.arg):
        return False
    ab = list(di.arg)
    di.set(Ins(Onop))
    x = Mem()
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = x
    emit(D.Oamsub, i.cls, i.to, CON_Z, MEM(fn.nmem - 1))
    fixarg(ab, 0, i.cls, 0, fn)
    fixarg(ab, 1, i.cls, 0, fn)
    x.base, x.index = ab
    return True


# acsel's kinds, in Mem.scale, which also holds whether the condition is inverted (<< 4)
CSEL, CSINC, CSINV, CSNEG = range(4)


def _cselform(a, b, k, fn):
    """(first operand, second operand, kind, invert) for "a if cc else b": cinc, cneg, cinv when an
    arm is the other plus one, negated or complemented, as clang selects; xzr for a zero."""
    if KBASE(k) == 1:
        return a, b, CSEL, 0
    for x, y, inv in ((a, b, 1), (b, a, 0)):  # x is the changed value, y the plain one
        found = _only(x, fn)
        if found is None or found[1].cls != k or found[1].arg[0] != y or not _plain(y, fn):
            continue
        di = found[1]
        kind = None
        if di.op == Oadd and _conbits(di.arg[1], fn) == 1:
            kind = CSINC
        elif di.op == Oneg:
            kind = CSNEG
        elif di.op == Oxor and _conbits(di.arg[1], fn) in (-1, 0xffffffff if k == Kw else -1):
            kind = CSINV
        if kind is not None:
            di.set(Ins(Onop))
            chuse(y, +1, fn)
            # csinc d, y, y, c is: y if c, else y + 1. So c must hold when y is wanted
            return y, y, kind, inv
    m = 0xffffffff if k == Kw else M64
    if _conbits(b, fn) == 0 and (_conbits(a, fn) or 0) & m == m:
        return CON_Z, CON_Z, CSINV, 1  # csetm: csinv d, xzr, xzr, !cc
    if _conbits(a, fn) == 0 and (_conbits(b, fn) or 0) & m == m:
        return CON_Z, CON_Z, CSINV, 0
    if _conbits(b, fn) == 0:
        return CON_Z, a, CSEL, 1  # csel d, xzr, a, !cc
    return a, b, CSEL, 0


def selcsel(i, fn):
    """cimo (ext.isel): "sel0 c; r = sel1 a, b" as a comparison and an acsel. A condition computed
    by a comparison in this block is compared again right here, for the flags; the comparison
    itself goes with sel0, once it has no other use. a and b come from before the comparison."""
    d = selpos
    while selblk.ins[d].op != D.Osel0:
        d -= 1
    c = selblk.ins[d].arg[0]
    cd = _defin(c, d) if rtype(c) == RTmp else None
    cm = iscmp(cd[1].op) if cd is not None else None
    k = i.cls
    x = Mem()
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = x
    x.offset.type = CBits
    a, b0, kind, inv = _cselform(i.arg[0], i.arg[1], k, fn)
    x.scale = kind | inv << 4
    emit(D.Oacsel, k, i.to, a, MEM(fn.nmem - 1))
    iarg = D.insb[D.curi].arg
    if _conbits(a, fn) != 0 or KBASE(k) == 1:
        fixarg(iarg, 0, k, 0, fn)
    b = [b0]
    if _conbits(b0, fn) != 0 or KBASE(k) == 1:
        fixarg(b, 0, k, 0, fn)
    x.base = b[0]
    if cm:
        ck, cc = cm
        args = list(cd[1].arg)
        for a in args:
            chuse(a, +1, fn)
        if selcmp(args, ck, fn):
            cc = cmpop(cc)
    else:
        cc = None
        if cd is not None and cd[1].op in (Oand, Oor) and cd[1].cls == Kw and fn.tmp[rval(c)].nuse == 1:
            cc = _ccmp(selblk, cd[1], fn, keep=True)
        if cc is None:  # each sel1 reads c, where sel0 counts one use for all of them
            cc = Cine
            chuse(c, +1, fn)
            selcmp([c, CON_Z], Kw, fn)
    x.offset.i = cc


def callable(r, fn):
    if rtype(r) == RTmp:
        return 1
    if rtype(r) == RCon:
        c = fn.con[rval(r)]
        if c.type == CAddr:
            if c.i == 0:
                return 1
    return 0


_qdest = [R]  # what the q instruction that takes its third operand from the one before writes


def sel(i, fn):
    """i is a copy (C passes Ins by value)."""
    if i.op in (D.Oqfma0, D.Oqsel0):  # cimo (ext.vector): fmla and bsl keep their third operand in the destination
        d = _qdest[0]
        emit(i.op, D.Kq, R, d, R)
        emit(Ocopy, D.Kq, d, i.arg[0], R)
        return
    if i.op in (D.Oqfma1s, D.Oqfma1d, D.Oqsel1):
        _qdest[0] = i.to
    if D.isqop(i.op):  # cimo (ext.vector): registers, and a plain address
        emiti(i)
        iarg = D.insb[D.curi].arg
        for n in (0, 1):
            if iarg[n] != R and argcls(i, n) != D.Kx and not (i.op >= D.Oqshlb and i.op <= D.Oqsarl and n == 1) \
                    and not (i.op >= D.Oqgetb and i.op <= D.Oqgetd and n == 1):
                fixarg(iarg, n, argcls(i, n), 0, fn)
        return
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
    if i.op == D.Osel1:
        selcsel(i, fn)
        return
    if i.op == D.Osel0:  # selected with its sel1s
        c = i.arg[0]
        if rtype(c) == RTmp:
            chuse(c, -1, fn)
            found = _defin(c, selpos)
            if found is not None and iscmp(found[1].op):
                _sweep(c, found[0], fn)
            elif found is not None and found[1].op in (Oand, Oor) and fn.tmp[rval(c)].nuse == 0:
                parts = list(found[1].arg)  # a condition for ccmp, and the comparisons it read
                _sweep(c, found[0], fn)
                for r in parts:
                    d = _defin(r, found[0]) if rtype(r) == RTmp else None
                    if d is not None and iscmp(d[1].op):
                        _sweep(r, d[0], fn)
        return
    if i.op == Ofma0:
        return  # selected with its fma1
    immarg = False
    if _ext.isel:
        if i.op in (Oadd, Osub) and KBASE(i.cls) == 0 and selmadd(i, fn):
            return
        if i.op == Oadd and KBASE(i.cls) == 0 and selcincadd(i, fn):
            return
        if i.op == Omul and KBASE(i.cls) == 0:
            if selmulcon(i, fn):
                return
            wide = _widening(i, fn)
            if wide is not None:
                emit(D.Oasmull if wide[0] == 1 else D.Oaumull, Kl, i.to, wide[1], wide[2])
                return
        if i.op == Oshl and KBASE(i.cls) == 0 and selubfiz(i, fn):
            return
        if i.op in (Oand, Oor) and i.cls == Kw:  # a value from two comparisons, as cset after ccmp
            emit(Oflag, Kw, i.to, R, R)
            flag = D.insb[D.curi]
            cc = _ccmp(selblk, i, fn)
            if cc is not None:
                flag.op += cc
                return
            D.curi += 1
        if i.op in ALU and KBASE(i.cls) == 0:
            if selbits(i, fn) or selalu(i, fn):
                return
        elif i.op in (Oshr, Osar) and selbits(i, fn):
            return
        elif i.op == Oneg and selneg(i, fn):
            return
        elif i.op in _EXTLOAD and KBASE(i.cls) == 0 and (i.cls == Kl or i.op not in (Oextsw, Oextuw)) \
                and selextload(i, fn):
            return
        if i.op == Oextuw and i.cls == Kl and _plain(i.arg[0], fn):
            # writing a w register clears the top of the x register (LLVM's isDef32, GCC's ree)
            found = _defin(i.arg[0], selpos)
            if found is not None and found[1].cls == Kw and found[1].op in _DEF32:
                i.op = Ocopy
        if isstore(i.op) and _conbits(i.arg[0], fn) == 0:  # wzr, xzr; a float's zero too
            if i.op == Ostores:
                i.op = Ostorew
            elif i.op == Ostored:
                i.op = Ostorel
            foldaddr(i.arg, 1, accsize(i), fn)
            foldindex(i.arg, 1, accsize(i), fn)
            emiti(i)
            iarg = D.insb[D.curi].arg
            fixarg(iarg, 1, Kl, 0, fn)
            return
    if KBASE(i.cls) == 0 and rtype(i.arg[1]) == RCon \
            and fn.con[rval(i.arg[1])].type == CBits:
        c = fn.con[rval(i.arg[1])]
        if _ext.isel and i.op in (Oshl, Osar, Oshr):
            i.arg[1] = getcon(c.i & (63 if i.cls == Kl else 31), fn)
            immarg = True
        elif i.op in (Oadd, Osub):
            im, n = imm(c, i.cls)
            if im in (Inlo12, Inhi12):
                i.op = Osub if i.op == Oadd else Oadd
                i.arg[1] = getcon(n, fn)
                immarg = True
            elif im in (Iplo12, Iphi12):
                immarg = True
        elif i.op in (Oand, Oor, Oxor):
            n = c.i
            if _ext.isel:
                immarg = bool(arm64_logimm(n, i.cls))  # emitf writes these whole
            elif i.cls == Kl and ((n >> 24) or (n & ~0xfff) == 0) \
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
        if _ext.isel:
            foldindex(i.arg, 0, accsize(i), fn)
    if isstore(i.op):
        foldaddr(i.arg, 1, accsize(i), fn)
        if _ext.isel:
            foldindex(i.arg, 1, accsize(i), fn)
    emiti(i)
    iarg = D.insb[D.curi].arg  # fixarg() can change curi
    fixarg(iarg, 0, argcls(i, 0), 0, fn)
    fixarg(iarg, 1, argcls(i, 1), 0, fn)


# flags that make each integer condition false, for ccmp when the first comparison fails
_FALSEFLAGS = {D.Cieq: 0, D.Cine: 4, D.Cisge: 8, D.Cislt: 0, D.Cisgt: 4, D.Cisle: 0,
               D.Ciuge: 0, D.Ciult: 2, D.Ciugt: 0, D.Ciule: 2}
_NZCVTRUE = {D.Cieq: 4, D.Cine: 0, D.Cisge: 0, D.Cislt: 8, D.Cisgt: 0, D.Cisle: 4,
             D.Ciuge: 2, D.Ciult: 0, D.Ciugt: 2, D.Ciule: 4}


def _ccmp(b, ir, fn, keep=False):
    """cimo (ext.isel), LLVM's AArch64ConditionalCompares: (a < b) & (c < d) or (a < b) | (c < d), both
    comparisons only used here, as cmp; ccmp, whose flags then answer. For & the second comparison runs
    if the first held, else the flags are set so that its condition is false; for | it runs if the
    first failed, else the flags make it true. Emits them and returns the final condition, or None.
    keep: the comparisons stay until the and or or goes, as other selects may read it (see Osel0)."""
    op = ir.op

    def cmpdef(r):
        if not _plain(r, fn) or fn.tmp[rval(r)].nuse != 1:
            return None
        for n in range(b.nins):
            if b.ins[n].to == r:
                return b.ins[n] if iscmp(b.ins[n].op) else None
        return None
    d0, d1 = cmpdef(ir.arg[0]), cmpdef(ir.arg[1])
    if d0 is None or d1 is None:
        return None
    c0, c1 = iscmp(d0.op), iscmp(d1.op)
    if KBASE(c1[0]) != 0:
        d0, d1, c0, c1 = d1, d0, c1, c0
    if KBASE(c1[0]) != 0:
        return None
    k, cc1 = c1
    a, y = d1.arg
    if rtype(a) == RCon:
        a, y = y, a
        cc1 = cmpop(cc1)
    if not _plain(a, fn):
        return None
    n = _conbits(y, fn)
    if n is not None and not 0 <= (n & M64 if k == Kl else n & 0xffffffff) <= 31:
        n = None
    if n is None and not _plain(y, fn):
        return None
    m = Mem()
    fn.nmem += 1
    vgrow(fn.mem, fn.nmem)
    fn.mem[fn.nmem - 1] = m
    m.offset.type = CBits
    if n is not None:
        m.base = R
        m.offset.i = n & 31
    else:
        m.base = y
    args0 = list(d0.arg)
    cc0 = c0[1]
    emit(D.Oaccmp, k, R, a, MEM(fn.nmem - 1))
    if selcmp(args0, c0[0], fn):
        cc0 = cmpop(cc0)
    # scale: the flags, the first comparison's condition, and whether ccmp runs when it fails
    if op == Oand:
        m.scale = _FALSEFLAGS[cc1] | cc0 << 4
    else:
        m.scale = _NZCVTRUE[cc1] | cc0 << 4 | 1 << 12
    if keep:
        for r in list(d0.arg) + list(d1.arg):
            chuse(r, +1, fn)
    else:
        d0.set(Ins(Onop))
        d1.set(Ins(Onop))
    return cc1


def _tstjump(b, r, k, cc, fn):
    """cimo (ext.isel): a jump on (x & y) == 0 or != 0, the and (r) only used by it: tst, or tbz and
    tbnz for one bit (emit._cbz). seljmp runs before the block's instructions are selected, which is
    why the and is simply deleted. Returns True if selected."""
    if not _plain(r, fn) or fn.tmp[rval(r)].nuse != 1:
        return False
    d = None
    for n in range(b.nins):
        if b.ins[n].to == r:
            d = b.ins[n]
    if d is None or d.op != Oand or d.cls != k or not _plain(d.arg[0], fn):
        return False
    y = d.arg[1]
    n = _conbits(y, fn)
    if n is not None:
        n &= (1 << (64 if k == Kl else 32)) - 1
        if n == 0 or not arm64_logimm(n, k):
            return False
        y = getcon(s64(n), fn)
    elif not _plain(y, fn):
        return False
    x = d.arg[0]
    d.set(Ins(Onop))
    emit(D.Oatst, k, R, x, y)
    b.jmp.type = Jjf + cc
    return True


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
    if _ext.isel and ir is not None and use == 1:
        if ir.op in (Oand, Oor) and ir.cls == Kw:
            cc = _ccmp(b, ir, fn)
            if cc is not None:
                ir.set(Ins(Onop))
                b.jmp.type = Jjf + cc
                return
        if ir.op == Oand and ir.cls == Kw and _tstjump(b, r, Kw, Cine, fn):  # jnz tests a word
            return
        if cm and cm[1] in (Cieq, Cine) and KBASE(cm[0]) == 0 and _conbits(ir.arg[1], fn) == 0 \
                and _tstjump(b, ir.arg[0], cm[0], cm[1], fn):
            ir.set(Ins(Onop))
            return
    if cm:
        ck, cc = cm
        if selcmp(ir.arg, ck, fn):
            cc = cmpop(cc)
        b.jmp.type = Jjf + cc
        ir.set(Ins(Onop))
    else:
        selcmp([r, CON_Z], Kw, fn)
        b.jmp.type = Jjfine


def _reads(i, r, fn):
    for a in i.arg:
        if req(a, r):
            return True
        if rtype(a) == D.RMem and (req(fn.mem[rval(a)].base, r) or req(fn.mem[rval(a)].index, r)):
            return True
    return False


def sinkinc(fn):
    """cimo (ext.isel): "q = p + c" after the accesses at p that follow it, unless they read q. p then
    dies where q starts, so they share a register (no copy in the loop) and emit._writeback can merge
    the add into the access: *p++ reads p after its add was computed."""
    b = fn.start
    while b:
        n = b.nins - 2
        while n >= 0:
            i = b.ins[n]
            c = _conbits(i.arg[1], fn) if i.op in (Oadd, Osub) and i.cls == Kl else None
            if c is None or not -256 <= c <= 255 or not _plain(i.arg[0], fn) or not _plain(i.to, fn):
                n -= 1
                continue
            p, q = i.arg[0], i.to
            last = None
            for m in range(n + 1, b.nins):
                j = b.ins[m]
                if _reads(j, q, fn):
                    break
                if D.isload(j.op) and req(j.arg[0], p) or D.isstore(j.op) and req(j.arg[1], p):
                    last = m
            if last is not None:
                b.ins[n:last + 1] = b.ins[n + 1:last + 1] + [b.ins[n]]
            n -= 1
        b = b.link


def qacrosscalls(fn):
    """cimo (ext.vector): a call keeps at most 64 bits of any FP register, so a 128-bit temporary live across one
    goes to memory: stored where it's defined, loaded before each use. Needs rpo, use; breaks use."""
    from ..live import filllive
    if not any(i.cls == D.Kq and rtype(i.to) == RTmp for b in fn.rpo[:fn.nblk] for i in b.ins[:b.nins]) \
            and not any(p.cls == D.Kq for b in fn.rpo[:fn.nblk] for p in _philist(b)):
        return
    filllive(fn)
    across = set()
    for b in fn.rpo[:fn.nblk]:
        live = set(t for t in _elems(b.out) if fn.tmp[t].cls == D.Kq)
        for r in (b.jmp.arg,):
            if rtype(r) == RTmp and rval(r) >= Tmp0 and fn.tmp[rval(r)].cls == D.Kq:
                live.add(rval(r))
        for x in reversed(b.ins[:b.nins]):
            if rtype(x.to) == RTmp:
                live.discard(rval(x.to))
            if x.op == Ocall:
                across |= live
            for a in x.arg[:2]:
                if rtype(a) == RTmp and rval(a) >= Tmp0 and fn.tmp[rval(a)].cls == D.Kq:
                    live.add(rval(a))
    if not across:
        return
    slots = {}
    allocs = []
    for t in sorted(across):
        s = newtmp('qs', Kl, fn)
        slots[t] = s
        allocs.append(Ins(Oalloc + 2, Kl, s, getcon(_ext.qbytes(), fn), R))
    for b in fn.rpo[:fn.nblk]:
        new = []
        p = b.phi
        while p:
            if rval(p.to) in slots:
                new.append(Ins(D.Oqstore, D.Kq, R, p.to, slots[rval(p.to)]))
            for n in range(p.narg):
                a = p.arg[n]
                if rtype(a) == RTmp and rval(a) in slots:
                    q = p.blk[n]
                    r = newtmp('qr', D.Kq, fn)
                    q.ins = q.ins[:q.nins] + [Ins(D.Oqload, D.Kq, r, slots[rval(a)], R)]
                    q.nins = len(q.ins)
                    p.arg[n] = r
            p = p.link
        b.ins = new + b.ins[:b.nins]
        b.nins = len(b.ins)
    for b in fn.rpo[:fn.nblk]:
        new = []
        for x in b.ins[:b.nins]:
            isslotop = x.op == D.Oqload and x.arg[0] in slots.values() or x.op == D.Oqstore and x.arg[1] in slots.values()
            if not isslotop:
                for n in (0, 1):
                    a = x.arg[n]
                    if rtype(a) == RTmp and rval(a) in slots:
                        r = newtmp('qr', D.Kq, fn)
                        new.append(Ins(D.Oqload, D.Kq, r, slots[rval(a)], R))
                        x.arg[n] = r
            new.append(x)
            if not isslotop and rtype(x.to) == RTmp and rval(x.to) in slots:
                new.append(Ins(D.Oqstore, D.Kq, R, x.to, slots[rval(x.to)]))
        b.ins = new
        b.nins = len(new)
        a = b.jmp.arg
        if rtype(a) == RTmp and rval(a) in slots:
            die("a 128-bit value in a jump")
    s0 = fn.start
    s0.ins = [x for x in s0.ins[:s0.nins] if D.ispar(x.op)] + allocs + [x for x in s0.ins[:s0.nins] if not D.ispar(x.op)]
    s0.nins = len(s0.ins)


def _philist(b):
    p = b.phi
    while p:
        yield p
        p = p.link


def _elems(bs):
    from ..util import bselems
    return bselems(bs, Tmp0)


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

    if _ext.isel:
        sinkinc(fn)
    if _ext.ldp and not fn.hasvolatile:
        from .pair import schedule
        schedule(fn)

    if D.debug['I']:
        sys.stderr.write("\n> After instruction selection:\n")
        printfn(fn, sys.stderr)
