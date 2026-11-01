"""Port of QBE amd64/emit.c."""

from .. import defs as D
from ..defs import (R, RTmp, RCon, RSlot, RMem, CBits, CAddr, CUndef, Con, SGlo, SThr, SExt,
                    SExtThr, Kw, Kl, Ks, Kd, KWIDE, KBASE, NCmpI, NCmp, Ins, TMP, BIT,
                    rtype, rval, rsval, optab, NOp, isxsel,
                    Cieq, Cine, Cisge, Cisgt, Cisle, Cislt, Ciuge, Ciugt, Ciule, Ciult,
                    Cfle, Cflt, Cfgt, Cfge, Cfo, Cfuo, Jhlt, Jret0, Jjmp, Jjf,
                    Oadd, Osub, Oand, Oor, Oxor, Osar, Oshr, Oshl, Omul, Odiv, Ostorel,
                    Ostorew, Ostoreh, Ostoreb, Ostores, Ostored, Oload, Oloadsw, Oloaduw,
                    Oloadsh, Oloaduh, Oloadsb, Oloadub, Oextsw, Oextuw, Oextsh, Oextuh,
                    Oextsb, Oextub, Oexts, Otruncd, Ostosi, Odtosi, Oswtof, Osltof, Ocast,
                    Oaddr, Oswap, Osign, Oxdiv, Oxidiv, Oxcmp, Oxtest, Oflag, Oflagfeq,
                    Oflagfne, Onop, Oneg, Ocopy, Ocall, Osalloc, Odbgloc, Oxsel)
from ..util import die, isreg, str_
from ..emit import emitfnlnk, emitdbgloc, elf_emitfnfin, stashbits
from .regs import (RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8, R9, R10, R11, R12, R13, R14,
                   R15, XMM0, XMM15, NCLR_SYSV, NCLR_WIN)


class E:
    __slots__ = ('f', 'fn', 'fp', 'fsz', 'nclob')

    def __init__(self, f, fn):
        self.f = f
        self.fn = fn
        self.fp = 0
        self.fsz = 0
        self.nclob = 0


# (cmp, set/jump suffix, negated suffix)
_CMP = [
    (Ciule, "be", "a"),
    (Ciult, "b", "ae"),
    (Cisle, "le", "g"),
    (Cislt, "l", "ge"),
    (Cisgt, "g", "le"),
    (Cisge, "ge", "l"),
    (Ciugt, "a", "be"),
    (Ciuge, "ae", "b"),
    (Cieq, "z", "nz"),
    (Cine, "nz", "z"),
    (NCmpI + Cfle, "?", "?"),
    (NCmpI + Cflt, "?", "?"),
    (NCmpI + Cfgt, "a", "be"),
    (NCmpI + Cfge, "ae", "b"),
    (NCmpI + Cfo, "np", "p"),
    (NCmpI + Cfuo, "p", "np"),
]

SLong, SWord, SShort, SByte = 0, 1, 2, 3
Ki = -1  # matches Kw and Kl
Ka = -2  # matches all classes

# Instruction format strings:
#
# if the format string starts with -, the instruction
# is assumed to be 3-address and is put in 2-address
# mode using an extra mov if necessary
#
# if the format string starts with +, the same as the
# above applies, but commutativity is also assumed
#
# %k  is used to set the class of the instruction,
#     it'll expand to "l", "q", "ss", "sd", depending
#     on the instruction class
# %0  designates the first argument
# %1  designates the second argument
# %=  designates the result
#
# if %k is not used, a prefix to 0, 1, or = must be
# added, it can be:
#   M - memory reference
#   L - long  (64 bits)
#   W - word  (32 bits)
#   H - short (16 bits)
#   B - byte  (8 bits)
#   S - single precision float
#   D - double precision float
omap = [
    (Oadd, Ka, "+add%k %1, %="),
    (Osub, Ka, "-sub%k %1, %="),
    (Oand, Ki, "+and%k %1, %="),
    (Oor, Ki, "+or%k %1, %="),
    (Oxor, Ki, "+xor%k %1, %="),
    (Osar, Ki, "-sar%k %B1, %="),
    (Oshr, Ki, "-shr%k %B1, %="),
    (Oshl, Ki, "-shl%k %B1, %="),
    (Omul, Ki, "+imul%k %1, %="),
    (Omul, Ks, "+mulss %1, %="),
    (Omul, Kd, "+mulsd %1, %="),
    (Odiv, Ka, "-div%k %1, %="),
    (Ostorel, Ka, "movq %L0, %M1"),
    (Ostorew, Ka, "movl %W0, %M1"),
    (Ostoreh, Ka, "movw %H0, %M1"),
    (Ostoreb, Ka, "movb %B0, %M1"),
    (Ostores, Ka, "movss %S0, %M1"),
    (Ostored, Ka, "movsd %D0, %M1"),
    (Oload, Ka, "mov%k %M0, %="),
    (Oloadsw, Kl, "movslq %M0, %L="),
    (Oloadsw, Kw, "movl %M0, %W="),
    (Oloaduw, Ki, "movl %M0, %W="),
    (Oloadsh, Ki, "movsw%k %M0, %="),
    (Oloaduh, Ki, "movzw%k %M0, %="),
    (Oloadsb, Ki, "movsb%k %M0, %="),
    (Oloadub, Ki, "movzb%k %M0, %="),
    (Oextsw, Kl, "movslq %W0, %L="),
    (Oextuw, Kl, "movl %W0, %W="),
    (Oextsh, Ki, "movsw%k %H0, %="),
    (Oextuh, Ki, "movzw%k %H0, %="),
    (Oextsb, Ki, "movsb%k %B0, %="),
    (Oextub, Ki, "movzb%k %B0, %="),

    (Oexts, Kd, "cvtss2sd %0, %="),
    (Otruncd, Ks, "cvtsd2ss %0, %="),
    (Ostosi, Ki, "cvttss2si%k %0, %="),
    (Odtosi, Ki, "cvttsd2si%k %0, %="),
    (Oswtof, Ka, "cvtsi2%k %W0, %="),
    (Osltof, Ka, "cvtsi2%k %L0, %="),
    (Ocast, Ki, "movq %D0, %L="),
    (Ocast, Ka, "movq %L0, %D="),

    (Oaddr, Ki, "lea%k %M0, %="),
    (Oswap, Ki, "xchg%k %0, %1"),
    (Osign, Kl, "cqto"),
    (Osign, Kw, "cltd"),
    (Oxdiv, Ki, "div%k %0"),
    (Oxidiv, Ki, "idiv%k %0"),
    (Oxcmp, Ks, "ucomiss %S0, %S1"),
    (Oxcmp, Kd, "ucomisd %D0, %D1"),
    (Oxcmp, Ki, "cmp%k %0, %1"),
    (Oxtest, Ki, "test%k %0, %1"),
]
omap += [(Oflag + c, Ki, "set" + s + " %B=\n\tmovzb%k %B=, %=") for c, s, _ in _CMP]
omap += [
    (Oflagfeq, Ki, "setz %B=\n\tmovzb%k %B=, %="),
    (Oflagfne, Ki, "setnz %B=\n\tmovzb%k %B=, %="),
]

cmov = {}
for _c, _s0, _s1 in _CMP:
    cmov[_c] = ("cmov" + _s0 + " %0, %=", "cmov" + _s1 + " %1, %=")
del _c, _s0, _s1

_ctoa = {c: (s, n) for c, s, n in _CMP}

rname = {
    RAX: ("rax", "eax", "ax", "al"),
    RBX: ("rbx", "ebx", "bx", "bl"),
    RCX: ("rcx", "ecx", "cx", "cl"),
    RDX: ("rdx", "edx", "dx", "dl"),
    RSI: ("rsi", "esi", "si", "sil"),
    RDI: ("rdi", "edi", "di", "dil"),
    RBP: ("rbp", "ebp", "bp", "bpl"),
    RSP: ("rsp", "esp", "sp", "spl"),
    R8: ("r8", "r8d", "r8w", "r8b"),
    R9: ("r9", "r9d", "r9w", "r9b"),
    R10: ("r10", "r10d", "r10w", "r10b"),
    R11: ("r11", "r11d", "r11w", "r11b"),
    R12: ("r12", "r12d", "r12w", "r12b"),
    R13: ("r13", "r13d", "r13w", "r13b"),
    R14: ("r14", "r14d", "r14w", "r14b"),
    R15: ("r15", "r15d", "r15w", "r15b"),
}


def slot(r, e):
    s = rsval(r)
    assert s <= e.fn.slot
    # specific to NAlign == 3
    if s < 0:
        if e.fp == RSP:
            return 4 * -s - 8 + e.fsz + e.nclob * 8
        else:
            return 4 * -s
    elif e.fp == RSP:
        return 4 * s + e.nclob * 8
    elif e.fn.vararg:
        if D.T.windows:
            return -4 * (e.fn.slot - s)
        else:
            return -176 + -4 * (e.fn.slot - s)
    else:
        return -4 * (e.fn.slot - s)


def emitcon(con, e):
    if con.type == CAddr:
        l = str_(con.sym.id)
        p = "" if l[0] == '"' else D.T.assym
        if con.sym.type == SThr:
            assert not D.T.apple
            e.f.write("%%fs:%s%s@tpoff" % (p, l))
        else:
            assert (con.sym.type & ~SExt) == SGlo
            e.f.write("%s%s" % (p, l))
        if con.i:
            e.f.write("%+d" % con.i)
    elif con.type == CBits:
        e.f.write("%d" % con.i)
    else:
        die("unreachable")


def regtoa(reg, sz):
    assert reg <= XMM15
    if reg >= XMM0:
        return "xmm%d" % (reg - XMM0)
    else:
        return rname[reg][sz]


def getarg(c, i):
    if c == '0':
        return i.arg[0]
    elif c == '1':
        return i.arg[1]
    elif c == '=':
        return i.to
    else:
        die("invalid arg letter %c", c)


def emitcopy(r1, r2, k, e):
    icp = Ins(Ocopy, k, r1, r2)
    emitins(icp, e)


_clstoa = ("l", "q", "ss", "sd")


def emitf(s, i, e):
    """i is modified (the caller's copy, as in C)."""
    f = e.f
    if s[0] in '+-':
        if s[0] == '+':
            if i.arg[1] == i.to:
                ref = i.arg[0]
                i.arg[0] = i.arg[1]
                i.arg[1] = ref
        # fall through
        assert (i.arg[1] != i.to or i.arg[0] == i.to), "cannot convert to 2-address"
        emitcopy(i.to, i.arg[0], i.cls, e)
        s = s[1:]

    out = ['\t']
    n = len(s)
    p = 0
    while True:
        # Next:
        q = s.find('%', p)
        if q < 0:
            out.append(s[p:])
            out.append('\n')
            f.write(''.join(out))
            return
        out.append(s[p:q])
        c = s[q + 1]
        p = q + 2
        if c == '%':
            out.append('%')
            continue
        if c == 'k':
            out.append(_clstoa[i.cls])
            continue
        if c in '0=1':
            sz = SLong if KWIDE(i.cls) else SWord
            p -= 1
        elif c in 'DS':
            sz = SLong  # does not matter for floats
        elif c == 'L':
            sz = SLong
        elif c == 'W':
            sz = SWord
        elif c == 'H':
            sz = SShort
        elif c == 'B':
            sz = SByte
        elif c == 'M':
            c = s[p]
            p += 1
            ref = getarg(c, i)
            t = rtype(ref)
            if t == RMem:
                _mem(ref, e, out)
            elif t == RSlot:
                out.append("%d(%%%s)" % (slot(ref, e), regtoa(e.fp, SLong)))
            elif t == RCon:
                off = e.fn.con[rval(ref)].copy()
                f.write(''.join(out))
                out = []
                emitcon(off, e)
                if off.type == CAddr:
                    if off.sym.type != SThr:
                        out.append("(%rip)")
            elif t == RTmp:
                assert isreg(ref)
                out.append("(%%%s)" % regtoa(rval(ref), SLong))
            else:
                die("unreachable")
            continue
        else:
            die("invalid format specifier %%%c", c)
        # Ref:
        c = s[p]
        p += 1
        ref = getarg(c, i)
        t = rtype(ref)
        if t == RTmp:
            assert isreg(ref)
            out.append("%%%s" % regtoa(rval(ref), sz))
        elif t == RSlot:
            out.append("%d(%%%s)" % (slot(ref, e), regtoa(e.fp, SLong)))
        elif t == RMem:
            _mem(ref, e, out)
        elif t == RCon:
            out.append('$')
            f.write(''.join(out))
            out = []
            emitcon(e.fn.con[rval(ref)], e)
        else:
            die("unreachable")


def _mem(ref, e, out):
    m = e.fn.mem[rval(ref)]
    if rtype(m.base) == RSlot:
        off = Con(CBits, slot(m.base, e))
        from ..util import addcon
        addcon(m.offset, off, 1)
        m.base = TMP(e.fp)
    if m.offset.type != CUndef:
        e.f.write(''.join(out))
        del out[:]
        emitcon(m.offset, e)
    out.append('(')
    if m.base != R:
        out.append("%%%s" % regtoa(rval(m.base), SLong))
    elif m.offset.type == CAddr:
        out.append("%rip")
    if m.index != R:
        out.append(", %%%s, %d" % (regtoa(rval(m.index), SLong), m.scale))
    out.append(')')


negmask = {Ks: 0x80000000, Kd: 0x8000000000000000}


def _table(i, e):
    # most instructions are just pulled out of
    # the table omap[], some special cases are
    # detailed below
    for op, cls, fmt in omap:
        if op == i.op:
            if cls == i.cls or (cls == Ki and KBASE(i.cls) == 0) or cls == Ka:
                emitf(fmt, i, e)
                return
    die("no match for %s(%c)", optab[i.op].name, "wlsd"[i.cls])


def emitins(i, e):
    """i: passed by value (copied here)."""
    i = i.copy()
    f = e.f
    op = i.op
    if op == Onop:
        # just do nothing for nops, they are inserted
        # by some passes
        pass
    elif op == Omul:
        # here, we try to use the 3-addresss form
        # of multiplication when possible
        if rtype(i.arg[1]) == RCon:
            r = i.arg[0]
            i.arg[0] = i.arg[1]
            i.arg[1] = r
        if (KBASE(i.cls) == 0  # only available for ints
                and rtype(i.arg[0]) == RCon
                and rtype(i.arg[1]) == RTmp):
            emitf("imul%k %0, %1, %=", i, e)
        else:
            _table(i, e)
    elif op == Osub:
        # we have to use the negation trick to handle
        # some 3-address subtractions
        if i.to == i.arg[1] and i.arg[0] != i.to:
            ineg = Ins(Oneg, i.cls, i.to, i.to)
            emitins(ineg, e)
            emitf("add%k %0, %=", i, e)
        else:
            _table(i, e)
    elif op == Oneg:
        if i.to != i.arg[0]:
            emitf("mov%k %0, %=", i, e)
        if KBASE(i.cls) == 0:
            emitf("neg%k %=", i, e)
        else:
            f.write("\txorp%c %sfp%d(%%rip), %%%s\n" % (
                "xxsd"[i.cls], D.T.asloc, stashbits(negmask[i.cls], 16),
                regtoa(rval(i.to), SLong)))
    elif op == Odiv:
        # use xmm15 to adjust the instruction when the
        # conversion to 2-address in emitf() would fail
        if i.to == i.arg[1]:
            i.arg[1] = TMP(XMM0 + 15)
            emitf("mov%k %=, %1", i, e)
            emitf("mov%k %0, %=", i, e)
            i.arg[0] = i.to
        _table(i, e)
    elif op == Ocopy:
        # copies are used for many things; see my note
        # to understand how to load big constants:
        # https://c9x.me/notes/2015-09-19.html
        assert rtype(i.to) != RMem
        if i.to == R or i.arg[0] == R:
            return
        if i.to == i.arg[0]:
            return
        t0 = rtype(i.arg[0])
        if i.cls == Kl and t0 == RCon and e.fn.con[rval(i.arg[0])].type == CBits:
            val = e.fn.con[rval(i.arg[0])].i
            if isreg(i.to):
                if 0 <= val <= 0xffffffff:
                    emitf("movl %W0, %W=", i, e)
                    return
            if rtype(i.to) == RSlot:
                if val < -(1 << 31) or val > (1 << 31) - 1:
                    emitf("movl %0, %=", i, e)
                    emitf("movl %0>>32, 4+%=", i, e)
                    return
        if isreg(i.to) and t0 == RCon and e.fn.con[rval(i.arg[0])].type == CAddr:
            emitf("lea%k %M0, %=", i, e)
            return
        if rtype(i.to) == RSlot and (t0 == RSlot or t0 == RMem):
            i.cls = Kd if KWIDE(i.cls) else Ks
            i.arg[1] = TMP(XMM0 + 15)
            emitf("mov%k %0, %1", i, e)
            emitf("mov%k %1, %=", i, e)
            return
        # conveniently, the assembler knows if it
        # should use movabsq when reading movq
        emitf("mov%k %0, %=", i, e)
    elif op == Oaddr:
        if rtype(i.arg[0]) != RCon:
            _table(i, e)
            return
        con = e.fn.con[rval(i.arg[0])]
        assert isreg(i.to) and con.type == CAddr
        sym = str_(con.sym.id)
        pre = "" if sym[0] == '"' else D.T.assym
        if D.T.apple and (con.sym.type & SThr):
            f.write("\tmovq %s%s@tlvp(%%rip), %%%s\n" % (
                pre, sym, regtoa(rval(i.to), SLong)))
            return
        if D.T.windows and con.sym.type != SGlo:
            die("extern/thread unsupported on amd64_win")
        st = con.sym.type
        if st == SThr:
            # derive the symbol address from the TCB
            # address at offset 0 of %fs
            emitf("movq %%fs:0, %L=", i, e)
            f.write("\tleaq %s%s@tpoff" % (pre, sym))
            if con.i:
                f.write("%+d" % con.i)
            f.write("(%%%s), %%%s\n" % (regtoa(rval(i.to), SLong), regtoa(rval(i.to), SLong)))
        elif st == SExtThr:
            # initial-exec TLS: load offset from
            # GOT, add to thread-base register
            assert not con.i
            emitf("movq %%fs:0, %L=", i, e)
            f.write("\taddq %s%s@gottpoff(%%rip), %%%s\n" % (
                pre, sym, regtoa(rval(i.to), SLong)))
        elif st == SExt:
            # load address from the GOT
            assert not con.i
            f.write("\tmovq %s%s@gotpcrel(%%rip), %%%s\n" % (
                pre, sym, regtoa(rval(i.to), SLong)))
        else:
            _table(i, e)
    elif op == Ocall:
        # calls simply have a weird syntax in AT&T
        # assembly...
        t = rtype(i.arg[0])
        if t == RCon:
            con = e.fn.con[rval(i.arg[0])]
            f.write("\tcallq ")
            emitcon(con, e)
            if con.type == CAddr and (con.sym.type & SExt) and not D.T.apple:
                f.write("@plt")
            f.write("\n")
        elif t == RTmp:
            emitf("callq *%L0", i, e)
        else:
            die("invalid call argument")
    elif op == Osalloc:
        # there is no good reason why this is here
        # maybe we should split Osalloc in 2 different
        # instructions depending on the result
        assert e.fp == RBP
        emitf("subq %L0, %%rsp", i, e)
        if i.to != R:
            emitcopy(i.to, TMP(RSP), Kl, e)
    elif op == Oswap:
        if KBASE(i.cls) == 0:
            _table(i, e)
            return
        # for floats, there is no swap instruction
        # so we use xmm15 as a temporary
        emitcopy(TMP(XMM0 + 15), i.arg[0], i.cls, e)
        emitcopy(i.arg[0], i.arg[1], i.cls, e)
        emitcopy(i.arg[1], TMP(XMM0 + 15), i.cls, e)
    elif op == Odbgloc:
        emitdbgloc(rval(i.arg[0]), rval(i.arg[1]), f)
    elif isxsel(op):
        if i.to == i.arg[1]:
            emitf(cmov[op - Oxsel][0], i, e)
        else:
            if i.to != i.arg[0]:
                emitf("mov %0, %=", i, e)
            emitf(cmov[op - Oxsel][1], i, e)
    else:
        _table(i, e)


def sysv_framesz(e):
    from .sysv import amd64_sysv_rclob
    # specific to NAlign == 3
    o = 0
    if not e.fn.leaf:
        for k in range(NCLR_SYSV):
            o ^= e.fn.reg >> amd64_sysv_rclob[k]
        o &= 1
    f = e.fn.slot
    f = (f + 3) & -4
    if f > 0 and e.fp == RSP and e.fn.salign == 4:
        f += 2
    e.fsz = 4 * f + 8 * o + 176 * e.fn.vararg


_id0_sysv = 0


def amd64_sysv_emitfn(fn, f):
    global _id0_sysv
    from .sysv import amd64_sysv_rsave, amd64_sysv_rclob
    id0 = _id0_sysv
    e = E(f, fn)
    emitfnlnk(fn.name, fn.lnk, f)
    f.write("\tendbr64\n")
    if not fn.leaf or fn.vararg or fn.dynalloc:
        e.fp = RBP
        f.write("\tpushq %rbp\n\tmovq %rsp, %rbp\n")
    else:
        e.fp = RSP
    sysv_framesz(e)
    if e.fsz:
        f.write("\tsubq $%d, %%rsp\n" % e.fsz)
    if fn.vararg:
        o = -176
        for r in amd64_sysv_rsave[:6]:
            f.write("\tmovq %%%s, %d(%%rbp)\n" % (rname[r][0], o))
            o += 8
        for n in range(8):
            f.write("\tmovaps %%xmm%d, %d(%%rbp)\n" % (n, o))
            o += 16
    for r in amd64_sysv_rclob[:NCLR_SYSV]:
        if fn.reg & BIT(r):
            itmp = Ins(0, 0, R, TMP(r))
            emitf("pushq %L0", itmp, e)
            e.nclob += 1

    lbl = 0
    b = fn.start
    while b:
        if lbl or b.npred > 1:
            for p in range(b.npred):
                if b.pred[p].id >= b.id:
                    break
            else:
                p = b.npred
            if p != b.npred:
                f.write(".p2align 4\n")
            f.write("%sbb%d:\n" % (D.T.asloc, id0 + b.id))
        for i in b.ins[:b.nins]:
            emitins(i, e)
        lbl = _emitjmp(b, fn, e, id0, amd64_sysv_rclob, NCLR_SYSV, 0)
        b = b.link
    _id0_sysv = id0 + fn.nblk
    if not D.T.apple:
        elf_emitfnfin(fn.name, f)


def _emitjmp(b, fn, e, id0, rclob, nclr, win):
    """The jump at the end of b. Returns lbl: 0 when b falls through to the next block (whose
    label is then only printed if it has several predecessors)."""
    f = e.f
    jt = b.jmp.type
    if jt == Jhlt:
        f.write("\tud2\n")
    elif jt == Jret0:
        if fn.dynalloc:
            f.write("\tmovq %%rbp, %%rsp\n\tsubq $%d, %%rsp\n" % (e.fsz + e.nclob * 8))
        for r in reversed(rclob[:nclr]):
            if fn.reg & BIT(r):
                itmp = Ins(0, 0, R, TMP(r))
                emitf("popq %L0", itmp, e)
        if e.fp == RBP:
            f.write("\tleave\n")
        elif e.fsz:
            f.write("\taddq $%d, %%rsp\n" % e.fsz)
        f.write("\tret\n")
    elif jt == Jjmp:
        if b.s1 is not b.link:
            f.write("\tjmp %sbb%d\n" % (D.T.asloc, id0 + b.s1.id))
        else:
            return 0
    else:
        c = jt - Jjf
        if 0 <= c <= NCmp:
            if b.link is b.s2 or (win and c >= NCmpI):
                s = b.s1
                b.s1 = b.s2
                b.s2 = s
                n = 0
            else:
                n = 1
            f.write("\tj%s %sbb%d\n" % (_ctoa[c][n], D.T.asloc, id0 + b.s2.id))
            if b.s1 is not b.link:
                f.write("\tjmp %sbb%d\n" % (D.T.asloc, id0 + b.s1.id))
            else:
                return 0
        else:
            die("unhandled jump %d", jt)
    return 1


def winabi_framesz(e):
    from .winabi import amd64_winabi_rclob
    # specific to NAlign == 3
    o = 0
    if not e.fn.leaf:
        for k in range(NCLR_WIN):
            o ^= e.fn.reg >> amd64_winabi_rclob[k]
        o &= 1
    f = e.fn.slot
    f = (f + 3) & -4
    if f > 0 and e.fp == RSP and e.fn.salign == 4:
        f += 2
    e.fsz = 4 * f + 8 * o


_id0_win = 0


def amd64_winabi_emitfn(fn, f):
    global _id0_win
    from .winabi import amd64_winabi_rclob
    id0 = _id0_win
    e = E(f, fn)
    emitfnlnk(fn.name, fn.lnk, f)
    f.write("\tendbr64\n")
    if fn.vararg:
        f.write("\tmovq %rcx, 0x8(%rsp)\n")
        f.write("\tmovq %rdx, 0x10(%rsp)\n")
        f.write("\tmovq %r8, 0x18(%rsp)\n")
        f.write("\tmovq %r9, 0x20(%rsp)\n")
    if not fn.leaf or fn.vararg or fn.dynalloc:
        e.fp = RBP
        f.write("\tpushq %rbp\n\tmovq %rsp, %rbp\n")
    else:
        e.fp = RSP
    winabi_framesz(e)
    if e.fsz:
        f.write("\tsubq $%d, %%rsp\n" % e.fsz)
    for r in amd64_winabi_rclob[:NCLR_WIN]:
        if fn.reg & BIT(r):
            itmp = Ins(0, 0, R, TMP(r))
            emitf("pushq %L0", itmp, e)
            e.nclob += 1

    lbl = 0
    b = fn.start
    while b:
        if lbl or b.npred > 1:
            f.write("%sbb%d:\n" % (D.T.asloc, id0 + b.id))
        for i in b.ins[:b.nins]:
            emitins(i, e)
        lbl = _emitjmp(b, fn, e, id0, amd64_winabi_rclob, NCLR_WIN, 1)
        b = b.link
    _id0_win = id0 + fn.nblk
