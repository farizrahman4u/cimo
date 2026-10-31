"""Port of QBE arm64/emit.c."""

from .. import defs as D
from ..defs import (R, RTmp, RCon, RSlot, RMem, CBits, CAddr, SGlo, SThr, SExt, SExtThr,
                    Kx, Kw, Kl, Ks, Kd, KWIDE, KBASE, M64, U32, NCmpI, NCmp, Ins, TMP,
                    SLOT, req, rtype, rval, rsval, s32, isload, isstore, optab, NOp, BIT,
                    Cieq, Cine, Cisge, Cisgt, Cisle, Cislt, Ciuge, Ciugt, Ciule, Ciult,
                    Cfeq, Cfge, Cfgt, Cfle, Cflt, Cfne, Cfo, Cfuo, Jhlt, Jret0, Jjmp, Jjf,
                    Oflag, Oadd, Osub, Oneg, Oand, Oor, Oxor, Osar, Oshr, Oshl, Omul, Odiv,
                    Oudiv, Orem, Ourem, Ocopy, Oswap, Ostoreb, Ostoreh, Ostorew, Ostorel,
                    Ostores, Ostored, Oloadsb, Oloadub, Oloadsh, Oloaduh, Oloadsw, Oloaduw,
                    Oload, Oextsb, Oextub, Oextsh, Oextuh, Oextsw, Oextuw, Oexts, Otruncd,
                    Ocast, Ostosi, Ostoui, Odtosi, Odtoui, Oswtof, Ouwtof, Osltof, Oultof,
                    Ocall, Oacmp, Oacmn, Oafcmp, Onop, Oaddr, Osalloc, Odbgloc, Oafmadd,
                    Oafmsub, Oafnmadd, Oafnmsub)
from ..util import die, isreg, str_
from ..emit import emitfnlnk, emitdbgloc, elf_emitfnfin
from .regs import R0, R18, IP1, FP, LR, SP, V0, V30
from .targ import arm64_rclob
from .. import ext as _ext


def _logimm(x, k):
    from .isel import arm64_logimm
    return arm64_logimm(x, k)


def _loadsz(i):
    from ..load import loadsz
    return loadsz(i)


def _storesz(i):
    from ..load import storesz
    return storesz(i)


class E:
    __slots__ = ('f', 'fn', 'frame', 'padding')

    def __init__(self, f, fn):
        self.f = f
        self.fn = fn
        self.frame = 0
        self.padding = 0


_CMP = [
    (Cieq, "eq", "ne"),
    (Cine, "ne", "eq"),
    (Cisge, "ge", "lt"),
    (Cisgt, "gt", "le"),
    (Cisle, "le", "gt"),
    (Cislt, "lt", "ge"),
    (Ciuge, "cs", "cc"),
    (Ciugt, "hi", "ls"),
    (Ciule, "ls", "hi"),
    (Ciult, "cc", "cs"),
    (NCmpI + Cfeq, "eq", "ne"),
    (NCmpI + Cfge, "ge", "lt"),
    (NCmpI + Cfgt, "gt", "le"),
    (NCmpI + Cfle, "ls", "hi"),
    (NCmpI + Cflt, "mi", "pl"),
    (NCmpI + Cfne, "ne", "eq"),
    (NCmpI + Cfo, "vc", "vs"),
    (NCmpI + Cfuo, "vs", "vc"),
]

Ki = -1  # matches Kw and Kl
Ka = -2  # matches all classes

omap = [
    (Oadd, Ki, "add %=, %0, %1"),
    (Oadd, Ka, "fadd %=, %0, %1"),
    (Osub, Ki, "sub %=, %0, %1"),
    (Osub, Ka, "fsub %=, %0, %1"),
    (Oneg, Ki, "neg %=, %0"),
    (Oneg, Ka, "fneg %=, %0"),
    (Oand, Ki, "and %=, %0, %1"),
    (Oor, Ki, "orr %=, %0, %1"),
    (Oxor, Ki, "eor %=, %0, %1"),
    (Osar, Ki, "asr %=, %0, %1"),
    (Oshr, Ki, "lsr %=, %0, %1"),
    (Oshl, Ki, "lsl %=, %0, %1"),
    (Omul, Ki, "mul %=, %0, %1"),
    (Omul, Ka, "fmul %=, %0, %1"),
    (Odiv, Ki, "sdiv %=, %0, %1"),
    (Odiv, Ka, "fdiv %=, %0, %1"),
    (Oudiv, Ki, "udiv %=, %0, %1"),
    (Orem, Ki, "sdiv %?, %0, %1\n\tmsub\t%=, %?, %1, %0"),
    (Ourem, Ki, "udiv %?, %0, %1\n\tmsub\t%=, %?, %1, %0"),
    (Ocopy, Ki, "mov %=, %0"),
    (Ocopy, Ka, "fmov %=, %0"),
    (Oswap, Ki, "mov %?, %0\n\tmov\t%0, %1\n\tmov\t%1, %?"),
    (Oswap, Ka, "fmov %?, %0\n\tfmov\t%0, %1\n\tfmov\t%1, %?"),
    (Ostoreb, Kw, "strb %W0, %M1"),
    (Ostoreh, Kw, "strh %W0, %M1"),
    (Ostorew, Kw, "str %W0, %M1"),
    (Ostorel, Kw, "str %L0, %M1"),
    (Ostores, Kw, "str %S0, %M1"),
    (Ostored, Kw, "str %D0, %M1"),
    (Oloadsb, Ki, "ldrsb %=, %M0"),
    (Oloadub, Ki, "ldrb %W=, %M0"),
    (Oloadsh, Ki, "ldrsh %=, %M0"),
    (Oloaduh, Ki, "ldrh %W=, %M0"),
    (Oloadsw, Kw, "ldr %=, %M0"),
    (Oloadsw, Kl, "ldrsw %=, %M0"),
    (Oloaduw, Ki, "ldr %W=, %M0"),
    (Oload, Ka, "ldr %=, %M0"),
    (Oextsb, Ki, "sxtb %=, %W0"),
    (Oextub, Ki, "uxtb %W=, %W0"),
    (Oextsh, Ki, "sxth %=, %W0"),
    (Oextuh, Ki, "uxth %W=, %W0"),
    (Oextsw, Ki, "sxtw %L=, %W0"),
    (Oextuw, Ki, "mov %W=, %W0"),
    (Oexts, Kd, "fcvt %=, %S0"),
    (Otruncd, Ks, "fcvt %=, %D0"),
    (Ocast, Kw, "fmov %=, %S0"),
    (Ocast, Kl, "fmov %=, %D0"),
    (Ocast, Ks, "fmov %=, %W0"),
    (Ocast, Kd, "fmov %=, %L0"),
    (Ostosi, Ka, "fcvtzs %=, %S0"),
    (Ostoui, Ka, "fcvtzu %=, %S0"),
    (Odtosi, Ka, "fcvtzs %=, %D0"),
    (Odtoui, Ka, "fcvtzu %=, %D0"),
    (Oswtof, Ka, "scvtf %=, %W0"),
    (Ouwtof, Ka, "ucvtf %=, %W0"),
    (Osltof, Ka, "scvtf %=, %L0"),
    (Oultof, Ka, "ucvtf %=, %L0"),
    (Ocall, Kw, "blr %L0"),

    (Oacmp, Ki, "cmp %0, %1"),
    (Oacmn, Ki, "cmn %0, %1"),
    (Oafcmp, Ka, "fcmpe %0, %1"),
    # cimo math builtins (ext.math)
    (D.Ofsqrt, Ka, "fsqrt %=, %0"),
    (D.Ofabs, Ka, "fabs %=, %0"),
    (D.Ofmin, Ka, "fminnm %=, %0, %1"),
    (D.Ofmax, Ka, "fmaxnm %=, %0, %1"),
    (D.Offloor, Ka, "frintm %=, %0"),
    (D.Ofceil, Ka, "frintp %=, %0"),
    (D.Oftrunc, Ka, "frintz %=, %0"),
    (D.Ofround, Ka, "frinta %=, %0"),
] + [(Oflag + _c, Ki, "cset %=, " + _s) for _c, _s, _ in _CMP] + [
    (NOp, 0, None),
]

V31 = 0x1fffffff  # local name for V31

_FMA = {Oafmadd: "fmadd", Oafmsub: "fmsub", Oafnmadd: "fnmadd", Oafnmsub: "fnmsub"}


def rname(r, k):
    if r == SP:
        assert k == Kl
        return "sp"
    elif R0 <= r <= LR:
        if k == Kw:
            return "w%d" % (r - R0)
        elif k == Kx or k == Kl:
            return "x%d" % (r - R0)
        die("invalid class")
    elif V0 <= r <= V30:
        if k == Ks:
            return "s%d" % (r - V0)
        elif k == Kx or k == Kd:
            return "d%d" % (r - V0)
        die("invalid class")
    elif r == V31:
        if k == Ks:
            return "s31"
        elif k == Kd:
            return "d31"
        die("invalid class")
    else:
        die("invalid register")


def _vasave(T):
    """Bytes a variadic function saves above its frame record: x0-x7 and q0-q7 (AAPCS64),
    x0-x7 only on Windows (cimo extension), nothing on Apple."""
    return 0 if T.apple else 64 if T.windows else 192


def _probe(e, size):
    """cimo extension, Windows: touch each page from sp down to sp - size before the
    prologue allocates it. Windows grows a stack through one guard page, so a frame larger
    than a page must not skip over it (MSVC calls __chkstk). Uses x16, x17."""
    global _nprobe
    _nprobe += 1
    e.f.write("\tmov\tx16, #%d\n" % (size & 0xFFFF))
    if size >> 16:
        e.f.write("\tmovk\tx16, #%d, lsl #16\n" % (size >> 16))
    e.f.write("\tsub\tx16, sp, x16\n"
              "\tmov\tx17, sp\n"
              ".Lprobe%d:\n"
              "\tsub\tx17, x17, #4096\n"
              "\tldr\txzr, [x17]\n"
              "\tcmp\tx17, x16\n"
              "\tb.hi\t.Lprobe%d\n" % (_nprobe, _nprobe))


_nprobe = 0


def slot(r, e):
    s = rsval(r)
    if s == -1:
        return 16 + e.frame
    if s < 0:
        if e.fn.vararg and not D.T.apple:
            return 16 + e.frame + _vasave(D.T) - (s + 2)
        else:
            return 16 + e.frame - (s + 2)
    else:
        return 16 + e.padding + 4 * s


def emitf(s, i, e):
    f = e.f
    f.write('\t')
    ns = len(s)
    p = 0
    sp = 0
    while True:
        k = i.cls
        while True:
            c = s[p] if p < ns else ''
            p += 1
            if c == '%':
                break
            if c == ' ' and not sp:
                f.write('\t')
                sp = 1
            elif not c:
                f.write('\n')
                return
            else:
                f.write(c)
        while True:  # Switch
            c = s[p]
            p += 1
            if c == 'W':
                k = Kw
                continue
            if c == 'L':
                k = Kl
                continue
            if c == 'S':
                k = Ks
                continue
            if c == 'D':
                k = Kd
                continue
            break
        if c == '?':
            if KBASE(k) == 0:
                f.write(rname(IP1, k))
            else:
                f.write(rname(V31, k))
        elif c == '=' or c == '0':
            r = i.to if c == '=' else i.arg[0]
            assert isreg(r) or req(r, TMP(V31))
            f.write(rname(rval(r), k))
        elif c == '1':
            r = i.arg[1]
            t = rtype(r)
            if t == RTmp:
                assert isreg(r)
                f.write(rname(rval(r), k))
            elif t == RCon:
                pc = e.fn.con[rval(r)]
                n = pc.i & M64
                assert pc.type == CBits
                if n >> 24:
                    assert _logimm(n, k)
                    f.write("#%d" % n)
                elif n & 0xfff000:
                    assert not (n & ~0xfff000)
                    f.write("#%d, lsl #12" % (n >> 12))
                else:
                    assert not (n & ~0xfff)
                    f.write("#%d" % n)
            else:
                die("invalid second argument")
        elif c == 'M':
            c = s[p]
            p += 1
            assert c == '0' or c == '1' or c == '='
            r = i.to if c == '=' else i.arg[ord(c) - ord('0')]
            t = rtype(r)
            if t == RTmp:
                assert isreg(r)
                f.write("[%s]" % rname(rval(r), Kl))
            elif t == RSlot:
                f.write("[x29, %d]" % slot(r, e))
            elif t == RMem:
                pm = e.fn.mem[rval(r)]
                assert isreg(pm.base) and req(pm.index, R)
                if pm.offset.i:
                    f.write("[%s, %d]" % (rname(rval(pm.base), Kl), pm.offset.i))
                else:
                    f.write("[%s]" % rname(rval(pm.base), Kl))
            else:
                die("todo (arm emit): unhandled ref")
        else:
            die("invalid escape")


def loadaddr(c, rn, e):
    T = D.T
    st = c.sym.type
    if st == SGlo:
        if T.apple:
            s = ("\tadrp\tR, S@pageO\n"
                 "\tadd\tR, R, S@pageoffO\n")
        else:
            s = ("\tadrp\tR, SO\n"
                 "\tadd\tR, R, #:lo12:SO\n")
    elif st == SExtThr or st == SThr:
        if st == SExtThr and not T.apple:
            die("extern thread unavailable on arm64")
        if T.apple:
            s = ("\tadrp\tR, S@tlvppage\n"
                 "\tldr\tR, [R, S@tlvppageoff]\n")
        else:
            s = ("\tmrs\tR, tpidr_el0\n"
                 "\tadd\tR, R, #:tprel_hi12:SO, lsl #12\n"
                 "\tadd\tR, R, #:tprel_lo12_nc:SO\n")
    elif st == SExt:
        assert c.i == 0
        if T.apple:
            s = ("\tadrp\tR, S@gotpage\n"
                 "\tldr\tR, [R, S@gotpageoff]\n")
        else:
            s = ("\tadrp\tR, :got:S\n"
                 "\tldr\tR, [R, #:got_lo12:S]\n")
    else:
        die("unreachable")

    l = str_(c.sym.id)
    p = "" if l[0] == '"' else T.assym
    f = e.f
    for ch in s:
        if ch == 'R':
            f.write(rn)
        elif ch == 'S':
            f.write(p)
            f.write(l)
        elif ch == 'O':
            if c.i:
                # todo, handle large offsets
                f.write("+%d" % c.i)
        else:
            f.write(ch)


def loadcon(c, r, k, e):
    w = KWIDE(k)
    rn = rname(r, k)
    n = c.i
    if c.type == CAddr:
        rn = rname(r, Kl)
        loadaddr(c, rn, e)
        return
    assert c.type == CBits
    if not w:
        n = s32(n)
    if (n | 0xffff) == -1 or _logimm(n & M64, k):
        e.f.write("\tmov\t%s, #%d\n" % (rn, n))
    else:
        e.f.write("\tmov\t%s, #%d\n" % (rn, s32(n & 0xffff)))
        sh = 16
        while True:
            n >>= 16
            if not n:
                break
            if (not w and sh == 32) or sh == 64:
                break
            e.f.write("\tmovk\t%s, #0x%x, lsl #%d\n" % (rn, n & 0xffff, sh))
            sh += 16


def fixarg(arg, a, sz, t, e):
    """fixarg(&arg[a], ...)"""
    r = arg[a]
    if rtype(r) == RSlot:
        s = slot(r, e)
        if s > (sz * 4095) & U32:
            if t < 0:
                return 1
            i = Ins(Oaddr, Kl, TMP(t), r)
            emitins(i, e)
            arg[a] = TMP(t)
    return 0


def _table(i, e):
    # most instructions are just pulled out of
    # the table omap[], some special cases are
    # detailed below
    o = 0
    while True:
        # this linear search should really be a binary
        # search
        op, cls, fmt = omap[o]
        if op == NOp:
            die("no match for %s(%c)", optab[i.op].name, "wlsd"[i.cls])
        if op == i.op:
            if cls == i.cls or cls == Ka or (cls == Ki and KBASE(i.cls) == 0):
                break
        o += 1
    emitf(fmt, i, e)


def emitins(i, e):
    T = D.T
    op = i.op
    if op == Onop:
        pass
    elif op == Ocopy:
        if req(i.to, i.arg[0]):
            return
        if rtype(i.to) == RSlot:
            r = i.to
            if not isreg(i.arg[0]):
                i.to = TMP(IP1)
                emitins(i, e)
                i.arg[0] = i.to
            i.op = Ostorew + i.cls
            i.cls = Kw
            i.arg[1] = r
            emitins(i, e)
            return
        assert isreg(i.to)
        t = rtype(i.arg[0])
        if t == RCon:
            c = e.fn.con[rval(i.arg[0])]
            loadcon(c, rval(i.to), i.cls, e)
        elif t == RSlot:
            i.op = Oload
            emitins(i, e)
        else:
            assert rval(i.to) != IP1
            _table(i, e)
    elif op == Oaddr:
        assert rtype(i.arg[0]) == RSlot
        rn = rname(rval(i.to), Kl)
        s = slot(i.arg[0], e)
        if s <= 4095:
            e.f.write("\tadd\t%s, x29, #%d\n" % (rn, s))
        elif s <= 65535:
            e.f.write("\tmov\t%s, #%d\n"
                      "\tadd\t%s, x29, %s\n" % (rn, s, rn, rn))
        else:
            e.f.write("\tmov\t%s, #%d\n"
                      "\tmovk\t%s, #%d, lsl #16\n"
                      "\tadd\t%s, x29, %s\n" % (rn, s & 0xFFFF, rn, s >> 16, rn, rn))
    elif op == Ocall:
        if rtype(i.arg[0]) != RCon:
            _table(i, e)
            return
        c = e.fn.con[rval(i.arg[0])]
        if c.type != CAddr or (c.sym.type & SThr) or c.i:
            die("invalid call argument")
        l = str_(c.sym.id)
        p = "" if l[0] == '"' else T.assym
        e.f.write("\tbl\t%s%s\n" % (p, l))
    elif op == Osalloc and D.T.windows:
        # cimo extension: probe each page (see _probe); x17 is reserved (IP1)
        global _nprobe
        _nprobe += 1
        emitf("sub x17, sp, %0", i, e)
        e.f.write(".Lprobe%d:\n"
                  "\tcmp\tsp, x17\n"
                  "\tb.ls\t.Lprobed%d\n"
                  "\tsub\tsp, sp, #4096\n"
                  "\tldr\txzr, [sp]\n"
                  "\tb\t.Lprobe%d\n"
                  ".Lprobed%d:\n"
                  "\tmov\tsp, x17\n" % (_nprobe, _nprobe, _nprobe, _nprobe))
        if not req(i.to, R):
            emitf("mov %=, sp", i, e)
    elif op == Osalloc:
        emitf("sub sp, sp, %0", i, e)
        if not req(i.to, R):
            emitf("mov %=, sp", i, e)
    elif op == Odbgloc:
        emitdbgloc(rval(i.arg[0]), rval(i.arg[1]), e.f)
    elif op in _FMA:
        # cimo extension: r = a*b +- c, with b and c in a Mem (see isel.selfma)
        m = e.fn.mem[rval(i.arg[1])]
        k = i.cls
        e.f.write("\t%s\t%s, %s, %s, %s\n" % (_FMA[op], rname(rval(i.to), k),
                                               rname(rval(i.arg[0]), k),
                                               rname(rval(m.base), k), rname(rval(m.index), k)))
    else:
        if isload(i.op):
            fixarg(i.arg, 0, _loadsz(i), IP1, e)
        if isstore(i.op):
            t = -1 if T.apple or T.windows else R18  # x18 is the platform register
            if fixarg(i.arg, 1, _storesz(i), t, e):
                if req(i.arg[0], TMP(IP1)):
                    e.f.write("\tfmov\t%s31, %s17\n"
                              % ("ds"[i.cls == Kw], "xw"[i.cls == Kw]))
                    i.arg[0] = TMP(V31)
                    i.op = Ostores + (i.cls - Kw)
                fixarg(i.arg, 1, _storesz(i), IP1, e)
        _table(i, e)


# cimo extension (ext.ldp): pair adjacent accesses into ldp/stp, see pair.py

_PK = {'w': Kw, 'x': Kl, 's': Ks, 'd': Kd}


def _paddr(r, e):
    """(base register, offset) of an address operand after allocation, or None."""
    t = rtype(r)
    if t == RTmp:
        return rval(r), 0
    if t == RSlot:
        return FP, slot(r, e)
    if t == RMem:
        m = e.fn.mem[rval(r)]
        if req(m.index, R) and m.offset.type in (CBits, D.CUndef):
            return rval(m.base), (m.offset.i if m.offset.type == CBits else 0)
    return None


def emitpair(i1, i2, e):
    """Emit i1; i2 as one ldp/stp if they fit one; returns 1 if done."""
    from .pair import kind, inrange, SIZE
    k1 = kind(i1)
    if not k1 or k1 != kind(i2):
        return 0
    ld = k1[0] == 'l'
    n = 0 if ld else 1
    a1 = _paddr(i1.arg[n], e)
    a2 = _paddr(i2.arg[n], e)
    if not a1 or not a2 or a1[0] != a2[0]:
        return 0
    sz = SIZE[k1[1]]
    if abs(a1[1] - a2[1]) != sz or not inrange(min(a1[1], a2[1]), sz):
        return 0
    r1 = i1.to if ld else i1.arg[0]
    r2 = i2.to if ld else i2.arg[0]
    if not isreg(r1) or not isreg(r2):
        return 0
    if ld and (rval(r1) == rval(r2) or rval(r1) == a1[0]):
        return 0  # ldp needs distinct registers; i2 must see the base i1 saw
    if a2[1] < a1[1]:
        r1, r2, a1 = r2, r1, a2
    k = _PK[k1[1]]
    base = "x29" if a1[0] == FP else rname(a1[0], Kl)
    off = (", %d" % a1[1]) if a1[1] else ""
    e.f.write("\t%s\t%s, %s, [%s%s]\n" % ("ldp" if ld else "stp", rname(rval(r1), k),
                                           rname(rval(r2), k), base, off))
    return 1


def emitinsl(ins, e):
    """emitins() for each instruction in a list, pairing accesses when ext.ldp is on."""
    k = 0
    n = len(ins)
    while k < n:
        if _ext.ldp and k + 1 < n and emitpair(ins[k], ins[k + 1], e):
            k += 2
            continue
        emitins(ins[k], e)
        k += 1


# cimo extension (ext.layout): a block that ends in "jmp H", where H is a small block
# earlier in the layout that ends in a conditional jump (a loop header with the loop test),
# gets a copy of H's code and conditional jump instead, so the loop runs one conditional
# branch per iteration instead of an unconditional and a conditional one.

MAXDUP = 4


def layoutplan(fn):
    """{latch block: (copied ins of H, jmp type, s1, s2)} and the set of blocks that need
    a label because a copied jump targets them. Snapshots are taken before emission, which
    mutates instructions and swaps successors."""
    dup = {}
    lbl = set()
    b = fn.start
    while b:
        h = b.s1
        if (b.jmp.type == Jjmp and h is not None and h is not b.link and h is not b
                and h.id < b.id and Jjf <= h.jmp.type <= Jjf + NCmp - 1 and h.nins <= MAXDUP
                and h.s1 is not None and h.s2 is not None):
            dup[b] = ([i.copy() for i in h.ins[:h.nins]], h.jmp.type, h.s1, h.s2)
            lbl.add(h.s1)
            lbl.add(h.s2)
        b = b.link
    return dup, lbl


def emitcondjmp(b, jt, s1, s2, e):
    """Conditional jump to s1 (condition true) or s2 from the end of b (b.link follows).
    Returns 1 if control falls through to b.link."""
    T = D.T
    c = jt - Jjf
    if b.link is s2:
        e.f.write("\tb%s\t%s%d\n" % (ctoa[c][0], T.asloc, _id0 + s1.id))
        return 1
    if b.link is s1:
        e.f.write("\tb%s\t%s%d\n" % (ctoa[c][1], T.asloc, _id0 + s2.id))
        return 1
    # neither follows: branch conditionally backwards (a loop), unconditionally forwards
    if s2.id <= b.id < s1.id:
        e.f.write("\tb%s\t%s%d\n" % (ctoa[c][1], T.asloc, _id0 + s2.id))
        e.f.write("\tb\t%s%d\n" % (T.asloc, _id0 + s1.id))
    else:
        e.f.write("\tb%s\t%s%d\n" % (ctoa[c][0], T.asloc, _id0 + s1.id))
        e.f.write("\tb\t%s%d\n" % (T.asloc, _id0 + s2.id))
    return 0


def framelayout(e):
    o = 0
    for r in arm64_rclob:
        if r < 0:
            break
        o += 1 & (e.fn.reg >> r)
    f = e.fn.slot
    f = (f + 3) & -4
    o += o & 1
    e.padding = (4 * (f - e.fn.slot)) & U32
    e.frame = 4 * f + 8 * o


#
#  Stack-frame layout:
#
#  +=============+
#  | varargs     |
#  |  save area  |
#  +-------------+
#  | callee-save |  ^
#  |  registers  |  |
#  +-------------+  |
#  |    ...      |  |
#  | spill slots |  |
#  |    ...      |  | e->frame
#  +-------------+  |
#  |    ...      |  |
#  |   locals    |  |
#  |    ...      |  |
#  +-------------+  |
#  | e->padding  |  v
#  +-------------+
#  |  saved x29  |
#  |  saved x30  |
#  +=============+ <- x29
#

ctoa = {c: (s, n) for c, s, n in _CMP}
_id0 = 0


def arm64_emitfn(fn, out):
    global _id0
    T = D.T
    e = E(out, fn)
    if T.apple:
        e.fn.lnk.align = 4
    emitfnlnk(e.fn.name, e.fn.lnk, e.f)
    e.f.write("\thint\t#34\n")
    framelayout(e)

    if e.fn.vararg and not T.apple:
        if not T.windows:
            for n in range(7, -1, -1):
                e.f.write("\tstr\tq%d, [sp, -16]!\n" % n)
        for n in range(7, -1, -2):
            e.f.write("\tstp\tx%d, x%d, [sp, -16]!\n" % (n - 1, n))
    if T.windows and e.frame + 16 > 4096:
        _probe(e, e.frame + 16)

    if e.frame + 16 <= 512:
        e.f.write("\tstp\tx29, x30, [sp, -%d]!\n" % (e.frame + 16))
    elif e.frame <= 4095:
        e.f.write("\tsub\tsp, sp, #%d\n"
                  "\tstp\tx29, x30, [sp, -16]!\n" % e.frame)
    elif e.frame <= 65535:
        e.f.write("\tmov\tx16, #%d\n"
                  "\tsub\tsp, sp, x16\n"
                  "\tstp\tx29, x30, [sp, -16]!\n" % e.frame)
    else:
        e.f.write("\tmov\tx16, #%d\n"
                  "\tmovk\tx16, #%d, lsl #16\n"
                  "\tsub\tsp, sp, x16\n"
                  "\tstp\tx29, x30, [sp, -16]!\n" % (e.frame & 0xFFFF, e.frame >> 16))
    e.f.write("\tmov\tx29, sp\n")
    if _ext.stkchk and e.fn.name in _ext.stkchk:  # cimo extension: stack guard
        e.f.write("\tbl\t%s__cimo_stkchk\n" % T.assym)
    s = s32((e.frame - e.padding) // 4)
    save = []
    for r in arm64_rclob:
        if r < 0:
            break
        if e.fn.reg & BIT(r):
            s -= 2
            i = Ins(Ostored if r >= V0 else Ostorel, 0, R, TMP(r), SLOT(s))
            save.append(i)
    emitinsl(save, e)

    dup, dlbl = layoutplan(e.fn) if _ext.layout else ({}, ())
    lbl = 0
    b = e.fn.start
    while b:
        if lbl or b.npred > 1 or b in dlbl:
            e.f.write("%s%d:\n" % (T.asloc, _id0 + b.id))
        emitinsl(b.ins[:b.nins], e)
        lbl = 1
        jt = b.jmp.type
        if b in dup:
            hins, hjt, hs1, hs2 = dup[b]
            emitinsl([i.copy() for i in hins], e)
            if emitcondjmp(b, hjt, hs1, hs2, e):
                lbl = 0
        elif jt == Jhlt:
            if _ext.trap:  # cimo extension: unwind to Python
                e.f.write("\tbl\t%s__cimo_trap\n" % T.assym)
            else:
                e.f.write("\tbrk\t#1000\n")
        elif jt == Jret0:
            s = s32((e.frame - e.padding) // 4)
            rest = []
            for r in arm64_rclob:
                if r < 0:
                    break
                if e.fn.reg & BIT(r):
                    s -= 2
                    i = Ins(Oload, 0, TMP(r), SLOT(s))
                    i.cls = Kd if r >= V0 else Kl
                    rest.append(i)
            emitinsl(rest, e)
            if e.fn.dynalloc:
                e.f.write("\tmov sp, x29\n")
            o = e.frame + 16
            if e.fn.vararg and not T.apple:
                o += _vasave(T)
            if o <= 504:
                e.f.write("\tldp\tx29, x30, [sp], %d\n" % o)
            elif o - 16 <= 4095:
                e.f.write("\tldp\tx29, x30, [sp], 16\n"
                          "\tadd\tsp, sp, #%d\n" % (o - 16))
            elif o - 16 <= 65535:
                e.f.write("\tldp\tx29, x30, [sp], 16\n"
                          "\tmov\tx16, #%d\n"
                          "\tadd\tsp, sp, x16\n" % (o - 16))
            else:
                e.f.write("\tldp\tx29, x30, [sp], 16\n"
                          "\tmov\tx16, #%d\n"
                          "\tmovk\tx16, #%d, lsl #16\n"
                          "\tadd\tsp, sp, x16\n" % ((o - 16) & 0xFFFF, (o - 16) >> 16))
            e.f.write("\tret\n")
        else:
            if jt != Jjmp and _ext.layout:
                c = jt - Jjf
                if c < 0 or c > NCmp:
                    die("unhandled jump %d", jt)
                if emitcondjmp(b, jt, b.s1, b.s2, e):
                    lbl = 0
                b = b.link
                continue
            if jt != Jjmp:
                c = jt - Jjf
                if c < 0 or c > NCmp:
                    die("unhandled jump %d", jt)
                if b.link is b.s2:
                    t = b.s1
                    b.s1 = b.s2
                    b.s2 = t
                    n = 0
                else:
                    n = 1
                e.f.write("\tb%s\t%s%d\n" % (ctoa[c][n], T.asloc, _id0 + b.s2.id))
            # Jmp:
            if b.s1 is not b.link:
                e.f.write("\tb\t%s%d\n" % (T.asloc, _id0 + b.s1.id))
            else:
                lbl = 0
        b = b.link
    _id0 += e.fn.nblk
    if not T.apple:
        elf_emitfnfin(fn.name, out)
