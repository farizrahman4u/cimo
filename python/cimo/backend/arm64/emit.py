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
from ..defs import bitsd, bitsf
from ..util import die, isreg, str_, intern
from ..defs import Con, Mem
from ..util import vgrow
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
    __slots__ = ('f', 'fn', 'frame', 'padding', 'tbzok', 'qthird')

    def __init__(self, f, fn):
        self.f = f
        self.fn = fn
        self.frame = 0
        self.padding = 0
        self.tbzok = False


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

    (D.Oafnmul, Ka, "fnmul %=, %0, %1"),
    (D.Oasmull, Kl, "smull %=, %W0, %W1"),
    (D.Oaumull, Kl, "umull %=, %W0, %W1"),
    (D.Osmulh, Kl, "smulh %=, %0, %1"),
    (D.Oumulh, Kl, "umulh %=, %0, %1"),
    (Oacmp, Ki, "cmp %0, %1"),
    (D.Oaadds, Ki, "adds %=, %0, %1"),
    (D.Oasubs, Ki, "subs %=, %0, %1"),
    (D.Oaands, Ki, "ands %=, %0, %1"),
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
    (D.Ofrint, Ka, "frintx %=, %0"),
    (D.Ofnearbyint, Ka, "frinti %=, %0"),
    # cimo bit builtins (ext.math); popcount through v31 (no scalar cnt before FEAT_CSSC)
    (D.Obclz, Ki, "clz %=, %0"),
    (D.Obctz, Ki, "rbit %=, %0\n\tclz %=, %="),
    (D.Obswap, Ki, "rev %=, %0"),
    (D.Obswap16, Kw, "rev16 %=, %0\n\tand %=, %=, #0xffff"),
    (D.Obpopcnt, Kw, "fmov %S?, %0\n\tcnt v31.8b, v31.8b\n\taddv b31, v31.8b\n\tfmov %=, %S?"),
    (D.Obpopcnt, Kl, "fmov %D?, %0\n\tcnt v31.8b, v31.8b\n\taddv b31, v31.8b\n\tfmov %W=, %S?"),
] + [(Oflag + _c, Ki, "cset %=, " + _s) for _c, _s, _ in _CMP] + [
    (NOp, 0, None),
]

V31 = 0x1fffffff  # local name for V31

_FMA = {Oafmadd: "fmadd", Oafmsub: "fmsub", Oafnmadd: "fnmadd", Oafnmsub: "fnmsub"}


_LOGICOPS = frozenset((Oand, Oor, Oxor, D.Oaands))
_POSTINDEX, _PREINDEX = -1, -2  # Mem.scale of an access that also updates its base (_writeback)


def _zr(r, k):
    """The name of a register operand, or of the zero register for the constant 0 (ext.isel)."""
    if rtype(r) == RCon:
        return "wzr" if k == Kw else "xzr"
    return rname(rval(r), k)


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
            if c == '0' and rtype(r) == RCon:  # cimo (ext.isel): storing zero
                assert e.fn.con[rval(r)].i == 0
                f.write("wzr" if k == Kw else "xzr")
            else:
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
                if _ext.isel and i.op in _LOGICOPS:
                    n &= 0xffffffff if k == Kw else M64
                    assert _logimm(n, k)
                    f.write("#%d" % n)
                elif n >> 24:
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
                assert isreg(pm.base)
                if pm.scale == _POSTINDEX:  # see _writeback
                    f.write("[%s], %d" % (rname(rval(pm.base), Kl), pm.offset.i))
                elif pm.scale == _PREINDEX:
                    f.write("[%s, %d]!" % (rname(rval(pm.base), Kl), pm.offset.i))
                elif not req(pm.index, R):  # cimo (ext.isel): see isel.foldindex
                    shift, ext = pm.scale & 15, pm.scale >> 4
                    b = rname(rval(pm.base), Kl)
                    x = rname(rval(pm.index), Kw if ext else Kl)
                    how = ("lsl", "sxtw", "uxtw")[ext]
                    if ext:
                        f.write("[%s, %s, %s%s]" % (b, x, how, " #%d" % shift if shift else ""))
                    elif shift:
                        f.write("[%s, %s, lsl #%d]" % (b, x, shift))
                    else:
                        f.write("[%s, %s]" % (b, x))
                elif pm.offset.i:
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


_QLANE = {'b': '16b', 'h': '8h', 'w': '4s', 'l': '2d', 's': '4s', 'd': '2d'}
_QINT = {'qadd': 'add', 'qsub': 'sub', 'qmul': 'mul', 'qmin': 'smin', 'qmax': 'smax', 'qumin': 'umin', 'qumax': 'umax',
         'qceq': 'cmeq', 'qcgt': 'cmgt', 'qcge': 'cmge', 'qcugt': 'cmhi', 'qcuge': 'cmhs', 'qneg': 'neg'}
_QFP = {'qadd': 'fadd', 'qsub': 'fsub', 'qmul': 'fmul', 'qdiv': 'fdiv', 'qmin': 'fminnm', 'qmax': 'fmaxnm',
        'qceq': 'fcmeq', 'qcgt': 'fcmgt', 'qcge': 'fcmge', 'qneg': 'fneg', 'qabs': 'fabs', 'qsqrt': 'fsqrt'}


def _qv(r, lane='16b'):
    return 'v%d.%s' % (rval(r) - V0, lane)


def _qaddr(r, e):
    """The memory operand of a 128-bit access: a slot's offset may not fit, so through x17."""
    if rtype(r) == RSlot:
        off = slot(r, e)
        base = "x29"
        if off < 4096:
            e.f.write("\tadd\tx17, %s, #%d\n" % (base, off))
        else:
            e.f.write("\tmov\tx17, #%d\n\tadd\tx17, %s, x17\n" % (off, base))
        return "[x17]"
    return "[%s]" % rname(rval(r), Kl)


def _qins(i, e):
    """cimo (ext.vector): one instruction on 128 bits; True if it was one."""
    f = e.f
    op = i.op
    if op == Ocopy and i.cls == D.Kq:
        if rtype(i.to) == RSlot or rtype(i.arg[0]) == RSlot:  # register allocation's moves, of what's spilled
            if req(i.to, i.arg[0]):
                return True
            tmpreg = 31
            if rtype(i.arg[0]) == RSlot:
                if rtype(i.to) == RSlot:
                    f.write("\tldr\tq31, %s\n" % _qaddr(i.arg[0], e))
                else:
                    tmpreg = rval(i.to) - V0
                    f.write("\tldr\tq%d, %s\n" % (tmpreg, _qaddr(i.arg[0], e)))
                    return True
            else:
                tmpreg = rval(i.arg[0]) - V0
            f.write("\tstr\tq%d, %s\n" % (tmpreg, _qaddr(i.to, e)))
            return True
        if not req(i.to, i.arg[0]):
            f.write("\tmov\t%s, %s\n" % (_qv(i.to), _qv(i.arg[0])))
        return True
    if op == Oswap and i.cls == D.Kq:
        a, b = _qv(i.arg[0]), _qv(i.arg[1])
        f.write("\tmov\tv31.16b, %s\n\tmov\t%s, %s\n\tmov\t%s, v31.16b\n" % (a, a, b, b))
        return True
    if (op == D.Oload and i.cls == D.Kq) or op == D.Oqload:
        f.write("\tldr\tq%d, %s\n" % (rval(i.to) - V0, _qaddr(i.arg[0], e)))
        return True
    if op == D.Oqstore:
        f.write("\tstr\tq%d, %s\n" % (rval(i.arg[0]) - V0, _qaddr(i.arg[1], e)))
        return True
    if not D.isqop(op):
        return False
    name = optab[op].name
    if name in ('qsel0', 'qfma0'):
        return True  # with the next one
    if name in ('qand', 'qor', 'qxor'):
        f.write("\t%s\t%s, %s, %s\n" % ({'qand': 'and', 'qor': 'orr', 'qxor': 'eor'}[name], _qv(i.to), _qv(i.arg[0]), _qv(i.arg[1])))
        return True
    if name in ('qsaddl', 'qsaddh', 'quaddl', 'quaddh'):
        high = name[-1] == 'h'
        mn = ('saddw' if name[1] == 's' else 'uaddw') + ('2' if high else '')
        f.write("\t%s\t%s, %s, %s\n" % (mn, _qv(i.to, '2d'), _qv(i.arg[0], '2d'), _qv(i.arg[1], '4s' if high else '2s')))
        return True
    if name == 'qnot':
        f.write("\tmvn\t%s, %s\n" % (_qv(i.to), _qv(i.arg[0])))
        return True
    if name in ('qsel1', 'qfma1s', 'qfma1d'):
        third = e.qthird
        l = _QLANE[name[-1]] if name != 'qsel1' else '16b'
        mn = 'bsl' if name == 'qsel1' else 'fmla'
        if name == 'qsel1':  # bsl: the mask is the destination
            if rval(i.to) in (rval(i.arg[0]), rval(i.arg[1])) or not req(i.to, third):
                f.write("\tmov\tv31.16b, %s\n\tbsl\tv31.16b, %s, %s\n\tmov\t%s, v31.16b\n" % (_qv(third), _qv(i.arg[0]), _qv(i.arg[1]), _qv(i.to)))
            else:
                f.write("\tbsl\t%s, %s, %s\n" % (_qv(i.to), _qv(i.arg[0]), _qv(i.arg[1])))
            return True
        if req(i.to, third):  # fmla: the addend is the destination
            f.write("\tfmla\t%s, %s, %s\n" % (_qv(i.to, l), _qv(i.arg[0], l), _qv(i.arg[1], l)))
        elif rval(i.to) in (rval(i.arg[0]), rval(i.arg[1])):
            f.write("\tmov\tv31.16b, %s\n\tfmla\tv31.%s, %s, %s\n\tmov\t%s, v31.16b\n" % (_qv(third), l, _qv(i.arg[0], l), _qv(i.arg[1], l), _qv(i.to)))
        else:
            f.write("\tmov\t%s, %s\n\tfmla\t%s, %s, %s\n" % (_qv(i.to), _qv(third), _qv(i.to, l), _qv(i.arg[0], l), _qv(i.arg[1], l)))
        return True
    x = name[-1]
    base = name[:-1]
    l = _QLANE[x]
    if base == 'qdup':
        if x in 'sd':
            f.write("\tdup\t%s, v%d.%s[0]\n" % (_qv(i.to, l), rval(i.arg[0]) - V0, x))
        else:
            f.write("\tdup\t%s, %s\n" % (_qv(i.to, l), rname(rval(i.arg[0]), Kl if x == 'l' else Kw)))
        return True
    if base == 'qget':
        n = e.fn.con[rval(i.arg[1])].i
        if x in 'sd':
            f.write("\tdup\t%s, v%d.%s[%d]\n" % (rname(rval(i.to), i.cls), rval(i.arg[0]) - V0, x, n))
        else:
            lane = {'b': 'b', 'h': 'h', 'w': 's', 'l': 'd'}[x]
            f.write("\tumov\t%s, v%d.%s[%d]\n" % (rname(rval(i.to), i.cls), rval(i.arg[0]) - V0, lane, n))
        return True
    if base in ('qshl', 'qshr', 'qsar'):
        n = e.fn.con[rval(i.arg[1])].i
        f.write("\t%s\t%s, %s, #%d\n" % ({'qshl': 'shl', 'qshr': 'ushr', 'qsar': 'sshr'}[base], _qv(i.to, l), _qv(i.arg[0], l), n))
        return True
    mn = (_QFP if x in 'sd' else _QINT)[base]
    if i.arg[1] == R:
        f.write("\t%s\t%s, %s\n" % (mn, _qv(i.to, l), _qv(i.arg[0], l)))
    else:
        f.write("\t%s\t%s, %s, %s\n" % (mn, _qv(i.to, l), _qv(i.arg[0], l), _qv(i.arg[1], l)))
    return True


def emitins(i, e):
    T = D.T
    op = i.op
    if i.cls == D.Kq or D.isqop(op):
        if op == D.Oqsel0 or op == D.Oqfma0:
            e.qthird = i.arg[0]
        if _qins(i, e):
            return
    if op == Onop:
        pass
    elif op == Ocopy:
        if req(i.to, i.arg[0]):
            return
        if rtype(i.to) == RSlot:
            r = i.to
            if not isreg(i.arg[0]):
                # fix over C QBE: a float through v31 (it used x17, an invalid class), as the
                # slot-to-slot copies of a large parallel move need
                i.to = TMP(V31 if KBASE(i.cls) == 1 else IP1)
                if rtype(i.arg[0]) == RSlot:
                    i.op = Oload
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
        # a size past the bottom of the address space runs to the guard page, like any too big:
        # sp can't take the wrapped value, or the kernel has no stack to report the fault on
        emitf("subs x17, sp, %0", i, e)
        e.f.write("\tcsel\tx17, xzr, x17, lo\n"
                  ".Lprobe%d:\n"
                  "\tcmp\tsp, x17\n"
                  "\tb.ls\t.Lprobed%d\n"
                  "\tsub\tsp, sp, #4096\n"
                  "\tldr\txzr, [sp]\n"
                  "\tb\t.Lprobe%d\n"
                  ".Lprobed%d:\n"
                  "\tmov\tsp, x17\n" % (_nprobe, _nprobe, _nprobe, _nprobe))
        if not req(i.to, R):
            emitf("mov %=, sp", i, e)
    elif op == D.Ogetsp:  # cimo extension (ext.vla)
        emitf("mov %=, sp", i, e)
    elif op == D.Osetsp:
        emitf("mov sp, %0", i, e)
    elif op == Osalloc:
        emitf("sub sp, sp, %0", i, e)
        if not req(i.to, R):
            emitf("mov %=, sp", i, e)
    elif op == Odbgloc:
        emitdbgloc(rval(i.arg[0]), rval(i.arg[1]), e.f)
    elif op == D.Oamadd or op == D.Oamsub:  # cimo (ext.isel), see isel.selmadd
        m = e.fn.mem[rval(i.arg[1])]
        k = i.cls
        wide = m.scale  # 0: as is; 1, 2: of 32-bit values, signed or not (isel._widening)
        mn = {0: ("madd", "msub"), 1: ("smaddl", "smsubl"), 2: ("umaddl", "umsubl")}[wide][op != D.Oamadd]
        e.f.write("\t%s\t%s, %s, %s, %s\n" % (mn, rname(rval(i.to), k), rname(rval(m.base), Kw if wide else k),
                                               rname(rval(m.index), Kw if wide else k), _zr(i.arg[0], k)))
    elif op == D.Oaalu:  # cimo (ext.isel), see isel.selalu
        m = e.fn.mem[rval(i.arg[1])]
        k = i.cls
        how = m.scale >> 4
        x = rname(rval(m.base), Kw if how >= 4 else k)
        n = m.offset.i
        e.f.write("\t%s\t%s, %s, %s%s\n" % (
            ("add", "sub", "and", "orr", "eor", "bic", "orn", "eon")[m.scale & 15],
            rname(rval(i.to), k), _zr(i.arg[0], k), x,
            (", %s #%d" % (("lsl", "lsr", "asr", "ror", "sxtw", "uxtw")[how], n)) if n or how >= 4 else ""))
    elif op == D.Oaubfx or op == D.Oasbfx:
        k = i.cls
        c = e.fn.con[rval(i.arg[1])].i
        e.f.write("\t%s\t%s, %s, #%d, #%d\n" % ("ubfx" if op == D.Oaubfx else "sbfx", rname(rval(i.to), k),
                                                 rname(rval(i.arg[0]), k), c & 255, c >> 8))
    elif op == D.Oabfi:
        k = i.cls
        m = e.fn.mem[rval(i.arg[1])]
        to, a, b = rval(i.to), rval(i.arg[0]), rval(m.base)
        if b == to and a != to:  # the field's source would be overwritten
            e.f.write("\tmov\t%s, %s\n" % (rname(IP1, k), rname(b, k)))
            b = IP1
        if a != to:
            e.f.write("\tmov\t%s, %s\n" % (rname(to, k), rname(a, k)))
        lsb, width = m.offset.i & 255, m.offset.i >> 8
        if m.scale:
            e.f.write("\tbfxil\t%s, %s, #%d, #%d\n" % (rname(to, k), rname(b, k), m.scale, width))
        elif lsb:
            e.f.write("\tbfi\t%s, %s, #%d, #%d\n" % (rname(to, k), rname(b, k), lsb, width))
        else:
            e.f.write("\tbfxil\t%s, %s, #0, #%d\n" % (rname(to, k), rname(b, k), width))
    elif op == D.Oaror:
        k = i.cls
        e.f.write("\tror\t%s, %s, #%d\n" % (rname(rval(i.to), k), rname(rval(i.arg[0]), k),
                                             e.fn.con[rval(i.arg[1])].i))
    elif op == D.Oafcon:
        k = i.cls
        bits = e.fn.con[rval(i.arg[0])].i
        if bits == 0:
            e.f.write("\tfmov\t%s, %s\n" % (rname(rval(i.to), k), "xzr" if k == Kd else "wzr"))
        else:
            v = bitsd(bits) if k == Kd else bitsf(bits)
            e.f.write("\tfmov\t%s, #%s\n" % (rname(rval(i.to), k), repr(v)))
    elif op == D.Oacsel:  # cimo (ext.isel), see isel.selcsel
        m = e.fn.mem[rval(i.arg[1])]
        k = i.cls
        c = ctoa[m.offset.i][m.scale >> 4]
        if KBASE(k):
            mn = "fcsel"
        else:
            mn = ("csel", "csinc", "csinv", "csneg")[m.scale & 15]
        e.f.write("\t%s\t%s, %s, %s, %s\n" % (mn, rname(rval(i.to), k), _zr(i.arg[0], k),
                                               _zr(m.base, k), c))
    elif op == D.Oaccmp:
        m = e.fn.mem[rval(i.arg[1])]
        k = i.cls
        b = rname(rval(m.base), k) if isreg(m.base) else "#%d" % m.offset.i
        e.f.write("\tccmp\t%s, %s, #%d, %s\n" % (rname(rval(i.arg[0]), k), b, m.scale & 15,
                                                  ctoa[(m.scale >> 4) & 255][m.scale >> 12]))
    elif op == D.Oaubfiz:
        k = i.cls
        c = e.fn.con[rval(i.arg[1])].i
        e.f.write("\tubfiz\t%s, %s, #%d, #%d\n" % (rname(rval(i.to), k), rname(rval(i.arg[0]), k), c & 255, c >> 8))
    elif op == D.Oatst:
        k = i.cls
        r = i.arg[1]
        e.f.write("\ttst\t%s, %s\n" % (rname(rval(i.arg[0]), k), rname(rval(r), k) if isreg(r)
                                        else "#%d" % (e.fn.con[rval(r)].i & (M64 if k == Kl else 0xffffffff))))
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
                    # fix over C QBE: the width comes from the op (a spill's storel has cls
                    # Kw, see Ocopy above, and stored only its low 4 bytes); only word and
                    # doubleword stores get their value in x17
                    assert i.op in (Ostorew, Ostorel)
                    w = i.op == Ostorew
                    e.f.write("\tfmov\t%s31, %s17\n" % ("ds"[w], "xw"[w]))
                    i.arg[0] = TMP(V31)
                    i.op = Ostores if w else Ostored
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
        if req(m.index, R) and m.offset.type in (CBits, D.CUndef) and m.scale >= 0:
            return rval(m.base), (m.offset.i if m.offset.type == CBits else 0)
    return None


def emitpair(i1, i2, e, after=()):
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
    c = None
    if _ext.isel and a1[1] == 0 and after and a1[0] != FP and a1[0] != SP and rval(r1) != a1[0] and rval(r2) != a1[0]:
        for j in after:  # the next that touches the base, if it's an add to it
            if j.op == Ocall:
                break
            if _refers(j, a1[0], e):
                c = _incby(j, a1[0], e, sz)
                if c is not None:  # it goes in too (LLVM's AArch64LoadStoreOpt)
                    j.set(Ins(Onop))
                break
    if c is not None:
        e.f.write("\t%s\t%s, %s, [%s], %d\n" % ("ldp" if ld else "stp", rname(rval(r1), k), rname(rval(r2), k), base, c))
        return 1
    off = (", %d" % a1[1]) if a1[1] else ""
    e.f.write("\t%s\t%s, %s, [%s%s]\n" % ("ldp" if ld else "stp", rname(rval(r1), k),
                                           rname(rval(r2), k), base, off))
    return 1


def emitinsl(ins, e):
    """emitins() for each instruction in a list, pairing accesses when ext.ldp is on."""
    k = 0
    n = len(ins)
    while k < n:
        if _ext.ldp and not e.fn.hasvolatile and k + 1 < n and emitpair(ins[k], ins[k + 1], e, ins[k + 2:k + 10]):
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


def emitcondjmp(b, jt, s1, s2, e, cz=None):
    """Conditional jump to s1 (condition true) or s2 from the end of b (b.link follows).
    Returns 1 if control falls through to b.link."""
    T = D.T
    c = jt - Jjf
    def branch(n, to):
        if cz is None:
            e.f.write("\tb%s\t%s%d\n" % (ctoa[c][n], T.asloc, _id0 + to.id))
        else:  # the block's last instruction, comparing with 0, was not emitted (see _cbz)
            e.f.write("\t%s\t%s%s%d\n" % (cz[0] if (c == Cieq) != n else cz[1], cz[2],
                                           T.asloc, _id0 + to.id))
    if b.link is s2:
        branch(0, s1)
        return 1
    if b.link is s1:
        branch(1, s2)
        return 1
    # neither follows: branch conditionally backwards (a loop), unconditionally forwards
    if s2.id <= b.id < s1.id:
        branch(1, s2)
        e.f.write("\tb\t%s%d\n" % (T.asloc, _id0 + s1.id))
    else:
        branch(0, s1)
        e.f.write("\tb\t%s%d\n" % (T.asloc, _id0 + s2.id))
    return 0


def _cbz(ins, jt, e):
    """cimo (ext.isel): (instructions, register) without a last "cmp x, #0" if jt tests it for
    (in)equality: cbz or cbnz does both; else (ins, None). Nothing after the jump reads flags."""
    c = jt - Jjf
    sign = c == Cislt or c == Cisge
    if not _ext.isel or (c != Cieq and c != Cine and not (sign and e.tbzok)):
        return ins, None
    n = len(ins)
    while n > 0 and ins[n - 1].op == Onop:
        n -= 1
    if n == 0 or rtype(ins[n - 1].arg[1]) != RCon or not isreg(ins[n - 1].arg[0]):
        return ins, None
    i = ins[n - 1]
    pc = e.fn.con[rval(i.arg[1])]
    if pc.type != CBits:
        return ins, None
    if sign:  # x < 0, x >= 0: the top bit, as emitcondjmp's second name is the one for "true"
        if i.op == Oacmp and pc.i == 0:
            at = "%s, #%d, " % (rname(rval(i.arg[0]), i.cls), 63 if i.cls == Kl else 31)
            return ins[:n - 1], ("tbz", "tbnz", at) if c == Cislt else ("tbnz", "tbz", at)
        return ins, None
    if i.op == Oacmp and pc.i == 0:
        return ins[:n - 1], ("cbz", "cbnz", "%s, " % rname(rval(i.arg[0]), i.cls))
    if i.op == D.Oatst and e.tbzok:
        v = pc.i & (M64 if i.cls == Kl else 0xffffffff)
        if v & (v - 1) == 0:
            bit = 0
            while (v >> bit) != 1:
                bit += 1
            return ins[:n - 1], ("tbz", "tbnz", "%s, #%d, " % (rname(rval(i.arg[0]), i.cls), bit))
    return ins, None


def _refers(i, r, e):
    """Instruction i reads or writes register r."""
    t = D.TMP(r)
    if req(i.to, t):
        return True
    for a in i.arg:
        if req(a, t):
            return True
        if rtype(a) == RMem:
            m = e.fn.mem[rval(a)]
            if req(m.base, t) or req(m.index, t):
                return True
    return False


def _incby(i, r, e, pair=0):
    """The constant of "r = r + c" or "r = r - c" on 64 bits, if i is one and c fits an access (for a pair:
    a multiple of the size pair, in 7 bits)."""
    if i.op not in (Oadd, Osub) or i.cls != Kl or not req(i.to, D.TMP(r)) or not req(i.arg[0], D.TMP(r)) \
            or rtype(i.arg[1]) != RCon or e.fn.con[rval(i.arg[1])].type != CBits:
        return None
    c = e.fn.con[rval(i.arg[1])].i
    c = c if i.op == Oadd else -c
    if pair:
        return c if c % pair == 0 and -64 <= c // pair <= 63 else None
    return c if -256 <= c <= 255 else None


def _indexed(i, r, c, how, e):
    m = Mem()
    e.fn.nmem += 1
    vgrow(e.fn.mem, e.fn.nmem)
    e.fn.mem[e.fn.nmem - 1] = m
    m.base = D.TMP(r)
    m.scale = how
    m.offset.type = CBits
    m.offset.i = c
    i.arg[0 if isload(i.op) else 1] = D.MEM(e.fn.nmem - 1)


def _access(i, e):
    """(address register, offset) of a plain load or store, whose data isn't that register."""
    if not (isload(i.op) or isstore(i.op)):
        return None
    a = i.arg[0 if isload(i.op) else 1]
    if rtype(a) == RTmp:
        r, off = rval(a), 0
    elif rtype(a) == RMem and req(e.fn.mem[rval(a)].index, R) and e.fn.mem[rval(a)].offset.type in (CBits, D.CUndef) \
            and e.fn.mem[rval(a)].scale >= 0 and isreg(e.fn.mem[rval(a)].base):
        m = e.fn.mem[rval(a)]
        r, off = rval(m.base), (m.offset.i if m.offset.type == CBits else 0)
    else:
        return None
    data = i.to if isload(i.op) else i.arg[0]
    if r == SP or r == FP or not isreg(a if rtype(a) == RTmp else e.fn.mem[rval(a)].base) or req(data, D.TMP(r)) \
            or (rtype(data) != RCon and not isreg(data)):
        return None
    return r, off


def _writeback(e):
    """cimo (ext.isel), LLVM's AArch64LoadStoreOpt: an access and an add to its base with nothing else
    at that register between them as one instruction. ldr x, [p]; add p, p, #c is ldr x, [p], #c;
    ldr x, [p, #c]; add p, p, #c and add p, p, #c; ldr x, [p] are ldr x, [p, #c]!."""
    b = e.fn.start
    while b:
        ins = [i for i in b.ins[:b.nins] if i.op not in (Onop, D.Odbgloc)]
        for n, i in enumerate(ins):
            if i.op == Onop:
                continue
            acc = _access(i, e)
            if acc is not None:
                r, off = acc
                for j in ins[n + 1:n + 9]:
                    if j.op == Ocall:
                        break
                    if _refers(j, r, e):
                        c = _incby(j, r, e)
                        if c is not None and off == 0:
                            _indexed(i, r, c, _POSTINDEX, e)
                            j.set(Ins(Onop))
                        elif c is not None and c == off:
                            _indexed(i, r, c, _PREINDEX, e)
                            j.set(Ins(Onop))
                        break
            elif i.op in (Oadd, Osub) and isreg(i.to):
                r = rval(i.to)
                c = _incby(i, r, e)
                if c is None or r == SP or r == FP:
                    continue
                for j in ins[n + 1:n + 9]:
                    if j.op == Ocall:
                        break
                    if _refers(j, r, e):
                        acc = _access(j, e)
                        if acc == (r, 0):
                            _indexed(j, r, c, _PREINDEX, e)
                            i.set(Ins(Onop))
                        break
        b = b.link


_SAFEFLAGS = {Onop, Ocopy, Oadd, Osub, Oand, Oor, Oxor, Osar, Oshr, Oshl, Omul, D.Oaubfx, D.Oasbfx, D.Oaubfiz,
              D.Oaalu, D.Oamadd, D.Oamsub, D.Odbgloc}


def _flagsfrom(e):
    """cimo (ext.isel), GCC's compare_elim, LLVM's optimizeCompareInstr: "cmp r, #0" after r = a + b,
    a - b, a & b is ands, adds or subs. The flags say the same of zero; subs and adds have other carry
    and overflow, and ands other carry, so only what reads them decides. A jump on zero alone has
    cbz, tested first."""
    b = e.fn.start
    while b:
        ins = [i for i in b.ins[:b.nins] if i.op not in (Onop, D.Odbgloc)]
        for n, i in enumerate(ins):
            if i.op != Oacmp or not isreg(i.arg[0]) or rtype(i.arg[1]) != RCon:
                continue
            pc = e.fn.con[rval(i.arg[1])]
            if pc.type != CBits or pc.i != 0:
                continue
            readers = []
            for j in ins[n + 1:]:
                if Oflag <= j.op < Oflag + NCmp:
                    readers.append(j.op - Oflag)
                elif j.op == D.Oacsel:
                    readers.append(e.fn.mem[rval(j.arg[1])].offset.i)
                elif j.op not in _SAFEFLAGS and not isload(j.op) and not isstore(j.op):
                    if j.op in (Oacmp, Oacmn, D.Oatst, Oafcmp):
                        break
                    readers = None
                    break
            else:
                jt = b.jmp.type
                if Jjf <= jt <= Jjf + NCmp - 1:
                    readers.append(jt - Jjf)
                    if len(readers) == 1 and readers[0] in (Cieq, Cine, Cislt, Cisge):
                        readers = None  # cbz, cbnz, tbz, tbnz
            if not readers:
                continue
            for j in reversed(ins[:n]):
                if j.op == Onop:
                    continue
                if req(j.to, i.arg[0]):
                    if j.cls == i.cls and j.op in (Oadd, Osub, Oand) and isreg(j.arg[0]):
                        wide = {Cieq, Cine}
                        if j.op == Oand:
                            wide |= {Cislt, Cisge, Cisgt, Cisle}
                        if all(c in wide for c in readers):
                            j.op = {Oadd: D.Oaadds, Osub: D.Oasubs, Oand: D.Oaands}[j.op]
                            i.op = Onop
                    break
                if j.op not in _SAFEFLAGS and not isload(j.op) and not isstore(j.op):
                    break
                a = j.arg[0 if isload(j.op) else 1] if isload(j.op) or isstore(j.op) else R
                if rtype(a) == RMem and e.fn.mem[rval(a)].scale < 0 and req(e.fn.mem[rval(a)].base, i.arg[0]):
                    break  # an access that wrote its base back is the other write there can be
        b = b.link


def _dropflags(e):
    """cimo (ext.isel), GCC's compare_elim: a block's only predecessor ended in the same comparison
    of the same registers, with nothing between: the flags are already right. cimo's switch does this
    (cmp w0, #3; b.eq ...; cmp w0, #3; b.lo ...)."""
    b = e.fn.start
    while b:
        if b.npred == 1:
            p = b.pred[0]
            first = next((i for i in b.ins[:b.nins] if i.op != Onop), None)
            last = next((i for i in reversed(p.ins[:p.nins]) if i.op != Onop), None)
            if first is not None and last is not None and first.op == Oacmp and last.op == Oacmp \
                    and Jjf <= p.jmp.type <= Jjf + NCmp - 1 and first.cls == last.cls \
                    and req(first.arg[0], last.arg[0]) and req(first.arg[1], last.arg[1]) \
                    and _cbz(p.ins[:p.nins], p.jmp.type, e)[1] is None:
                first.op = Onop
        b = b.link


# at least this many cases, each table at least 10% full, as LLVM except for the count: it says 4, but
# measured per element, with a tree / with a table: 4 cases at random 5.0 / 6.5 ns, 8 cases 7.5 / 7.2,
# 16 cases 9.0 / 7.6, 32 cases 11.3 / 7.8; when a few cases dominate the tree stays ahead, 4.5 / 5.6
JT_MINCASES = 16
JT_MINDENSITY = 0.1
JT_MAXSIZE = 4096
VT_MINCASES = 4
VT_MAXSIZE = 256
_njt = 0


def _switchnode(fn, b, reg, k, lo, hi, cases, leaves, nodes):
    """Add the part of a compare tree at b to cases {value: block} and leaves. A tree is what the
    frontend writes for a switch: b.eq on a constant, then b.lo to the left, b.hi... to the right, both
    on one register, and the other outcomes all reach one default block. False if it isn't one."""
    if lo == hi:  # the tree leaves out the last compare: only this value gets here
        cases[lo] = b
        return True
    live = [i for i in b.ins[:b.nins] if i.op not in (Onop, D.Odbgloc)]
    i = live[-1] if live else None
    isnode = (b.npred == 1 and len(live) == 1 and i.op == Oacmp and req(i.arg[0], reg) and i.cls == k
              and rtype(i.arg[1]) == RCon and e_con(fn, i.arg[1]) is not None
              and b.jmp.type in (Jjf + Cieq, Jjf + Cine))
    if not isnode:
        leaves.add(b)
        return True
    c = e_con(fn, i.arg[1])
    if not lo <= c <= hi or c in cases:
        return False
    eq, ne = (b.s1, b.s2) if b.jmp.type == Jjf + Cieq else (b.s2, b.s1)
    cases[c] = eq
    nodes.add(b)
    if ne.npred == 1 and not any(x.op not in (Onop, D.Odbgloc) for x in ne.ins[:ne.nins]) \
            and ne.jmp.type == Jjf + Ciult:
        nodes.add(ne)
        return (_switchnode(fn, ne.s1, reg, k, lo, c - 1, cases, leaves, nodes)
                and _switchnode(fn, ne.s2, reg, k, c + 1, hi, cases, leaves, nodes))
    return _switchnode(fn, ne, reg, k, lo, hi, cases, leaves, nodes)


def _constcase(c):
    """(register, class, value, block to go to) of a block that only sets a register to a constant."""
    live = [x for x in c.ins[:c.nins] if x.op not in (Onop, D.Odbgloc)]
    if len(live) != 1 or live[0].op != Ocopy or not isreg(live[0].to) or c.jmp.type != Jjmp \
            or rtype(live[0].arg[0]) != RCon or KBASE(live[0].cls) != 0:
        return None
    return live[0].to, live[0].cls, live[0].arg[0], c.s1


def _valuetable(fn, cases, d):
    """LLVM's SwitchToLookupTable: every case only sets one register to a constant and all go on to the
    same block: (register, class, values, where to go, the case blocks that go) or None. Holes take the
    default's value if it is the same kind of block, or there's no table. Density as LLVM's 40%."""
    size = max(cases) + 1
    if len(cases) < VT_MINCASES or size > VT_MAXSIZE or len(cases) * 10 < size * 4:
        return None
    shapes = {v: _constcase(c) for v, c in cases.items()}
    first = next(iter(shapes.values()))
    if first is None:
        return None
    to, k, _, join = first
    for s in shapes.values():
        if s is None or not req(s[0], to) or s[1] != k or s[3] is not join:
            return None
    def bits(r):
        c = fn.con[rval(r)]
        return c.i if c.type == CBits else None
    values = {v: bits(s[2]) for v, s in shapes.items()}
    if None in values.values():
        return None
    dv = None
    ds = _constcase(d)
    if ds is not None and req(ds[0], to) and ds[1] == k and ds[3] is join:
        dv = bits(ds[2])
    if dv is None and len(cases) != size:
        return None
    mask = (1 << (64 if k == Kl else 32)) - 1
    gone = {c for c in cases.values() if sum(1 for x in cases.values() if x is c) == c.npred}
    return to, k, [values.get(v, dv) & mask for v in range(size)], join, gone


def e_con(fn, r):
    c = fn.con[rval(r)]
    return c.i if c.type == CBits and c.i >= 0 else None


def _jumptables(e):
    """cimo (ext.isel): {root block: (position of its compare, register, class, table, default)} for
    compare trees worth a table, and the blocks of the tree nothing else reaches. Found here, after
    register allocation, because QBE's blocks have two successors at most."""
    fn = e.fn
    tables = {}
    dead = set()
    b = fn.start
    while b:
        n = b.nins
        while n > 0 and b.ins[n - 1].op in (Onop, D.Odbgloc):
            n -= 1
        i = b.ins[n - 1] if n else None
        if b not in dead and i is not None and i.op == Oacmp and isreg(i.arg[0]) and rtype(i.arg[1]) == RCon \
                and e_con(fn, i.arg[1]) is not None and b.jmp.type in (Jjf + Cieq, Jjf + Cine):
            cases, leaves, nodes = {}, set(), set()
            top = (1 << (64 if i.cls == Kl else 32)) - 1
            c = e_con(fn, i.arg[1])
            eq, ne = (b.s1, b.s2) if b.jmp.type == Jjf + Cieq else (b.s2, b.s1)
            cases[c] = eq
            ok = True
            if ne.npred == 1 and not any(x.op not in (Onop, D.Odbgloc) for x in ne.ins[:ne.nins]) \
                    and ne.jmp.type == Jjf + Ciult:
                nodes.add(ne)
                ok = (_switchnode(fn, ne.s1, i.arg[0], i.cls, 0, c - 1, cases, leaves, nodes)
                      and _switchnode(fn, ne.s2, i.arg[0], i.cls, c + 1, top, cases, leaves, nodes))
            else:
                ok = _switchnode(fn, ne, i.arg[0], i.cls, 0, top, cases, leaves, nodes)
            vt = _valuetable(fn, cases, next(iter(leaves))) if ok and len(leaves) == 1 else None
            if vt is not None:
                tables[b] = (n - 1, i.arg[0], i.cls, [next(iter(leaves)), vt[3]], next(iter(leaves)), vt)
                dead |= nodes | vt[4]
            elif ok and len(leaves) == 1 and len(cases) >= JT_MINCASES:
                size = max(cases) + 1
                if size <= JT_MAXSIZE and len(cases) >= JT_MINDENSITY * size and size - 1 <= 4095:
                    d = next(iter(leaves))
                    tables[b] = (n - 1, i.arg[0], i.cls, [cases.get(v, d) for v in range(size)], d, None)
                    dead |= nodes
        b = b.link
    return tables, dead


def _emitjumptable(root, jt, e):
    """Replaces the compare and branch at the end of root: bounds check, then a jump through the
    table. x17 is never allocated. Returns the (label, table) to emit after the function."""
    global _njt
    T = D.T
    _, reg, k, table, default, values = jt
    _njt += 1
    name = "\"%sjt%d\"" % (T.asloc, _njt)
    r = rname(rval(reg), k)
    e.f.write("\tcmp\t%s, #%d\n" % (r, (len(values[2]) if values is not None else len(table)) - 1))
    e.f.write("\tb.hi\t%s%d\n" % (T.asloc, _id0 + default.id))
    c = Con(CAddr)
    c.sym.id = intern(name)
    loadaddr(c, "x17", e)
    if values is not None:
        to, cls, data, join, _ = values
        sh = 3 if cls == Kl else 2
        e.f.write("\tldr\t%s, [x17, %s]\n" % (rname(rval(to), cls), "%s, uxtw #%d" % (r, sh) if k == Kw
                                                else "%s, lsl #%d" % (r, sh)))
        e.f.write("\tb\t%s%d\n" % (T.asloc, _id0 + join.id))
        return name, data, 8 if cls == Kl else 4
    e.f.write("\tldr\tx17, [x17, %s]\n" % ("%s, uxtw #3" % r if k == Kw else "%s, lsl #3" % r))
    e.f.write("\tbr\tx17\n")
    return name, table, 0


def _siblingcall(b, e):
    """cimo (ext.isel): the direct call a function ends with, if it can be a jump after the epilogue
    (LLVM's and gcc's sibling calls): nothing lives in this frame by then, as it has no stack slots,
    and arguments on the stack would be followed by an instruction that frees them."""
    fn = e.fn
    if not _ext.isel or fn.slot or fn.vararg or fn.dynalloc:
        return None
    if b.jmp.type != Jret0:  # or a jump to a block that only returns
        s = b.s1
        if b.jmp.type != Jjmp or s is None or s.jmp.type != Jret0 or any(
                i.op != Onop for i in s.ins[:s.nins]):
            return None
    n = b.nins
    while n > 0 and (b.ins[n - 1].op == Onop or (b.ins[n - 1].op == Ocopy
                                                 and req(b.ins[n - 1].to, b.ins[n - 1].arg[0]))):
        n -= 1  # a register copied to itself: the result stays in x0
    if n == 0 or b.ins[n - 1].op != Ocall or rtype(b.ins[n - 1].arg[0]) != RCon:
        return None
    c = fn.con[rval(b.ins[n - 1].arg[0])]
    if c.type != CAddr or (c.sym.type & SThr) or c.i:
        return None
    return n - 1


def _frameless(e):
    """cimo (ext.isel): a leaf that keeps nothing on the stack needs no frame: no saved register
    (x30 is allocatable), slot, stack argument, call, stack pointer change, or trap (which
    unwinds through the frame chain)."""
    fn = e.fn
    if (not _ext.isel or e.frame or fn.vararg or fn.dynalloc or fn.slot or (fn.reg & BIT(LR))
            or (_ext.stkchk and fn.name in _ext.stkchk)):
        return False
    b = fn.start
    while b:
        if b.jmp.type == Jhlt:
            return False
        for i in b.ins[:b.nins]:
            if i.op in (Ocall, Oaddr, Osalloc) or rtype(i.to) == RSlot \
                    or rtype(i.arg[0]) == RSlot or rtype(i.arg[1]) == RSlot:
                return False
        b = b.link
    return True


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

    frameless = _frameless(e)
    if frameless:
        pass
    elif e.frame + 16 <= 512:
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
    if not frameless:
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
    if _ext.isel:
        # tbz reaches 32 KB: only in a function certainly smaller, at 16 bytes per instruction at most
        nins = 0
        b = e.fn.start
        while b:
            nins += b.nins + 1
            b = b.link
        e.tbzok = nins < 1500
        _writeback(e)
        _flagsfrom(e)
        _dropflags(e)
        tables, deadblocks = _jumptables(e)
        for jt in tables.values():
            dlbl = set(dlbl) | set(jt[3]) | {jt[4]}
    else:
        e.tbzok = False
        tables, deadblocks = {}, ()
    emitted = []
    lbl = 0
    b = e.fn.start
    while b:
        if b in deadblocks:
            lbl = 1
            b = b.link
            continue
        if lbl or b.npred > 1 or b in dlbl:
            e.f.write("%s%d:\n" % (T.asloc, _id0 + b.id))
        if b in tables:
            emitinsl(b.ins[:tables[b][0]], e)
            emitted.append(_emitjumptable(b, tables[b], e))
            lbl = 1
            b = b.link
            continue
        jt = b.jmp.type
        body, cz = b.ins[:b.nins], None
        sib = _siblingcall(b, e)
        if sib is not None:
            sibling = body[sib]
            body = body[:sib]
            jt = Jret0
        if jt != Jjmp and jt != Jhlt and jt != Jret0 and _ext.layout and b not in dup:
            body, cz = _cbz(body, jt, e)
        emitinsl(body, e)
        lbl = 1
        if b in dup:
            hins, hjt, hs1, hs2 = dup[b]
            hins, hcz = _cbz(hins, hjt, e)
            emitinsl([i.copy() for i in hins], e)
            if emitcondjmp(b, hjt, hs1, hs2, e, hcz):
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
            if frameless:
                e.f.write("\tret\n")
                b = b.link
                continue
            tailto = None
            if sib is not None:
                l = str_(e.fn.con[rval(sibling.arg[0])].sym.id)
                tailto = ("" if l[0] == '"' else T.assym) + l
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
            e.f.write("\tb\t%s\n" % tailto if tailto else "\tret\n")
        else:
            if jt != Jjmp and _ext.layout:
                c = jt - Jjf
                if c < 0 or c > NCmp:
                    die("unhandled jump %d", jt)
                if emitcondjmp(b, jt, b.s1, b.s2, e, cz):
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
    for name, table, valsize in emitted:
        e.f.write(".section __DATA,__const\n" if T.apple else ".section .rodata\n")
        e.f.write(".p2align 3\n%s:\n" % name)
        for t in table:
            if valsize:
                e.f.write("\t%s\t%d\n" % (".quad" if valsize == 8 else ".int", t))
            else:
                e.f.write("\t.quad\t%s%d\n" % (T.asloc, _id0 + t.id))
        e.f.write(".text\n")
    _id0 += e.fn.nblk
    if not T.apple:
        elf_emitfnfin(fn.name, out)
