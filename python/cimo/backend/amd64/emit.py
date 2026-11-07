"""Port of QBE amd64/emit.c."""

from .. import defs as D
from .. import ext as _ext
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
                    Oflagfne, Onop, Oneg, Ocopy, Ocall, Osalloc, Odbgloc, Oxsel, Ogetsp,
                    Osetsp, Ofsqrt, Ofabs, Obclz, Obctz, Obswap, Obswap16)
from ..util import die, isreg, str_
from ..emit import emitfnlnk, emitdbgloc, elf_emitfnfin, stashbits
from .regs import (RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8, R9, R10, R11, R12, R13, R14,
                   R15, XMM0, XMM15, NCLR_SYSV, NCLR_WIN)


def _cimowin():
    """cimo's Windows x64 target (targ.T_amd64_windows), not QBE's amd64_win: xmm6-xmm15
    callee-saved, stack probes (r11 is reserved for them), externs through the GOT."""
    return D.T.name == 'amd64_windows'


class E:
    __slots__ = ('f', 'fn', 'fp', 'fsz', 'nclob', 'xsave', 'xarea', 'qthird', 'vzu')

    def __init__(self, f, fn):
        self.f = f
        self.fn = fn
        self.fp = 0
        self.fsz = 0
        self.nclob = 0
        self.xsave = []  # cimo (amd64_windows): callee-saved xmm registers, saved below the
        self.xarea = 0   # pushes in xarea bytes
        self.qthird = R  # cimo (ext.vector): qfma0's addend, for the qfma1 after it
        # cimo (AVX2): vzeroupper before each call and the return of a function that uses ymm registers, so code
        # around it, SSE in callers and callees, doesn't pay for their dirty upper halves; and inside it, no legacy
        # SSE at all (_Vex): Sapphire Rapids ran SSE after 256-bit code up to 250x slower (nbody_acc, conv3x3, gemm)
        self.vzu = _ext.qbytes() == 32 and _usesq(fn)
        if self.vzu:
            self.f = _Vex(f)


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
    (D.Oxcmpe, Ks, "comiss %S0, %S1"),  # cimo: the signaling comparison (isel.selcmp)
    (D.Oxcmpe, Kd, "comisd %D0, %D1"),
    (Oxtest, Ki, "test%k %0, %1"),
    (D.Oxaddm, Ki, "add%k %0, %M1"),
    (D.Oxsubm, Ki, "sub%k %0, %M1"),
    (D.Oxandm, Ki, "and%k %0, %M1"),
    (D.Oxorm, Ki, "or%k %0, %M1"),
    (D.Oxxorm, Ki, "xor%k %0, %M1"),
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
            return 4 * -s - 8 + e.fsz + e.nclob * 8 + e.xarea
        else:
            return 4 * -s
    elif e.fp == RSP:
        return 4 * s + e.nclob * 8 + e.xarea
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


_FALSEDEP = frozenset((Ofsqrt, Oswtof, Osltof, Oexts, Otruncd))


def _xreg(r):
    return rtype(r) == RTmp and isreg(r) and rval(r) >= XMM0


def _xcopy(to, a, e):
    """cimo: movss and movsd between registers keep the destination's upper bits, so they wait for what last wrote it,
    which behind a division or a square root is most of the loop (nbody_acc on Intel). movaps doesn't, as gcc and clang."""
    e.f.write("\tmovaps %%xmm%d, %%xmm%d\n" % (rval(a) - XMM0, rval(to) - XMM0))


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


_QX15 = "%xmm15"
# (mnemonic, commutes). Two addresses: dest = dest op source
_QTWO = {'qaddb': ('paddb', 1), 'qaddh': ('paddw', 1), 'qaddw': ('paddd', 1), 'qaddl': ('paddq', 1),
         'qadds': ('addps', 1), 'qaddd': ('addpd', 1), 'qsubb': ('psubb', 0), 'qsubh': ('psubw', 0),
         'qsubw': ('psubd', 0), 'qsubl': ('psubq', 0), 'qsubs': ('subps', 0), 'qsubd': ('subpd', 0),
         'qmulh': ('pmullw', 1), 'qmuls': ('mulps', 1), 'qmuld': ('mulpd', 1), 'qdivs': ('divps', 0), 'qdivd': ('divpd', 0),
         'qand': ('pand', 1), 'qor': ('por', 1), 'qxor': ('pxor', 1), 'qxandn': ('pandn', 0),
         'qminh': ('pminsw', 1), 'qmaxh': ('pmaxsw', 1), 'qumin' + 'b': ('pminub', 1), 'qumaxb': ('pmaxub', 1),
         'qceqb': ('pcmpeqb', 1), 'qceqh': ('pcmpeqw', 1), 'qceqw': ('pcmpeqd', 1),
         'qcgtb': ('pcmpgtb', 0), 'qcgth': ('pcmpgtw', 0), 'qcgtw': ('pcmpgtd', 0),
         'qxmuludq': ('pmuludq', 1), 'qxlo': ('punpckldq', 0), 'qxhi': ('punpckhdq', 0),
         'qxmins': ('minps', 0), 'qxmind': ('minpd', 0), 'qxmaxs': ('maxps', 0), 'qxmaxd': ('maxpd', 0),
         'qceqs': ('cmpps $0,', 1), 'qceqd': ('cmppd $0,', 1), 'qxunords': ('cmpps $3,', 1), 'qxunordd': ('cmppd $3,', 1),
         'qtbl': ('pshufb', 0)}  # qtbl: only with AVX2 (vpshufb: within each 16-byte half; the frontend's 16-byte vectors)
_QSWAPPED = {'qcgts': 'cmpps $1,', 'qcgtd': 'cmppd $1,', 'qcges': 'cmpps $2,', 'qcged': 'cmppd $2,'}  # b < a, b <= a
_QSHIFT = {'qshlh': 'psllw', 'qshlw': 'pslld', 'qshll': 'psllq', 'qshrh': 'psrlw', 'qshrw': 'psrld', 'qshrl': 'psrlq',
           'qsarh': 'psraw', 'qsarw': 'psrad'}


def _qx(r):
    return "%%xmm%d" % (rval(r) - XMM0)


def _qr(r, sz):
    return "%%%s" % regtoa(rval(r), sz)


def _qtwo(mn, to, a, b, commutes, f):
    """to = a op b, in two addresses, through xmm15 if b is in the way."""
    if to == b and to != a:
        if commutes:
            a, b = b, a
        else:
            f.write("\tmovaps %s, %s\n\tmovaps %s, %s\n" % (_qx(b), _QX15, _qx(a), _qx(to)))
            f.write("\t%s %s, %s\n" % (mn, _QX15, _qx(to)))
            return
    if to != a:
        f.write("\tmovaps %s, %s\n" % (_qx(a), _qx(to)))
    f.write("\t%s %s, %s\n" % (mn, _qx(b), _qx(to)))


def _qcopy(i, e):
    f = e.f
    to, a = i.to, i.arg[0]
    if to == R or a == R or to == a:
        return
    if rtype(to) == RSlot and rtype(a) == RSlot:
        f.write("\tmovups %d(%%%s), %s\n" % (slot(a, e), regtoa(e.fp, SLong), _QX15))
        f.write("\tmovups %s, %d(%%%s)\n" % (_QX15, slot(to, e), regtoa(e.fp, SLong)))
    elif rtype(to) == RSlot:
        emitf("movups %0, %M=", i, e)
    elif rtype(a) == RSlot:
        emitf("movups %M0, %=", i, e)
    else:
        f.write("\tmovaps %s, %s\n" % (_qx(a), _qx(to)))


# AVX2, 256 bits: three addresses, dest = a op b, the VEX form of each SSE2 one, and what qlower leaves for AVX2
_QY = dict((k, 'v' + mn) for k, (mn, _) in _QTWO.items())
_QY.update({'qmulw': 'vpmulld', 'qceql': 'vpcmpeqq', 'qcgtl': 'vpcmpgtq', 'qminb': 'vpminsb', 'qminw': 'vpminsd',
            'qmaxb': 'vpmaxsb', 'qmaxw': 'vpmaxsd', 'quminh': 'vpminuw', 'quminw': 'vpminud', 'qumaxh': 'vpmaxuw',
            'qumaxw': 'vpmaxud', 'qshlvw': 'vpsllvd', 'qshlvl': 'vpsllvq', 'qshrvw': 'vpsrlvd', 'qshrvl': 'vpsrlvq',
            'qsarvw': 'vpsravd'})
_QYSWAPPED = dict((k, 'v' + mn) for k, mn in _QSWAPPED.items())


def _qy(r):
    return "%%ymm%d" % (rval(r) - XMM0)


# __builtin_convertvector's ops (frontend vecops 'convertx'), AVX2 on 32 bytes: widen the low 16 bytes (or, through
# ymm15, the high), narrow a pair into one (packed within each half, then vpermq puts the halves in order), convert
_QYW = {'b': 'bw', 'h': 'wd', 'w': 'dq'}
_QYCVT = frozenset(['q%s%s%s' % (s, h, x) for s in ('sx', 'ux') for h in 'lh' for x in 'bhw'] +
                   ['qfcvtl', 'qfcvth', 'qfcvtn', 'qnrwh', 'qnrww', 'qnrwl', 'qitofs', 'qftois'])


def _qx15(r):
    return "%%xmm%d" % (rval(r) - XMM0)


def _qycvt(f, name, to, a, b):
    w = f.write
    if name in ('qitofs', 'qftois'):
        w("\t%s %s, %s\n" % ('vcvtdq2ps' if name == 'qitofs' else 'vcvttps2dq', _qy(a), _qy(to)))
    elif name in ('qfcvtl', 'qfcvth'):
        src = _qx15(a)
        if name == 'qfcvth':
            w("\tvextractf128 $1, %s, %%xmm15\n" % _qy(a))
            src = '%xmm15'
        w("\tvcvtps2pd %s, %s\n" % (src, _qy(to)))
    elif name == 'qfcvtn':  # lo's 4 doubles, then hi's
        w("\tvcvtpd2ps %s, %%xmm15\n\tvcvtpd2ps %s, %s\n\tvinsertf128 $1, %s, %%ymm15, %s\n"
          % (_qy(a), _qy(b), _qx15(to), _qx15(to), _qy(to)))
    elif name[:3] in ('qsx', 'qux'):
        mn = ('vpmovsx' if name[1] == 's' else 'vpmovzx') + _QYW[name[-1]]
        src = _qx15(a)
        if name[3] == 'h':
            w("\tvextracti128 $1, %s, %%xmm15\n" % _qy(a))
            src = '%xmm15'
        w("\t%s %s, %s\n" % (mn, src, _qy(to)))
    elif name in ('qnrwh', 'qnrww'):  # the low byte or half of each lane, lo's then hi's
        bits, sl, sr, pack = (8, 'vpsllw', 'vpsrlw', 'vpackuswb') if name == 'qnrwh' else (16, 'vpslld', 'vpsrld', 'vpackusdw')
        w("\t%s $%d, %s, %%ymm15\n\t%s $%d, %%ymm15, %%ymm15\n" % (sl, bits, _qy(a), sr, bits))
        w("\t%s $%d, %s, %s\n\t%s $%d, %s, %s\n" % (sl, bits, _qy(b), _qy(to), sr, bits, _qy(to), _qy(to)))
        w("\t%s %s, %%ymm15, %s\n\tvpermq $0xd8, %s, %s\n" % (pack, _qy(to), _qy(to), _qy(to), _qy(to)))
    else:  # qnrwl: the low dword of each qword
        w("\tvpshufd $0x08, %s, %%ymm15\n\tvpermq $0x08, %%ymm15, %%ymm15\n" % _qy(a))
        w("\tvpshufd $0x08, %s, %s\n\tvpermq $0x08, %s, %s\n" % (_qy(b), _qy(to), _qy(to), _qy(to)))
        w("\tvinserti128 $1, %s, %%ymm15, %s\n" % (_qx15(to), _qy(to)))


def _narrowmasks(k):
    """vpshufb masks for channel k of 8 pixels (24 bytes) as 16-bit lanes: from the 16 bytes at 0 and the 16 at 8;
    0x80 makes a zero byte, so the lanes come out zero-extended."""
    a, b = [0x80] * 16, [0x80] * 16
    for i in range(8):
        x = 3 * i + k
        if x < 16:
            a[2 * i] = x
        else:
            b[2 * i] = x - 8
    return a, b


_LOWBYTES = list(range(0, 16, 2)) + [0x80] * 8  # each 16-bit lane's low byte, packed in the low 8


def _narrowins(i, e, y):
    """cimo: vectorize's loops over bytes (qldbh, qld3bh, qstbh), VEX; y: 16 lanes in a ymm, else 8 in an xmm."""
    from ..emit import stashvec
    f = e.f
    op = i.op
    r = (lambda x: "%%ymm%d" % (rval(x) - XMM0)) if y else (lambda x: "%%xmm%d" % (rval(x) - XMM0))
    s15 = "%ymm15" if y else "%xmm15"
    twice = (lambda m: m + m) if y else (lambda m: m)
    if op == D.Oqxwin:
        base = "%%%s" % regtoa(rval(i.arg[0]), SLong)
        off = e.fn.con[rval(i.arg[1])].i
        if y:
            f.write("\tvmovdqu %d(%s), %%xmm15\n\tvinserti128 $1, %d(%s), %%ymm15, %s\n" % (off, base, off + 24, base, r(i.to)))
        else:
            f.write("\tvmovdqu %d(%s), %s\n" % (off, base, r(i.to)))
    elif op == D.Oqxshufb:
        m = e.fn.con[rval(i.arg[1])].i
        mask = _narrowmasks(m >> 1)[m & 1]
        f.write("\tvpshufb %s(%%rip), %s, %s\n" % (stashvec(twice(mask)), r(i.arg[0]), r(i.to)))
    elif op == D.Oqldbh:
        f.write("\tvpmovzxbw %s, %s\n" % (_memtext(i.arg[0], e), r(i.to)))
    elif op == D.Oqld3bh:
        base = "%%%s" % regtoa(rval(i.arg[0]), SLong)
        ma, mb = _narrowmasks(e.fn.con[rval(i.arg[1])].i)
        for off, m, dst in ((0, ma, r(i.to)), (8, mb, s15)):
            f.write("\tvmovdqu %d(%s), %%xmm15\n" % (off, base))
            if y:  # the next 8 pixels, 24 bytes on, in the upper half
                f.write("\tvinserti128 $1, %d(%s), %%ymm15, %%ymm15\n" % (off + 24, base))
            f.write("\tvpshufb %s(%%rip), %s, %s\n" % (stashvec(twice(m)), s15, dst))
        f.write("\tvpor %s, %s, %s\n" % (s15, r(i.to), r(i.to)))
    else:  # qstbh
        f.write("\tvpshufb %s(%%rip), %s, %s\n" % (stashvec(twice(_LOWBYTES)), r(i.arg[0]), s15))
        if y:
            f.write("\tvpermq $8, %%ymm15, %%ymm15\n\tvmovdqu %%xmm15, %s\n" % _memtext(i.arg[1], e))
        else:
            f.write("\tvmovsd %%xmm15, %s\n" % _memtext(i.arg[1], e))
    return True


def _memtext(ref, e):
    """A memory operand as emitf writes it (%M)."""
    import io
    f, e.f = e.f, io.StringIO()
    try:
        emitf("%M0", Ins(D.Onop, D.Kl, R, ref, R), e)
        return e.f.getvalue().strip()
    finally:
        e.f = f


def _emitf_y(fmt, i, e):
    """emitf, with the vector register named as a ymm (memory operands only name general registers)."""
    import io
    f, e.f = e.f, io.StringIO()
    try:
        emitf(fmt, i, e)
        text = e.f.getvalue()
    finally:
        e.f = f
    f.write(text.replace('%xmm', '%ymm'))


def _qcopy_y(i, e):
    f = e.f
    to, a = i.to, i.arg[0]
    if to == R or a == R or to == a:
        return
    if rtype(to) == RSlot and rtype(a) == RSlot:
        f.write("\tvmovups %d(%%%s), %%ymm15\n" % (slot(a, e), regtoa(e.fp, SLong)))
        f.write("\tvmovups %%ymm15, %d(%%%s)\n" % (slot(to, e), regtoa(e.fp, SLong)))
    elif rtype(to) == RSlot:
        _emitf_y("vmovups %0, %M=", i, e)
    elif rtype(a) == RSlot:
        _emitf_y("vmovups %M0, %=", i, e)
    else:
        f.write("\tvmovaps %s, %s\n" % (_qy(a), _qy(to)))


def _qswapslot(i, e, copy):
    """A swap of a register and a stack slot, through xmm15/ymm15 (register allocation's cycle of moves through a
    spilled phi). Two slots would need two scratch registers."""
    a, b = i.arg[0], i.arg[1]
    if rtype(a) == RSlot and rtype(b) == RSlot:
        _swap2slots(i, e)
        return
    reg, sl = (b, a) if rtype(a) == RSlot else (a, b)
    x15 = TMP(XMM0 + 15)
    for to, frm in ((x15, sl), (sl, reg), (reg, x15)):
        copy(Ins(Ocopy, D.Kq, to, frm, R), e)


def _swap2slots(i, e):
    """A swap of two stack slots, 8 bytes (or 4) at a time through r11 (kept for stack probes, never allocated):
    xmm15 can't be used, a float copy between slots goes through it."""
    n = _ext.qbytes() if i.cls == D.Kq else 8 if KWIDE(i.cls) else 4
    fp = regtoa(e.fp, SLong)
    sfx, r = ('q', '%r11') if n >= 8 else ('l', '%r11d')
    for k in range(0, n, 8 if n >= 8 else 4):
        a, b = slot(i.arg[0], e) + k, slot(i.arg[1], e) + k
        e.f.write("\tmov%s %d(%%%s), %s\n\txchg%s %s, %d(%%%s)\n\tmov%s %s, %d(%%%s)\n"
                  % (sfx, a, fp, r, sfx, r, b, fp, sfx, r, a, fp))


def _qins_y(i, e):
    """_qins on 256 bits (AVX2, ext.qbytes() == 32)."""
    f = e.f
    op = i.op
    if op == Ocopy and i.cls == D.Kq:
        _qcopy_y(i, e)
        return True
    if op == Oswap and i.cls == D.Kq:
        if rtype(i.arg[0]) == RSlot or rtype(i.arg[1]) == RSlot:  # a cycle of moves through a spilled phi
            _qswapslot(i, e, _qcopy_y)
            return True
        a, b = _qy(i.arg[0]), _qy(i.arg[1])
        f.write("\tvmovaps %s, %%ymm15\n\tvmovaps %s, %s\n\tvmovaps %%ymm15, %s\n" % (a, b, a, b))
        return True
    if op in (D.Oqldbh, D.Oqld3bh, D.Oqstbh, D.Oqxwin, D.Oqxshufb):
        return _narrowins(i, e, True)
    if (op == Oload and i.cls == D.Kq) or op == D.Oqload:
        _emitf_y("vmovups %M0, %=", i, e)
        return True
    if op == D.Oqload16:  # 16 bytes in both halves
        _emitf_y("vbroadcasti128 %M0, %=", i, e)
        return True
    if op == D.Oqldh:  # F16C: 8 halves to floats
        _emitf_y("vcvtph2ps %M0, %=", i, e)
        return True
    if op == D.Oqsth:  # 8 floats rounded (MXCSR's mode) to halves
        f.write("\tvcvtps2ph $4, %s, %s\n" % (_qy(i.arg[0]), _memtext(i.arg[1], e)))
        return True
    if op == D.Oqstore16:  # the lower half
        emitf("vmovups %0, %M1", i, e)
        return True
    if op == D.Oqstore:
        _emitf_y("vmovups %0, %M1", i, e)
        return True
    if not D.isqop(op):
        return False
    name = optab[op].name
    to, a, b = i.to, i.arg[0], i.arg[1]
    if name == 'qfma0':
        e.qthird = a
        return True
    if name in ('qfma1s', 'qfma1d'):
        _fma3(f, 'p' + name[-1], to, a, b, e.qthird, y=True)
        return True
    if name in ('qxfmams', 'qxfmamd'):
        _fma3(f, 'p' + name[-1], to, a, _memtext(b, e), e.qthird, y=True)
        return True
    if name in _QYCVT:
        _qycvt(f, name, to, a, b)
    elif name in _QY:
        f.write("\t%s %s, %s, %s\n" % (_QY[name], _qy(b), _qy(a), _qy(to)))
    elif name in _QYSWAPPED:
        f.write("\t%s %s, %s, %s\n" % (_QYSWAPPED[name], _qy(a), _qy(b), _qy(to)))
    elif name in _QSHIFT:
        f.write("\tv%s $%d, %s, %s\n" % (_QSHIFT[name], e.fn.con[rval(b)].i, _qy(a), _qy(to)))
    elif name == 'qxshufd':
        f.write("\tvpshufd $%d, %s, %s\n" % (e.fn.con[rval(b)].i, _qy(a), _qy(to)))
    elif name in ('qsqrts', 'qsqrtd'):
        f.write("\tvsqrtp%s %s, %s\n" % (name[-1], _qy(a), _qy(to)))
    elif name in ('qxtruncd', 'qxfloord'):
        f.write("\tvroundpd $%d, %s, %s\n" % (3 if name == 'qxtruncd' else 1, _qy(a), _qy(to)))
    elif name.startswith('qdup'):
        x = name[-1]
        y = _qy(to)
        if rtype(a) == RCon:  # isel: 0 and -1 only
            f.write("\t%s %s, %s, %s\n" % ("vpxor" if e.fn.con[rval(a)].i == 0 else "vpcmpeqd", y, y, y))
        elif x in 'sd':
            f.write("\tvbroadcasts%s %s, %s\n" % (x, _qx(a), y))
        else:
            f.write("\t%s %s, %%xmm15\n" % (("movq", _qr(a, SLong)) if x == 'l' else ("movd", _qr(a, SWord))))
            f.write("\tvpbroadcast%s %%xmm15, %s\n" % ({'b': 'b', 'h': 'w', 'w': 'd', 'l': 'q'}[x], y))
    elif name.startswith('qget'):
        x = name[-1]
        n = e.fn.con[rval(b)].i
        half = 16 // {'b': 1, 'h': 2, 'w': 4, 'l': 8, 's': 4, 'd': 8}[x]
        if n >= half:  # the upper 128 bits first
            f.write("\tvextracti128 $1, %s, %%xmm15\n" % _qy(a))
            _qget(x, i, e, "%xmm15", n - half)
        else:
            _qget(x, i, e, _qx(a), n)
    else:
        die("no AVX2 instruction for %s", name)
    return True


def _qins(i, e):
    """cimo (ext.vector): one instruction on 128 bits; amd64/qlower.py has left only those SSE2 has. True if it was one."""
    if _ext.qbytes() == 32:
        return _qins_y(i, e)
    f = e.f
    op = i.op
    if op == Ocopy and i.cls == D.Kq:
        _qcopy(i, e)
        return True
    if op == Oswap and i.cls == D.Kq:
        if rtype(i.arg[0]) == RSlot or rtype(i.arg[1]) == RSlot:  # a cycle of moves through a spilled phi
            _qswapslot(i, e, _qcopy)
            return True
        a, b = _qx(i.arg[0]), _qx(i.arg[1])
        f.write("\tmovaps %s, %s\n\tmovaps %s, %s\n\tmovaps %s, %s\n" % (a, _QX15, b, a, _QX15, b))
        return True
    if op in (D.Oqldbh, D.Oqld3bh, D.Oqstbh, D.Oqxwin, D.Oqxshufb):
        return _narrowins(i, e, False)
    if (op == Oload and i.cls == D.Kq) or op == D.Oqload:
        emitf("movups %M0, %=", i, e)
        return True
    if op == D.Oqldh:  # F16C (with AVX): 4 halves to floats
        emitf("vcvtph2ps %M0, %=", i, e)
        return True
    if op == D.Oqsth:
        f.write("\tvcvtps2ph $4, %s, %s\n" % (_qx(i.arg[0]), _memtext(i.arg[1], e)))
        return True
    if op == D.Oqstore:
        emitf("movups %0, %M1", i, e)
        return True
    if not D.isqop(op):
        return False
    name = optab[op].name
    if name == 'qfma0':
        e.qthird = i.arg[0]  # with the next one
        return True
    if name in ('qfma1s', 'qfma1d'):
        _fma3(f, 'p' + name[-1], i.to, i.arg[0], i.arg[1], e.qthird)
        return True
    if name in ('qxfmams', 'qxfmamd'):
        _fma3(f, 'p' + name[-1], i.to, i.arg[0], _memtext(i.arg[1], e), e.qthird)
        return True
    if name in _QTWO:
        mn, commutes = _QTWO[name]
        _qtwo(mn, i.to, i.arg[0], i.arg[1], commutes, f)
    elif name in _QSWAPPED:
        _qtwo(_QSWAPPED[name], i.to, i.arg[1], i.arg[0], 0, f)
    elif name in _QSHIFT:
        if i.to != i.arg[0]:
            f.write("\tmovaps %s, %s\n" % (_qx(i.arg[0]), _qx(i.to)))
        f.write("\t%s $%d, %s\n" % (_QSHIFT[name], e.fn.con[rval(i.arg[1])].i, _qx(i.to)))
    elif name == 'qxshufd':
        f.write("\tpshufd $%d, %s, %s\n" % (e.fn.con[rval(i.arg[1])].i, _qx(i.arg[0]), _qx(i.to)))
    elif name in ('qsqrts', 'qsqrtd'):
        f.write("\tsqrtp%s %s, %s\n" % (name[-1], _qx(i.arg[0]), _qx(i.to)))
    elif name.startswith('qdup'):
        _qdup(name[-1], i, e)
    elif name.startswith('qget'):
        _qget(name[-1], i, e)
    else:
        die("no x86-64 instruction for %s", name)
    return True


def _qdup(x, i, e):
    f = e.f
    to = _qx(i.to)
    a = i.arg[0]
    if rtype(a) == RCon:  # isel: 0 and -1 only
        f.write("\t%s %s, %s\n" % ("pxor" if e.fn.con[rval(a)].i == 0 else "pcmpeqd", to, to))
    elif x in 'sd':
        f.write("\tpshufd $%d, %s, %s\n" % (0 if x == 's' else 0x44, _qx(a), to))
    elif x == 'l':
        f.write("\tmovq %s, %s\n\tpunpcklqdq %s, %s\n" % (_qr(a, SLong), to, to, to))
    else:
        f.write("\tmovd %s, %s\n" % (_qr(a, SWord), to))
        if x == 'b':
            f.write("\tpunpcklbw %s, %s\n" % (to, to))
        if x in 'bh':
            f.write("\tpshuflw $0, %s, %s\n" % (to, to))
        f.write("\tpshufd $0, %s, %s\n" % (to, to))


def _qget(x, i, e, a=None, n=None):
    """Lane n of a (a register's name), 128 bits."""
    f = e.f
    if n is None:
        n = e.fn.con[rval(i.arg[1])].i
    if a is None:
        a = _qx(i.arg[0])
    if x == 's':
        f.write("\tpshufd $%d, %s, %s\n" % (n, a, _qx(i.to)))
    elif x == 'd':
        f.write("\tpshufd $%d, %s, %s\n" % (0xee if n else 0x44, a, _qx(i.to)))
    elif x == 'w':
        if n:
            f.write("\tpshufd $%d, %s, %s\n" % (n, a, _QX15))
            a = _QX15
        f.write("\tmovd %s, %s\n" % (a, _qr(i.to, SWord)))
    elif x == 'l':
        if n:
            f.write("\tpshufd $238, %s, %s\n" % (a, _QX15))
            a = _QX15
        f.write("\tmovq %s, %s\n" % (a, _qr(i.to, SLong)))
    elif x == 'h':
        f.write("\tpextrw $%d, %s, %s\n" % (n, a, _qr(i.to, SWord)))
    else:
        f.write("\tpextrw $%d, %s, %s\n" % (n >> 1, a, _qr(i.to, SWord)))
        if n & 1:
            f.write("\tshrl $8, %s\n" % _qr(i.to, SWord))
        else:
            f.write("\tmovzbl %s, %s\n" % (_qr(i.to, SByte), _qr(i.to, SWord)))


def emitins(i, e):
    """i: passed by value (copied here)."""
    i = i.copy()
    f = e.f
    op = i.op
    if (i.cls == D.Kq or D.isqop(op)) and _qins(i, e):
        return
    if op in _FALSEDEP and _xreg(i.to) and i.to != i.arg[0]:  # as _xcopy: these keep the upper bits too, as gcc's pxor
        x = rval(i.to) - XMM0
        f.write("\txorps %%xmm%d, %%xmm%d\n" % (x, x))
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
            emitcopy(i.to, i.arg[0], i.cls, e)
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
            emitcopy(i.arg[1], i.to, i.cls, e)
            emitcopy(i.to, i.arg[0], i.cls, e)
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
        if _xreg(i.to) and _xreg(i.arg[0]):
            _xcopy(i.to, i.arg[0], e)
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
        if D.T.windows and con.sym.type != SGlo and not (_cimowin() and con.sym.type == SExt):
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
            if e.vzu:
                f.write("\tvzeroupper\n")
            f.write("\tcallq ")
            emitcon(con, e)
            if con.type == CAddr and (con.sym.type & SExt) and not D.T.apple:
                f.write("@plt")
            f.write("\n")
        elif t == RTmp:
            if e.vzu:
                f.write("\tvzeroupper\n")
            emitf("callq *%L0", i, e)
        else:
            die("invalid call argument")
    elif op == Osalloc:
        # there is no good reason why this is here
        # maybe we should split Osalloc in 2 different
        # instructions depending on the result
        assert e.fp == RBP
        if _cimowin() and not (rtype(i.arg[0]) == RCon
                               and e.fn.con[rval(i.arg[0])].i <= 4096):
            # cimo: Windows commits the stack through a guard page; touch every page
            emitf("movq %%rsp, %%r11", i, e)
            emitf("subq %L0, %%r11", i, e)
            # a size past the bottom of the address space runs to the guard page, like any too
            # big: rsp can't take the wrapped value, or the kernel has no stack for the fault
            global _nprobe
            _nprobe += 1
            f.write("\tjnc .Lnowrap%d\n\txorl %%r11d, %%r11d\n.Lnowrap%d:\n" % (_nprobe, _nprobe))
            _probe(f)
        else:
            emitf("subq %L0, %%rsp", i, e)
        if i.to != R:
            emitcopy(i.to, TMP(RSP), Kl, e)
    elif op == Oswap:
        if rtype(i.arg[0]) == RSlot and rtype(i.arg[1]) == RSlot:  # cimo: no instruction swaps two slots
            _swap2slots(i, e)
            return
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
    # cimo extensions (ext.vla, ext.math); SSE2 only, so baseline x86-64 runs them
    elif op == Ogetsp:
        emitf("movq %%rsp, %L=", i, e)
    elif op == Osetsp:
        emitf("movq %L0, %%rsp", i, e)
    elif op in (D.Ordfpcr, D.Ordfpsr):  # cimo (<fenv.h>): MXCSR through 8 bytes below the stack pointer; lea
        f.write("\tleaq -8(%rsp), %rsp\n\tstmxcsr (%rsp)\n")  # keeps the flags
        emitf("movl (%%rsp), %W=", i, e)
        f.write("\tleaq 8(%rsp), %rsp\n")
    elif op in (D.Owrfpcr, D.Owrfpsr):
        f.write("\tleaq -8(%rsp), %rsp\n")
        emitf("movl %W0, (%%rsp)", i, e)
        f.write("\tldmxcsr (%rsp)\n\tleaq 8(%rsp), %rsp\n")
    elif op == Ofsqrt:
        emitf("sqrt%k %0, %=", i, e)
    elif op == Ofabs:
        if i.to != i.arg[0]:
            emitcopy(i.to, i.arg[0], i.cls, e)
        f.write("\tandp%c %sfp%d(%%rip), %%%s\n" % (
            "xxsd"[i.cls], D.T.asloc, stashbits(absmask[i.cls], 16),
            regtoa(rval(i.to), SLong)))
    elif op == Obswap:
        if i.to != i.arg[0]:
            emitf("mov%k %0, %=", i, e)
        emitf("bswap%k %=", i, e)
    elif op == Obswap16:
        emitf("movzwl %H0, %W=", i, e)
        emitf("rolw $8, %H=", i, e)
    elif op == Obclz:  # undefined for 0, as __builtin_clz
        if 'lzcnt' in _ext.x86:
            _clear(i, e)
            emitf("lzcnt%k %0, %=", i, e)
        else:
            emitf("bsr%k %0, %=", i, e)
            emitf("xor%k $63, %=" if i.cls == Kl else "xor%k $31, %=", i, e)
    elif op == Obctz:
        if 'bmi1' in _ext.x86:
            _clear(i, e)
            emitf("tzcnt%k %0, %=", i, e)
        else:
            emitf("bsf%k %0, %=", i, e)
    elif op == D.Oh2f:  # ext.x86 f16c: the half's bits through xmm15
        emitf("vmovd %W0, %%xmm15", i, e)
        emitf("vcvtph2ps %%xmm15, %S=", i, e)
    elif op == D.Of2h:  # rounded in MXCSR's mode ($4), the 16 bits zero-extended
        emitf("vcvtps2ph $4, %S0, %%xmm15", i, e)
        emitf("vpextrw $0, %%xmm15, %W=", i, e)
    elif op == D.Obpopcnt:  # only with ext.x86 popcnt: the frontend emits it then
        _clear(i, e)
        emitf("popcnt%k %0, %=", i, e)
    elif op in _FMA3:
        _fma(i, e)
    else:
        _table(i, e)


def _clear(i, e):
    """popcnt, lzcnt and tzcnt wait for their destination's old value on several Intel cores, as
    gcc's tuning knows: cleared first, unless it's the source."""
    if i.to != i.arg[0] and rtype(i.to) == RTmp:
        r = regtoa(rval(i.to), SWord)
        e.f.write("\txorl %%%s, %%%s\n" % (r, r))


_FMA3 = {D.Oafmadd: 'vfmadd', D.Oafmsub: 'vfnmadd', D.Oafnmsub: 'vfmsub', D.Oafnmadd: 'vfnmsub'}


def _fma(i, e):
    """r = afmadd a, MEM{base=b, index=c}: r = a*b + c, rounded once (FMA3); afmsub c - a*b, afnmsub a*b - c,
    afnmadd -(a*b) - c."""
    m = e.fn.mem[rval(i.arg[1])]
    _fma3(e.f, 's' + ('d' if i.cls == Kd else 's'), i.to, i.arg[0], m.base, m.index, _FMA3[i.op])


def _fma3(f, x, to, a, b, c, mn='vfmadd', y=False):
    """to = ±(a*b) ± c in x ('ss', 'sd', 'ps', 'pd'), whichever of them to shares a register with: the 231 form
    computes src1*src2 ± dst, the 213 form src1*dst ± src2, so the sign of each part is the mnemonic's either way."""
    q = lambda r: r if isinstance(r, str) else "%%%smm%d" % ('y' if y else 'x', rval(r) - XMM0)
    if isinstance(b, str):  # b is memory (qxfmam): only the last source can be
        if to == a:
            f.write("\t%s132%s %s, %s, %s\n" % (mn, x, b, q(c), q(to)))  # to = to * b + c
        else:
            if to != c:
                f.write("\t%s %s, %s\n" % ('vmovaps' if y else 'movaps', q(c), q(to)))
            f.write("\t%s231%s %s, %s, %s\n" % (mn, x, b, q(a), q(to)))
        return
    if to == c:
        f.write("\t%s231%s %s, %s, %s\n" % (mn, x, q(b), q(a), q(to)))
    elif to == a:
        f.write("\t%s213%s %s, %s, %s\n" % (mn, x, q(c), q(b), q(to)))
    elif to == b:
        f.write("\t%s213%s %s, %s, %s\n" % (mn, x, q(c), q(a), q(to)))
    else:
        f.write("\t%s %s, %s\n\t%s231%s %s, %s, %s\n" % ('vmovaps' if y else 'movaps', q(c), q(to), mn, x, q(b), q(a), q(to)))


absmask = {Ks: 0x7fffffff, Kd: 0x7fffffffffffffff}
_nprobe = 0


def _probe(f):
    """cimo (amd64_windows): move rsp down to r11 a page at a time, touching each page, as
    __chkstk does: Windows grows a stack through one guard page, so a frame or alloca that
    skips a page faults."""
    global _nprobe
    _nprobe += 1
    f.write(".Lprobe%d:\n"
            "\tcmpq %%r11, %%rsp\n"
            "\tjbe .Lprobed%d\n"
            "\tsubq $4096, %%rsp\n"
            "\ttestq %%rsp, (%%rsp)\n"
            "\tjmp .Lprobe%d\n"
            ".Lprobed%d:\n"
            "\tmovq %%r11, %%rsp\n" % (_nprobe, _nprobe, _nprobe, _nprobe))


def _stkchk(fn, f):
    """cimo (ext.stkchk): the stack guard, first thing in the prologue (cimo.execution)."""
    if _ext.stkchk and fn.name in _ext.stkchk:
        f.write("\tcallq %s__cimo_stkchk\n" % D.T.assym)


# cimo: a loop's header on 64 bytes, not QBE's 16: where a short loop lands within a 64-byte line moved gemm by 1.1 to
# 1.5x on Ice Lake, Sapphire Rapids and Zen 4 when one instruction changed (FMA), and the baseline build is no slower
_LOOPALIGN = ".p2align 6\n"


def _ops(text):
    """Operands of an AT&T instruction: split at the commas outside parentheses."""
    out, depth, cur = [], 0, ''
    for c in text:
        if c == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
            continue
        depth += (c == '(') - (c == ')')
        cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


# legacy SSE -> VEX. Destination also the first source, so the meaning is the SSE instruction's (bits above the
# scalar from the destination); VEX zeroes the rest of the ymm register, which no one reads
_VEX3 = frozenset('addss addsd subss subsd mulss mulsd divss divsd minss minsd maxss maxsd sqrtss sqrtsd cvtss2sd '
                  'cvtsd2ss andps andpd andnps andnpd orps orpd xorps xorpd pxor pand por pandn punpcklbw punpcklqdq '
                  'punpckldq cvtsi2ss cvtsi2sd cvtsi2ssl cvtsi2ssq cvtsi2sdl cvtsi2sdq'.split())
_VEX2 = frozenset('movaps movups movapd movupd movdqa movdqu ucomiss ucomisd comiss comisd movd movq pshufd pshuflw '
                  'pextrw cvttss2si cvttsd2si cvttss2sil cvttss2siq cvttsd2sil cvttsd2siq cvtss2si cvtsd2si'.split())


def _vexline(line):
    """An emitted line, with its SSE instruction as the VEX one (lines without an xmm operand unchanged)."""
    if '%xmm' not in line or not line.startswith('\t'):
        return line
    mn, _, rest = line[1:].partition(' ')
    if mn in _VEX3:
        o = _ops(rest)
        if len(o) == 2:
            return '\tv%s %s, %s, %s' % (mn, o[0], o[1], o[1])
    elif mn in _VEX2:
        return '\tv' + line[1:]
    elif mn in ('movss', 'movsd'):
        o = _ops(rest)
        if len(o) == 2 and o[0].startswith('%xmm') and o[1].startswith('%xmm'):
            return '\tv%s %s, %s, %s' % (mn, o[0], o[1], o[1])
        return '\tv' + line[1:]
    elif mn in ('roundss', 'roundsd'):
        o = _ops(rest)
        if len(o) == 3:
            return '\tv%s %s, %s, %s, %s' % (mn, o[0], o[1], o[2], o[2])
    return line  # already VEX, or one this doesn't know: correct, only slower


class _Vex:
    """An output stream that writes the VEX form of each SSE instruction (_vexline), line by line."""

    def __init__(self, f):
        self.f = f
        self.buf = ''

    def write(self, text):
        self.buf += text
        if '\n' in self.buf:
            *lines, self.buf = self.buf.split('\n')
            self.f.write(''.join(_vexline(l) + '\n' for l in lines))

    def getvalue(self):  # an io.StringIO's, for _emitf_y
        return self.f.getvalue()


def _usesq(fn):
    """Whether fn has q values (ext.vector)."""
    b = fn.start
    while b:
        for i in b.ins[:b.nins]:
            if i.cls == D.Kq or D.isqop(i.op):
                return True
        b = b.link
    return False


def _usesxmm15(fn):
    """Whether emitins uses xmm15 as a scratch register in fn (see Odiv, Ocopy, Oswap)."""
    b = fn.start
    while b:
        for i in b.ins[:b.nins]:
            if i.cls == D.Kq or D.isqop(i.op):  # ext.vector: moves, lanes, operands in the way
                return True
            if i.op == Odiv and KBASE(i.cls) == 1 and i.to == i.arg[1]:
                return True
            if i.op == Ocopy and rtype(i.to) == RSlot and rtype(i.arg[0]) in (RSlot, RMem):
                return True
            if i.op == Oswap and KBASE(i.cls) == 1:
                return True
            if i.op in (D.Oh2f, D.Of2h):  # f16c through xmm15
                return True
        b = b.link
    return False


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
    _stkchk(fn, f)
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
                f.write(_LOOPALIGN)
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
        if _ext.trap:  # cimo: unwind to Python (cimo.execution) instead of killing it
            f.write("\tcallq %s__cimo_trap\n" % D.T.assym)
        else:
            f.write("\tud2\n")
    elif jt == Jret0:
        if fn.dynalloc:
            f.write("\tmovq %%rbp, %%rsp\n\tsubq $%d, %%rsp\n"
                    % (e.fsz + e.nclob * 8 + e.xarea))
        if e.xarea:
            for k, x in enumerate(e.xsave):
                f.write("\t%smovdqu %d(%%rsp), %%xmm%d\n" % ("v" if e.vzu else "", 16 * k, x - XMM0))
            f.write("\taddq $%d, %%rsp\n" % e.xarea)
        for r in reversed(rclob[:nclr]):
            if fn.reg & BIT(r):
                itmp = Ins(0, 0, R, TMP(r))
                emitf("popq %L0", itmp, e)
        if e.fp == RBP:
            f.write("\tleave\n")
        elif e.fsz:
            f.write("\taddq $%d, %%rsp\n" % e.fsz)
        if e.vzu:
            f.write("\tvzeroupper\n")
        f.write("\tret\n")
    elif jt == Jjmp:
        if b.s1 is not b.link:
            f.write("\tjmp %sbb%d\n" % (D.T.asloc, id0 + b.s1.id))
        else:
            return 0
    else:
        c = jt - Jjf
        if 0 <= c <= NCmp:
            # cimo: a loop's back edge is the conditional jump, not a jump after one that's rarely taken
            if b.link is b.s2 or (win and c >= NCmpI) or (b.s1 is not b.link and b.s1.id <= b.id < b.s2.id):
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
    _stkchk(fn, f)
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
    if e.fsz > 4096 and _cimowin():
        f.write("\tleaq -%d(%%rsp), %%r11\n" % e.fsz)
        _probe(f)
    elif e.fsz:
        f.write("\tsubq $%d, %%rsp\n" % e.fsz)
    for r in amd64_winabi_rclob[:NCLR_WIN]:
        if fn.reg & BIT(r):
            itmp = Ins(0, 0, R, TMP(r))
            emitf("pushq %L0", itmp, e)
            e.nclob += 1
    if _cimowin():
        # cimo: xmm6-xmm15 are callee-saved on Windows (QBE treats them as scratch)
        from .winabi import WIN_XCLOB
        e.xsave = [x for x in WIN_XCLOB if fn.reg & BIT(x)]
        if _usesxmm15(fn):
            e.xsave.append(XMM15)
        e.xarea = 16 * len(e.xsave)
        if e.xarea:
            f.write("\tsubq $%d, %%rsp\n" % e.xarea)
            for k, x in enumerate(e.xsave):
                f.write("\t%smovdqu %%xmm%d, %d(%%rsp)\n" % ("v" if e.vzu else "", x - XMM0, 16 * k))

    lbl = 0
    b = fn.start
    while b:
        if lbl or b.npred > 1:
            if any(b.pred[p].id >= b.id for p in range(b.npred)):
                f.write(_LOOPALIGN)
            f.write("%sbb%d:\n" % (D.T.asloc, id0 + b.id))
        for i in b.ins[:b.nins]:
            emitins(i, e)
        lbl = _emitjmp(b, fn, e, id0, amd64_winabi_rclob, NCLR_WIN, 1)
        b = b.link
    _id0_win = id0 + fn.nblk
    if _cimowin():
        elf_emitfnfin(fn.name, f)
