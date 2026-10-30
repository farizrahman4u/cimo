"""Port of QBE parse.c."""

import math
import re

from . import defs as D
from .defs import (R, UNDEF, RTmp, RCon, RInt, RType, RSlot, RCall, RMem, Tmp0, NIns,
                   NPubOp, NField, Ins, Phi, Con, Typ, Field, Dat, Lnk, Fn, CBits, CAddr,
                   CUndef, SExt, SThr, Kx, Kw, Kl, Ks, Kd, Ke, optab, jtoa, TMP, TYPE,
                   INT, KBASE, req, rtype, rval, rsval, isstore, isret, s64)
from .util import (err, hash, intern, str_, newtmp, newcon, vgrow, idup, clsmerge,
                   bsinit, bszero, bsset, bshas, bsequal)
from . import cfg
from . import ext as _ext

Ksb = 4  # matches Oarg/Opar/Jret
Kub = 5
Ksh = 6
Kuh = 7
Kc = 8
K0 = 9

PXXX, PLbl, PPhi, PIns, PEnd = range(5)

Txxx = 0
_toks = ('loadw loadl loads loadd alloc1 alloc2 blit call env phi jmp jnz ret hlt '
         'export thread extern common func type data section align dbgfile l w sh uh h '
         'sb ub b d s z int flts fltd tmp lbl glo typ str plus eq comma lparen rparen '
         'lbrace rbrace nl dots eof').split()
for _i, _t in enumerate(_toks):
    globals()['T' + _t] = NPubOp + _i
Ntok = NPubOp + len(_toks)
del _i, _t
Tfma = Ntok  # cimo extension: only a keyword when ext.fma is on
Tmath = Ntok + 1  # cimo extension: the math instructions, keywords when ext.math is on
_MATHOPS = ('fsqrt', 'fabs', 'fmin', 'fmax', 'ffloor', 'fceil', 'ftrunc', 'fround')

kwmap = {
    Tloadw: "loadw", Tloadl: "loadl", Tloads: "loads", Tloadd: "loadd",
    Talloc1: "alloc1", Talloc2: "alloc2", Tblit: "blit", Tcall: "call", Tenv: "env",
    Tphi: "phi", Tjmp: "jmp", Tjnz: "jnz", Tret: "ret", Thlt: "hlt",
    Texport: "export", Tthread: "thread", Textern: "extern", Tcommon: "common",
    Tfunc: "function", Ttype: "type", Tdata: "data", Tsection: "section",
    Talign: "align", Tdbgfile: "dbgfile", Tsb: "sb", Tub: "ub", Tsh: "sh", Tuh: "uh",
    Tb: "b", Th: "h", Tw: "w", Tl: "l", Ts: "s", Td: "d", Tz: "z", Tdots: "...",
}
for _i in range(NPubOp):
    if optab[_i].name:
        kwmap[_i] = optab[_i].name
_kwlookup = {v: k for k, v in kwmap.items()}

NPred = 63
EOF_ = -1

# scanf %f / %lf: optional sign, then decimal, hex, inf or nan
_fltre = re.compile(rb'[ \t\n\v\f\r]*([+-]?(?:0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)'
                    rb'(?:[pP][+-]?[0-9]+)?|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
                    rb'|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN]))')



# cimo: declared return class of each function parsed (K0 for none, Kc for aggregates),
# for targets whose function types need it even when no ret is reached (wasm)
retcls = {}

def _tofloat(s):
    s = s.decode('latin-1')
    if 'x' in s or 'X' in s:
        return float.fromhex(s)
    return float(s)


class _State:
    pass


S = _State()


def _isblank(c):
    return c == 32 or c == 9


def _isdigit(c):
    return 48 <= c <= 57


def _isalpha(c):
    return 65 <= c <= 90 or 97 <= c <= 122


def fgetc():
    p = S.pos
    if p >= S.n:
        S.pos = p + 1
        return EOF_
    S.pos = p + 1
    return S.buf[p]


def ungetc(c):
    if c != EOF_:
        S.pos -= 1
    else:
        S.pos -= 1


def getint():
    n = 0
    c = fgetc()
    m = c == 45  # '-'
    if m:
        c = fgetc()
        if not _isdigit(c):
            err("integer expected")
    while True:
        n = (10 * n + (c - 48)) & D.M64
        c = fgetc()
        if not _isdigit(c):
            break
    ungetc(c)
    if m:
        n = (1 + ~n) & D.M64
    return s64(n)


def _scanflt():
    """fscanf(inf, "_%f"): returns float or None."""
    p = S.pos
    if p >= S.n or S.buf[p] != 95:  # '_'
        return None
    m = _fltre.match(S.buf, p + 1)
    if not m:
        S.pos = p + 1
        return None
    S.pos = m.end()
    return _tofloat(m.group(1))


def lex():
    while True:
        c = fgetc()
        if not _isblank(c):
            break
    t = Txxx
    S.tokval_chr = c
    goto_alpha = False
    if c == EOF_:
        return Teof
    elif c == 44:
        return Tcomma
    elif c == 40:
        return Tlparen
    elif c == 41:
        return Trparen
    elif c == 123:
        return Tlbrace
    elif c == 125:
        return Trbrace
    elif c == 61:
        return Teq
    elif c == 43:
        return Tplus
    elif c == 115:  # s
        f = _scanflt()
        if f is not None:
            S.tokval_flts = D.f32(f)
            return Tflts
    elif c == 100:  # d
        f = _scanflt()
        if f is not None:
            S.tokval_fltd = f
            return Tfltd
    elif c == 37:  # %
        t = Ttmp
        c = fgetc()
        goto_alpha = True
    elif c == 64:  # @
        t = Tlbl
        c = fgetc()
        goto_alpha = True
    elif c == 36:  # $
        t = Tglo
        c = fgetc()
        if c == 34:
            return _quoted(c, t)
        goto_alpha = True
    elif c == 58:  # :
        t = Ttyp
        c = fgetc()
        goto_alpha = True
    elif c == 35:  # #
        while True:
            c = fgetc()
            if c == 10 or c == EOF_:
                break
        D.lnum += 1
        return Tnl
    elif c == 10:
        D.lnum += 1
        return Tnl
    if not goto_alpha:
        if _isdigit(c) or c == 45:
            ungetc(c)
            S.tokval_num = getint()
            return Tint
        if c == 34:
            return _quoted(c, Tstr)
    # Alpha:
    if not _isalpha(c) and c != 46 and c != 95:
        err("invalid character %c (%d)", chr(c) if c >= 0 else '\xff', c)
    start = S.pos - 1
    while True:
        c = fgetc()
        if not (_isalpha(c) or c == 36 or c == 46 or c == 95 or _isdigit(c)):
            break
    ungetc(c)
    S.tokval_str = S.buf[start:S.pos].decode('latin-1')
    if t != Txxx:
        return t
    t = _kwlookup.get(S.tokval_str)
    if t is None:
        if S.tokval_str == 'fma' and _ext.fma:
            return Tfma
        if _ext.math and S.tokval_str in _MATHOPS:
            S.mathop = getattr(D, 'O' + S.tokval_str)
            return Tmath
        err("unknown keyword %s", S.tokval_str)
    return t


def _quoted(c, t):
    start = S.pos - 1
    esc = 0
    while True:
        c = fgetc()
        if c == EOF_:
            err("unterminated string")
        if c == 34 and not esc:
            S.tokval_str = S.buf[start:S.pos].decode('latin-1')
            return t
        esc = c == 92 and not esc


def peek():
    if S.thead == Txxx:
        S.thead = lex()
    return S.thead


def next_():
    t = peek()
    S.thead = Txxx
    return t


def nextnl():
    while True:
        t = next_()
        if t != Tnl:
            return t


_ttoa = {Tlbl: "label", Tcomma: ",", Teq: "=", Tnl: "newline", Tlparen: "(",
         Trparen: ")", Tlbrace: "{", Trbrace: "}"}


def expect(t):
    t1 = next_()
    if t == t1:
        return
    s1 = _ttoa.get(t, "??")
    s2 = _ttoa.get(t1, "??")
    err("%s expected, got %s instead" % (s1, s2))


def tmpref():
    t = S.tmph.get(S.tokval_str)
    if t is not None:
        return TMP(t)
    t = S.curf.ntmp
    S.tmph[S.tokval_str] = t
    newtmp(None, Kx, S.curf)
    S.curf.tmp[t].name = S.tokval_str
    return TMP(t)


def parseref():
    c = Con()
    tok = next_()
    if tok == Ttmp:
        return tmpref()
    elif tok == Tint:
        c.type = CBits
        c.i = S.tokval_num
    elif tok == Tflts:
        c.type = CBits
        c.s = S.tokval_flts
        c.flt = 1
    elif tok == Tfltd:
        c.type = CBits
        c.d = S.tokval_fltd
        c.flt = 2
    else:
        if tok != Tglo:
            while True:
                if tok == Textern:
                    c.sym.type |= SExt
                    tok = next_()
                    continue
                if tok == Tthread:
                    c.sym.type |= SThr
                    tok = next_()
                    continue
                break
            if tok != Tglo:
                return R
        c.type = CAddr
        c.sym.id = intern(S.tokval_str)
    return newcon(c, S.curf)


def findtyp(i):
    i -= 1
    while i >= 0:
        if S.tokval_str == D.typ[i].name:
            return i
        i -= 1
    err("undefined type :%s", S.tokval_str)


def parsecls():
    """Returns (k, tyn); tyn is set only for Kc."""
    t = next_()
    if t == Ttyp:
        return Kc, findtyp(S.ntyp)
    m = {Tsb: Ksb, Tub: Kub, Tsh: Ksh, Tuh: Kuh, Tw: Kw, Tl: Kl, Ts: Ks, Td: Kd}
    if t in m:
        return m[t], None
    err("invalid class specifier")


def _put(i):
    D.insb[D.curi] = i
    D.curi += 1


def parserefl(arg):
    hasenv = 0
    vararg = 0
    expect(Tlparen)
    while peek() != Trparen:
        if D.curi >= NIns:
            err("too many instructions")
        if not arg and vararg:
            err("no parameters allowed after '...'")
        p = peek()
        skip = False
        if p == Tdots:
            if vararg:
                err("only one '...' allowed")
            vararg = 1
            if arg:
                _put(Ins(D.Oargv))
            next_()
            skip = True
        elif p == Tenv:
            if hasenv:
                err("only one environment allowed")
            hasenv = 1
            env = 1
            next_()
            k = Kl
        else:
            env = 0
            k, ty = parsecls()
        if not skip:
            r = parseref()
            if req(r, R):
                err("invalid argument")
            if not arg and rtype(r) != RTmp:
                err("invalid function parameter")
            if env:
                if arg:
                    _put(Ins(D.Oarge, k, R, r))
                else:
                    _put(Ins(D.Opare, k, r, R))
            elif k == Kc:
                if arg:
                    _put(Ins(D.Oargc, Kl, R, TYPE(ty), r))
                else:
                    _put(Ins(D.Oparc, Kl, r, TYPE(ty)))
            elif k >= Ksb:
                if arg:
                    _put(Ins(D.Oargsb + (k - Ksb), Kw, R, r))
                else:
                    _put(Ins(D.Oparsb + (k - Ksb), Kw, r, R))
            else:
                if arg:
                    _put(Ins(D.Oarg, k, R, r))
                else:
                    _put(Ins(D.Opar, k, r, R))
        # Next:
        if peek() == Trparen:
            break
        expect(Tcomma)
    expect(Trparen)
    return vararg


def findblk():
    b = S.blkh.get(S.tokval_str)
    if b is not None:
        return b
    b = cfg.newblk()
    b.id = S.nblk
    S.nblk += 1
    b.name = S.tokval_str
    S.blkh[S.tokval_str] = b
    return b


def closeblk():
    idup(S.curb, 0, D.curi)
    S.blink = (S.curb, 'link')
    D.curi = 0


def _setlink(pl, v):
    setattr(pl[0], pl[1], v)


def parseline(ps):
    arg = [R] * NPred
    blk = [None] * NPred
    curf = S.curf

    t = nextnl()
    if ps == PLbl and t != Tlbl and t != Trbrace:
        err("label or } expected")
    r = R
    k = Kw
    ty = None
    if t == Ttmp:
        r = tmpref()
        expect(Teq)
        k, ty = parsecls()
        op = next_()
    elif t == Trbrace:
        return PEnd
    elif t == Tlbl:
        b = findblk()
        if S.curb and S.curb.jmp.type == D.Jxxx:
            closeblk()
            S.curb.jmp.type = D.Jjmp
            S.curb.s1 = b
        if b.jmp.type != D.Jxxx:
            err("multiple definitions of block @%s", b.name)
        _setlink(S.blink, b)
        S.curb = b
        S.plink = (b, 'phi')
        expect(Tnl)
        return PPhi
    elif t == Tret:
        curb = S.curb
        curb.jmp.type = D.Jretw + S.rcls
        if peek() == Tnl:
            curb.jmp.type = D.Jret0
        elif S.rcls != K0:
            r = parseref()
            if req(r, R):
                err("invalid return value")
            curb.jmp.arg = r
        expect(Tnl)
        closeblk()
        return PLbl
    elif t == Tjmp or t == Tjnz:
        curb = S.curb
        if t == Tjmp:
            curb.jmp.type = D.Jjmp
        else:
            curb.jmp.type = D.Jjnz
            r = parseref()
            if req(r, R):
                err("invalid argument for jnz jump")
            curb.jmp.arg = r
            expect(Tcomma)
        expect(Tlbl)
        curb.s1 = findblk()
        if curb.jmp.type != D.Jjmp:
            expect(Tcomma)
            expect(Tlbl)
            curb.s2 = findblk()
        if curb.s1 is curf.start or (curb.s2 is not None and curb.s2 is curf.start):
            err("invalid jump to the start block")
        expect(Tnl)
        closeblk()
        return PLbl
    elif t == Thlt:
        S.curb.jmp.type = D.Jhlt
        expect(Tnl)
        closeblk()
        return PLbl
    elif t == D.Odbgloc:
        op = t
        k = Kw
        r = R
        expect(Tint)
        arg[0] = INT(S.tokval_num)
        if rval(arg[0]) != S.tokval_num:
            err("line number too big")
        if peek() == Tcomma:
            next_()
            expect(Tint)
            arg[1] = INT(S.tokval_num)
            if rval(arg[1]) != S.tokval_num:
                err("column number too big")
        else:
            arg[1] = INT(0)
        return _ins(op, k, r, arg)
    elif isstore(t) or t == Tblit or t == Tcall or t == D.Ovastart:
        # operations without result
        r = R
        k = Kw
        op = t
    else:
        err("label, instruction or jump expected")

    if op == Tcall:
        curf.leaf = 0
        arg[0] = parseref()
        parserefl(1)
        op = D.Ocall
        expect(Tnl)
        if k == Kc:
            k = Kl
            arg[1] = TYPE(ty)
        if k >= Ksb:
            k = Kw
        return _ins(op, k, r, arg)
    if op == Tloadw:
        op = D.Oloadsw
    if Tloadl <= op <= Tloadd:
        op = D.Oload
    if op == Talloc1 or op == Talloc2:
        op = D.Oalloc
    if op == D.Ovastart and not curf.vararg:
        err("cannot use vastart in non-variadic function")
    if k >= Ksb:
        err("size class must be w, l, s, or d")
    i = 0
    if peek() != Tnl:
        while True:
            if i == NPred:
                err("too many arguments")
            if op == Tphi:
                expect(Tlbl)
                blk[i] = findblk()
            arg[i] = parseref()
            if req(arg[i], R):
                err("invalid instruction argument")
            i += 1
            t = peek()
            if t == Tnl:
                break
            if t != Tcomma:
                err(", or end of line expected")
            next_()
    next_()
    if op == Tphi:
        if ps != PPhi or S.curb is curf.start:
            err("unexpected phi instruction")
        phi = Phi()
        phi.to = r
        phi.cls = k
        phi.arg = arg[:i]
        phi.blk = blk[:i]
        phi.narg = i
        _setlink(S.plink, phi)
        S.plink = (phi, 'link')
        return PPhi
    elif op == Tfma:
        if i != 3 or KBASE(k) != 1:
            err("fma needs three arguments and class s or d")
        if D.curi >= NIns - 1:
            err("too many instructions")
        _put(Ins(D.Ofma0, k, R, arg[2], R))
        _put(Ins(D.Ofma1, k, r, arg[0], arg[1]))
        return PIns
    elif op == Tmath:
        mop = S.mathop
        if i != (2 if mop in (D.Ofmin, D.Ofmax) else 1) or KBASE(k) != 1:
            err("%s needs %s and class s or d", optab[mop].name,
                "two arguments" if mop in (D.Ofmin, D.Ofmax) else "one argument")
        return _ins(mop, k, r, arg)
    elif op == Tblit:
        if D.curi >= NIns - 1:
            err("too many instructions")
        _put(Ins(D.Oblit0, 0, R, arg[0], arg[1]))
        if rtype(arg[2]) != RCon:
            err("blit size must be constant")
        c = curf.con[rval(arg[2])]
        r = INT(c.i)
        if c.type != CBits or rsval(r) < 0 or rsval(r) != c.i:
            err("invalid blit size")
        _put(Ins(D.Oblit1, 0, R, r, R))
        return PIns
    else:
        if op >= NPubOp:
            err("invalid instruction")
        return _ins(op, k, r, arg)


def _ins(op, k, r, arg):
    if D.curi >= NIns:
        err("too many instructions")
    _put(Ins(op, k, r, arg[0], arg[1]))
    return PIns


def usecheck(r, k, fn):
    return (rtype(r) != RTmp or fn.tmp[rval(r)].cls == k
            or (fn.tmp[rval(r)].cls == Kl and k == Kw))


def typecheck(fn):
    pb = D.BSet()
    ppb = D.BSet()
    cfg.fillpreds(fn)
    bsinit(pb, fn.nblk)
    bsinit(ppb, fn.nblk)
    b = fn.start
    while b:
        p = b.phi
        while p:
            fn.tmp[rval(p.to)].cls = p.cls
            p = p.link
        for i in b.ins[:b.nins]:
            if rtype(i.to) == RTmp:
                t = fn.tmp[rval(i.to)]
                t.cls, e = clsmerge(t.cls, i.cls)
                if e:
                    err("temporary %%%s is assigned with multiple types", t.name)
        b = b.link
    b = fn.start
    while b:
        bszero(pb)
        for n in range(b.npred):
            bsset(pb, b.pred[n].id)
        p = b.phi
        while p:
            bszero(ppb)
            t = fn.tmp[rval(p.to)]
            for n in range(p.narg):
                k = t.cls
                if bshas(ppb, p.blk[n].id):
                    err("multiple entries for @%s in phi %%%s", p.blk[n].name, t.name)
                if not usecheck(p.arg[n], k, fn):
                    err("invalid type for operand %%%s in phi %%%s",
                        fn.tmp[rval(p.arg[n])].name, t.name)
                bsset(ppb, p.blk[n].id)
            if not bsequal(pb, ppb):
                err("predecessors not matched in phi %%%s", t.name)
            p = p.link
        for i in b.ins[:b.nins]:
            for n in range(2):
                k = optab[i.op].argcls[n][i.cls]
                r = i.arg[n]
                t = fn.tmp[rval(r)]
                if k == Ke:
                    err("invalid instruction type in %s", optab[i.op].name)
                if rtype(r) == RType:
                    continue
                if rtype(r) != -1 and k == Kx:
                    err("no %s operand expected in %s",
                        "second" if n == 1 else "first", optab[i.op].name)
                if rtype(r) == -1 and k != Kx:
                    err("missing %s operand in %s",
                        "second" if n == 1 else "first", optab[i.op].name)
                if not usecheck(r, k, fn):
                    err("invalid type for %s operand %%%s in %s",
                        "second" if n == 1 else "first", t.name, optab[i.op].name)
        r = b.jmp.arg
        jerr = False
        if isret(b.jmp.type):
            if b.jmp.type == D.Jretc:
                k = Kl
            elif b.jmp.type >= D.Jretsb:
                k = Kw
            else:
                k = b.jmp.type - D.Jretw
            if not usecheck(r, k, fn):
                jerr = True
        if jerr or (b.jmp.type == D.Jjnz and not usecheck(r, Kw, fn)):
            err("invalid type for jump argument %%%s in block @%s",
                fn.tmp[rval(r)].name, b.name)
        if b.s1 and b.s1.jmp.type == D.Jxxx:
            err("block @%s is used undefined", b.s1.name)
        if b.s2 and b.s2.jmp.type == D.Jxxx:
            err("block @%s is used undefined", b.s2.name)
        b = b.link


def parsefn(lnk):
    S.curb = None
    S.nblk = 0
    S.tmph = {}
    S.blkh = {}
    D.curi = 0
    curf = S.curf = Fn()
    curf.ntmp = 0
    curf.ncon = 2
    curf.tmp = []
    curf.con = [Con(), Con()]
    T = D.T
    for i in range(Tmp0):
        if T.fpr0 <= i < T.fpr0 + T.nfpr:
            newtmp(None, Kd, curf)
        else:
            newtmp(None, Kl, curf)
    curf.con[0].type = CBits
    curf.con[0].i = 0xdeaddead  # UNDEF
    curf.con[1].type = CBits
    curf.lnk = lnk.copy()
    curf.leaf = 1
    S.blink = (curf, 'start')
    curf.retty = Kx
    if peek() != Tglo:
        S.rcls, ty = parsecls()
        if ty is not None:
            curf.retty = ty
    else:
        S.rcls = K0
    if next_() != Tglo:
        err("function name expected")
    curf.name = S.tokval_str
    retcls[curf.name] = S.rcls  # cimo: the wasm backend types functions by it
    curf.vararg = parserefl(0)
    if nextnl() != Tlbrace:
        err("function body must start with {")
    ps = PLbl
    while True:
        ps = parseline(ps)
        if ps == PEnd:
            break
    if not S.curb:
        err("empty function")
    if S.curb.jmp.type == D.Jxxx:
        err("last block misses jump")
    curf.mem = []
    curf.nmem = 0
    curf.nblk = S.nblk
    curf.rpo = [None] * S.nblk
    b = curf.start
    while b:
        b.dlink = None  # was trashed by findblk()
        b = b.link
    typecheck(curf)
    return curf


def parsefields(fld, ty, t):
    n = 0
    sz = 0
    al = ty.align
    while t != Trbrace:
        ty1 = None
        if t == Td:
            type_, s, a = D.Fd, 8, 3
        elif t == Tl:
            type_, s, a = D.Fl, 8, 3
        elif t == Ts:
            type_, s, a = D.Fs, 4, 2
        elif t == Tw:
            type_, s, a = D.Fw, 4, 2
        elif t == Th:
            type_, s, a = D.Fh, 2, 1
        elif t == Tb:
            type_, s, a = D.Fb, 1, 0
        elif t == Ttyp:
            type_ = D.FTyp
            ty1i = findtyp(S.ntyp - 1)
            ty1 = D.typ[ty1i]
            s = ty1.size
            a = ty1.align
        else:
            err("invalid type member specifier")
        if a > al:
            al = a
        a = (1 << a) - 1
        a = ((sz + a) & ~a) - sz
        if a:
            if n < NField:
                # padding
                fld[n].type = D.FPad
                fld[n].len = a
                n += 1
        t = nextnl()
        if t == Tint:
            c = D.s32(S.tokval_num)
            t = nextnl()
        else:
            c = 1
        sz = (sz + a + c * s) & D.M64
        if type_ == D.FTyp:
            s = ty1i
        while c > 0 and n < NField:
            fld[n].type = type_
            fld[n].len = s
            c -= 1
            n += 1
        if t != Tcomma:
            break
        t = nextnl()
    if t != Trbrace:
        err(", or } expected")
    fld[n].type = D.FEnd
    a = 1 << al
    if sz < ty.size:
        sz = ty.size
    ty.size = (sz + a - 1) & -a & D.M64
    ty.align = al


def _newfields():
    return [Field() for _ in range(NField + 1)]


def parsetyp():
    ty = Typ()
    D.typ.append(ty)
    S.ntyp += 1
    ty.isdark = 0
    ty.isunion = 0
    ty.align = -1
    ty.size = 0
    if nextnl() != Ttyp or nextnl() != Teq:
        err("type name and then = expected")
    ty.name = S.tokval_str
    t = nextnl()
    if t == Talign:
        if nextnl() != Tint:
            err("alignment expected")
        al = 0
        num = S.tokval_num
        while True:
            num = -(-num // 2) if num < 0 else num // 2
            if not num:
                break
            al += 1
        S.tokval_num = num
        ty.align = al
        t = nextnl()
    if t != Tlbrace:
        err("type body must start with {")
    t = nextnl()
    if t == Tint:
        ty.isdark = 1
        ty.size = S.tokval_num
        if ty.align == -1:
            err("dark types need alignment")
        if nextnl() != Trbrace:
            err("} expected")
        return
    n = 0
    ty.fields = [_newfields()]
    if t == Tlbrace:
        ty.isunion = 1
        while True:
            if t != Tlbrace:
                err("invalid union member")
            vgrow(ty.fields, n + 1, _newfields)
            parsefields(ty.fields[n], ty, nextnl())
            n += 1
            t = nextnl()
            if t == Trbrace:
                break
    else:
        parsefields(ty.fields[n], ty, t)
        n += 1
    ty.nunion = n


def parsedatref(d):
    d.isref = 1
    d.ref_name = S.tokval_str
    d.ref_off = 0
    t = peek()
    if t == Tplus:
        next_()
        if next_() != Tint:
            err("invalid token after offset in ref")
        d.ref_off = S.tokval_num


def parsedatstr(d):
    d.isstr = 1
    d.str = S.tokval_str


def parsedat(cb, lnk):
    if nextnl() != Tglo or nextnl() != Teq:
        err("data name, then = expected")
    name = S.tokval_str
    t = nextnl()
    lnk.align = 8
    if t == Talign:
        if nextnl() != Tint:
            err("alignment expected")
        num = S.tokval_num
        if num <= 0 or num > 127 or (num & (num - 1)) != 0:
            err("invalid alignment")
        lnk.align = num
        t = nextnl()
    d = Dat()
    d.type = D.DStart
    d.name = name
    d.lnk = lnk
    cb(d)

    if t != Tlbrace:
        err("expected data contents in { .. }")
    done = False
    while not done:
        t = nextnl()
        m = {Tl: D.DL, Tw: D.DW, Th: D.DH, Tb: D.DB, Ts: D.DW, Td: D.DL, Tz: D.DZ}
        if t == Trbrace:
            break
        if t not in m:
            err("invalid size specifier %c in data", chr(S.tokval_chr & 0xff))
        d.type = m[t]
        t = nextnl()
        while True:
            d.isstr = 0
            d.isref = 0
            d.num = 0
            d.fltd = 0.0
            d.flts = 0.0
            d.str = None
            d.ref_name = None
            d.ref_off = 0
            if t == Tflts:
                d.flts = S.tokval_flts
                d.num = D.fbits(S.tokval_flts)
            elif t == Tfltd:
                d.fltd = S.tokval_fltd
                d.num = D.dbits(S.tokval_fltd)
            elif t == Tint:
                d.num = S.tokval_num
            elif t == Tglo:
                parsedatref(d)
            elif t == Tstr:
                parsedatstr(d)
            else:
                err("constant literal expected")
            cb(d)
            t = nextnl()
            if not (t == Tint or t == Tflts or t == Tfltd or t == Tstr or t == Tglo):
                break
        if t == Trbrace:
            break
        if t != Tcomma:
            err(", or } expected")
    d.type = D.DEnd
    cb(d)


def parselnk(lnk):
    haslnk = 0
    while True:
        t = nextnl()
        if t == Texport:
            lnk.export = 1
        elif t == Tthread:
            lnk.thread = 1
        elif t == Tcommon:
            lnk.common = 1
        elif t == Tsection:
            if lnk.sec:
                err("only one section allowed")
            if next_() != Tstr:
                err("section \"name\" expected")
            lnk.sec = S.tokval_str
            if peek() == Tstr:
                next_()
                lnk.secf = S.tokval_str
        else:
            if t == Tfunc and lnk.thread:
                err("only data may have thread linkage")
            if haslnk and t != Tdata and t != Tfunc:
                err("only data and function have linkage")
            return t
        haslnk = 1


def parse(buf, path, dbgfile, data, func):
    """buf: the whole input as bytes."""
    S.buf = buf
    S.n = len(buf)
    S.pos = 0
    D.inpath = path
    D.lnum = 1
    S.thead = Txxx
    S.ntyp = 0
    S.tokval_chr = 0
    S.tokval_num = 0
    S.tokval_flts = 0.0
    S.tokval_fltd = 0.0
    S.tokval_str = ''
    S.curb = None
    D.typ = []
    while True:
        lnk = Lnk()
        t = parselnk(lnk)
        if t == Tdbgfile:
            expect(Tstr)
            dbgfile(S.tokval_str)
        elif t == Tfunc:
            lnk.align = 16
            func(parsefn(lnk))
        elif t == Tdata:
            parsedat(data, lnk)
        elif t == Ttype:
            parsetyp()
        elif t == Teof:
            return
        else:
            err("top-level definition expected")


def _fmtf(x):
    """printf("%f"): glibc prints the sign of NaN."""
    if x != x:
        return '-nan' if math.copysign(1.0, x) < 0 else 'nan'
    return '%f' % x


def printcon(c, f):
    if c.type == CUndef:
        pass
    elif c.type == CAddr:
        if c.sym.type & SExt:
            f.write("extern ")
        if c.sym.type & SThr:
            f.write("thread ")
        f.write("$%s" % str_(c.sym.id))
        if c.i:
            f.write("%+d" % c.i)
    elif c.type == CBits:
        if c.flt == 1:
            f.write("s_%s" % _fmtf(c.s))
        elif c.flt == 2:
            f.write("d_%s" % _fmtf(c.d))
        else:
            f.write("%d" % c.i)


def printref(r, fn, f):
    rt = rtype(r)
    if rt == RTmp:
        if rval(r) < Tmp0:
            f.write("R%d" % rval(r))
        else:
            f.write("%%%s" % fn.tmp[rval(r)].name)
    elif rt == RCon:
        if req(r, UNDEF):
            f.write("UNDEF")
        else:
            printcon(fn.con[rval(r)], f)
    elif rt == RSlot:
        f.write("S%d" % rsval(r))
    elif rt == RCall:
        f.write("%04x" % rval(r))
    elif rt == RType:
        f.write(":%s" % D.typ[rval(r)].name)
    elif rt == RMem:
        i = 0
        m = fn.mem[rval(r)]
        f.write('[')
        if m.offset.type != CUndef:
            printcon(m.offset, f)
            i = 1
        if not req(m.base, R):
            if i:
                f.write(" + ")
            printref(m.base, fn, f)
            i = 1
        if not req(m.index, R):
            if i:
                f.write(" + ")
            f.write("%d * " % m.scale)
            printref(m.index, fn, f)
        f.write(']')
    elif rt == RInt:
        f.write("%d" % rsval(r))
    elif rt == -1:
        f.write("R")


_ktoc = "wlsd"
_clsops = None


def printfn(fn, f):
    global _clsops
    if _clsops is None:
        _clsops = {D.Oarg, D.Oswap, D.Oxcmp, D.Oacmp, D.Oacmn, D.Oafcmp, D.Oxtest,
                   D.Oxdiv, D.Oxidiv}
    f.write("function $%s() {\n" % fn.name)
    b = fn.start
    while b:
        f.write("@%s\n" % b.name)
        p = b.phi
        while p:
            f.write("\t")
            printref(p.to, fn, f)
            f.write(" =%c phi " % _ktoc[p.cls])
            assert p.narg
            n = 0
            while True:
                f.write("@%s " % p.blk[n].name)
                printref(p.arg[n], fn, f)
                if n == p.narg - 1:
                    f.write("\n")
                    break
                else:
                    f.write(", ")
                n += 1
            p = p.link
        for n, i in enumerate(b.ins[:b.nins]):
            if i.op == D.Ofma0 and n + 1 < b.nins and b.ins[n + 1].op == D.Ofma1:
                continue  # printed with its fma1, as "fma a, b, c"
            f.write("\t")
            if i.op == D.Ofma1 and n > 0 and b.ins[n - 1].op == D.Ofma0:
                printref(i.to, fn, f)
                f.write(" =%c fma " % _ktoc[i.cls])
                printref(i.arg[0], fn, f)
                f.write(", ")
                printref(i.arg[1], fn, f)
                f.write(", ")
                printref(b.ins[n - 1].arg[0], fn, f)
                f.write("\n")
                continue
            if not req(i.to, R):
                printref(i.to, fn, f)
                f.write(" =%c " % _ktoc[i.cls])
            assert optab[i.op].name
            f.write(optab[i.op].name)
            if req(i.to, R) and i.op in _clsops:
                f.write(_ktoc[i.cls])
            if not req(i.arg[0], R):
                f.write(" ")
                printref(i.arg[0], fn, f)
            if not req(i.arg[1], R):
                f.write(", ")
                printref(i.arg[1], fn, f)
            f.write("\n")
        jt = b.jmp.type
        if isret(jt) or jt == D.Jretc:
            f.write("\t%s" % jtoa[jt])
            if jt != D.Jret0 or not req(b.jmp.arg, R):
                f.write(" ")
                printref(b.jmp.arg, fn, f)
            if jt == D.Jretc:
                f.write(", :%s" % D.typ[fn.retty].name)
            f.write("\n")
        elif jt == D.Jhlt:
            f.write("\thlt\n")
        elif jt == D.Jjmp:
            if b.s1 is not b.link:
                f.write("\tjmp @%s\n" % b.s1.name)
        else:
            f.write("\t%s " % jtoa[jt])
            if jt == D.Jjnz:
                printref(b.jmp.arg, fn, f)
                f.write(", ")
            assert b.s1 and b.s2
            f.write("@%s, @%s\n" % (b.s1.name, b.s2.name))
        b = b.link
    f.write("}\n")
