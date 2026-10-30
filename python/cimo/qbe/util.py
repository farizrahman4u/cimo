"""Port of QBE util.c."""

from . import defs as D
from .defs import (R, RTmp, RCon, Tmp0, NBit, U32, NIns, Ins, Tmp, Con, CBits, CAddr,
                   CUndef, Kx, Kw, Kl, Ks, Kd, Ocmpw, Ocmpw1, Ocmpl, Ocmpl1, Ocmps,
                   Ocmps1, Ocmpd, Ocmpd1, NCmpI, NCmp, Oblit0, Oblit1, Ocall, Osel0,
                   Osel1, Onop, Osalloc, Oand, Oadd, ispar, isarg, optab, TMP, CON,
                   rtype, rval, s64, QbeError)

IBits = 12
IMask = (1 << IBits) - 1


def hash(s):
    h = 0
    for c in s.encode('latin-1'):
        h = (c + 17 * h) & U32
    return h


def die(fmt, *args):
    raise QbeError('dying: ' + (fmt % args if args else fmt))


def err(fmt, *args):
    raise QbeError('qbe:%s:%d: %s' % (D.inpath, D.lnum, fmt % args if args else fmt))


def vgrow(lst, n, factory=None):
    """Grow list lst to at least n elements (new elements from factory())."""
    while len(lst) < n:
        lst.append(factory() if factory else None)


def addins(b_or_list, i):
    """addins(&b->ins, &b->nins, i) is addins(b, i); appends a copy of i to b."""
    if i.op == Onop:
        return
    b = b_or_list
    b.ins[b.nins:] = [i.copy()]
    b.nins += 1


def addbins(b, bs):
    """addbins(&b->ins, &b->nins, bs)"""
    for k in range(bs.nins):
        addins(b, bs.ins[k])


def strf(fmt, *args):
    return fmt % args if args else fmt


_itbl = [[] for _ in range(IMask + 1)]


def intern(s):
    h = hash(s) & IMask
    b = _itbl[h]
    for i, s1 in enumerate(b):
        if s1 == s:
            return h + (i << IBits)
    if len(b) == 1 << (32 - IBits):
        die("interning table overflow")
    b.append(s)
    return h + ((len(b) - 1) << IBits)


def str_(id):
    b = _itbl[id & IMask]
    assert id >> IBits < len(b)
    return b[id >> IBits]


def isreg(r):
    return rtype(r) == RTmp and rval(r) < Tmp0


def iscmp(op):
    """Returns (k, c) if op is a comparison, else None."""
    if Ocmpw <= op <= Ocmpw1:
        return Kw, op - Ocmpw
    if Ocmpl <= op <= Ocmpl1:
        return Kl, op - Ocmpl
    if Ocmps <= op <= Ocmps1:
        return Ks, NCmpI + op - Ocmps
    if Ocmpd <= op <= Ocmpd1:
        return Kd, NCmpI + op - Ocmpd
    return None


def igroup(b, i):
    """i is an index in b.ins; returns (i0, i1) indices."""
    ins = b.ins
    ib = 0
    ie = b.nins
    op = ins[i].op
    if op == Oblit0:
        return i, i + 2
    if op == Oblit1:
        return i - 1, i + 1
    if ispar(op):
        while i > ib and ispar(ins[i - 1].op):
            i -= 1
        i0 = i
        while i < ie and ispar(ins[i].op):
            i += 1
        return i0, i
    if op == Ocall or isarg(op):
        while i > ib and isarg(ins[i - 1].op):
            i -= 1
        i0 = i
        while i < ie and ins[i].op != Ocall:
            i += 1
        assert i < ie
        return i0, i + 1
    if op == Osel1:
        while i > ib and ins[i - 1].op == Osel1:
            i -= 1
        assert ins[i].op == Osel0
        op = Osel0
    if op == Osel0:
        i0 = i
        i += 1
        while i < ie and ins[i].op == Osel1:
            i += 1
        return i0, i
    return i, i + 1


def argcls(i, n):
    return optab[i.op].argcls[n][i.cls]


def emit(op, k, to, arg0, arg1):
    if D.curi == 0:
        die("emit, too many instructions")
    D.curi -= 1
    D.insb[D.curi] = Ins(op, k, to, arg0, arg1)


def emiti(i):
    emit(i.op, i.cls, i.to, i.arg[0], i.arg[1])


def idup(b, s, n):
    """idup(b, &insb[s], n): b->ins becomes a copy of insb[s:s+n]."""
    insb = D.insb
    b.ins = [insb[k].copy() for k in range(s, s + n)]
    b.nins = n


def icpy(d, di, s, si, n):
    """icpy(&d[di], &s[si], n) on two lists; returns di + n. Copies Ins values
    (memmove semantics: safe when d is s)."""
    tmp = [s[k].copy() for k in range(si, si + n)]
    if len(d) < di + n:
        d.extend([None] * (di + n - len(d)))
    d[di:di + n] = tmp
    return di + n


# negation, swap
_cmptab = {
    D.Ciule: (D.Ciugt, D.Ciuge),
    D.Ciult: (D.Ciuge, D.Ciugt),
    D.Ciugt: (D.Ciule, D.Ciult),
    D.Ciuge: (D.Ciult, D.Ciule),
    D.Cisle: (D.Cisgt, D.Cisge),
    D.Cislt: (D.Cisge, D.Cisgt),
    D.Cisgt: (D.Cisle, D.Cislt),
    D.Cisge: (D.Cislt, D.Cisle),
    D.Cieq: (D.Cine, D.Cieq),
    D.Cine: (D.Cieq, D.Cine),
    NCmpI + D.Cfle: (-1, NCmpI + D.Cfge),
    NCmpI + D.Cflt: (-1, NCmpI + D.Cfgt),
    NCmpI + D.Cfgt: (-1, NCmpI + D.Cflt),
    NCmpI + D.Cfge: (-1, NCmpI + D.Cfle),
    NCmpI + D.Cfeq: (-1, NCmpI + D.Cfeq),
    NCmpI + D.Cfne: (-1, NCmpI + D.Cfne),
    NCmpI + D.Cfo: (-1, NCmpI + D.Cfo),
    NCmpI + D.Cfuo: (-1, NCmpI + D.Cfuo),
}


def cmpop(c):
    assert 0 <= c < NCmp
    return _cmptab[c][1]


def cmpwlneg(op):
    if Ocmpw <= op <= Ocmpw1:
        return _cmptab[op - Ocmpw][0] + Ocmpw
    if Ocmpl <= op <= Ocmpl1:
        return _cmptab[op - Ocmpl][0] + Ocmpl
    die("not a wl comparison")


def clsmerge(k1, k):
    """clsmerge(&k1, k): returns (new k1, result)."""
    if k1 == Kx:
        return k, 0
    if (k1 == Kw and k == Kl) or (k1 == Kl and k == Kw):
        return Kw, 0
    return k1, int(k1 != k)


def phicls(t, tmp):
    t1 = tmp[t].phi
    if not t1:
        return t
    t1 = phicls(t1, tmp)
    tmp[t].phi = t1
    return t1


def phiargn(p, b):
    if p:
        for n in range(p.narg):
            if p.blk[n] is b:
                return n
    return U32


def phiarg(p, b):
    n = phiargn(p, b)
    assert n != U32, "block not found"
    return p.arg[n]


_newtmp_n = 0


def newtmp(prfx, k, fn):
    global _newtmp_n
    t = fn.ntmp
    fn.ntmp += 1
    vgrow(fn.tmp, fn.ntmp)
    tm = Tmp()
    fn.tmp[t] = tm
    if prfx:
        _newtmp_n += 1
        tm.name = "%s.%d" % (prfx, _newtmp_n)
    tm.cls = k
    tm.slot = -1
    tm.nuse = 1
    tm.ndef = 1
    return TMP(t)


def chuse(r, du, fn):
    if rtype(r) == RTmp:
        fn.tmp[rval(r)].nuse += du


def symeq(s0, s1):
    return s0.type == s1.type and s0.id == s1.id


def newcon(c0, fn):
    for i in range(1, fn.ncon):
        c1 = fn.con[i]
        if c0.type == c1.type and symeq(c0.sym, c1.sym) and c0.i == c1.i:
            return CON(i)
    i = fn.ncon
    fn.ncon += 1
    vgrow(fn.con, fn.ncon, Con)
    fn.con[i] = c0.copy()
    return CON(i)


def getcon(val, fn):
    for c in range(1, fn.ncon):
        if fn.con[c].type == CBits and fn.con[c].i == val:
            return CON(c)
    c = fn.ncon
    fn.ncon += 1
    vgrow(fn.con, fn.ncon, Con)
    fn.con[c] = Con(CBits, val)
    return CON(c)


def addcon(c0, c1, m):
    if m != 1 and c1.type == CAddr:
        return 0
    if c0.type == CUndef:
        c0.set(c1)
        c0.i = s64(c0.i * m)
    else:
        if c1.type == CAddr:
            if c0.type == CAddr:
                return 0
            c0.type = CAddr
            c0.sym = c1.sym.copy()
        c0.i = s64(c0.i + c1.i * m)
    return 1


def isconbits(fn, r):
    """Returns the constant's bits, or None."""
    if rtype(r) == RCon:
        c = fn.con[rval(r)]
        if c.type == CBits:
            return c.i
    return None


def salloc(rt, rs, fn):
    fn.dynalloc = 1
    if rtype(rs) == RCon:
        sz = fn.con[rval(rs)].i
        if sz < 0 or sz >= 0x7fffffff - 15:
            err("invalid alloc size %d", sz)
        sz = (sz + 15) & -16
        emit(Osalloc, Kl, rt, getcon(sz, fn), R)
    else:
        r0 = newtmp("isel", Kl, fn)
        r1 = newtmp("isel", Kl, fn)
        emit(Osalloc, Kl, rt, r0, R)
        emit(Oand, Kl, r0, r1, getcon(-16, fn))
        emit(Oadd, Kl, r1, rs, getcon(15, fn))
        if fn.tmp[rval(rs)].slot != -1:
            err("unlikely alloc argument %%%s for %%%s",
                fn.tmp[rval(rs)].name, fn.tmp[rval(rt)].name)


# bitsets: BSet.t is a Python int

def bsinit(bs, n):
    bs.nt = (n + NBit - 1) // NBit
    bs.t = 0


def bszero(bs):
    bs.t = 0


def bscount(bs):
    return bin(bs.t).count('1')


def bsset(bs, elt):
    assert elt < bs.nt * NBit
    bs.t |= 1 << elt


def bsclr(bs, elt):
    assert elt < bs.nt * NBit
    bs.t &= ~(1 << elt)


def bscopy(a, b):
    assert a.nt == b.nt
    a.t = b.t


def bsunion(a, b):
    assert a.nt == b.nt
    a.t |= b.t


def bsinter(a, b):
    assert a.nt == b.nt
    a.t &= b.t


def bsdiff(a, b):
    assert a.nt == b.nt
    a.t &= ~b.t


def bsequal(a, b):
    assert a.nt == b.nt
    return a.t == b.t


def bshas(bs, elt):
    assert elt < bs.nt * NBit
    return (bs.t >> elt) & 1


def bsiter(bs, i):
    """Smallest element >= i, or -1. C: for (i=0; bsiter(bs, &i); i++) is
    i = bsiter(bs, 0); while i >= 0: ...; i = bsiter(bs, i+1)"""
    if i >= bs.nt * NBit:
        return -1
    b = bs.t >> i
    if not b:
        return -1
    return i + (b & -b).bit_length() - 1


def bselems(bs, i=0):
    """Iterate the elements >= i (snapshot of the set at call time)."""
    b = bs.t >> i
    while b:
        lo = b & -b
        k = lo.bit_length() - 1
        yield i + k
        b ^= lo


def dumpts(bs, tmp, f):
    f.write("[")
    for t in bselems(bs, Tmp0):
        f.write(" %s" % tmp[t].name)
    f.write(" ]\n")
