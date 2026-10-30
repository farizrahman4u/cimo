"""Port of QBE simpl.c."""

from . import defs as D
from .defs import (R, RCon, CBits, KBASE, Kl, Kw, M64, NIns, Ostorel, Ostorew, Ostoreh,
                   Ostoreb, Oload, Oloaduh, Oloadub, Oadd, Oblit0, Oblit1, Oudiv, Ourem,
                   Oand, Oshr, rtype, rval, rsval)
from .util import newtmp, getcon, emit, emiti, icpy, idup

_blittbl = [
    (Ostorel, Oload, Kl, 8),
    (Ostorew, Oload, Kw, 4),
    (Ostoreh, Oloaduh, Kw, 2),
    (Ostoreb, Oloadub, Kw, 1),
]


def blit(sd, sz, fn):
    fwd = sz >= 0
    sz = abs(sz)
    off = sz if fwd else 0
    k = 0
    while sz:
        st, ld, cls, size = _blittbl[k]
        n = size
        while sz >= n:
            off -= n if fwd else 0
            r = newtmp("blt", Kl, fn)
            r1 = newtmp("blt", Kl, fn)
            ro = getcon(off, fn)
            emit(st, 0, R, r, r1)
            emit(Oadd, Kl, r1, sd[1], ro)
            r1 = newtmp("blt", Kl, fn)
            emit(ld, cls, r, r1, R)
            emit(Oadd, Kl, r1, sd[0], ro)
            off += 0 if fwd else n
            sz -= n
        k += 1


ulog2_tab64 = [
    63, 0, 1, 41, 37, 2, 16, 42,
    38, 29, 32, 3, 12, 17, 43, 55,
    39, 35, 30, 53, 33, 21, 4, 23,
    13, 9, 18, 6, 25, 44, 48, 56,
    62, 40, 36, 15, 28, 31, 11, 54,
    34, 52, 20, 22, 8, 5, 24, 47,
    61, 14, 27, 10, 51, 19, 7, 46,
    60, 26, 50, 45, 59, 49, 58, 57,
]


def ulog2(pow2):
    return ulog2_tab64[((pow2 * 0x5b31ab928877a7e) & M64) >> 58]


def ispow2(v):
    return v and (v & (v - 1)) == 0


def ins(k, new, b, fn):
    """ins(&i, &new, b, fn) with i the index k in b.ins; returns (k, new)."""
    i = b.ins[k]
    # simplify more instructions here;
    # copy 0 into xor, bit rotations,
    # etc.
    if i.op == Oblit1:
        assert k > 0
        assert b.ins[k - 1].op == Oblit0
        if not new:
            D.curi = NIns
            ni = b.nins - (k + 1)
            D.curi -= ni
            icpy(D.insb, D.curi, b.ins, k + 1, ni)
            new = 1
        blit(b.ins[k - 1].arg, rsval(i.arg[0]), fn)
        return k - 1, new
    elif i.op == Oudiv or i.op == Ourem:
        r = i.arg[1]
        if KBASE(i.cls) == 0:
            if rtype(r) == RCon:
                c = fn.con[rval(r)]
                if c.type == CBits:
                    v = c.i & M64
                    if ispow2(v):
                        n = ulog2(v)
                        if i.op == Ourem:
                            i.op = Oand
                            i.arg[1] = getcon((1 << n) - 1, fn)
                        else:
                            i.op = Oshr
                            i.arg[1] = getcon(n, fn)
    if new:
        emiti(i)
    return k, new


def simpl(fn):
    b = fn.start
    while b:
        new = 0
        k = b.nins
        while k != 0:
            k -= 1
            k, new = ins(k, new, b, fn)
        if new:
            idup(b, D.curi, NIns - D.curi)
        b = b.link
