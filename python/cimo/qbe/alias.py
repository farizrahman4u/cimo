"""Port of QBE alias.c."""

from .defs import (R, RTmp, RCon, RType, RInt, CAddr, NBit, M64, ABot, ALoc, ACon, AEsc,
                   ASym, AUnk, NoAlias, MayAlias, MustAlias, Alias, Oalloc, Oalloc1,
                   Ocopy, Oadd, Oblit0, Oblit1, Oargc, Jretc, CBits, BIT, astack, isload,
                   isstore, rtype, rval, rsval, s32, s64)
from .util import die, symeq


def getalias(a, r, fn):
    """getalias(a, r, fn): fills the Alias a in place."""
    t = rtype(r)
    if t == RTmp:
        a.set(fn.tmp[rval(r)].alias)
        if astack(a.type):
            a.type = a.slot.type
        assert a.type != ABot
    elif t == RCon:
        c = fn.con[rval(r)]
        if c.type == CAddr:
            a.type = ASym
            a.sym = c.sym.copy()
        else:
            a.type = ACon
        a.offset = c.i
        a.slot = None
    else:
        die("unreachable")


def alias(p, op, sp, q, sq, fn):
    """alias(p, op, sp, q, sq, &delta, fn): returns (result, delta)."""
    ap = Alias()
    aq = Alias()
    getalias(ap, p, fn)
    getalias(aq, q, fn)
    ap.offset = s64(ap.offset + op)
    # when delta is meaningful (ovlap == 1),
    # we do not overflow int because sp and
    # sq are bounded by 2^28
    delta = s32(ap.offset - aq.offset)
    ovlap = ap.offset < aq.offset + sq and aq.offset < ap.offset + sp

    if astack(ap.type) and astack(aq.type):
        # if both are offsets of the same
        # stack slot, they alias iif they
        # overlap
        if ap.base == aq.base and ovlap:
            return MustAlias, delta
        return NoAlias, delta

    if ap.type == ASym and aq.type == ASym:
        # they conservatively alias if the
        # symbols are different, or they
        # alias for sure if they overlap
        if not symeq(ap.sym, aq.sym):
            return MayAlias, delta
        if ovlap:
            return MustAlias, delta
        return NoAlias, delta

    if ((ap.type == ACon and aq.type == ACon)
            or (ap.type == aq.type and ap.base == aq.base)):
        assert ap.type == ACon or ap.type == AUnk
        # if they have the same base, we
        # can rely on the offsets only
        if ovlap:
            return MustAlias, delta
        return NoAlias, delta

    # if one of the two is unknown
    # there may be aliasing unless
    # the other is provably local
    if ap.type == AUnk and aq.type != ALoc:
        return MayAlias, delta
    if aq.type == AUnk and ap.type != ALoc:
        return MayAlias, delta

    return NoAlias, delta


def escapes(r, fn):
    if rtype(r) != RTmp:
        return 1
    a = fn.tmp[rval(r)].alias
    return int(not astack(a.type) or a.slot.type == AEsc)


def esc(r, fn):
    assert rtype(r) <= RType
    if rtype(r) == RTmp:
        a = fn.tmp[rval(r)].alias
        if astack(a.type):
            a.slot.type = AEsc


def store(r, sz, fn):
    if rtype(r) == RTmp:
        a = fn.tmp[rval(r)].alias
        if a.slot:
            assert astack(a.type)
            off = a.offset
            if sz >= NBit or (off < 0 or off >= NBit):
                m = M64
            else:
                m = ((BIT(sz) - 1) << off) & M64
            a.slot.loc_m |= m


# C locals a0, a1 of fillalias(); kept across calls like stack garbage
_a0 = Alias()
_a1 = Alias()


def fillalias(fn):
    from .load import storesz
    a0 = _a0
    a1 = _a1
    for t in range(fn.ntmp):
        fn.tmp[t].alias.type = ABot
    for n in range(fn.nblk):
        b = fn.rpo[n]
        p = b.phi
        while p:
            assert rtype(p.to) == RTmp
            a = fn.tmp[rval(p.to)].alias
            assert a.type == ABot
            a.type = AUnk
            a.base = rval(p.to)
            a.offset = 0
            a.slot = None
            p = p.link
        ii = 0
        while ii < b.nins:
            i = b.ins[ii]
            a = None
            if i.to != R:
                assert rtype(i.to) == RTmp
                a = fn.tmp[rval(i.to)].alias
                assert a.type == ABot
                if Oalloc <= i.op <= Oalloc1:
                    a.type = ALoc
                    a.slot = a
                    a.loc_sz = -1
                    if rtype(i.arg[0]) == RCon:
                        c = fn.con[rval(i.arg[0])]
                        x = c.i
                        if c.type == CBits:
                            if 0 <= x <= NBit:
                                a.loc_sz = x
                else:
                    a.type = AUnk
                    a.slot = None
                a.base = rval(i.to)
                a.offset = 0
            if i.op == Ocopy:
                assert a
                getalias(a, i.arg[0], fn)
            if i.op == Oadd:
                getalias(a0, i.arg[0], fn)
                getalias(a1, i.arg[1], fn)
                if a0.type == ACon:
                    a.set(a1)
                    a.offset = s64(a.offset + a0.offset)
                elif a1.type == ACon:
                    a.set(a0)
                    a.offset = s64(a.offset + a1.offset)
            if i.to == R or a.type == AUnk:
                if i.op != Oblit0:
                    if not isload(i.op):
                        esc(i.arg[0], fn)
                    if not isstore(i.op):
                        if i.op != Oargc:
                            esc(i.arg[1], fn)
            if i.op == Oblit0:
                ii += 1
                i = b.ins[ii]
                assert i.op == Oblit1
                assert rtype(i.arg[0]) == RInt
                sz = abs(rsval(i.arg[0]))
                store(b.ins[ii - 1].arg[1], sz, fn)
            if isstore(i.op):
                store(i.arg[1], storesz(i), fn)
            ii += 1
        if b.jmp.type != Jretc:
            esc(b.jmp.arg, fn)
    b = fn.start
    while b:
        p = b.phi
        while p:
            for n in range(p.narg):
                esc(p.arg[n], fn)
            p = p.link
        b = b.link
