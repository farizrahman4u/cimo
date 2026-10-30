"""Port of QBE copy.c."""

from . import defs as D
from . import gvn as G
from .defs import (R, RTmp, CON_Z, BIT, KBASE, KWIDE, M64, Ins, Kw, Jjmp, Jjnz, UPhi,
                   UIns, Ocopy, Oand, Oor, Oxor, Oshr, Osar, Onop, Oextsb, Oextuh, Oextub,
                   Oextsw, Oextuw, WFull, Wsb, Wub, Wsh, Wuh, Wsw, Wuw, isext, ispar,
                   optab, rtype, rval)
from .util import isconbits, iscmp, newtmp, argcls, phiarg

# zext, nopw (is a no-op if arg width is <= nopw), usew (uses only the low usew bits of arg)
_exttbl = [
    (0, 7, 8),     # extsb
    (1, 8, 8),     # extub
    (0, 15, 16),   # extsh
    (1, 16, 16),   # extuh
    (0, 31, 32),   # extsw
    (1, 32, 32),   # extuw
]


def ext(i):
    """Returns (zext, nopw, usew) or None."""
    if not isext(i.op):
        return None
    return _exttbl[i.op - Oextsb]


def bitwidth(v):
    return (v & M64).bit_length()


def visit(fn, r, w, f):
    ret = f(fn, r, w)
    b = fn.start
    while b:
        p = b.phi
        while p:
            p.visit = 0
            p = p.link
        b = b.link
    return ret


def uwl(fn, r, w):
    assert rtype(r) == RTmp
    t = fn.tmp[rval(r)]
    for k in range(t.nuse):
        u = t.use[k]
        if u.type == UPhi:
            p = u.u
            # during gvn, phi nodes may be
            # replaced by other temps; in
            # this case, the replaced phi
            # uses are added to the
            # replacement temp uses and
            # Phi.to is set to R
            if p.visit or p.to == R:
                continue
            p.visit = 1
            if uwl(fn, p.to, w):
                continue
        elif u.type == UIns:
            i = u.u
            assert i is not None
            if i.op == Ocopy:
                if uwl(fn, i.to, w):
                    continue
            e = ext(i)
            if e:
                if e[2] <= w:
                    continue
                if uwl(fn, i.to, w):
                    continue
            if i.op == Oand:
                if r == i.arg[0]:
                    rc = i.arg[1]
                else:
                    assert r == i.arg[1]
                    rc = i.arg[0]
                v = isconbits(fn, rc)
                if v is not None and bitwidth(v) <= w:
                    continue
        return 0
    return 1


def usewidthle(fn, r, w):
    """no more than w bits are used"""
    return visit(fn, r, w, uwl)


def dwl(fn, r, w):
    v = isconbits(fn, r)
    if v is not None and bitwidth(v) <= w:
        return 1
    if w <= 0:
        return 0
    if rtype(r) != RTmp:
        return 0
    t = fn.tmp[rval(r)]
    if t.cls != Kw:
        return 0

    if not t.def_:
        # phi def
        p = fn.rpo[t.bid].phi
        while p:
            if p.to == r:
                break
            p = p.link
        assert p
        if p.visit and p.visit <= w:
            return 1
        p.visit = w
        for n in range(p.narg):
            if not dwl(fn, p.arg[n], w):
                return 0
        return 1

    i = t.def_
    if i.op == Ocopy:
        return dwl(fn, i.arg[0], w)
    if i.op == Oshr or i.op == Osar:
        v = isconbits(fn, i.arg[1])
        if v is not None:
            if 0 < v <= 32:
                if i.op == Oshr and w + v >= 32:
                    return 1
                if w < 32:
                    if i.op == Osar:
                        w = min(31, w + v)
                    else:
                        w = min(32, w + v)
        return dwl(fn, i.arg[0], w)
    if iscmp(i.op):
        return int(w >= 1)
    if i.op == Oand:
        if dwl(fn, i.arg[0], w) or dwl(fn, i.arg[1], w):
            return 1
        return 0
    if i.op == Oor or i.op == Oxor:
        if dwl(fn, i.arg[0], w) and dwl(fn, i.arg[1], w):
            return 1
        return 0
    e = ext(i)
    if e:
        if e[0] and e[2] <= w:
            return 1
        w = min(w, e[1])
        return dwl(fn, i.arg[0], w)

    return 0


def defwidthle(fn, r, w):
    """is the ref narrower than w bits"""
    return visit(fn, r, w, dwl)


def isw1(fn, r):
    return defwidthle(fn, r, 1)


def narrowpars(fn):
    """insert early extub/extuh instructions
    for pars used only narrowly; this
    helps factoring extensions out of
    loops

    needs use; breaks use"""
    # only useful for functions with loops
    loop = 0
    b = fn.start
    while b:
        if b.loop > 1:
            loop = 1
            break
        b = b.link
    if not loop:
        return

    b = fn.start

    npar = 0
    for i in b.ins[:b.nins]:
        if not ispar(i.op):
            break
        npar += 1
    if npar == 0:
        return

    nins = b.nins + npar
    ins = [None] * nins
    for k in range(npar):
        ins[k] = b.ins[k].copy()
    for k in range(b.nins - npar):
        ins[2 * npar + k] = b.ins[npar + k].copy()
    b.ins = ins
    b.nins = nins

    for k in range(b.nins):
        i = b.ins[k]
        if not ispar(i.op):
            break
        e = Ins(Onop)
        if i.cls == Kw:
            if usewidthle(fn, i.to, 16):
                e.op = Oextuh
                if usewidthle(fn, i.to, 8):
                    e.op = Oextub
                r = newtmp("vw", i.cls, fn)
                e.cls = i.cls
                e.to = i.to
                e.arg[0] = r
                i.to = r
        b.ins[k + npar] = e


# which extensions are copies for a given
# argument width
_extcpy = {
    WFull: 0,
    Wsb: BIT(Wsb) | BIT(Wsh) | BIT(Wsw),
    Wub: BIT(Wub) | BIT(Wuh) | BIT(Wuw),
    Wsh: BIT(Wsh) | BIT(Wsw),
    Wuh: BIT(Wuh) | BIT(Wuw),
    Wsw: BIT(Wsw),
    Wuw: BIT(Wuw),
}


def copyref(fn, b, i):
    con01 = G.con01
    if i.op == Ocopy:
        return i.arg[0]

    op = optab[i.op]
    # op identity value
    if (op.hasid
            and KBASE(i.cls) == 0  # integer only - fp NaN!
            and i.arg[1] == con01[op.idval]
            and (not op.cmpeqwl or isw1(fn, i.arg[0]))):
        return i.arg[0]

    # idempotent op with identical args
    if op.idemp and i.arg[0] == i.arg[1]:
        return i.arg[0]

    # integer cmp with identical args
    if (op.cmpeqwl or op.cmplgtewl) and i.arg[0] == i.arg[1]:
        return con01[op.eqval]

    # cmpeq/ne 0 with 0/non-0 inference
    if op.cmpeqwl and i.arg[1] == CON_Z:
        z = G.zeroval(fn, b, i.arg[0], argcls(i, 0))
        if z is not None:
            return con01[op.eqval ^ z ^ 1]

    # redundant and mask
    if i.op == Oand:
        v = isconbits(fn, i.arg[1])
        if (v is not None
                and (v > 0 and ((v + 1) & v) == 0)
                and defwidthle(fn, i.arg[0], bitwidth(v))):
            return i.arg[0]

    if i.cls == Kw and (i.op == Oextsw or i.op == Oextuw):
        return i.arg[0]

    e = ext(i)
    if e and rtype(i.arg[0]) == RTmp:
        t = fn.tmp[rval(i.arg[0])]
        assert KBASE(t.cls) == 0

        # do not break typing by returning
        # a narrower temp
        if KWIDE(i.cls) > KWIDE(t.cls):
            return R

        w = Wsb + (i.op - Oextsb)
        if BIT(w) & _extcpy[t.width]:
            return i.arg[0]

        # avoid eliding extensions of params
        # inserted in the start block; their
        # point is to make further extensions
        # redundant
        if ((not t.def_ or not ispar(t.def_.op))
                and usewidthle(fn, i.to, e[2])):
            return i.arg[0]

        if defwidthle(fn, i.arg[0], e[1]):
            return i.arg[0]

    return R


def phieq(pa, pb):
    assert pa.narg == pb.narg
    for n in range(pa.narg):
        r = phiarg(pb, pa.blk[n])
        if pa.arg[n] != r:
            return 0
    return 1


def phicopyref(fn, b, p):
    con01 = G.con01
    # identical args
    r = R
    n = 0
    while n < p.narg:
        if p.arg[n] != p.to:
            if r == R:
                r = p.arg[n]
            elif p.arg[n] != r:
                break
        n += 1
    if n == p.narg:
        return r

    # same as a previous phi
    p1 = b.phi
    while p1 is not p:
        assert p1
        if phieq(p1, p):
            return p1.to
        p1 = p1.link

    # can be replaced by a
    # dominating jnz arg
    d = b.idom
    if (p.narg != 2
            or d.jmp.type != Jjnz
            or not isw1(fn, d.jmp.arg)):
        return R

    s = [None, None]
    for n in range(2):
        for c in range(2):
            if p.arg[n] == con01[c]:
                s[c] = p.blk[n]

    # if s1 ends with a jnz on either b
    # or s2; the inference below is wrong
    # without the jump type checks
    if (d.s1 is s[1] and d.s2 is s[0]
            and d.s1.jmp.type == Jjmp
            and d.s2.jmp.type == Jjmp):
        return d.jmp.arg

    return R
