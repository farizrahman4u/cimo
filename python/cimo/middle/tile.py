"""Loop tiling of k in an i, k, j nest (gemm): the i loop runs inside blocks of KB values of k.

    for (i...)                          for (kk = k0; kk < bk; kk = ke) {
      for (k = k0; k < bk; k++) {  ->      ke = bk - kk > KB ? kk + KB : bk;     (unsigned: no overflow)
        pre(i, k);                         for (i...)
        for (j...) body(i, k, j);            for (k = kk; k < ke; k++) { pre(i, k); for (j...) body(i, k, j); }
      }                                  }

The block of b's rows a block of k reads (KB rows) stays in L1 across every i, instead of all of b streaming from L2
for each row of c: gemm at n = 384 2.3 against 3.7 ms on Ice Lake, 1.55 against 2.8 on Sapphire Rapids, 4.0 against 5.6
on Graviton 3.

Iterations (i, k) now run in the order (block of k, i, k), so for i1 < i2 an iteration of i2 can come before one of i1.
That is legal when nothing one row of i touches is touched by another: every store, and every access that may alias
it, is at one address form, j-strided by its own size, whose part that varies with i steps by at least the bytes j
covers (a row: c[i*n + j] with j in [0, n), or c[i][j] with a constant row); it doesn't depend on k; loads outside the
innermost loop alias no store. Each element of c then sees the same updates in the same order (k ascending): results
are bit for bit the same.

Shape: i loop {Hi, Bi, k loop, Ei}, k loop {Hk, Bk, j loop, Lk}, j innermost (inner.Shape); Hi, Hk with induction
variables only; k counts up by 1 (slt or ult) to a bound invariant in the nest, from an init invariant in it; Bi, Ei,
Bk, Lk with no stores; nothing the nest computes used after it.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ins, rval
from ..backend.util import newtmp
from . import ir
from .ir import Bail, Emitter
from .affine import Analysis
from .inner import Shape, _loadinfo, _storeinfo
import cimo.middle as M

KB = 16  # rows of k a block: 16 to 32 was best on every machine measured (n = 384), 16 a multiple of jam's 4

_LT = {D.Ocsltl: (Kl, D.Ocugtl), D.Ocsltw: (Kw, D.Ocugtw), D.Ocultl: (Kl, D.Ocugtl), D.Ocultw: (Kw, D.Ocugtw)}


def _defin(fn, r, blocks):
    """The instruction defining r in one of blocks, else None."""
    if not ir.istmp(r):
        return None
    for b in blocks:
        for i in ir.ins(b):
            if i.to == r:
                return i
    return None


def _iv(fn, p, latch):
    """p's constant step when its latch value is p + constant, else None."""
    nx = ir.phiargof(p, latch)
    i = None
    for b in ir.blocks(fn):
        for x in ir.ins(b):
            if x.to == nx:
                i = x
                break
        if i is not None:
            break
    if i is None or i.op != D.Oadd or i.arg[0] != p.to:
        return None
    return ir.conval(fn, i.arg[1])


def _single_succ(b, target):
    return b.jmp.type == D.Jjmp and b.s1 is target


def transform(fn, I, rts, done):
    if len(I.children) != 1 or len(I.children[0].children) != 1 or I.children[0].children[0].children:
        return False
    K = I.children[0]
    J = K.children[0]
    Hi, Hk = I.header, K.header
    outside = lambda r, lp: not (ir.istmp(r) and fn.tmp[rval(r)].bid in {b.id for b in lp.blocks})
    iblk = I.blocks

    # ---- shape
    pre = I.outside_preds()
    if len(pre) != 1 or pre[0].jmp.type != D.Jjmp or len(I.latches) != 1 or len(K.latches) != 1:
        raise Bail('tile: loop shape')
    pre = pre[0]
    if Hi.jmp.type != D.Jjnz:
        raise Bail('tile: outer header')
    Bi, Xi = (Hi.s1, Hi.s2) if Hi.s1 in iblk else (Hi.s2, Hi.s1)
    if Bi not in iblk or Xi in iblk or not _single_succ(Bi, Hk):
        raise Bail('tile: outer body')
    Ei = I.latches[0]
    if Hk.jmp.type != D.Jjnz or Ei.npred != 1 or not _single_succ(Ei, Hi):
        raise Bail('tile: middle header')
    if Hk.s1 in K.blocks and Hk.s2 is Ei:
        Bk = Hk.s1
    else:
        raise Bail('tile: middle exit')
    Lk = K.latches[0]
    sh = Shape(fn, J)
    if sh.pre is not Bk or sh.E is not Lk or Lk.npred != 1 or not _single_succ(Lk, Hk):
        raise Bail('tile: inner shape')
    if iblk != {Hi, Bi, Ei} | K.blocks or K.blocks != {Hk, Bk, Lk} | J.blocks:
        raise Bail('tile: extra blocks')
    for b in iblk:
        for i in ir.ins(b):
            if not ir.copyable(i):
                raise Bail('tile: uncopyable')
            if ir.isstore(i.op) and b not in J.blocks:
                raise Bail('tile: store outside the inner loop')
    # nothing the nest computes is used after it
    for b in ir.blocks(fn):
        if b in iblk:
            continue
        refs = [a for p in ir.phis(b) for a in p.arg[:p.narg]] + [a for i in ir.ins(b) for a in i.arg] + [b.jmp.arg]
        if any(ir.istmp(r) and not outside(r, I) for r in refs):
            raise Bail('tile: value used after the nest')

    # ---- induction variables: i any constant step, k step 1 tested k < bk
    istep = {}
    for p in ir.phis(Hi):
        s = _iv(fn, p, Ei)
        if s is None or s == 0 or not outside(ir.phiargof(p, pre), I):
            raise Bail('tile: outer carried value')
        istep[p.to] = s
    kphi = None
    for p in ir.phis(Hk):
        s = _iv(fn, p, Lk)
        if s is None or not outside(ir.phiargof(p, Bi), I):
            raise Bail('tile: middle carried value')
        if s == 1:
            kphi = p if kphi is None else kphi
    test = _defin(fn, Hk.jmp.arg, [Hk])
    if kphi is None or test is None or test.op not in _LT or test.arg[0] != kphi.to or fn.tmp[rval(test.to)].nuse != 1:
        raise Bail('tile: middle test')
    kcls, ugt = _LT[test.op]
    bk = test.arg[1]
    if kphi.cls != kcls or not outside(bk, I):
        raise Bail('tile: middle bound')
    if any(p is not kphi for p in ir.phis(Hk)):  # another one would restart from its init in every block
        raise Bail('tile: middle loop with another induction variable')
    hi_test = _defin(fn, Hi.jmp.arg, [Hi])
    if hi_test is None or any(not outside(a, I) and a not in istep for a in hi_test.arg if a != R):
        raise Bail('tile: outer test')

    # ---- memory
    A = Analysis(fn, J, sh.order)
    A.fn_con0 = ir.getcon(0, fn)
    jtest = _defin(fn, sh.H.jmp.arg, [sh.H])
    jphi = [p for p in ir.phis(sh.H)]
    if jtest is None or jtest.op not in (D.Ocsltl, D.Ocsltw) or not any(jtest.arg[0] == p.to for p in jphi):
        raise Bail('tile: inner test')
    jp = [p for p in jphi if p.to == jtest.arg[0]][0]
    j0 = ir.conval(fn, ir.phiargof(jp, Bk))
    bj = jtest.arg[1]
    if j0 is None or _iv(fn, jp, sh.latch) != 1 or not outside(bj, I):
        raise Bail('tile: inner counter')

    def ipart(r, depth=0):
        """r as (ci, cm, m): ci*i + cm*(i*m) + something the same for every i and k, else None."""
        if depth > 8:
            return None
        if not ir.istmp(r) or outside(r, I):
            return (0, 0, None)
        if r in istep:
            return (1, 0, None)
        x = _defin(fn, r, [Bi])
        if x is None or x.cls != Kl:
            return None
        if x.op == D.Ocopy:
            return ipart(x.arg[0], depth + 1)
        if x.op in (D.Oextsw, D.Oextuw):  # of i itself: i only runs from its init up or down, never wrapping
            return (1, 0, None) if x.arg[0] in istep else None
        if x.op in (D.Oadd, D.Osub):
            a, b = ipart(x.arg[0], depth + 1), ipart(x.arg[1], depth + 1)
            if a is None or b is None or (a[1] and b[1] and a[2] != b[2]):
                return None
            k = 1 if x.op == D.Oadd else -1
            return (a[0] + k * b[0], a[1] + k * b[1], a[2] or b[2])
        if x.op in (D.Omul, D.Oshl):
            c = ir.conval(fn, x.arg[1])
            if x.op == D.Oshl:
                if c is None or not 0 <= c < 32:
                    return None
                c = 1 << c
            if c is not None:
                a = ipart(x.arg[0], depth + 1)
                return None if a is None else (a[0] * c, a[1] * c, a[2])
            c = ir.conval(fn, x.arg[0]) if x.op == D.Omul else None
            if c is not None:
                a = ipart(x.arg[1], depth + 1)
                return None if a is None else (a[0] * c, a[1] * c, a[2])
            for u, v in ((x.arg[0], x.arg[1]), (x.arg[1], x.arg[0])):
                if u in istep and outside(v, I):
                    return (0, 1, v)
        return None

    def rows(f, size):
        """Whether f's addresses for different i never meet: steps by at least the bytes j covers."""
        if not f.S.isconst() or f.S.c != size or size <= 0:
            return False
        ci = cm = 0
        m = None
        for (r, kind), coef in f.L.t.items():
            if outside(r, I):
                continue
            if kind != 'l':
                return False
            part = ipart(r)
            if part is None or (part[1] and m is not None and part[2] != m):
                return False
            ci += coef * part[0]
            cm += coef * part[1]
            m = m or part[2]
        step = abs(list(istep.values())[0]) if len(istep) == 1 else None
        if step is None or (ci and cm) or not (ci or cm):
            return False
        if ci:
            n = ir.conval(fn, bj)
            return n is not None and abs(ci) * step >= size * (n - j0)
        return cm == size and m == bj and j0 == 0

    stores, others = [], []
    for b in sh.order:
        for i in ir.ins(b):
            if ir.isstore(i.op):
                stores.append((i, i.arg[1], _storeinfo(i.op)[1]))
            elif ir.isload(i.op):
                info = _loadinfo(i.op, i.cls) if i.cls in (Kw, Kl, D.Ks, D.Kd) else None
                others.append((i, i.arg[0], info[0] if info else 8))
    if not stores:
        raise Bail('tile: no stores')
    for st, a, size in stores:
        f = A.form(a, 64)
        if f is None or not rows(f, size):
            raise Bail('tile: rows of the stores')
        for x, xa, xsize in stores + others:
            if x is st or not rts.may_alias(a, xa):
                continue
            g = A.form(xa, 64)
            if g is None or xsize != size or g.L.ident() != f.L.ident() or g.S.ident() != f.S.ident():
                raise Bail('tile: an access that may alias a store')
    for b in (Hi, Bi, Ei, Hk, Bk, Lk):
        for i in ir.ins(b):
            if ir.isload(i.op) and any(rts.may_alias(i.arg[0], a) for _, a, _ in stores):
                raise Bail('tile: a load outside the inner loop may alias a store')

    # ---- transform
    TH = ir.newblock(fn, pre, 'th')    # kk = phi(k0, ke); kk < bk ?
    TB = ir.newblock(fn, TH, 'tb')     # ke
    TL = ir.newblock(fn, TB, 'tl')     # the next block
    k0 = ir.phiargof(kphi, Bi)
    kk = newtmp('t', kcls, fn)
    eb = Emitter(fn, TB)
    rem = eb.sub(kcls, bk, kk)
    more = eb.raw(ugt, Kw, rem, eb.con(KB))
    nxt = eb.addc(kcls, kk, KB)
    eb.ins(Ins(D.Osel0, Kw, R, more, R))
    ke = eb.raw(D.Osel1, kcls, nxt, bk)
    eb.flush()
    ir.setjmp(TB, Hi)
    eh = Emitter(fn, TH)
    c = eh.raw(test.op, Kw, kk, bk)
    eh.flush()
    ir.setjnz(TH, c, TB, Xi)
    ir.addphi(TH, kcls, kk, [(pre, k0), (TL, ke)])
    ir.setjmp(TL, TH)
    ir.retarget(pre, Hi, TH)
    ir.renamepred(Hi, pre, TB)
    ir.retarget(Hi, Xi, TL)
    ir.renamepred(Xi, Hi, TH)
    ir.setphiarg(kphi, Bi, kk)
    test.arg[1] = ke
    done.update([Hi, TH])
    M.count('tiled')
    return True
