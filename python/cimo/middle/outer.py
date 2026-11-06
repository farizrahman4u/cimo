"""Outer-loop vectorization (nbody_acc): the lanes are iterations of the outer loop, each running the whole inner loop.

    for (i...) {                         for (i = i0; bound - i >= vf; i += vf) {     lanes i .. i+vf-1
      pre(i);                              pre(i .. i+vf-1);
      for (j...) s = f(s, i, j);           for (j...) s[0..vf-1] = f(s, i .. i+vf-1, j);
      post(s, i);                          post(...);
    }                                    }
                                         the scalar loop, from i, for what is left

Each lane does the scalar instructions of its own i, in the same order, so every sum is added in C's order and results
are bit for bit the scalar loop's: no reduction is reassociated, unlike vectorizing the inner loop (whose in-order
sums have to add their lanes one by one).

Values are uniform (the same for every i: computed from j and invariants, kept scalar and broadcast where a lane needs
them), affine in i (addresses: lane 0's is computed with i = the vector loop's counter), or varying (one per lane, a
vector). A load is uniform when its address is, or a vector load when its address steps by its size with i; a store
is a vector store at such an address. Varying values have one class (lanes of one width).

Legal when: the inner loop's trip count is the same for every i (uniform bounds); the inner loop stores nothing; no
store in the nest may alias a load in it (the lanes' stores come after all of their inner loops); the outer loop
counts up by 1 to a uniform bound; nothing computed in the nest is used after it.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ks, Kd, Kq, Ins, rval
from ..backend.util import newtmp
from . import ir
from .ir import Bail, Emitter
from .inner import Shape, _loadinfo, _storeinfo
from . import vectorize
from ..backend import ext as _ext
import cimo.middle as M

_LANE = {Ks: 's', Kd: 'd', Kw: 'w', Kl: 'l'}
_SIZE = {Ks: 4, Kd: 8, Kw: 4, Kl: 8}


def _kof(kind, a, inside):
    if not ir.istmp(a) or not inside(a):
        return 'u'
    return kind[rval(a)]


def _classify(fn, ip, iphis, varying, order, inside, Bpre, sh):
    """Kinds of the nest's values with the inner phis in varying taken as varying, the others as uniform."""
    kind = {rval(ip.to): 'a'}
    coef = {rval(ip.to): 1}
    vcls = set()
    for p in iphis:
        kind[rval(p.to)] = 'v' if rval(p.to) in varying else 'u'
    k = lambda a: _kof(kind, a, inside)
    for b in order:
        third = None
        for i in ir.ins(b):
            if i.op in (D.Onop, D.Odbgloc):
                continue
            if i.op == D.Ofma0:
                third = i.arg[0]
                continue
            if ir.isload(i.op):
                ka = k(i.arg[0])
                if ka == 'u':
                    kind[rval(i.to)] = 'u'
                elif ka == 'a':
                    info = _loadinfo(i.op, i.cls) if i.cls in _LANE else None
                    if info is None or info[1] is not None or coef[rval(i.arg[0])] != info[0]:
                        raise Bail('outer: load not consecutive in i')
                    kind[rval(i.to)] = 'v'
                    vcls.add(i.cls)
                else:
                    raise Bail('outer: load at a varying address')
                continue
            if ir.isstore(i.op):
                if k(i.arg[1]) != 'a' or coef[rval(i.arg[1])] != _storeinfo(i.op)[1] or k(i.arg[0]) == 'a':
                    raise Bail('outer: store not consecutive in i')
                if k(i.arg[0]) == 'v':
                    vcls.add({D.Ostored: Kd, D.Ostores: Ks, D.Ostorel: Kl, D.Ostorew: Kw}.get(i.op, 0))
                continue
            if not ir.istmp(i.to):
                raise Bail('outer: instruction without a result')
            args = [a for a in i.arg if a != R]
            ks = [k(a) for a in args]
            if i.op == D.Ofma1:
                if third is None:
                    raise Bail('outer: fma')
                ks.append(k(third))
            if all(x == 'u' for x in ks):
                kind[rval(i.to)] = 'u'
            elif 'v' in ks:
                if 'a' in ks:
                    raise Bail('outer: an address in arithmetic')
                kind[rval(i.to)] = 'v'
                vcls.add(i.cls)
            else:  # affine: integer arithmetic of i with uniform constants
                c = None
                if i.cls == Kl and i.op in (D.Oadd, D.Osub):
                    ca = coef[rval(i.arg[0])] if ks[0] == 'a' else 0
                    cb = coef[rval(i.arg[1])] if ks[1] == 'a' else 0
                    c = ca + cb if i.op == D.Oadd else ca - cb
                    if 'a' not in ks:
                        c = None
                elif i.cls == Kl and i.op == D.Omul and ks.count('a') == 1:
                    m = ir.conval(fn, i.arg[1] if ks[0] == 'a' else i.arg[0])
                    if m is not None:
                        c = coef[rval(i.arg[0] if ks[0] == 'a' else i.arg[1])] * m
                elif i.cls == Kl and i.op == D.Oshl and ks[0] == 'a' and ir.conval(fn, i.arg[1]) is not None:
                    c = coef[rval(i.arg[0])] << ir.conval(fn, i.arg[1])
                if c is None:
                    raise Bail('outer: affine value')
                kind[rval(i.to)] = 'a'
                coef[rval(i.to)] = c
    return kind, coef, vcls


def _single_succ(b, target):
    return b.jmp.type == D.Jjmp and b.s1 is target


def transform(fn, O, rts, done):
    if not (_ext.vector and D.T.name.startswith(('arm64', 'amd64')) and M.vectorizing):
        return False
    if len(O.children) != 1 or O.children[0].children:
        return False
    I = O.children[0]
    Ho = O.header
    oblk = O.blocks
    inside = lambda r: ir.istmp(r) and fn.tmp[rval(r)].bid in {b.id for b in oblk}

    # ---- shape: Ho (i, test) -> Bpre -> inner loop -> Ej (latch) -> Ho
    pre = O.outside_preds()
    if len(pre) != 1 or pre[0].jmp.type != D.Jjmp or len(O.latches) != 1 or Ho.jmp.type != D.Jjnz:
        raise Bail('outer: shape')
    pre = pre[0]
    Bpre, Xo = (Ho.s1, Ho.s2) if Ho.s1 in oblk else (Ho.s2, Ho.s1)
    if Bpre not in oblk or Xo in oblk:
        raise Bail('outer: header')
    sh = Shape(fn, I)
    Ej = O.latches[0]
    if sh.pre is not Bpre or sh.E is not Ej or not _single_succ(Bpre, sh.H) or not _single_succ(Ej, Ho) \
            or Ej.npred != 1 or oblk != {Ho, Bpre, Ej} | I.blocks:
        raise Bail('outer: blocks')
    for b in oblk:
        for i in ir.ins(b):
            if not ir.copyable(i):
                raise Bail('outer: uncopyable')
            if ir.isstore(i.op) and b is not Ej:
                raise Bail('outer: store before or in the inner loop')
    for b in ir.blocks(fn):
        if b in oblk:
            continue
        refs = [a for p in ir.phis(b) for a in p.arg[:p.narg]] + [a for i in ir.ins(b) for a in i.arg] + [b.jmp.arg]
        if any(inside(r) for r in refs):
            raise Bail('outer: value used after the nest')

    # ---- the outer counter: the only phi of Ho, step 1, i < bound
    ophis = list(ir.phis(Ho))
    if len(ophis) != 1:
        raise Bail('outer: carried value')
    ip = ophis[0]
    if ip.cls != Kl:
        raise Bail('outer: counter not 64-bit')
    nxt = ir.phiargof(ip, Ej)
    test = None
    for b in oblk:
        for i in ir.ins(b):
            if i.to == nxt and not (i.op == D.Oadd and i.arg[0] == ip.to and ir.conval(fn, i.arg[1]) == 1):
                raise Bail('outer: step')
    for i in ir.ins(Ho):
        if i.to == Ho.jmp.arg:
            test = i
    if test is None or test.op not in (D.Ocsltl, D.Ocultl) or test.arg[0] != ip.to or inside(test.arg[1]) \
            or Ho.s1 is not Bpre or fn.tmp[rval(test.to)].nuse != 1:
        raise Bail('outer: test')
    bound = test.arg[1]
    i0 = ir.phiargof(ip, pre)
    if any(i.to != test.to for i in ir.ins(Ho)):
        raise Bail('outer: header instructions')

    # ---- classify: 'u' uniform, 'a' affine in i (coefficient), 'v' varying; inner phis varying (accumulators) when
    # their init or latch value is, found by iterating from all uniform
    iphis = list(ir.phis(sh.H))
    order = [Bpre] + sh.order + [Ej]
    varying = set()
    for _ in range(len(iphis) + 1):
        kind, coef, vcls = _classify(fn, ip, iphis, varying, order, inside, Bpre, sh)
        now = {rval(p.to) for p in iphis
               if rval(p.to) in varying or _kof(kind, ir.phiargof(p, sh.latch), inside) == 'v'
               or _kof(kind, ir.phiargof(p, Bpre), inside) in ('v', 'a')}
        if now == varying:
            break
        varying = now
    else:
        raise Bail('outer: classification')
    k = lambda a: _kof(kind, a, inside)
    for p in iphis:
        if rval(p.to) in varying:
            if _kof(kind, ir.phiargof(p, Bpre), inside) == 'a':
                raise Bail('outer: an address as an accumulator')
            if _kof(kind, ir.phiargof(p, sh.latch), inside) != 'v':
                raise Bail('outer: a uniform latch value for an accumulator')
            vcls.add(p.cls)
    if len(vcls) != 1:
        raise Bail('outer: lanes of one width')
    cls = vcls.pop()
    lane = _LANE[cls]
    vf = _ext.qbytes() // _SIZE[cls]
    # the inner loop's control is uniform; affine values are only addresses (and each other)
    if sh.test is not None and any(k(a) != 'u' for a in sh.test.arg if a != R):
        raise Bail('outer: inner trip count depends on i')
    for p in iphis:
        if kind[rval(p.to)] == 'u' and k(ir.phiargof(p, Bpre)) != 'u':
            raise Bail('outer: inner counter')
    for b in order:
        for i in ir.ins(b):
            for n, a in enumerate(i.arg):
                if a != R and k(a) == 'a' and not (ir.isload(i.op) and n == 0) and not (ir.isstore(i.op) and n == 1) \
                        and kind.get(rval(i.to), 'x') != 'a' and i.to != nxt:
                    raise Bail('outer: i used other than in addresses')
    # memory: stores alias no load
    loads = [i.arg[0] for b in order for i in ir.ins(b) if ir.isload(i.op)]
    stores = [i.arg[1] for i in ir.ins(Ej) if ir.isstore(i.op)]
    if not stores:
        raise Bail('outer: no stores')
    for s in stores:
        for a in loads + [x for x in stores if x is not s]:
            if rts.may_alias(s, a):
                raise Bail('outer: a store may alias a load')
    for b in order:
        for i in ir.ins(b):
            if kind.get(rval(i.to)) == 'v' and i.op not in (D.Ofma1,) and not ir.isload(i.op) \
                    and vectorize._opname(i) is None:
                raise Bail('outer: no vector form')

    # the inner loop is rotated below: its header must be the phis and the test alone
    if any(x is not sh.test for x in ir.ins(sh.H)) or sh.test is None or sh.B is None \
            or fn.tmp[rval(sh.test.to)].nuse != 1:
        raise Bail('outer: inner header')

    # ---- build
    #   VG: enough iterations for a vector?    VH: iv, pre(...), the inner loop's first test
    #   VP: what the inner loop computes the same each time (reached only when it runs)
    #   IB: the inner loop, bottom-tested      VE: post(...) with the inner phis' last values, iv + vf
    #   XR: into the scalar loop, from where the vectors stopped
    VG = ir.newblock(fn, pre, 'og')
    VH = ir.newblock(fn, VG, 'oh')
    VP = ir.newblock(fn, VH, 'op')
    IB = ir.newblock(fn, VP, 'oib')
    VE = ir.newblock(fn, IB, 'oe')
    XR = ir.newblock(fn, VE, 'ox')
    iv = newtmp('o', Kl, fn)
    evh, evp, eib, eve = Emitter(fn, VH), Emitter(fn, VP), Emitter(fn, IB), Emitter(fn, VE)
    env = {rval(ip.to): iv}
    jdep = set()    # new temporaries that change in the inner loop
    memo = {}

    def dup(x, em):
        if x in jdep:
            return em.raw(getattr(D, 'Oqdup' + lane), Kq, x)
        if ('dup', x) not in memo:
            memo[('dup', x)] = (evp if em in (evp, eib) else em).raw(getattr(D, 'Oqdup' + lane), Kq, x)
        return memo[('dup', x)]

    def val(a, em, want):
        if not ir.istmp(a) or not inside(a):
            return dup(a, em) if want == 'v' else a
        t = rval(a)
        x = env[t]
        return dup(x, em) if want == 'v' and kind[t] == 'u' else x

    def clone(b, em, inner):
        """b's instructions; in the inner loop (em is eib) what doesn't change there goes to VP instead."""
        third = None
        for i in ir.ins(b):
            if i.op in (D.Onop, D.Odbgloc) or i is sh.test:
                continue
            if i.op == D.Ofma0:
                third = i.arg[0]
                continue
            t = rval(i.to) if ir.istmp(i.to) else None
            srcs = [a for a in i.arg if a != R] + ([third] if i.op == D.Ofma1 else [])
            moving = inner and any(ir.istmp(a) and inside(a) and env.get(rval(a)) in jdep for a in srcs)
            e = (eib if moving else evp) if inner else em
            if ir.isstore(i.op):
                e.ins(Ins(D.Oqstore, Kq, R, val(i.arg[0], e, 'v'), val(i.arg[1], e, 'u')))
                continue
            kt = kind[t]
            if ir.isload(i.op) and kt == 'v':
                r = e.raw(D.Oqload, Kq, val(i.arg[0], e, 'u'))
            elif kt in ('u', 'a'):
                args = [val(a, e, 'u') if a != R else R for a in i.arg]
                if i.op == D.Ofma1:
                    e.ins(Ins(D.Ofma0, i.cls, R, val(third, e, 'u'), R))
                r = e.raw(i.op, i.cls, args[0], args[1])
            elif i.op == D.Ofma1:
                a0, a1, a2 = val(third, e, 'v'), val(i.arg[0], e, 'v'), val(i.arg[1], e, 'v')
                e.ins(Ins(D.Oqfma0, Kq, R, a0, R))
                r = e.raw(getattr(D, 'Oqfma1' + lane), Kq, a1, a2)
            else:
                op = getattr(D, 'O' + vectorize._opname(i))
                if i.arg[1] == R:
                    r = e.raw(op, Kq, val(i.arg[0], e, 'v'))
                else:
                    r = e.raw(op, Kq, val(i.arg[0], e, 'v'), val(i.arg[1], e, 'v'))
            env[t] = r
            if moving:
                jdep.add(r)

    def first_test(em, phival):
        """The inner loop's test with its phis replaced by phival(p)."""
        args = []
        for a in sh.test.arg[:2]:
            p = [q for q in iphis if q.to == a]
            args.append(phival(p[0]) if p else val(a, em, 'u'))
        return em.raw(sh.test.op, sh.test.cls, args[0], args[1])

    # VG: i0 < bound and bound - i0 >= vf (unsigned once i0 < bound)
    eg = Emitter(fn, VG)
    c0 = eg.raw(test.op, Kw, i0, bound)
    big = eg.raw(D.Ocugel, Kw, eg.sub(Kl, bound, i0), eg.con(vf))
    ok = eg.raw(D.Oand, Kw, c0, big)
    eg.flush()
    ir.setjnz(VG, ok, VH, XR)
    # VH: pre(...), inits, first test
    clone(Bpre, evh, False)
    inits = {}
    for p in iphis:
        t = rval(p.to)
        inits[t] = val(ir.phiargof(p, Bpre), evh, 'v' if kind[t] == 'v' else 'u')
    runs = first_test(evh, lambda p: inits[rval(p.to)])
    # IB: phis, body, the next test
    cur = {}
    for p in iphis:
        t = rval(p.to)
        cur[t] = newtmp('o', Kq if kind[t] == 'v' else p.cls, fn)
        env[t] = cur[t]
        jdep.add(cur[t])
    clone(sh.B, eib, True)
    nxtv = {}
    for p in iphis:
        t = rval(p.to)
        lv = ir.phiargof(p, sh.latch)
        nxtv[t] = val(lv, eib, 'v' if kind[t] == 'v' else 'u')
    again = first_test(eib, lambda p: nxtv[rval(p.to)])
    for p in iphis:
        t = rval(p.to)
        ir.addphi(IB, Kq if kind[t] == 'v' else p.cls, cur[t], [(VP, inits[t]), (IB, nxtv[t])])
    # VE: the inner phis' values on the way out, then post(...)
    for p in iphis:
        t = rval(p.to)
        out = newtmp('o', Kq if kind[t] == 'v' else p.cls, fn)
        ir.addphi(VE, Kq if kind[t] == 'v' else p.cls, out, [(VH, inits[t]), (IB, nxtv[t])])
        env[t] = out
    jdep.clear()
    clone(Ej, eve, False)
    iv2 = eve.addc(Kl, iv, vf)
    more = eve.raw(D.Ocugel, Kw, eve.sub(Kl, bound, iv2), eve.con(vf))  # iv2 never passes bound
    for e, b in ((evh, VH), (evp, VP), (eib, IB), (eve, VE)):
        e.flush()
    cont_t, cont_f = (IB, VE) if sh.cont_true else (VE, IB)
    ir.setjnz(VH, runs, VP, VE) if sh.cont_true else ir.setjnz(VH, runs, VE, VP)
    ir.setjmp(VP, IB)
    ir.setjnz(IB, again, cont_t, cont_f)
    ir.setjnz(VE, more, VH, XR)
    ir.addphi(VH, Kl, iv, [(VG, i0), (VE, iv2)])
    start = newtmp('o', Kl, fn)
    ir.addphi(XR, Kl, start, [(VG, i0), (VE, iv2)])
    ir.setjmp(XR, Ho)
    ir.retarget(pre, Ho, VG)
    ir.renamepred(Ho, pre, XR)
    ir.setphiarg(ip, XR, start)
    done.update([Ho, VH, IB])
    M.count('outer_vectorized')
    return True
