"""Unroll-and-jam by 2 of a loop around an innermost loop (gemm's k loop).

    for (k...) { pre(k); for (j...) body(k, j); post(k); }
becomes
    while (test(k) && test(k')) {             // k' = k's next value, same arithmetic
        pre(k); pre(k');
        for (j...) { body(k, j); body(k', j); }
        post(k); post(k');
        k = k'';
    }
    original loop, from k                     // the 0 or 1 iterations left, and the exit

Each fused iteration runs body(k, j) before body(k', j), as C does, so floating-point
results are unchanged; what moves is body(k', j) before body(k, j') for j' > j, pre(k')
before the inner loop of k, and post(k) after the inner loop of k'. That is legal when no
store involved may touch what the other access touches:
  * the alias analysis says the two addresses don't alias, or
  * both are the same affine function of j alone (base and offset not depending on k) with
    a stride at least the access size, so different j never overlap.
The point is the second case: c[i*n + j] is then loaded and stored once per two updates,
after the inner loop pass forwards the store of body(k, j) to the load of body(k', j) and
deletes the first store.

Shape required: outer loop {Ho, Bpre, inner loop, Ej} where Ho (pure instructions, the exit
test) -> Bpre -> inner header; the inner loop exits to Ej, which jumps back to Ho. Outer
phis must be induction variables; inner phis induction variables with inits from outside the
outer loop (so both copies share them); no inner value is used after the inner loop.
"""

from ..backend import defs as D
from ..backend.defs import R, RTmp, Kw, Kl, Ins, rtype, rval
from ..backend.util import newtmp
from . import ir
from .ir import Bail, Emitter
from .affine import Analysis
from .inner import Shape, Control
import cimo.middle as M

MAXINNER = 40


def _clone(em, i, env):
    args = [env.get(rval(a), a) if ir.istmp(a) else a for a in i.arg]
    if not ir.istmp(i.to):
        em.ins(Ins(i.op, i.cls, R, args[0], args[1]))
        return
    v = em.raw(i.op, i.cls, args[0], args[1])
    env[rval(i.to)] = v


def transform(fn, O, rts, done):
    if len(O.children) != 1 or O.children[0].children:
        raise Bail('not a two-deep nest')
    I = O.children[0]
    if len(O.latches) != 1:
        raise Bail('outer latches')
    Ho = O.header
    if Ho.jmp.type != D.Jjnz:
        raise Bail('outer header')
    Bpre = Ho.s1 if Ho.s1 in O.blocks else Ho.s2
    Eo = Ho.s2 if Bpre is Ho.s1 else Ho.s1
    if Eo in O.blocks or Bpre in I.blocks:
        raise Bail('outer exit shape')
    shi = Shape(fn, I)
    if shi.pre is not Bpre or Bpre.npred != 1 or Bpre.phi:
        raise Bail('inner preheader')
    Ej = shi.E
    if Ej not in O.blocks or Ej.npred != 1 or Ej.jmp.type != D.Jjmp or Ej.s1 is not Ho \
            or Ej.phi or O.latches[0] is not Ej:
        raise Bail('outer latch shape')
    if O.blocks != {Ho, Bpre, Ej} | I.blocks:
        raise Bail('outer body shape')
    for b in (Ho, Bpre, Ej):
        for i in ir.ins(b):
            if not ir.copyable(i):
                raise Bail('uncopyable')
    for i in ir.ins(Ho):
        if ir.isload(i.op) or ir.isstore(i.op):
            raise Bail('memory access in the outer header')
    innerins = [i for b in shi.order for i in ir.ins(b)]
    if shi.B is not None and any(ir.isload(i.op) or ir.isstore(i.op) for i in ir.ins(shi.H)):
        raise Bail('memory access in the inner header')
    if len(innerins) > MAXINNER:
        raise Bail('inner body too big')

    # outer: induction variables only, and a normal exit test
    Ao = Analysis(fn, O, [Ho, Bpre] + shi.order + [Ej])
    for p in ir.phis(Ho):
        if rval(p.to) not in Ao.biv:
            raise Bail('outer carried value')

    class _Sh:
        pass
    sho = _Sh()
    sho.H, sho.test, sho.cont_true, sho.cond = Ho, None, Ho.s1 is Bpre, Ho.jmp.arg
    for i in ir.ins(Ho):
        if i.to == Ho.jmp.arg:
            from ..backend.util import iscmp
            if iscmp(i.op) is not None:
                sho.test = i
    Ao.fn_con0 = ir.getcon(0, fn)
    Control(Ao, sho)    # raises Bail if not a counted loop

    # inner: induction variables with outer-invariant inits, outer-invariant trip count
    Ai = Analysis(fn, I, shi.order)
    Ai.fn_con0 = Ao.fn_con0
    ctli = Control(Ai, shi)
    oblocks = set(b.id for b in O.blocks)

    def outer_inv_lin(L):
        return all(not (ir.istmp(k[0]) and fn.tmp[rval(k[0])].bid in oblocks) for k in L.t)

    for p in ir.phis(shi.H):
        t = rval(p.to)
        if t not in Ai.biv:
            raise Bail('inner carried value')
        init = ir.phiargof(p, Bpre)
        if ir.istmp(init) and fn.tmp[rval(init)].bid in oblocks:
            raise Bail('inner init depends on the outer loop')
        if not outer_inv_lin(Ai.biv[t][1]):
            raise Bail('inner step depends on the outer loop')
    if not (outer_inv_lin(ctli.x.L) and outer_inv_lin(ctli.b.L)):
        raise Bail('inner trip count depends on the outer loop')
    iblocks = set(b.id for b in I.blocks)
    for b in ir.blocks(fn):
        if b in I.blocks:
            continue
        refs = [a for p in ir.phis(b) for a in p.arg[:p.narg]]
        refs += [a for i in ir.ins(b) for a in i.arg] + [b.jmp.arg]
        for r in refs:
            if ir.istmp(r) and fn.tmp[rval(r)].bid in iblocks:
                raise Bail('inner value used after the inner loop')

    # dependences
    mem_in = [i for i in innerins if ir.isload(i.op) or ir.isstore(i.op)]
    mem_out = [i for b in (Bpre, Ej) for i in ir.ins(b) if ir.isload(i.op) or ir.isstore(i.op)]
    if any(ir.isstore(i.op) for i in mem_out):
        raise Bail('store outside the inner loop')

    def addr(i):
        return i.arg[1] if ir.isstore(i.op) else i.arg[0]

    def size(i):
        if ir.isstore(i.op):
            return ir.access_size(i.op)
        from .inner import _loadinfo
        li = _loadinfo(i.op, i.cls)
        return li[0] if li else 8

    def jinvariant_same(x, y):
        fx, fy = Ai.form(addr(x), 64), Ai.form(addr(y), 64)
        if fx is None or fy is None or not fx.S.isconst() or fx.inv():
            return False
        if fx.L.ident() != fy.L.ident() or fx.S.ident() != fy.S.ident():
            return False
        if not (outer_inv_lin(fx.L) and outer_inv_lin(fx.S)):
            return False
        return abs(fx.S.c) >= max(size(x), size(y))

    reuse = False
    for x in mem_in:
        for y in mem_in:
            if not (ir.isstore(x.op) or ir.isstore(y.op)):
                continue
            if not rts.may_alias(addr(x), addr(y)):
                continue
            if jinvariant_same(x, y):
                reuse = True
                continue
            raise Bail('dependence between outer iterations')
    for x in mem_out:
        for y in mem_in:
            if ir.isstore(y.op) and rts.may_alias(addr(x), addr(y)):
                raise Bail('dependence with the outer body')
    if not reuse:
        raise Bail('nothing to reuse')

    JH = _emit(fn, O, I, Ho, Bpre, Ej, Eo, shi, Ao)
    done.update([Ho, JH])
    M.count('jam')
    return True


def _emit(fn, O, I, Ho, Bpre, Ej, Eo, shi, Ao):
    """Build the jammed loop in front of the original one."""
    pre = [p for p in Ho.pred[:Ho.npred] if p not in O.blocks]
    if len(pre) != 1 or pre[0].jmp.type != D.Jjmp:
        raise Bail('outer preheader')
    pre = pre[0]
    JH = ir.newblock(fn, pre, 'jh')
    J2 = ir.newblock(fn, JH, 'j2')
    JP = ir.newblock(fn, J2, 'jp')       # pre(k), pre(k')
    FH = ir.newblock(fn, JP, 'jih')      # fused inner header
    FB = ir.newblock(fn, FH, 'jib')      # fused inner body
    JE = ir.newblock(fn, FB, 'je')       # post(k), post(k'), latch
    SL = ir.newblock(fn, JE, 'jslow')    # into the original loop
    latch_o = O.latches[0]

    # outer phis of the jammed loop
    ophis = list(ir.phis(Ho))
    cur = {}
    for p in ophis:
        cur[rval(p.to)] = newtmp('m', p.cls, fn)
    # JH: P_o(k), test
    e = Emitter(fn, JH)
    env0 = dict(cur)
    for i in ir.ins(Ho):
        _clone(e, i, env0)
    e.flush()
    c1 = env0.get(rval(Ho.jmp.arg), Ho.jmp.arg)
    cont_true = Ho.s1 not in (Eo,)
    # next values k' = latch args, computed from k with the latch's own instructions
    e2 = Emitter(fn, J2)
    nxt = _next(fn, Ao, ophis, latch_o, env0, e2)
    env1 = dict(nxt)
    for i in ir.ins(Ho):
        _clone(e2, i, env1)
    e2.flush()
    c2 = env1.get(rval(Ho.jmp.arg), Ho.jmp.arg)
    if cont_true:
        ir.setjnz(JH, c1, J2, SL)
        ir.setjnz(J2, c2, JP, SL)
    else:
        ir.setjnz(JH, c1, SL, J2)
        ir.setjnz(J2, c2, SL, JP)
    # JP: pre(k), pre(k')
    ep = Emitter(fn, JP)
    for env in (env0, env1):
        for i in ir.ins(Bpre):
            _clone(ep, i, env)
    ep.flush()
    ir.setjmp(JP, FH)
    # fused inner loop: shared inner phis, header and body of both copies
    iphis = list(ir.phis(shi.H))
    ilatch = shi.latch
    shared = {}
    for p in iphis:
        shared[rval(p.to)] = newtmp('m', p.cls, fn)
    envs = []
    eh = Emitter(fn, FH)
    for env in (env0, env1):
        en = dict(env)
        en.update(shared)
        for i in ir.ins(shi.H):
            _clone(eh, i, en)
        envs.append(en)
    eh.flush()
    ci = envs[0].get(rval(shi.H.jmp.arg), shi.H.jmp.arg)
    if shi.B is None:
        body = FH
        eb = eh
    else:
        body = FB
        eb = Emitter(fn, FB)
        for en in envs:
            for i in ir.ins(shi.B):
                _clone(eb, i, en)
        eb.flush()
    if shi.B is not None:
        ir.setjmp(FB, FH)
        if shi.cont_true:
            ir.setjnz(FH, ci, FB, JE)
        else:
            ir.setjnz(FH, ci, JE, FB)
    else:
        if shi.cont_true:
            ir.setjnz(FH, ci, FH, JE)
        else:
            ir.setjnz(FH, ci, JE, FH)
    for p in iphis:
        t = rval(p.to)
        init = ir.phiargof(p, Bpre)
        nx = ir.phiargof(p, ilatch)
        nxv = envs[0].get(rval(nx), nx) if ir.istmp(nx) else nx
        ir.addphi(FH, p.cls, shared[t], [(JP, init), (body, nxv)])
    # JE: post(k), post(k'), then k'' = next(k')
    ee = Emitter(fn, JE)
    for env in (env0, env1):
        for i in ir.ins(Ej):
            _clone(ee, i, env)
    nn = _next(fn, Ao, ophis, latch_o, env1, ee)
    ee.flush()
    ir.setjmp(JE, JH)
    for p in ophis:
        t = rval(p.to)
        ir.addphi(JH, p.cls, cur[t], [(pre, ir.phiargof(p, pre)), (JE, nn[t])])
    # into the original loop with the current values
    ir.setjmp(SL, Ho)
    for p in ophis:
        t = rval(p.to)
        v = newtmp('m', p.cls, fn)
        ir.addphi(SL, p.cls, v, [(JH, cur[t]), (J2, cur[t])])
        ir.setphiarg(p, pre, v)
    ir.renamepred(Ho, pre, SL)
    ir.retarget(pre, Ho, JH)
    return JH


def _next(fn, Ao, ophis, latch, env, em):
    """Values of the outer phis in the next iteration, from their affine forms (they are
    induction variables: next = cur + step, in the phi's width)."""
    out = {}
    for p in ophis:
        t = rval(p.to)
        init, step = Ao.biv[t]
        W = 32 if p.cls == Kw else 64
        k = Kw if W == 32 else Kl
        v = env[t]
        if step.isconst():
            out[t] = em.addc(k, v, step.c)
        else:
            acc = v
            for key, coef in sorted(step.t.items()):
                r, kind = key
                if kind == 's':
                    lv = em.op(D.Oextsw, Kl, r)
                elif kind == 'u':
                    lv = em.op(D.Oextuw, Kl, r)
                else:
                    lv = r
                acc = em.add(k, acc, em.mulc(k, lv, coef))
            out[t] = em.addc(k, acc, step.c)
    return out
