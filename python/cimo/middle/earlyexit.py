"""Vectorizing loops that leave early (LLVM's early-exit vectorization): a counted loop whose body loads, computes a
condition on what it loaded, and leaves when the condition holds.

    for (i = 0; i < n; i++) if (a[i] == v) return i;

A vector loop goes in front of it. Each iteration checks a vector's worth of iterations at once: it loads them, works
out the condition on every lane, and goes on only if no lane holds. Otherwise the original loop runs those iterations
one by one, from the first of them, so it leaves exactly where C would, with exactly C's values; at the end of them, if
it hasn't left, it goes back to the vector loop. Nothing after the loop changes: every value used there still comes from
the original loop.

    P:  limit = bound - (chunk - 1) * step, if that doesn't wrap: a chunk of COPIES vectors passes the loop's test
    VH: v = phi(start, VB: v + chunk, L: next)   v < limit (or <=): the whole chunk passes
    VC: each load's chunk in one 4 KB page        the scalar loop reads the first of them, so nothing can fault
    VB: lanes loaded, the condition per lane      none: v + chunk, back to VH; any: H, to the end of the chunk
    H:  the original loop, with phis for where to come back (stop) and whether to (flag)
    L:  its latch, which goes back to VH at stop

Reading past where C stops is what makes this a transformation and not a copy: the scalar loop would never have read
lanes after the one that leaves. A vector load inside the page of an element that C does read can't fault, whatever
the array's length; a chunk that would cross a page runs scalar. Pages are at least 4 KB on every target.

Taken: the header holds the loop's test of an induction variable against a bound the loop doesn't change, with `<`
or `<=` (signed or unsigned), stepping up by a constant, and arithmetic on induction variables; then a chain of up to
four blocks, each ending in a way out (`a || b` is two), of loads with consecutive addresses, in the first block only
(the one the scalar loop surely runs), operations with lane forms (cimo.middle.vectorize) and comparisons; a latch of
induction updates. No stores, no calls, no reductions yet.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ks, Kd, Kq, Ins, rval
from ..backend.util import newtmp
from ..backend import ext as _ext
from . import ir, vectorize as V
from .affine import Analysis
import cimo.middle as M

PAGE = 4096
COPIES = 4  # vectors a chunk: one test of all their lanes, one page check of all their bytes
_SIGNED = {D.Ocsltw: 'lt', D.Ocsltl: 'lt', D.Ocslew: 'le', D.Ocslel: 'le'}
_UNSIGNED = {D.Ocultw: 'lt', D.Ocultl: 'lt', D.Oculew: 'le', D.Oculel: 'le'}
_FLIP = {D.Ocsgtw: D.Ocsltw, D.Ocsgtl: D.Ocsltl, D.Ocsgew: D.Ocslew, D.Ocsgel: D.Ocslel,
         D.Ocugtw: D.Ocultw, D.Ocugtl: D.Ocultl, D.Ocugew: D.Oculew, D.Ocugel: D.Oculel}


class _No(Exception):
    pass


def _need(c):
    if not c:
        raise _No


def transform(fn, lp):
    """Put a vector loop in front of lp if it is a loop that leaves early (needs fillcfg, filluse, a preheader);
    returns True if it changed."""
    if lp.children or not (M.vectorizing and _ext.vector):
        return False
    try:
        plan = _plan(fn, lp)
    except _No:
        return False
    _emit(fn, lp, plan)
    M.count('earlyexit')
    return True


class _Plan:
    pass


def _plan(fn, lp):
    H = lp.header
    _need(len(lp.latches) == 1)
    L = lp.latches[0]
    _need(H.jmp.type == D.Jjnz and L.jmp.type == D.Jjmp and L.s1 is H and L.npred == 1 and L.phi is None)
    _need(H.s1 in lp.blocks and H.s2 not in lp.blocks)
    # the body: a chain of blocks, each leaving the loop or going on (a || b is two), then the latch
    chain = []
    exits = []  # (block, leave when its condition holds)
    b = H.s1
    while b is not L:
        _need(b in lp.blocks and b.npred == 1 and b.phi is None and b.jmp.type == D.Jjnz and ir.istmp(b.jmp.arg))
        if b.s1 not in lp.blocks and b.s2 in lp.blocks:
            exits.append((b, True))
            nxt = b.s2
        elif b.s2 not in lp.blocks and b.s1 in lp.blocks:
            exits.append((b, False))
            nxt = b.s1
        else:
            raise _No
        chain.append(b)
        b = nxt
        _need(len(chain) <= 4)
    _need(chain and len(lp.blocks) == len(chain) + 2)
    outs = lp.outside_preds()
    _need(len(outs) == 1 and outs[0].jmp.type == D.Jjmp)
    P = outs[0]
    A = Analysis(fn, lp, [H] + chain + [L])
    _need(not A.checks)
    # the induction variables: all of the header's phis, stepping up by constants
    ivs = []
    for p in ir.phis(H):
        t = rval(p.to)
        _need(t in A.biv and p.cls in (Kw, Kl))
        init, d = A.biv[t]
        _need(d.isconst() and d.c > 0)
        ivs.append((p, ir.phiargof(p, P), ir.phiargof(p, L), d.c))
    _need(ivs)
    phis = {rval(p.to) for p, _, _, _ in ivs}
    inv = lambda x: not ir.istmp(x) or not A.inloop(x)
    # the header: the test, and what addresses are made of
    scalar = set(phis)    # values computed from the induction variables and invariants
    test = None
    pre = []
    for i in ir.ins(H):
        if i.op in (D.Onop, D.Odbgloc):
            continue
        if i.to == H.jmp.arg:
            test = i
            continue
        _need(ir.istmp(i.to) and i.op in ir.PURE_SAFE and V._compare(i) is None)
        _need(all(inv(x) or rval(x) in scalar for x in i.arg[:2] if x != R))
        scalar.add(rval(i.to))
        pre.append(i)
    _need(test is not None)
    op, a, b = test.op, test.arg[0], test.arg[1]
    if op in _FLIP:
        op, a, b = _FLIP[op], b, a
    _need(op in _SIGNED or op in _UNSIGNED)
    ctl = next((iv for iv in ivs if iv[0].to == a), None)
    _need(ctl is not None and inv(b))
    # the latch: updates of the induction variables, from them
    lto = [j.to for j in ir.ins(L)]
    for i in ir.ins(L):
        _need(i.op in ir.PURE_SAFE and all(inv(x) or rval(x) in phis or x in lto for x in i.arg[:2]))
    vector = {}           # tmp -> ('mask' | class) of what has a lane form
    loads = []
    widths = set()
    body = list(pre)
    for n, blk in enumerate(chain):
        for i in ir.ins(blk):
            if i.op in (D.Onop, D.Odbgloc):
                continue
            body.append(i)
            _need(ir.istmp(i.to))
            args = [x for x in i.arg[:2] if x != R]
            if all(inv(x) or rval(x) in scalar for x in args) and i.op in ir.PURE_SAFE and V._compare(i) is None:
                scalar.add(rval(i.to))
                continue
            size = V._loadsize(i)
            if size is not None:
                # only where the scalar loop surely loads: the first block, which every iteration runs
                _need(n == 0)
                f = A.form(i.arg[0], 64)
                # consecutive (the step is per iteration): a vector's worth of iterations reads one vector
                _need(f is not None and f.S.isconst() and f.S.c == size and not f.chk and A.avlevel(f) != 'X')
                _need(inv(i.arg[0]) or rval(i.arg[0]) in scalar)
                loads.append(i)
                widths.add(size)
                vector[rval(i.to)] = i.cls
                continue
            _need(not any(ir.istmp(x) and rval(x) in scalar for x in args))  # lanes don't take iteration numbers
            c = V._compare(i)
            if c is not None:
                _need(all(inv(x) or vector.get(rval(x)) == c[3] for x in args))
                widths.add(V._SIZE[c[3]])
                vector[rval(i.to)] = 'mask'
                continue
            if i.op in (D.Oand, D.Oor, D.Oxor) and i.cls == Kw and \
                    all(ir.istmp(x) and vector.get(rval(x)) == 'mask' for x in args):
                vector[rval(i.to)] = 'mask'
                continue
            name = V._opname(i)
            _need(name is not None and i.op not in (D.Oshl, D.Oshr, D.Osar))
            _need(all(inv(x) or vector.get(rval(x)) == i.cls for x in args))
            widths.add(V._SIZE[i.cls])
            vector[rval(i.to)] = i.cls
        _need(vector.get(rval(blk.jmp.arg)) == 'mask')
    _need(loads and len(widths) == 1)
    pl = _Plan()
    pl.H, pl.L, pl.P = H, L, P
    pl.ivs, pl.ctl = ivs, ctl
    pl.test, pl.op, pl.bound = test, op, b
    pl.body, pl.scalar, pl.vector, pl.loads = body, scalar, vector, loads
    pl.exits = exits
    pl.vf = 16 // widths.pop()
    return pl


def _emit(fn, lp, pl):
    H, L, P = pl.H, pl.L, pl.P
    vf = pl.vf
    VH = ir.newblock(fn, P, 'eh')
    VC = ir.newblock(fn, VH, 'ec')
    VB = ir.newblock(fn, VC, 'eb')
    C = vf * COPIES
    # P: the last start of a whole chunk, once
    p0, _, _, step0 = pl.ctl
    k = p0.cls
    span = (C - 1) * step0
    ep = ir.Emitter(fn, P)
    limit = ep.raw(D.Osub, k, pl.bound, ep.con(span))
    if pl.op in _SIGNED:
        low = -(1 << (31 if k == Kw else 63))
        room = ep.raw(D.Ocsgew if k == Kw else D.Ocsgel, Kw, pl.bound, ep.con(low + span))
    else:
        room = ep.raw(D.Ocugew if k == Kw else D.Ocugel, Kw, pl.bound, ep.con(span))
    ep.flush()
    # VH: where the vector loop stands, and whether a chunk is left
    v = {}
    for p, init, nxt, step in pl.ivs:
        v[rval(p.to)] = newtmp('ee', p.cls, fn)
    eh = ir.Emitter(fn, VH)
    cur = v[rval(p0.to)]
    go = eh.raw(D.Oand, Kw, room, eh.raw(pl.op, Kw, cur, limit))
    nxts = {}
    for p, init, nxt, step in pl.ivs:
        nxts[rval(p.to)] = eh.raw(D.Oadd, p.cls, v[rval(p.to)], eh.con(C * step))
    stop = nxts[rval(p0.to)]
    eh.flush()
    ir.setjnz(VH, go, VC, H)
    # VC: the addresses, and none of the vectors crosses a page
    ec = ir.Emitter(fn, VC)
    env = dict((t, v[t]) for t in v)

    def s(x):
        return env.get(rval(x), x) if ir.istmp(x) else x
    for i in pl.body:
        if rval(i.to) in pl.scalar:
            env[rval(i.to)] = ec.raw(i.op, i.cls, s(i.arg[0]), s(i.arg[1]))
    bad = None
    for i in pl.loads:
        off = ec.raw(D.Oand, Kl, s(i.arg[0]), ec.con(PAGE - 1))
        b = ec.raw(D.Ocugtl, Kw, off, ec.con(PAGE - 16 * COPIES))
        bad = b if bad is None else ec.raw(D.Oor, Kw, bad, b)
    ec.flush()
    ir.setjnz(VC, bad, H, VB)
    # VB: every lane's condition
    eb = ir.Emitter(fn, VB)
    q = {}
    memo = {}

    def vec(x, cls):
        if ir.istmp(x) and rval(x) in q:
            return q[rval(x)]  # this copy's
        key = (x, cls)
        if key not in memo:
            memo[key] = eb.raw(getattr(D, 'Oqdup' + V._LANE[cls]), Kq, x)
        return memo[key]
    hits = None
    for copy in range(COPIES):
        q.clear()
        for i in pl.body:
            t = rval(i.to)
            if t in pl.scalar:
                continue
            if V._loadsize(i) is not None:
                addr = s(i.arg[0])
                if copy:
                    addr = eb.raw(D.Oadd, Kl, addr, eb.con(16 * copy))
                q[t] = eb.raw(D.Oqload, Kq, addr)
                continue
            c = V._compare(i)
            if c is not None:
                name, swap, negate, kk = c
                a, b = vec(i.arg[0], kk), vec(i.arg[1], kk)
                m = eb.raw(getattr(D, 'O' + name + V._LANE[kk]), Kq, *((b, a) if swap else (a, b)))
                q[t] = eb.raw(D.Oqnot, Kq, m) if negate else m
                continue
            if i.op in (D.Oand, D.Oor, D.Oxor) and pl.vector.get(t) == 'mask':
                q[t] = eb.raw({D.Oand: D.Oqand, D.Oor: D.Oqor, D.Oxor: D.Oqxor}[i.op], Kq, q[rval(i.arg[0])],
                              q[rval(i.arg[1])])
                continue
            op = getattr(D, 'O' + V._opname(i))
            if i.arg[1] == R:
                q[t] = eb.raw(op, Kq, vec(i.arg[0], i.cls))
            else:
                q[t] = eb.raw(op, Kq, vec(i.arg[0], i.cls), vec(i.arg[1], i.cls))
        for blk, when in pl.exits:
            m = q[rval(blk.jmp.arg)]
            if not when:
                m = eb.raw(D.Oqnot, Kq, m)
            hits = m if hits is None else eb.raw(D.Oqor, Kq, hits, m)
    m = hits
    anylane = eb.raw(D.Oor, Kl, eb.raw(D.Oqgetl, Kl, m, eb.con(0)), eb.raw(D.Oqgetl, Kl, m, eb.con(1)))
    hit = eb.raw(D.Ocnel, Kw, anylane, eb.con(0))
    eb.flush()
    ir.setjnz(VB, hit, H, VH)
    # VH's phis: from the preheader, the vector loop, and the scalar loop's way back
    for p, init, nxt, step in pl.ivs:
        t = rval(p.to)
        ir.addphi(VH, p.cls, v[t], [(P, init), (VB, nxts[t]), (L, nxt)])
    ir.setjmp(P, VH)
    # H: enters from the vector loop, at v; the scalar loop goes back to VH at stop if it came from VB or VC
    zero = ir.getcon(0, fn)
    one = ir.getcon(1, fn)
    for p, init, nxt, step in pl.ivs:
        t = rval(p.to)
        pairs = [(x, a) for x, a in zip(p.blk[:p.narg], p.arg[:p.narg]) if x is not P]
        pairs += [(VH, v[t]), (VC, v[t]), (VB, v[t])]
        p.blk, p.arg, p.narg = [x for x, _ in pairs], [a for _, a in pairs], len(pairs)
    flag = newtmp('ee', Kw, fn)
    stopv = newtmp('ee', k, fn)
    ir.addphi(H, Kw, flag, [(VH, zero), (VC, one), (VB, one), (L, flag)])
    ir.addphi(H, k, stopv, [(VH, zero), (VC, stop), (VB, stop), (L, stopv)])
    el = ir.Emitter(fn, L)
    at = el.raw(D.Oceqw if k == Kw else D.Oceql, Kw, pl.ctl[2], stopv)
    back = el.raw(D.Oand, Kw, at, flag)
    el.flush()
    ir.setjnz(L, back, VH, H)
