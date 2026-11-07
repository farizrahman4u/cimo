"""cimo (ext.x86 fma): no fused multiply-add whose addend is on a loop's recurrence, as gcc's avoid_fma_chains tuning
for Zen and recent Intel cores, by the latency of the cycle. Through the addend an FMA costs its 4 cycles where an add
alone was 2 on Sapphire Rapids and 3 on Zen 4: `dot` ran 2.0x and 1.33x slower contracted, `iir` 1.34x and 1.22x, and
the same on Ice Lake, where both are 4. Through a factor it saves the add, so those stay fused (`mandelbrot`'s
2*zr*zi + ci). A product left out is a multiply and an add again, which rounds as C does uncontracted. Only products contracted from
`a*b + c` (IR `fmac`, Ins.contract); an `fma()` call is fused always, as C requires.
"""

from .. import defs as D
from ..defs import R, RTmp, Ins, rtype, rval, Ks, Kd


def _phis(b):
    p = b.phi
    while p:
        yield p
        p = p.link


def _tmps(refs):
    return [rval(r) for r in refs if rtype(r) == RTmp]


def split(fn):
    """Needs rpo; returns whether anything changed."""
    blocks = fn.rpo[:fn.nblk]
    ins = [(b, k, i) for b in blocks for k, i in enumerate(b.ins[:b.nins])]
    defs = {}
    for b, k, i in ins:
        if rtype(i.to) == RTmp:
            defs[rval(i.to)] = i
    phis = [p for b in blocks for p in _phis(b)]
    pdef = {rval(p.to): p for p in phis}

    def args(t):
        if t in pdef:
            p = pdef[t]
            return _tmps(p.arg[:p.narg])
        i = defs.get(t)
        if i is None or D.isload(i.op) or i.op == D.Ocall:
            return []
        return _tmps(i.arg[:2])

    fused = {}
    keep = set()
    for b, k, i in ins:
        if i.op == D.Ofma1 and k and b.ins[k - 1].op == D.Ofma0:
            fused[rval(i.to)] = (b, k)
            if not i.contract:  # an fma() call: always fused, but still an operand on the cycle
                keep.add(rval(i.to))
            # the addend is an operand too, for the closures below
    third = {rval(b.ins[k].to): b.ins[k - 1].arg[0] for t, (b, k) in fused.items()}

    def allargs(t):
        a = args(t)
        if t in third and rtype(third[t]) == RTmp:
            a = a + [rval(third[t])]
        return a

    users = {}
    for t in list(defs) + list(pdef):
        for a in allargs(t):
            users.setdefault(a, []).append(t)
    split_ = set()
    for h in blocks:
        top = [p for p in _phis(h) if p.cls in (Ks, Kd)]
        if not top or not any(h.pred[n].id >= h.id for n in range(len(h.pred))):
            continue
        down = set(rval(p.to) for p in top)  # what depends on the recurrences
        work = list(down)
        while work:
            for u in users.get(work.pop(), ()):
                if u not in down:
                    down.add(u)
                    work.append(u)
        up = set()  # what the values going round depend on
        work = [rval(p.arg[n]) for p in top for n in range(p.narg)
                if p.blk[n].id >= h.id and rtype(p.arg[n]) == RTmp]
        while work:
            t = work.pop()
            if t in up:
                continue
            up.add(t)
            work += allargs(t)
        cycle = down & up
        for t in cycle:
            c = third.get(t)
            if c is not None and rtype(c) == RTmp and rval(c) in cycle and t not in keep:
                split_.add(t)
    def negated(r, k):
        """x if r is neg x of class k, else None."""
        i = defs.get(rval(r)) if rtype(r) == RTmp else None
        return i.arg[0] if i is not None and i.op == D.Oneg and i.cls == k else None

    for t in split_:
        b, k = fused[t]
        i, i0 = b.ins[k], b.ins[k - 1]
        x, y, c = i.arg[0], i.arg[1], i0.arg[0]
        m = _newtmp(fn, i.cls)
        nc = negated(c, i.cls)
        nx = negated(x, i.cls)
        if nc is not None:  # a*b - c as the frontend wrote it, fma(a, b, -c): a sub again, no neg on the path
            b.ins[k - 1] = Ins(D.Omul, i.cls, m, x, y)
            b.ins[k] = Ins(D.Osub, i.cls, i.to, m, nc)
        elif nx is not None:  # c - a*b, fma(-a, b, c)
            b.ins[k - 1] = Ins(D.Omul, i.cls, m, nx, y)
            b.ins[k] = Ins(D.Osub, i.cls, i.to, c, m)
        else:
            b.ins[k - 1] = Ins(D.Omul, i.cls, m, x, y)
            b.ins[k] = Ins(D.Oadd, i.cls, i.to, m, c)
    return bool(split_)


def _newtmp(fn, k):
    from ..util import newtmp
    return newtmp('fma', k, fn)
