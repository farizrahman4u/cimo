"""cimo (ext.fma): a product C computes more than once is a multiply, not part of each fused multiply-add.

Contraction is per expression (the frontend's), so `zr*zr + zi*zi < 4` and `zr*zr - zi*zi + cr` each fuse zr*zr into
an fma; after promotion they multiply the same temporaries. As gcc, which contracts a multiplication only into its
single use: the product once, and plain adds. mandelbrot's zr*zr - zi*zi was an fnmsub whose addend came from the
last iteration's zi, 4 cycles on the loop's cycle where gcc had a 2-cycle subtraction (1.14 to 1.21x of gcc on
Graviton). A product used once stays fused, and code outside loops is contracted as written.
"""

from . import defs as D
from .defs import R, RTmp, Ins, rtype, rval, Ks, Kd
from .util import newtmp
from . import cfg


def _key(i):
    a, b = i.arg[0], i.arg[1]
    return (i.cls,) + tuple(sorted((a, b)))


def split(fn):
    """Needs nothing; returns whether anything changed (uses and the CFG's dominators are left to be refilled)."""
    cfg.fillrpo(fn)
    cfg.fillloop(fn)
    blocks = fn.rpo[:fn.nblk]
    count = {}
    for b in blocks:
        if b.loop <= 1:  # in loops only: elsewhere a*b + c is fused as written, like clang's -ffp-contract=on
            continue
        ins = b.ins[:b.nins]
        for n, i in enumerate(ins):
            if i.op == D.Ofma1 and n and ins[n - 1].op == D.Ofma0 and i.cls in (Ks, Kd):
                count[_key(i)] = count.get(_key(i), 0) + 1
            elif i.op == D.Omul and i.cls in (Ks, Kd):
                count[_key(i)] = count.get(_key(i), 0) + 1
    if not any(v > 1 for v in count.values()):
        return False
    cfg.filldom(fn)
    defs, uses = {}, {}
    for b in blocks:
        for i in b.ins[:b.nins]:
            if rtype(i.to) == RTmp:
                defs[rval(i.to)] = i
            for a in i.arg:
                if rtype(a) == RTmp:
                    uses[rval(a)] = uses.get(rval(a), 0) + 1
        p = b.phi
        while p:
            for a in p.arg[:p.narg]:
                if rtype(a) == RTmp:
                    uses[rval(a)] = uses.get(rval(a), 0) + 1
            p = p.link
        if rtype(b.jmp.arg) == RTmp:
            uses[rval(b.jmp.arg)] = uses.get(rval(b.jmp.arg), 0) + 1
    made = {}  # key -> [(block, temporary)] products made so far
    changed = False

    def product(b, key, a0, a1, cls, out):
        for blk, t in made.get(key, ()):
            if cfg.dom(blk, b):
                return t
        t = newtmp('fs', cls, fn)
        out.append(Ins(D.Omul, cls, t, a0, a1))
        made.setdefault(key, []).append((b, t))
        return t

    for b in blocks:  # reverse postorder: a block's dominators come first
        if b.loop <= 1:
            continue
        ins = b.ins[:b.nins]
        out = []
        n = 0
        while n < len(ins):
            i = ins[n]
            if i.op == D.Ofma0 and n + 1 < len(ins) and ins[n + 1].op == D.Ofma1 and count.get(_key(ins[n + 1]), 0) > 1:
                f = ins[n + 1]
                p = product(b, _key(f), f.arg[0], f.arg[1], f.cls, out)
                c = i.arg[0]
                d = defs.get(rval(c)) if rtype(c) == RTmp else None
                if d is not None and d.op == D.Oneg and d.cls == f.cls:  # p + -x: p - x, as C wrote it
                    out.append(Ins(D.Osub, f.cls, f.to, p, d.arg[0]))
                    if uses.get(rval(c)) == 1:
                        d.op, d.arg[0], d.to = D.Onop, R, R
                else:
                    out.append(Ins(D.Oadd, f.cls, f.to, p, c))
                n += 2
                changed = True
                continue
            if i.op == D.Omul and i.cls in (Ks, Kd) and count.get(_key(i), 0) > 1 and rtype(i.to) == RTmp:
                made.setdefault(_key(i), []).append((b, i.to))
            out.append(i)
            n += 1
        b.ins = out
        b.nins = len(out)
    return changed
