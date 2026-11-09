"""Address arithmetic in the form PTX addressing wants, before the cleanup after the loop passes: a constant moved out
of a scaled sum, mul(add(y, c), k) -> add(mul(y, k), c * k) (and shl alike), exact in modular arithmetic. Unrolled
loops index arrays as base + (i + c) * size; after this, gvn makes every such address one register plus a constant,
which the emitter puts in the instruction ([r+12]) and the driver's compiler merges into wide loads."""

from ....backend import defs as D
from ....backend.defs import Ins, RTmp, RCon, CBits, Kw, Kl, rtype, rval, s32, s64
from ....backend.util import newtmp, getcon


def _const(fn, r):
    if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
        return fn.con[rval(r)].i
    return None


def run(fn):
    """Returns True if it changed fn. Needs the definitions (filluse)."""
    changed = False
    b = fn.start
    while b:
        out = []
        for i in b.ins[:b.nins]:
            if i.op in (D.Omul, D.Oshl) and i.cls in (Kw, Kl) and rtype(i.arg[0]) == RTmp:
                k = _const(fn, i.arg[1])
                d = fn.tmp[rval(i.arg[0])].def_
                c = _const(fn, d.arg[1]) if d is not None and d.op == D.Oadd and d.cls == i.cls else None
                if k is not None and c is not None and (i.op == D.Omul or 0 <= k < (32 if i.cls == Kw else 64)):
                    scaled = c * k if i.op == D.Omul else c << k
                    scaled = s32(scaled & 0xffffffff) if i.cls == Kw else s64(scaled & 0xffffffffffffffff)
                    u = newtmp('cn', i.cls, fn)
                    out.append(Ins(i.op, i.cls, u, d.arg[0], i.arg[1]))
                    i.op, i.arg[0], i.arg[1] = D.Oadd, u, getcon(scaled, fn)
                    changed = True
            out.append(i)
        if len(out) != b.nins:
            b.ins = out
            b.nins = len(out)
        b = b.link
    return changed


def reassoc(fn):
    """Sums regrouped for loops: in a chain of adds (of one class, each intermediate used once, in one block), the
    terms defined outside the loop are added first, then the others, then the constants. gcm then hoists the
    invariant part as one value, so an address in a loop keeps one register live where the chain kept each term (and
    the constant stays last, for the instruction's offset). Needs filluse; returns True if it changed fn."""
    from ....backend.cfg import fillloop
    fillloop(fn)
    changed = False
    for b in fn.rpo[:fn.nblk]:
        out = []
        for i in b.ins[:b.nins]:
            if i.op == D.Oadd and i.cls in (Kw, Kl) and rtype(i.to) == RTmp:
                leaves, inner = [], 0
                stack = [i.arg[1], i.arg[0]]
                while stack:
                    r = stack.pop()
                    t = fn.tmp[rval(r)] if rtype(r) == RTmp else None
                    d = t.def_ if t is not None else None
                    if d is not None and d.op == D.Oadd and d.cls == i.cls and t.nuse == 1 and t.bid == b.id \
                            and len(leaves) + len(stack) < 12:
                        stack += [d.arg[1], d.arg[0]]
                        inner += 1
                        continue
                    leaves.append(r)
                if inner:
                    cons = [r for r in leaves if rtype(r) != RTmp]
                    tmps = [r for r in leaves if rtype(r) == RTmp]
                    inv = [r for r in tmps if fn.rpo[fn.tmp[rval(r)].bid].loop < b.loop]
                    var = [r for r in tmps if fn.rpo[fn.tmp[rval(r)].bid].loop >= b.loop]
                    if len(inv) >= 2 and var or (cons and leaves[-1] not in cons):
                        terms = inv + var + cons
                        acc = terms[0]
                        for r in terms[1:-1]:
                            u = newtmp('ra', i.cls, fn)
                            out.append(Ins(D.Oadd, i.cls, u, acc, r))
                            acc = u
                        i.arg[0], i.arg[1] = acc, terms[-1]
                        changed = True
            out.append(i)
        if len(out) != b.nins:
            b.ins = out
            b.nins = len(out)
    return changed


# what gpuid N can be (cimo/gpu.h): thread index, block size, block index, grid size, lane, warp, SM
_GPUID = {0: (0, 1023), 1: (0, 1023), 2: (0, 63), 4: (1, 1024), 5: (1, 1024), 6: (1, 64),
          8: (0, 0x7fffffff), 9: (0, 65535), 10: (0, 65535), 12: (1, 0x7fffffff), 13: (1, 65535), 14: (1, 65535),
          16: (0, 31), 17: (0, 63), 18: (0, 1023)}


def ranges(fn):
    """{temporary: (lo, hi)} for integer values computed from thread ids and constants, where known to fit."""
    rg = {}

    def of(r):
        if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
            v = fn.con[rval(r)].i
            return (v, v)
        if rtype(r) == RTmp:
            return rg.get(rval(r))
        return None

    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if rtype(i.to) != RTmp or i.cls not in (Kw, Kl):
                continue
            o, x, y = i.op, of(i.arg[0]), of(i.arg[1])
            r = None
            if o == D.Ogpuid and x is not None:
                r = _GPUID.get(x[0])
            elif x is None:
                pass
            elif o in (D.Oextsw, D.Ocopy):
                r = x
            elif o == D.Oextuw and x[0] >= 0:
                r = x
            elif y is None:
                pass
            elif o == D.Oadd:
                r = (x[0] + y[0], x[1] + y[1])
            elif o == D.Osub:
                r = (x[0] - y[1], x[1] - y[0])
            elif o == D.Omul:
                ps = (x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1])
                r = (min(ps), max(ps))
            elif o == D.Oshl and y[0] == y[1] and 0 <= y[0] < 31:
                r = (x[0] << y[0], x[1] << y[0])
            elif o in (D.Oshr, D.Osar) and y[0] == y[1] and 0 <= y[0] < 31 and x[0] >= 0:
                r = (x[0] >> y[0], x[1] >> y[0])
            elif o == D.Oand and y[0] == y[1] and y[0] >= 0:
                r = (0, y[0]) if x[0] < 0 else (0, min(x[1], y[0]))
            elif o in (D.Odiv, D.Oudiv) and y[0] == y[1] and y[0] > 0 and x[0] >= 0:
                r = (x[0] // y[0], x[1] // y[0])
            elif o in (D.Orem, D.Ourem) and y[0] == y[1] and y[0] > 0 and x[0] >= 0:
                r = (0, min(x[1], y[0] - 1))
            if r is not None and -0x80000000 <= r[0] and r[1] <= 0x7fffffff:
                rg[rval(i.to)] = r
    return rg


def divisions(fn):
    """Signed division and remainder by a positive constant d of a value known not to be negative: (y + c*d) / d is
    y / d + c, (y + c*d) % d is y % d, and by a power of two a shift or a mask (which the driver can't do without
    knowing the sign). Needs filluse; returns True if it changed fn."""
    rg = ranges(fn)
    changed = False
    for b in fn.rpo[:fn.nblk]:
        out = []
        for i in b.ins[:b.nins]:
            if i.op in (D.Odiv, D.Orem) and i.cls in (Kw, Kl) and rtype(i.arg[0]) == RTmp:
                d = _const(fn, i.arg[1])
                x = rg.get(rval(i.arg[0]))
                if d is not None and d > 0 and x is not None and x[0] >= 0:
                    a = fn.tmp[rval(i.arg[0])].def_
                    c = _const(fn, a.arg[1]) if a is not None and a.op == D.Oadd else None
                    y = rg.get(rval(a.arg[0])) if c is not None and rtype(a.arg[0]) == RTmp else None
                    if c is not None and c % d == 0 and y is not None and y[0] >= 0:
                        if i.op == D.Orem:
                            i.arg[0] = a.arg[0]
                        else:
                            u = newtmp('dv', i.cls, fn)
                            out.append(Ins(D.Odiv, i.cls, u, a.arg[0], i.arg[1]))
                            i.op, i.arg[0], i.arg[1] = D.Oadd, u, getcon(c // d, fn)
                        changed = True
                    elif d & (d - 1) == 0:
                        if i.op == D.Odiv:
                            i.op, i.arg[1] = D.Oshr, getcon(d.bit_length() - 1, fn)
                        else:
                            i.op, i.arg[1] = D.Oand, getcon(d - 1, fn)
                        changed = True
            out.append(i)
        if len(out) != b.nins:
            b.ins = out
            b.nins = len(out)
    return changed
