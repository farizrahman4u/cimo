"""If-conversion of small integer diamonds and triangles inside innermost loops.

    if (c) x = a; else x = b;        ->   m = -c;  x = b ^ ((a ^ b) & m)

The arms must hold only a few pure, non-trapping instructions (they run speculatively) and
the join's phis must be integers. QBE's own if-conversion (ifopt) needs a `sel` instruction,
which the arm64 backend doesn't have; this arithmetic select (4 instructions) is used
instead. It turns data-dependent branches, which mispredict, into straight-line code, and
gives the loop the one-block body the loop transformations need.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ins, rval
from ..backend.util import newtmp, iscmp
from . import ir
import cimo.middle as M

MAXARM = 4


def _okarm(b, ifb):
    if b is ifb:
        return True
    if b.phi is not None or b.nins > MAXARM:
        return False
    return all(i.op in ir.PURE_SAFE for i in ir.ins(b))


def _arm(b, ifb):
    """b is an arm (one predecessor, ifb; jumps on), returns its successor or None."""
    if b.npred == 1 and b.jmp.type == D.Jjmp and b.phi is None and b.nins <= MAXARM and \
            all(i.op in ir.PURE_SAFE for i in ir.ins(b)):
        return b.s1
    return None


def _shape(ifb):
    """(true pred, false pred, join): the blocks the join's phi arguments come from when
    the jnz is true / false."""
    if ifb.jmp.type != D.Jjnz or ifb.s1 is ifb.s2:
        return None
    s1, s2 = ifb.s1, ifb.s2
    j1, j2 = _arm(s1, ifb), _arm(s2, ifb)
    if j1 is not None and j1 is s2:
        return s1, ifb, s2           # if (c) { s1 } ; join = s2
    if j2 is not None and j2 is s1:
        return ifb, s2, s1           # if (!c) { s2 } ; join = s1
    if j1 is not None and j1 is j2:
        return s1, s2, j1
    return None


def convert_loop(fn, lp):
    """If-convert inside loop lp (needs fillcfg); returns True if something changed."""
    for ifb in list(lp.blocks):
        g = _shape(ifb)
        if not g:
            continue
        tb, fb, joinb = g
        if not all(x in lp.blocks for x in (tb, fb, joinb)):
            continue
        if joinb.npred != 2 or joinb is lp.header or joinb.phi is None:
            continue
        if any(p.cls not in (Kw, Kl) for p in ir.phis(joinb)):
            continue
        out = ir.ins(ifb)
        for b in (tb, fb):
            if b is not ifb:
                out = out + ir.ins(b)
        c = ifb.jmp.arg
        if not any(i.to == c and iscmp(i.op) is not None for i in out):
            c2 = newtmp('m', Kw, fn)
            out.append(Ins(D.Ocnew, Kw, c2, c, ir.getcon(0, fn)))
            c = c2
        masks = {}
        for p in ir.phis(joinb):
            a = ir.phiargof(p, tb)
            b = ir.phiargof(p, fb)
            k = p.cls
            if k not in masks:
                cc = c
                if k == Kl:
                    cc = newtmp('m', Kl, fn)
                    out.append(Ins(D.Oextuw, Kl, cc, c, R))
                m = newtmp('m', k, fn)
                out.append(Ins(D.Oneg, k, m, cc, R))
                masks[k] = m
            x = newtmp('m', k, fn)
            out.append(Ins(D.Oxor, k, x, a, b))
            y = newtmp('m', k, fn)
            out.append(Ins(D.Oand, k, y, x, masks[k]))
            out.append(Ins(D.Oxor, k, p.to, b, y))
        ir.setins(ifb, out)
        joinb.phi = None
        ir.setjmp(ifb, joinb)
        M.count('ifconv')
        return True
    return False
