"""If-conversion of small integer diamonds and triangles inside innermost loops.

    if (c) x = a; else x = b;        ->   m = -c;  x = b ^ ((a ^ b) & m)

The arms must hold only a few pure, non-trapping instructions (they run speculatively) and
the join's phis must be integers. QBE's own if-conversion (ifopt) needs a `sel` instruction,
which the arm64 backend doesn't have; this arithmetic select (4 instructions) is used
instead. It turns data-dependent branches, which mispredict, into straight-line code, and
gives the loop the one-block body the loop transformations need, so only loops that can get one
are converted.
"""

from ..backend import defs as D
from ..backend.defs import R, Kw, Kl, Ins, rval
from ..backend.util import newtmp, iscmp
from ..backend import ext as _ext
from . import ir
import cimo.middle as M

MAXARM = 4
# what an arm may hold: also a select an inner diamond became, so nested ones convert from the inside out
# (v < 0 ? 0 : v > 1 ? 1 : v); its sel0 and sel1 stay adjacent, as the blocks' instructions are copied in order
_ARMOPS = None


def _armops():
    global _ARMOPS
    if _ARMOPS is None:
        _ARMOPS = ir.PURE_SAFE | {D.Osel0, D.Osel1}
    return _ARMOPS


def _okarm(b, ifb):
    if b is ifb:
        return True
    if b.phi is not None or b.nins > MAXARM:
        return False
    return all(i.op in _armops() for i in ir.ins(b))


def _arm(b, ifb):
    """b is an arm (one predecessor, ifb; jumps on), returns its successor or None."""
    if b.npred == 1 and b.jmp.type == D.Jjmp and b.phi is None and b.nins <= MAXARM and \
            all(i.op in _armops() for i in ir.ins(b)):
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


def _can_be_one_block(lp):
    """A loop with a side exit (a failing bounds check) or a call never gets a one-block body:
    selects there only add instructions and registers, in code whose branches predict well.
    Measured, also with csel: scramjit's viewshed 8.17 ms kept to this rule, 14.38 ms not."""
    if len(lp.exits()) > 1:
        return False
    return not any(i.op == D.Ocall for b in lp.blocks for i in ir.ins(b))


_DIVS = (D.Odiv, D.Orem, D.Oudiv, D.Ourem)


def _waits_on_division(lp, c):
    """Whether c comes from a division in lp, a few instructions back: a select would wait for it, and what reads the
    select after it (an address, a load), where a predicted branch lets them run ahead (backend.ifopt.slowcond)."""
    defs = {i.to: i for b in lp.blocks for i in ir.ins(b) if ir.istmp(i.to)}
    work, seen = [c], set()
    while work and len(seen) < 12:
        r = work.pop()
        if r in seen or r not in defs:
            continue
        seen.add(r)
        d = defs[r]
        if d.op in _DIVS and d.cls in (Kw, Kl):
            return True
        work += [a for a in d.arg[:2] if ir.istmp(a)]
    return False


def convert_loop(fn, lp):
    """If-convert inside loop lp (needs fillcfg); returns True if something changed."""
    if not _can_be_one_block(lp):
        return False
    for ifb in list(lp.blocks):
        g = _shape(ifb)
        if not g:
            continue
        tb, fb, joinb = g
        if not all(x in lp.blocks for x in (tb, fb, joinb)):
            continue
        if joinb.npred != 2 or joinb is lp.header or joinb.phi is None:
            continue
        if _ext.isel and _waits_on_division(lp, ifb.jmp.arg):
            continue
        if any(p.cls not in (Kw, Kl) for p in ir.phis(joinb)) and not (_ext.isel or D.T.name.startswith('amd64')):
            continue  # cimo: x86-64 selects floats with a jump over a move (amd64.emit), so the vectorizer takes them
        out = ir.ins(ifb)
        for b in (tb, fb):
            if b is not ifb:
                out = out + ir.ins(b)
        c = ifb.jmp.arg
        arm = _ext.isel and D.T.name.startswith('arm64')  # csel: one instruction per value
        x86 = D.T.name.startswith('amd64')  # cmov, as QBE's ifopt; a float one a jump over a move
        if arm or x86:
            for p in ir.phis(joinb):  # a sel0 each: the loop passes copy them as pairs
                out.append(Ins(D.Osel0, Kw, R, c, R))
                out.append(Ins(D.Osel1, p.cls, p.to, ir.phiargof(p, tb), ir.phiargof(p, fb)))
            ir.setins(ifb, out)
            joinb.phi = None
            ir.setjmp(ifb, joinb)
            M.count('ifconv')
            return True
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
