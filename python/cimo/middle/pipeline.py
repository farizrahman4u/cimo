"""Driver of the middle-end passes for one function."""

import sys

from ..backend import defs as D
from ..backend.parse import printfn
import cimo.middle as M
from ..backend.defs import rval
from . import ir, loops, memdep, inner, jam, ifconv, unswitch, stride, earlyexit


def _restrict_names(fn):
    info = M.params.get(fn.name) or []
    return set(name.lstrip('%') for name, r in info if r)


def _region(lp):
    """The blocks a transformation of lp may change: its own, its preheaders and the blocks it leaves to."""
    out = set(lp.blocks)
    out.update(lp.outside_preds())
    out.update(t for _, t in lp.exits())
    return out


def _newtmps(lp, ntmp0):
    """Whether lp refers to a temporary made after ntmp0 (by a transformation earlier in the round)."""
    for b in lp.blocks:
        refs = [b.jmp.arg]
        for p in ir.phis(b):
            refs += p.arg[:p.narg]
            refs.append(p.to)
        for i in ir.ins(b):
            refs += i.arg[:2]
            refs.append(i.to)
        if any(ir.istmp(r) and rval(r) >= ntmp0 for r in refs):
            return True
    return False


def _phase(fn, rnames, pick, act, done):
    """Apply act to loops chosen by pick until nothing changes; returns True if any did.

    A round refreshes the function once (uses, blocks, dominators), finds its loops and computes its roots once, then
    takes every loop it can: a loop whose blocks, preheaders and exits no earlier transformation in the round changed,
    that refers to no temporary one made, and whose values no changed block uses (so its use lists are whole). For those the round's analyses still hold: a transformation never changes
    what an existing temporary means, it makes new ones, and block numbers only change at a refresh. The others wait for
    the next round. One transformation a round made the compile quadratic in the number of loops (scramjit: 60 loops
    4.0 s, 120 loops 14.7 s)."""
    changed = False
    for _ in range(200):
        ir.refresh(fn)
        if ir.merge_blocks(fn):
            ir.refresh(fn)
        progress = False
        rts = None  # the function's roots: one fixpoint over the whole function, computed once per round
        succs = {b: (b.s1, b.s2) for b in ir.blocks(fn)}
        ir.created[:] = []
        ntmp0 = fn.ntmp
        seen_blocks = set()
        touched = set()
        seen = set()  # temporaries the touched blocks refer to: their use lists may miss those uses
        for lp in loops.find_loops(fn):
            if lp.header in done or not pick(lp):
                continue
            if touched:
                new = touched - seen_blocks
                _refs(new, seen)
                seen_blocks.update(new)
                if any(_region(l) & touched for l in [lp] + lp.children) or _newtmps(lp, ntmp0) or \
                        _defines(lp, seen):
                    progress = True  # next round
                    continue
            try:
                added = False
                for l in [lp] + lp.children:
                    outs = l.outside_preds()
                    if not (len(outs) == 1 and outs[0].jmp.type == D.Jjmp):
                        loops.ensure_preheader(fn, l)
                        added = True
                if added:
                    progress = True
                    touched |= _changed(fn, succs, _region(lp)) | _region(lp)
                    continue
                if rts is None:
                    rts = memdep.Roots(fn, rnames)
                ok = act(fn, lp, rts)
            except ir.Bail as e:
                if M.debug:
                    sys.stderr.write('middle: %s: loop @%s: %s\n' % (fn.name, lp.header.name, e))
                done.add(lp.header)
                continue
            if ok:
                progress = changed = True
                touched |= _changed(fn, succs, _region(lp)) | _region(lp)
                continue
            done.add(lp.header)
        if not progress:
            break
    return changed


def _defines(lp, tmps):
    """Whether lp defines one of tmps."""
    for b in lp.blocks:
        if any(rval(p.to) in tmps for p in ir.phis(b)) or any(ir.istmp(i.to) and rval(i.to) in tmps for i in ir.ins(b)):
            return True
    return False


def _refs(blocks, out):
    """Add the temporaries the blocks refer to, to out."""
    for b in blocks:
        refs = [b.jmp.arg]
        for p in ir.phis(b):
            refs += p.arg[:p.narg]
        for i in ir.ins(b):
            refs += i.arg[:2]
        out.update(rval(r) for r in refs if ir.istmp(r))


def _changed(fn, succs, near):
    """Blocks made, or whose successors changed, since succs was taken, and their successors then and now (whose
    predecessors changed); succs is brought up to date. Only near (the transformed loop's region) and the blocks made
    since can have changed: looking at no others keeps a round linear."""
    out = set()
    made, ir.created[:] = list(ir.created), []
    for b in list(near) + made:
        now = (b.s1, b.s2)
        was = succs.get(b)
        if was != now:
            out.add(b)
            out.update(x for x in (was or ()) + now if x is not None)
            succs[b] = now
    return out


def optimize(fn):
    if M.debug:
        sys.stderr.write('\n> Before middle end:\n')
        printfn(fn, sys.stderr)
    rnames = _restrict_names(fn)
    done = set()
    innermost = lambda lp: not lp.children
    nest2 = lambda lp: len(lp.children) == 1 and not lp.children[0].children
    unswitch._count.pop(fn.name, None)
    stride._count.pop(fn.name, None)
    stride._versioned.pop(fn.name, None)
    changed = _phase(fn, rnames, innermost, lambda fn, lp, rts: stride.transform(fn, lp), set())
    changed |= _phase(fn, rnames, innermost, lambda fn, lp, rts: unswitch.transform(fn, lp), set())
    changed |= _phase(fn, rnames, innermost,
                      lambda fn, lp, rts: inner.transform(fn, lp, rts, done, 'full'), set())
    if M.unroll > 1:
        jdone = set()
        changed |= _phase(fn, rnames, nest2,
                          lambda fn, lp, rts: jam.transform(fn, lp, rts, jdone), jdone)
    changed |= _phase(fn, rnames, innermost,
                      lambda fn, lp, rts: ifconv.convert_loop(fn, lp), set())
    changed |= _phase(fn, rnames, innermost,
                      lambda fn, lp, rts: inner.transform(fn, lp, rts, done, 'opt'), done)
    changed |= _phase(fn, rnames, innermost, lambda fn, lp, rts: earlyexit.transform(fn, lp), set())
    ir.normalize_wcons(fn)
    if changed:
        ir.refresh(fn)
        ir.dce(fn)
        ir.refresh(fn)
    if M.debug:
        sys.stderr.write('\n> After middle end:\n')
        printfn(fn, sys.stderr)
    return changed
