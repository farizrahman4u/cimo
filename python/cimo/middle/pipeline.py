"""Driver of the middle-end passes for one function."""

import sys

from ..backend import defs as D
from ..backend.parse import printfn
import cimo.middle as M
from . import ir, loops, memdep, inner, jam, ifconv


def _restrict_names(fn):
    info = M.params.get(fn.name) or []
    return set(name.lstrip('%') for name, r in info if r)


def _phase(fn, rnames, pick, act, done):
    """Apply act to loops chosen by pick until nothing changes; returns True if any did."""
    changed = False
    for _ in range(200):
        ir.refresh(fn)
        if ir.merge_blocks(fn):
            ir.refresh(fn)
        progress = False
        for lp in loops.find_loops(fn):
            if lp.header in done or not pick(lp):
                continue
            try:
                for l in [lp] + lp.children:
                    outs = l.outside_preds()
                    if not (len(outs) == 1 and outs[0].jmp.type == D.Jjmp):
                        loops.ensure_preheader(fn, l)
                        progress = True
                if progress:
                    break
                rts = memdep.Roots(fn, rnames)
                ok = act(fn, lp, rts)
            except ir.Bail as e:
                if M.debug:
                    sys.stderr.write('middle: %s: loop @%s: %s\n' % (fn.name, lp.header.name, e))
                done.add(lp.header)
                continue
            if ok:
                progress = changed = True
                break
            done.add(lp.header)
        if not progress:
            break
    return changed


def optimize(fn):
    if M.debug:
        sys.stderr.write('\n> Before middle end:\n')
        printfn(fn, sys.stderr)
    rnames = _restrict_names(fn)
    done = set()
    innermost = lambda lp: not lp.children
    nest2 = lambda lp: len(lp.children) == 1 and not lp.children[0].children
    changed = _phase(fn, rnames, innermost,
                     lambda fn, lp, rts: inner.transform(fn, lp, rts, done, 'full'), set())
    if M.unroll > 1:
        jdone = set()
        changed |= _phase(fn, rnames, nest2,
                          lambda fn, lp, rts: jam.transform(fn, lp, rts, jdone), jdone)
    changed |= _phase(fn, rnames, innermost,
                      lambda fn, lp, rts: ifconv.convert_loop(fn, lp), set())
    changed |= _phase(fn, rnames, innermost,
                      lambda fn, lp, rts: inner.transform(fn, lp, rts, done, 'opt'), done)
    ir.normalize_wcons(fn)
    if changed:
        ir.refresh(fn)
        ir.dce(fn)
        ir.refresh(fn)
    if M.debug:
        sys.stderr.write('\n> After middle end:\n')
        printfn(fn, sys.stderr)
    return changed
