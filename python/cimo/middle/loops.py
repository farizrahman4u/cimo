"""Natural loops and preheaders.

A loop is found from its back edges (an edge b -> h where h dominates b); its body is h plus
every block that reaches a latch without going through h. Irreducible cycles have no such
header and are never transformed.
"""

from ..backend import defs as D
from ..backend.cfg import dom
from . import ir


class Loop:
    __slots__ = ('header', 'blocks', 'latches', 'parent', 'children', 'depth')

    def __init__(self, header):
        self.header = header
        self.blocks = set()
        self.latches = []
        self.parent = None
        self.children = []
        self.depth = 0

    def __contains__(self, b):
        return b in self.blocks

    def outside_preds(self):
        return [p for p in self.header.pred[:self.header.npred] if p not in self.blocks]

    def exits(self):
        """(from, to) edges leaving the loop."""
        out = []
        for b in self.blocks:
            for s in ir.succs(b):
                if s not in self.blocks:
                    out.append((b, s))
        return out

    def __repr__(self):
        return 'Loop(@%s, %d blocks)' % (self.header.name, len(self.blocks))


def find_loops(fn):
    """All natural loops, innermost first (needs fillcfg + filldom)."""
    loops = {}
    for h in fn.rpo[:fn.nblk]:
        for p in h.pred[:h.npred]:
            if dom(h, p):
                lp = loops.get(h)
                if lp is None:
                    lp = loops[h] = Loop(h)
                    lp.blocks.add(h)
                lp.latches.append(p)
                stack = [p]
                while stack:
                    b = stack.pop()
                    if b in lp.blocks:
                        continue
                    lp.blocks.add(b)
                    stack.extend(b.pred[:b.npred])
    lst = sorted(loops.values(), key=lambda l: len(l.blocks))
    for i, l in enumerate(lst):
        for o in lst[i + 1:]:
            if l.header in o.blocks:
                l.parent = o
                o.children.append(l)
                break
    for l in lst:
        d = 0
        p = l.parent
        while p:
            d += 1
            p = p.parent
        l.depth = d
    return lst


def ensure_preheader(fn, lp):
    """Give lp a dedicated preheader: a block outside the loop whose only successor is the
    header and which is the header's only outside predecessor. Returns it (the CFG is stale
    if a block was added)."""
    h = lp.header
    outs = lp.outside_preds()
    if len(outs) == 1 and outs[0].jmp.type == D.Jjmp and outs[0].s1 is h:
        return outs[0]
    if h is fn.start or not outs:
        raise ir.Bail('loop at function entry')
    pre = ir.newblock(fn, outs[-1], 'pre')
    ir.setjmp(pre, h)
    for o in outs:
        ir.retarget(o, h, pre)
    for p in ir.phis(h):
        inside = [(b, a) for b, a in zip(p.blk[:p.narg], p.arg[:p.narg]) if b in lp.blocks]
        outside = [(b, a) for b, a in zip(p.blk[:p.narg], p.arg[:p.narg]) if b not in lp.blocks]
        if len(set(a for _, a in outside)) == 1:
            v = outside[0][1]
        else:
            v = ir.newtmp('m', p.cls, fn)
            ir.addphi(pre, p.cls, v, outside)
        pairs = inside + [(pre, v)]
        p.blk = [b for b, _ in pairs]
        p.arg = [a for _, a in pairs]
        p.narg = len(pairs)
    return pre
