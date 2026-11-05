"""Loop unswitching (LLVM's SimpleLoopUnswitch, GCC's tree_unswitch): a small innermost loop that branches
on a value it doesn't change becomes two loops, one for each outcome, chosen once before them.

    for (...) { if (k) a; else b; }    ->    if (k) for (...) a; else for (...) b;

Each copy then has fewer blocks, which is what the loop passes need to unroll it. It doubles the code, so
only loops of at most MAXINS instructions (LLVM's threshold for this is 50), and at most MAXPERFN times in a
function. Every way out must lead to one block with no other predecessor: values of the loop used later
then meet in phis there.
"""

from ..backend import defs as D
from ..backend.defs import R, Ins, rval
from . import ir
import cimo.middle as M

MAXINS = 50
MAXPERFN = 8
_count = {}


def _condition(fn, lp):
    """(a block of lp with a jnz on a value that doesn't change in lp, the instruction to compute it
    before the loop or None if it's there already). gcm's sink copies a comparison next to its
    branch, so it can be in the loop while everything it reads isn't."""
    ids = {b.id for b in lp.blocks}
    for b in sorted(lp.blocks, key=lambda x: x.id):
        if b.jmp.type != D.Jjnz or b.s1 is b.s2 or not ir.istmp(b.jmp.arg):
            continue
        t = fn.tmp[rval(b.jmp.arg)]
        if t.def_ is None:
            continue
        if t.bid not in ids:
            return b, None
        d = t.def_
        if d.op in ir.PURE_SAFE and all(not ir.istmp(a) or fn.tmp[rval(a)].bid not in ids for a in d.arg):
            return b, d
    return None


def transform(fn, lp):
    """Unswitch lp once (needs fillcfg, filluse, a preheader); returns True if it changed."""
    n = _count.get(fn.name, 0)
    if n >= MAXPERFN or lp.children:
        return False
    if sum(len(ir.ins(b)) for b in lp.blocks) > MAXINS:
        return False
    if any(not ir.copyable(i) for b in lp.blocks for i in ir.ins(b)):
        return False
    c = _condition(fn, lp)
    if c is None:
        return False
    c, hoist = c
    g = shape(lp)
    if g is None:
        return False
    P, E, exits = g
    H = lp.header
    copy = clone(fn, lp, E, exits)
    # choose once, before them; each copy knows the outcome
    cond = c.jmp.arg
    if hoist is not None:
        cond = ir.newtmp('us', hoist.cls, fn)
        ir.setins(P, ir.ins(P) + [Ins(hoist.op, hoist.cls, cond, hoist.arg[0], hoist.arg[1])])
    ir.setjnz(P, cond, H, copy[H])
    s1, s2 = c.s1, c.s2
    ir.setjmp(c, s1)
    ir.setjmp(copy[c], copy.get(s2, s2))
    _count[fn.name] = n + 1
    M.count('unswitch')
    return True


def shape(lp, many=False, fn=None):
    """(the preheader, the block every way out leads to, the exits), if lp can be copied as clone() does. many: the
    ways out may lead to several blocks (a loop that returns from inside), given as a list in place of the one, when
    each is reached from the loop alone and values of the loop are used only in them."""
    exits = lp.exits()
    targets = list(dict.fromkeys(e for _, e in exits))
    if not targets or (len(targets) != 1 and not many):
        return None
    for E in targets:
        if any(p not in lp.blocks for p in E.pred[:E.npred]):
            return None
    outs = lp.outside_preds()
    if len(outs) != 1 or outs[0].jmp.type != D.Jjmp:
        return None
    if len(targets) == 1:
        return outs[0], targets[0], exits
    defined = set()
    for b in lp.blocks:
        defined.update(rval(p.to) for p in ir.phis(b))
        defined.update(rval(i.to) for i in ir.ins(b) if ir.istmp(i.to))
    for b, _ in _uses(fn, lp, defined, set(targets)):
        if b not in targets:
            return None
    return outs[0], targets, exits


def _uses(fn, lp, defined, targets):
    """(block, tmp) for each use outside lp of a value in defined, but the targets' phi arguments from lp."""
    out = []
    for b in ir.blocks(fn):
        if b in lp.blocks:
            continue
        refs = [b.jmp.arg]
        for i in ir.ins(b):
            refs += i.arg[:2]
        for p in ir.phis(b):
            for x, a in zip(p.blk[:p.narg], p.arg[:p.narg]):
                if not (b in targets and x in lp.blocks):
                    refs.append(a)
        out += [(b, rval(r)) for r in refs if ir.istmp(r) and rval(r) in defined]
    return out


def clone(fn, lp, E, exits, subst=None):
    """A copy of lp's blocks, for the caller to branch to, in which subst ({value: replacement}) holds; values of
    the loop used after it meet in phis at E. Returns {block: its copy}."""
    subst = subst or {}
    env = {}
    for b in lp.blocks:
        for p in ir.phis(b):
            env[rval(p.to)] = ir.newtmp('us', p.cls, fn)
        for i in ir.ins(b):
            if ir.istmp(i.to):
                env[rval(i.to)] = ir.newtmp('us', i.cls, fn)

    def m(r):
        return env.get(rval(r), subst.get(r, r)) if ir.istmp(r) else r

    order = sorted(lp.blocks, key=lambda x: x.id)
    copy = {}
    after = order[-1]
    for b in order:
        nb = ir.newblock(fn, after, 'us')
        copy[b] = nb
        after = nb
    for b in order:
        nb = copy[b]
        ir.setins(nb, [Ins(i.op, i.cls, m(i.to), m(i.arg[0]), m(i.arg[1])) for i in ir.ins(b)])
        for p in ir.phis(b):
            ir.addphi(nb, p.cls, m(p.to), [(copy.get(x, x), m(a)) for x, a in zip(p.blk[:p.narg], p.arg[:p.narg])])
        nb.jmp.type = b.jmp.type
        nb.jmp.arg = m(b.jmp.arg)
        nb.s1 = copy.get(b.s1, b.s1) if b.s1 else None
        nb.s2 = copy.get(b.s2, b.s2) if b.s2 else None
    # values of the loop used after it meet in phis where they leave it: at E, for everything after it, or at
    # each of several targets (shape(many=True)), for uses there
    targets = E if isinstance(E, list) else [E]
    inside = set(order) | set(copy.values())
    for T in targets:
        preds = [x for x, e in exits if e is T]
        for p in ir.phis(T):  # the target's own phis take the copy's value from the copy's edges
            for x in preds:
                a = ir.phiargof(p, x)
                p.blk.append(copy[x])
                p.arg.append(m(a))
                p.narg += 1
        where = [b for b in ir.blocks(fn) if b not in inside] if len(targets) == 1 else [T]
        uses = set()
        for b in where:
            refs = [b.jmp.arg]
            for i in ir.ins(b):
                refs += i.arg[:2]
            for p in ir.phis(b):
                for x, a in zip(p.blk[:p.narg], p.arg[:p.narg]):
                    if not (b is T and x in inside):
                        refs.append(a)
            uses.update(rval(r) for r in refs if ir.istmp(r) and rval(r) in env)
        joined = {}
        for t in sorted(uses):
            r = D.TMP(t)
            v = ir.newtmp('us', fn.tmp[t].cls, fn)
            ir.addphi(T, fn.tmp[t].cls, v, [(x, r) for x in preds] + [(copy[x], m(r)) for x in preds])
            joined[r] = v
        if joined:
            new = set(joined.values())
            for b in where:
                for i in ir.ins(b):
                    i.arg[0] = joined.get(i.arg[0], i.arg[0])
                    i.arg[1] = joined.get(i.arg[1], i.arg[1])
                b.jmp.arg = joined.get(b.jmp.arg, b.jmp.arg)
                for p in ir.phis(b):
                    if p.to in new:
                        continue
                    for k in range(p.narg):
                        if not (b is T and p.blk[k] in inside):
                            p.arg[k] = joined.get(p.arg[k], p.arg[k])
    return copy
