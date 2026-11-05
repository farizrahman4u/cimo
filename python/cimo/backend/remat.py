"""cimo: constants made near their uses, not once for the whole function.

gvn merges equal broadcasts of a constant (`qdupd 2.0`) from different loops into one value, which gcm then puts where
it dominates them all, near the start; arm64's isel loaded each float constant once, in the start block. Either way the
value is live through the whole function: with 120 small loops, 80% of every block's live-in set was such constants,
which made liveness, spilling and register allocation quadratic in the function's size, and held registers the code
needed.

Each use gets a home: the preheader of the innermost loop it's in (the header's immediate dominator), or its own block
outside loops. A constant is made once per home, at the home's start, so a loop finds it in a register, made before the
loop, and it isn't live past that loop. Homes further out (the outermost loop's preheader) kept registers busy all
through a nest of loops (mandelbrot: saves of d8-d12 and moves); made in each block, an inner loop remade its constants on
every iteration. Broadcasts of constants are cloned; float constant operands (arm64) become a copy per home, which isel
makes there (its float constants are per block now).
"""

from . import defs as D
from .defs import R, RCon, RTmp, CBits, Ins, KBASE, Kq, rtype, rval, UPhi, UIns
from .util import newtmp, argcls
from . import cfg

LARGE = 256  # blocks: smaller functions keep one constant for all, which register allocation is tuned around
homed = False  # whether the function being compiled has had its float constants given homes (arm64's isel asks)


def _isdup(i):
    return D.Oqdupb <= i.op <= D.Oqdupd and rtype(i.arg[0]) == RCon


def _first(b):
    """Where a home's constants go: after the parameters and allocations the start block has to begin with."""
    n = 0
    while n < b.nins and (D.ispar(b.ins[n].op) or D.isalloc(b.ins[n].op)):
        n += 1
    return n


def run(fn, floats):
    """floats: also give float constant operands a home (arm64, whose isel would load each one in the start block).
    Only for a function of LARGE blocks or more: in a small one a constant live all through costs little, and changing
    where it lives moved register allocation's choices in the benchmark kernels' loops (mandelbrot: two moves in the
    innermost loop). Needs fillcfg and filluse; leaves uses to be refilled. Returns True if it changed anything."""
    global homed
    homed = False
    if fn.nblk < LARGE:
        return False
    homed = floats
    cfg.filldom(fn)
    cfg.fillloop(fn)
    homes = {}

    def home(b):
        h = homes.get(b)
        if h is None:
            # the innermost loop's blocks share its weight (fillloop: 10 a loop); up the dominators to its header,
            # whose immediate dominator is its preheader
            x = b
            if b.loop > 1:
                while x.idom is not None and x.idom.loop >= b.loop:
                    x = x.idom
                x = x.idom if x.idom is not None else x
            homes[b] = h = x
        return h

    blocks = fn.rpo[:fn.nblk]
    made = {}  # home block -> [Ins] to put at its start
    changed = False
    # broadcasts of constants
    for b in blocks:
        for i in b.ins[:b.nins]:
            if not _isdup(i) or rtype(i.to) != RTmp:
                continue
            t = rval(i.to)
            tm = fn.tmp[t]
            groups = {}  # home -> [(kind, user, arg index or pred)]
            for u in (tm.use or [])[:tm.nuse]:
                ub = fn.rpo[u.bid]
                if u.type == UPhi:
                    p = u.u
                    for n in range(p.narg):
                        if p.arg[n] == i.to:
                            groups.setdefault(home(p.blk[n]), []).append((UPhi, p, n))
                elif u.type == UIns:
                    groups.setdefault(home(ub), []).append((UIns, u.u, None))
                else:
                    groups.setdefault(home(ub), []).append((None, ub, None))
            if not groups or (len(groups) == 1 and next(iter(groups)) is home(b) and home(b) is b):
                continue
            for h, us in groups.items():
                r = newtmp('rm', Kq, fn)
                made.setdefault(h, []).append(Ins(i.op, Kq, r, i.arg[0], R))
                for kind, x, n in us:
                    if kind == UPhi:
                        x.arg[n] = r
                    elif kind == UIns:
                        x.arg[0] = r if x.arg[0] == i.to else x.arg[0]
                        x.arg[1] = r if x.arg[1] == i.to else x.arg[1]
                    else:
                        x.jmp.arg = r
            i.set(Ins(D.Onop))
            changed = True
    # float constant operands
    if floats:
        cache = {}  # (home, constant index, class) -> temporary

        def ref(h, a, k):
            key = (h, rval(a), k)
            r = cache.get(key)
            if r is None:
                r = cache[key] = newtmp('rf', k, fn)
                made.setdefault(h, []).append(Ins(D.Ocopy, k, r, a, R))
            return r

        def isfcon(a, k):
            return rtype(a) == RCon and k in (D.Ks, D.Kd) and fn.con[rval(a)].type == CBits
        for b in blocks:
            h = home(b)
            for i in b.ins[:b.nins]:
                for n in (0, 1):
                    a = i.arg[n]
                    if a == R or rtype(a) != RCon:
                        continue
                    k = argcls(i, n)
                    if D.isstore(i.op) and fn.con[rval(a)].i == 0:
                        continue  # stored as xzr: no register at all
                    if isfcon(a, k):
                        i.arg[n] = ref(h, a, k)
                        changed = True
            p = b.phi
            while p:
                if p.cls in (D.Ks, D.Kd):
                    for n in range(p.narg):
                        if isfcon(p.arg[n], p.cls):
                            p.arg[n] = ref(home(p.blk[n]), p.arg[n], p.cls)
                            changed = True
                p = p.link
    for h, ins in made.items():
        n = _first(h)
        h.ins = h.ins[:n] + ins + h.ins[n:h.nins]
        h.nins = len(h.ins)
    return changed
