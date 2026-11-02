"""cimo: memory optimizations QBE doesn't have (ext.cse), from LLVM's GVNHoist, MergedLoadStoreMotion and
DeadStoreElimination and GCC's store_merging. All work on SSA after load elimination, and only within
what they can see: a block, or a branch and its two arms.

    hoist(fn)   the same load first in both arms: once, before the branch
    sink(fn)    the same store last in both arms: once, at the join, of a phi
    dse(fn)     a store overwritten in its block or on every way on, with nothing that could read it
    pre(fn)     a load known on all ways into its block but one: a load on that way and a phi
    merge(fn)   adjacent constant stores off one base: one wider store (arm64 and wasm allow unaligned)

hoist and sink require rpo, preds and use; dse and merge only the instructions. Each breaks use and
returns True if it changed anything.
"""

from . import defs as D
from .defs import R, RTmp, RCon, CBits, Kw, Kl, Ks, Kd, rtype, rval
from .gvn import replaceuses
from .util import newtmp, getcon

_STORESIZE = {D.Ostoreb: 1, D.Ostoreh: 2, D.Ostorew: 4, D.Ostorel: 8, D.Ostores: 4, D.Ostored: 8}
_STORECLS = {D.Ostoreb: Kw, D.Ostoreh: Kw, D.Ostorew: Kw, D.Ostorel: Kl, D.Ostores: Ks, D.Ostored: Kd}
_INTSTORE = {1: D.Ostoreb, 2: D.Ostoreh, 4: D.Ostorew, 8: D.Ostorel}


def _live(b):
    return [i for i in b.ins[:b.nins] if i.op not in (D.Onop, D.Odbgloc)]


def _arms(h):
    """The two arms of h's branch, each only reached from h; or None."""
    if h.jmp.type != D.Jjnz or h.s1 is h.s2:
        return None
    a, b = h.s1, h.s2
    if a.npred != 1 or b.npred != 1 or a.phi is not None or b.phi is not None:
        return None
    return a, b


def hoist(fn):
    changed = False
    for h in fn.rpo[:fn.nblk]:
        arms = _arms(h)
        if arms is None:
            continue
        a, b = arms
        while True:
            la, lb = _live(a), _live(b)
            if not la or not lb:
                break
            x, y = la[0], lb[0]
            if not D.isload(x.op) or x.op != y.op or x.cls != y.cls or x.arg[0] != y.arg[0]:
                break
            r = x.arg[0]
            if rtype(r) == RTmp and fn.tmp[rval(r)].bid in (a.id, b.id):
                break  # its address is computed in the arm
            replaceuses(fn, y.to, x.to)
            b.ins = [i for i in b.ins[:b.nins] if i is not y]
            b.nins = len(b.ins)
            a.ins = [i for i in a.ins[:a.nins] if i is not x]
            a.nins = len(a.ins)
            h.ins = h.ins[:h.nins] + [x]
            h.nins = len(h.ins)
            fn.tmp[rval(x.to)].bid = h.id
            changed = True
    return changed


def sink(fn):
    changed = False
    for h in fn.rpo[:fn.nblk]:
        arms = _arms(h)
        if arms is None:
            continue
        a, b = arms
        if a.jmp.type != D.Jjmp or b.jmp.type != D.Jjmp or a.s1 is not b.s1 or a.s1.npred != 2:
            continue
        j = a.s1
        la, lb = _live(a), _live(b)
        if not la or not lb:
            continue
        x, y = la[-1], lb[-1]
        if not D.isstore(x.op) or x.op != y.op or x.arg[1] != y.arg[1]:
            continue
        r = x.arg[1]
        if rtype(r) == RTmp and fn.tmp[rval(r)].bid in (a.id, b.id):
            continue
        k = _STORECLS[x.op]
        if x.arg[0] == y.arg[0]:
            v = x.arg[0]
        else:
            v = newtmp('sk', k, fn)
            p = D.Phi()
            p.to = v
            p.cls = k
            p.arg = [x.arg[0], y.arg[0]]
            p.blk = [a, b]
            p.narg = 2
            p.link = j.phi
            j.phi = p
        a.ins = [i for i in a.ins[:a.nins] if i is not x]
        a.nins = len(a.ins)
        b.ins = [i for i in b.ins[:b.nins] if i is not y]
        b.nins = len(b.ins)
        j.ins = [D.Ins(x.op, x.cls, R, v, r)] + j.ins[:j.nins]
        j.nins = len(j.ins)
        changed = True
    return changed


def _reads(i):
    """i can read memory or do anything else that lets a store be seen."""
    o = i.op
    if D.isload(o) or o in (D.Ocall, D.Oblit0, D.Oblit1, D.Ovastart, D.Ovaarg) or D.isvload(o) or D.isvstore(o):
        return True
    return o in (D.Ogetsp, D.Osetsp, D.Osalloc) or D.isalloc(o)


def _overwritten(b, r, n, depth):
    """Every way on from b stores at least n bytes at r before anything reads memory (LLVM's DSE, across
    blocks). Few blocks are followed: this is for a store and its overwrite in the arms of a branch. Only
    forwards: round a loop, r is the same temporary but another address."""
    if b.jmp.type not in (D.Jjmp, D.Jjnz) or depth == 0:
        return False
    for s in (b.s1, b.s2):
        if s is None:
            continue
        if s.id <= b.id:
            return False
        for i in s.ins[:s.nins]:
            if _reads(i):
                return False
            if D.isstore(i.op) and i.arg[1] == r and _STORESIZE[i.op] >= n:
                break
        else:
            if not _overwritten(s, r, n, depth - 1):
                return False
    return True


def dse(fn):
    changed = False
    b = fn.start
    while b:
        killed = {}  # address -> bytes overwritten from it, later in the block
        out = []
        ins = b.ins[:b.nins]
        drop = False
        clean = True  # nothing reads memory from here to the end of the block
        for i in reversed(ins):
            if _reads(i):
                killed = {}
                clean = False
            elif D.isstore(i.op):
                n = _STORESIZE[i.op]
                r = i.arg[1]
                if killed.get(r, 0) >= n or (clean and _overwritten(b, r, n, 4)):
                    drop = True
                    continue
                killed[r] = max(killed.get(r, 0), n)
            out.append(i)
        if drop:
            out.reverse()
            b.ins = out
            b.nins = len(out)
            changed = True
        b = b.link
    return changed


def _baseoff(fn, r):
    """(base, offset) of an address: x, or x + constant."""
    if rtype(r) == RTmp:
        d = fn.tmp[rval(r)].def_
        if d is not None and d.op == D.Oadd and d.cls == Kl and rtype(d.arg[1]) == RCon:
            c = fn.con[rval(d.arg[1])]
            if c.type == CBits:
                return d.arg[0], c.i
    return r, 0


MERGESPAN = 128  # instructions looked at from each store: a block of thousands of them was quadratic


def merge(fn):
    changed = False
    intstores = frozenset(_INTSTORE.values())
    b = fn.start
    while b:
        ins = b.ins[:b.nins]
        out = []
        n = 0
        did = False
        while n < len(ins):
            i = ins[n]
            if i.op not in intstores or rtype(i.arg[0]) != RCon or fn.con[rval(i.arg[0])].type != CBits:
                out.append(i)
                n += 1
                continue
            run = []
            k = n
            end = n
            while k < len(ins) and k - n < MERGESPAN:
                x = ins[k]
                if x.op in intstores and rtype(x.arg[0]) == RCon and fn.con[rval(x.arg[0])].type == CBits:
                    run.append(x)
                    k += 1
                    end = k
                    continue
                if D.isstore(x.op) or _reads(x) or x.to == R and x.op not in (D.Onop, D.Odbgloc):
                    break
                k += 1
            k = end
            stores = set(map(id, run))
            # pure instructions among them, such as the addresses, stay
            between = [x for x in ins[n:end] if id(x) not in stores]
            if len(run) < 2:
                out.append(i)
                n += 1
                continue
            bases = [_baseoff(fn, x.arg[1]) for x in run]
            base = bases[0][0]
            if any(p[0] != base for p in bases):
                out.append(i)
                n += 1
                continue
            spans = sorted((p[1], _STORESIZE[x.op], fn.con[rval(x.arg[0])].i) for p, x in zip(bases, run))
            if any(spans[m][0] + spans[m][1] != spans[m + 1][0] for m in range(len(spans) - 1)):
                out.append(i)  # gaps or overlaps: order would matter
                n += 1
                continue
            start = spans[0][0]
            total = spans[-1][0] + spans[-1][1] - start
            value = 0
            for off, size, v in spans:
                value |= (v & ((1 << (8 * size)) - 1)) << (8 * (off - start))
            rest = total
            # the widest stores that fit, each at the address of an old store when one has it
            byoff = {p[1]: x.arg[1] for p, x in zip(bases, run)}
            ok = True
            pieces = []
            off = start
            while rest:
                size = 8 if rest >= 8 else 4 if rest >= 4 else 2 if rest >= 2 else 1
                if off not in byoff:
                    ok = False
                if not ok:
                    break
                pieces.append((off, size))
                off += size
                rest -= size
            if not ok or len(pieces) >= len(run):
                out.append(i)
                n += 1
                continue
            out += between  # the merged stores go last: every address is computed by then
            for off, size in pieces:
                v = (value >> (8 * (off - start))) & ((1 << (8 * size)) - 1)
                out.append(D.Ins(_INTSTORE[size], Kw, R, getcon(D.s64(v), fn), byoff[off]))
            did = True
            n = k
        if did:
            b.ins = out
            b.nins = len(out)
            changed = True
        b = b.link
    return changed


_CLOBBER = frozenset((D.Ocall, D.Oblit0, D.Oblit1, D.Ovastart, D.Ovaarg))
_STOREOF = {Kw: D.Ostorew, Kl: D.Ostorel, Ks: D.Ostores, Kd: D.Ostored}


def _available(p, r, op, cls):
    """What the end of block p knows is at r for a load op of class cls: a load or store there of the same
    extent with nothing that may write after it; or None."""
    whole = op == D.Oload or (cls == Kw and op in (D.Oloadsw, D.Oloaduw))
    for i in reversed(p.ins[:p.nins]):
        if i.op == op and i.cls == cls and i.arg[0] == r:
            return i.to
        if D.isstore(i.op):
            return i.arg[0] if i.arg[1] == r and whole and i.op == _STOREOF[cls] else None
        if i.op in _CLOBBER or D.isvload(i.op) or D.isvstore(i.op):
            return None
    return None


def pre(fn):
    """LLVM's GVN load PRE, GCC's pre: a block's load whose value is known on all its ways in but one. A load
    on that way makes it a phi, so each path runs one load fewer or the same. Loads QBE's own elimination
    leaves: it inserts them for loops, not for branches."""
    changed = False
    for b in fn.rpo[:fn.nblk]:
        if b.npred < 2 or b.loop > 1:
            continue
        preds = b.pred[:b.npred]
        if len(set(preds)) != len(preds) or any(p.loop > b.loop for p in preds):
            continue
        for i in b.ins[:b.nins]:
            if D.isstore(i.op) or i.op in _CLOBBER or D.isvload(i.op) or D.isvstore(i.op):
                break
            if not D.isload(i.op) or rtype(i.arg[0]) == D.RSlot:
                continue
            r = i.arg[0]
            if rtype(r) == D.RTmp and fn.tmp[rval(r)].bid == b.id:
                continue  # its address is computed here
            vals = [_available(p, r, i.op, i.cls) for p in preds]
            missing = [p for p, v in zip(preds, vals) if v is None]
            if len(missing) != 1 or len(preds) == 1:
                continue
            m = missing[0]
            if m.jmp.type != D.Jjmp:
                continue
            t = newtmp('pre', i.cls, fn)
            m.ins = m.ins[:m.nins] + [D.Ins(i.op, i.cls, t, r, R)]
            m.nins = len(m.ins)
            fn.tmp[rval(t)].bid = m.id
            p = D.Phi()
            p.to = i.to
            p.cls = i.cls
            p.arg = [t if v is None else v for v in vals]
            p.blk = list(preds)
            p.narg = len(preds)
            p.link = b.phi
            b.phi = p
            i.set(D.Ins(D.Onop))
            changed = True
    return changed
