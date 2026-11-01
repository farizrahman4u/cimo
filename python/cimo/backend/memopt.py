"""cimo: memory optimizations QBE doesn't have (ext.cse), from LLVM's GVNHoist, MergedLoadStoreMotion and
DeadStoreElimination and GCC's store_merging. All work on SSA after load elimination, and only within
what they can see: a block, or a branch and its two arms.

    hoist(fn)   the same load first in both arms: once, before the branch
    sink(fn)    the same store last in both arms: once, at the join, of a phi
    dse(fn)     a store overwritten in its block, with nothing that could read it in between
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


def dse(fn):
    changed = False
    b = fn.start
    while b:
        killed = {}  # address -> bytes overwritten from it, later in the block
        out = []
        ins = b.ins[:b.nins]
        drop = False
        for i in reversed(ins):
            if _reads(i):
                killed = {}
            elif D.isstore(i.op):
                n = _STORESIZE[i.op]
                r = i.arg[1]
                if killed.get(r, 0) >= n:
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


def merge(fn):
    changed = False
    b = fn.start
    while b:
        ins = b.ins[:b.nins]
        out = []
        n = 0
        did = False
        while n < len(ins):
            i = ins[n]
            run = []
            between = []  # pure instructions among them, such as the addresses: they stay
            k = n
            end = n
            while k < len(ins):
                x = ins[k]
                if x.op in _INTSTORE.values() and rtype(x.arg[0]) == RCon \
                        and fn.con[rval(x.arg[0])].type == CBits:
                    run.append(x)
                    k += 1
                    end = k
                    continue
                if D.isstore(x.op) or _reads(x) or x.to == R and x.op not in (D.Onop, D.Odbgloc):
                    break
                between.append(x)
                k += 1
            k = end
            between = [x for x in ins[n:end] if x not in run]
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
