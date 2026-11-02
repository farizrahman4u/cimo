"""cimo: LLVM's SCCP and ADCE, GCC's ccp and cd_dce (ext.cse; not in QBE).

gvn folds what it sees is constant from its operands, and removes what has no uses. These two assume the
best of a loop until shown otherwise, so a value that only depends on itself goes too."""

from . import defs as D
from .defs import R, RCon, RTmp, CBits, Jjmp, Jjnz, KBASE, optab, rtype, rval
from .fold import foldref
from .gcm import pinned
from .gvn import replaceuses
from .util import iscmp

_BOT = -1  # not constant; absent from the map: nothing known yet


def sccp(fn):
    """Conditional constant propagation (Wegman and Zadeck), by sweeps: the lattice is 2 high. Requires
    use lists; returns True if anything changed (then the caller rebuilds the cfg)."""
    if not any(_constphi(fn, b) for b in fn.rpo[:fn.nblk]):
        return False
    val = {}  # temporary's number: constant ref or _BOT
    run = {fn.start}
    edge = set()

    def get(r):
        if rtype(r) == RCon:
            return r if fn.con[rval(r)].type == CBits else _BOT
        if rtype(r) != RTmp:
            return _BOT
        return val.get(rval(r))

    def put(to, v):
        n = rval(to)
        old = val.get(n)
        if old == _BOT or v is None or old == v:
            return False
        if old is not None and v != _BOT and fn.con[rval(old)].i == fn.con[rval(v)].i:
            return False
        val[n] = v if old is None else _BOT
        return True
    changed = True
    while changed:
        changed = False
        for b in fn.rpo[:fn.nblk]:
            if b not in run:
                continue
            p = b.phi
            while p:
                v = None
                if KBASE(p.cls) != 0:
                    v = _BOT
                else:
                    for a, f in zip(p.arg, p.blk):
                        if (f, b) not in edge:
                            continue
                        x = get(a)
                        if x is None:
                            continue
                        if x == _BOT or (v is not None and fn.con[rval(v)].i != fn.con[rval(x)].i):
                            v = _BOT
                            break
                        v = x
                changed = put(p.to, v) or changed
                p = p.link
            for i in b.ins[:b.nins]:
                if rtype(i.to) != RTmp:
                    continue
                if not optab[i.op].canfold or pinned(i) or KBASE(i.cls) != 0:
                    v = _BOT
                else:
                    x = get(i.arg[0])
                    y = D.CON_Z if i.arg[1] == R else get(i.arg[1])
                    if x == _BOT or y == _BOT:
                        v = _BOT
                    elif x is None or y is None:
                        v = None
                    else:
                        v = foldref(fn, D.Ins(i.op, i.cls, i.to, x, y if i.arg[1] != R else R))
                        v = _BOT if v == R or fn.con[rval(v)].type != CBits else v
                changed = put(i.to, v) or changed
            j = b.jmp
            if j.type == Jjmp:
                out = [b.s1]
            elif j.type == Jjnz:
                c = get(j.arg)
                if c is None:
                    out = []
                elif c == _BOT:
                    out = [b.s1, b.s2]
                else:
                    out = [b.s1 if fn.con[rval(c)].i & 0xffffffff else b.s2]
            else:
                out = []
            for t in out:
                if (b, t) not in edge:
                    edge.add((b, t))
                    run.add(t)
                    changed = True
    any_ = False
    for n, v in val.items():
        if v != _BOT and fn.tmp[n].nuse:
            replaceuses(fn, D.TMP(n), v)
            any_ = True
    for b in fn.rpo[:fn.nblk]:
        if b in run and b.jmp.type == Jjnz:
            c = get(b.jmp.arg)
            if c is not None and c != _BOT:
                t = b.s1 if fn.con[rval(c)].i & 0xffffffff else b.s2
                b.jmp.type, b.jmp.arg = Jjmp, R
                b.s1, b.s2 = t, None
                any_ = True
    return any_


def _constphi(fn, b):
    p = b.phi
    while p:
        if KBASE(p.cls) == 0 and any(rtype(a) == RCon for a in p.arg):
            return True
        p = p.link
    return False


def adce(fn):
    """Everything reached from what must happen, from nothing else: values in cycles of phis that only feed
    themselves go. Then a branch whose two ways are now the same goes, and what its condition needed."""
    changed = False
    for _ in range(4):
        if not _sweep(fn) and not _diamonds(fn) and not _deadloops(fn):
            break
        changed = True
    return changed


def _sweep(fn):
    phis = {}
    live = set()
    todo = []

    def need(r):
        if rtype(r) == RTmp and rval(r) >= D.Tmp0 and rval(r) not in live:
            live.add(rval(r))
            todo.append(rval(r))
    count = 0
    for b in fn.rpo[:fn.nblk]:
        p = b.phi
        while p:
            phis[rval(p.to)] = p
            count += 1
            p = p.link
        for i in b.ins[:b.nins]:
            if rtype(i.to) != RTmp or rval(i.to) < D.Tmp0 or (pinned(i) and not D.isload(i.op)):
                need(i.arg[0])
                need(i.arg[1])
                if rtype(i.to) == RTmp:
                    need(i.to)
            else:
                count += 1
        need(b.jmp.arg)
    while todo:
        t = todo.pop()
        p = phis.get(t)
        if p is not None:
            for a in p.arg:
                need(a)
            continue
        d = fn.tmp[t].def_
        if d is not None:
            need(d.arg[0])
            need(d.arg[1])
    removed = False
    for b in fn.rpo[:fn.nblk]:
        p, last = b.phi, None
        while p:
            if rval(p.to) not in live:
                removed = True
                if last is None:
                    b.phi = p.link
                else:
                    last.link = p.link
            else:
                last = p
            p = p.link
        for i in b.ins[:b.nins]:
            if rtype(i.to) == RTmp and rval(i.to) >= D.Tmp0 and rval(i.to) not in live \
                    and (not pinned(i) or D.isload(i.op)):
                i.set(D.Ins(D.Onop))
                removed = True
    if removed:
        from .ssa import filluse
        filluse(fn)
    return removed


def _deadloops(fn):
    """LLVM's LoopDeletion: a loop that does nothing anyone sees, with one way out, is that way out. It must
    count (C lets a compiler assume a loop ends when its test isn't a constant; `for (;;) if (x) break;` can
    run forever, and the IR has lost which kind it was)."""
    from .cfg import dom, fillcfg, filldom
    from .ssa import filluse
    filldom(fn)
    for h in fn.rpo[:fn.nblk]:
        back = [p for p in h.pred[:h.npred] if dom(h, p)]
        if not back:
            continue
        body = {h}
        todo = list(back)
        while todo and len(body) < 64:
            b = todo.pop()
            if b not in body:
                body.add(b)
                todo += b.pred[:b.npred]
        if todo:
            continue
        ok = True
        exit_ = None
        tests = []
        ids = set(x.id for x in body)
        for b in body:
            if any(pinned(i) and not D.isload(i.op) or (rtype(i.to) != RTmp and i.op not in (D.Onop, D.Odbgloc))
                   for i in b.ins[:b.nins]) or b.jmp.type not in (Jjmp, Jjnz):
                ok = False
                break
            for s in (b.s1, b.s2):
                if s is not None and s not in body:
                    if exit_ is not None and s is not exit_:
                        ok = False
                    exit_ = s
                    tests.append(b)
        if not ok or exit_ is None:
            continue
        phis = []
        for b in body:
            p = b.phi
            while p:
                phis.append(p)
                p = p.link
        if not any(_counted(fn, b.jmp.arg, phis) for b in tests):
            continue
        for b in body:  # nothing computed here is used elsewhere
            defs = [i.to for i in b.ins[:b.nins] if rtype(i.to) == RTmp]
            p = b.phi
            while p:
                defs.append(p.to)
                p = p.link
            for d in defs:
                t = fn.tmp[rval(d)]
                if any(u.bid not in ids for u in t.use[:t.nuse]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        p = exit_.phi
        while p:  # one value from all of the loop's ways out
            vals = {a for a, f in zip(p.arg, p.blk) if f in body}
            if len(vals) != 1:
                ok = False
                break
            p = p.link
        if not ok:
            continue
        p = exit_.phi
        while p:
            kept, seen = [], False
            for a, f in zip(p.arg, p.blk):
                if f in body:
                    if seen:
                        continue
                    seen = True
                    f = h
                kept.append((a, f))
            p.arg = [a for a, _ in kept]
            p.blk = [f for _, f in kept]
            p.narg = len(kept)
            p = p.link
        h.phi = None
        h.ins = []
        h.nins = 0
        h.jmp.type, h.jmp.arg = Jjmp, R
        h.s1, h.s2 = exit_, None
        fillcfg(fn)
        filluse(fn)
        return True
    return False


def _counted(fn, r, phis):
    """r compares a counter of the loop, a phi that adds a constant to itself each time round."""
    if rtype(r) != RTmp or rval(r) < D.Tmp0 or fn.tmp[rval(r)].def_ is None:
        return False
    d = fn.tmp[rval(r)].def_
    if iscmp(d.op) is None:
        return False
    for a in d.arg:
        if rtype(a) != RTmp or rval(a) < D.Tmp0:
            continue
        x = fn.tmp[rval(a)].def_
        if x is not None and x.op in (D.Oadd, D.Osub) and rtype(x.arg[1]) == RCon:
            a = x.arg[0]
        for p in phis:
            if p.to == a:
                for v in p.arg:
                    y = fn.tmp[rval(v)].def_ if rtype(v) == RTmp else None
                    if y is not None and y.op in (D.Oadd, D.Osub) and y.arg[0] == a and rtype(y.arg[1]) == RCon:
                        return True
    return False


def _empty(b):
    return b.phi is None and b.jmp.type == Jjmp and all(i.op in (D.Onop, D.Odbgloc) for i in b.ins[:b.nins])


def _diamonds(fn):
    """if (c) {} else {}: the same either way."""
    any_ = False
    for b in fn.rpo[:fn.nblk]:
        if b.jmp.type != Jjnz or b.s1 is b.s2:
            continue
        s1, s2 = b.s1, b.s2
        t1 = s1.s1 if _empty(s1) and s1.npred == 1 else s1
        t2 = s2.s1 if _empty(s2) and s2.npred == 1 else s2
        if t1 is not t2:
            continue
        same = True
        p = t1.phi
        while p:
            v1 = v2 = None
            for a, f in zip(p.arg, p.blk):
                if f is (s1 if t1 is not s1 else b):
                    v1 = a
                if f is (s2 if t2 is not s2 else b):
                    v2 = a
            if v1 is None or v2 is None or v1 != v2:
                same = False
                break
            p = p.link
        if not same:
            continue
        p = t1.phi
        while p:  # one entry from b, whichever way it came
            pairs = [(a, b if f is s1 or f is s2 else f) for a, f in zip(p.arg, p.blk)]
            seen = set()
            kept = []
            for a, f in pairs:
                if f is b:
                    if b in seen:
                        continue
                    seen.add(b)
                kept.append((a, f))
            p.arg = [a for a, _ in kept]
            p.blk = [f for _, f in kept]
            p.narg = len(kept)
            p = p.link
        b.jmp.type, b.jmp.arg = Jjmp, R
        b.s1, b.s2 = t1, None
        any_ = True
    if any_:
        from .cfg import fillcfg
        from .ssa import filluse
        fillcfg(fn)
        filluse(fn)
    return any_
