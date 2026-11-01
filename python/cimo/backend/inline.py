"""cimo: inlining of small functions (ext.inline; not in QBE).

Runs on the parsed functions of a module before abi0, so the IR is still target independent
and not yet in SSA form (the later passes accept temporaries defined more than once). A
function is copied into its callers when it is small: at most ext.inline instructions, not
counting parameters, stack slots and debug locations. Callees are processed before their
callers, so a helper's own inlined helpers come along.

    expand_all(fns, limit)  the whole module (main._Driver.finish)
    snapshot(fn, limit)     a copy of fn's body to inline later, or None if it can't be
    expand(fn, bodies)      inline the calls of fn to the functions in bodies {symbol id: Body}

The callee keeps its own definition (it may be called from Python, through a pointer, or
from a caller it is too big for). Its stack slots move to the caller's start block, where they
stay static allocations that mem.promote turns into temporaries. Calls are inlined only when
the arguments match the parameters one for one and the result class matches the return
class; aggregate parameters and returns, variadic functions, dynamic allocations, large stack
slots and calls of functions that return twice (setjmp) keep the call.
"""

from . import defs as D
from .defs import R, Kw, RTmp, RCon, Tmp0, CAddr, CON_Z, rtype, rval
from .util import newtmp, newcon, str_
from . import cfg

# callees after which a function may resume twice: never copied into another frame
_RETURNS_TWICE = frozenset(('setjmp', '_setjmp', '__setjmp', 'sigsetjmp', '__sigsetjmp',
                            'savectx', 'vfork', 'getcontext'))
# a caller stops taking inlined code once it has this many instructions
CALLER_LIMIT = 4000
# a callee's stack slots join the caller's frame, once per inlined call: functions with more
# bytes of them than this keep their own frame, so inlining can't overflow a stack
SLOTS_LIMIT = 256
_EXT = {D.Oparsb: D.Oextsb, D.Oparub: D.Oextub, D.Oparsh: D.Oextsh, D.Oparuh: D.Oextuh}
_RETEXT = {D.Jretsb: D.Oextsb, D.Jretub: D.Oextub, D.Jretsh: D.Oextsh, D.Jretuh: D.Oextuh}


class Body:
    """A function's body, copied: blocks [(name, phis, ins, jmp type, jmp arg, s1, s2)] with
    successors as indices (phis [(to, cls, args, block indices)]), its temporaries
    [(name, cls)] from Tmp0 on, constants, parameters [(op, cls, to)], the return value's
    class (None: void) and the instruction count."""
    __slots__ = ('name', 'blocks', 'tmps', 'cons', 'pars', 'retk', 'size')


def _symname(fn, r):
    """The global symbol a call goes to, or None (a pointer, an offset, a thread local). A
    call before the definition names the symbol `extern`; only the module's own functions are
    ever inlined, so that is the same function."""
    if rtype(r) != RCon:
        return None
    c = fn.con[rval(r)]
    if c.type != CAddr or c.i != 0 or c.sym.type not in (D.SGlo, D.SExt):
        return None
    return c.sym.id


def _count(fn):
    n = 0
    b = fn.start
    while b:
        n += len(b.ins)
        b = b.link
    return n


def snapshot(fn, limit):
    if fn.vararg or fn.dynalloc or fn.retty != D.Kx:
        return None
    index = {}
    b = fn.start
    while b:
        index[b] = len(index)
        b = b.link
    size = slots = 0
    retk = None
    pars = []
    blocks = []
    b = fn.start
    while b:
        start = b is fn.start
        for k, i in enumerate(b.ins):
            o = i.op
            if D.ispar(o):
                if not start or k != len(pars) or o in (D.Oparc, D.Opare):
                    return None
                pars.append((o, i.cls, i.to))
            elif D.isalloc(o):
                if not start or rtype(i.arg[0]) != RCon:
                    return None
                slots += fn.con[rval(i.arg[0])].i
                if slots > SLOTS_LIMIT:
                    return None
            elif o == D.Odbgloc:
                pass
            elif o in (D.Ovastart, D.Ogetsp, D.Osetsp, D.Oargc, D.Oarge):
                return None
            else:
                if o == D.Ocall:
                    s = _symname(fn, i.arg[0])
                    if s is not None and str_(s) in _RETURNS_TWICE:
                        return None
                size += 1
        j = b.jmp.type
        if D.isret(j):
            if j == D.Jretc:
                return None
            k = None if j == D.Jret0 else Kw if D.isretbh(j) else j - D.Jretw
            if k is not None:
                retk = k
        if size > limit:
            return None
        phis = []
        p = b.phi
        while p:
            phis.append((p.to, p.cls, list(p.arg), [index[x] for x in p.blk]))
            p = p.link
        blocks.append((b.name, phis, [i.copy() for i in b.ins], j, b.jmp.arg,
                       index[b.s1] if b.s1 else None, index[b.s2] if b.s2 else None))
        b = b.link
    body = Body()
    body.name = fn.name
    body.blocks = blocks
    body.tmps = [(t.name, t.cls) for t in fn.tmp[Tmp0:fn.ntmp]]
    body.cons = [c.copy() for c in fn.con[:fn.ncon]]
    body.pars = pars
    body.retk = retk
    body.size = size
    return body


def _args(b, j, body):
    """The index of the first argument instruction of the call b.ins[j] if its arguments and
    result fit body's parameters and return, else None."""
    k = j
    while k > 0 and D.isarg(b.ins[k - 1].op):
        k -= 1
    args = b.ins[k:j]
    if len(args) != len(body.pars):
        return None
    for a, (po, pk, _) in zip(args, body.pars):
        if a.op - D.Oarg != po - D.Opar or a.cls != pk:
            return None
    call = b.ins[j]
    if call.arg[1] != R:
        return None
    if call.to != R and body.retk is not None and call.cls != body.retk:
        return None
    return k


_serial = 0


def _expand1(fn, b, k, j, body):
    """Replace the call b.ins[j] (arguments from b.ins[k]) by a copy of body; returns the
    block holding the instructions after the call."""
    global _serial
    _serial += 1
    tag = '%s.i%d' % (body.name, _serial)
    call = b.ins[j]
    args = b.ins[k:j]
    tmap = []
    for name, cls in body.tmps:
        r = newtmp(None, cls, fn)
        fn.tmp[rval(r)].name = '%s.%s' % (name, tag) if name else None
        tmap.append(r)
    cmap = {}

    def ref(r):
        t = rtype(r)
        if t == RTmp:
            v = rval(r)
            return r if v < Tmp0 else tmap[v - Tmp0]
        if t == RCon:
            v = rval(r)
            c = cmap.get(v)
            if c is None:
                c = cmap[v] = newcon(body.cons[v], fn)
            return c
        return r

    blks = []
    for name, *_ in body.blocks:
        nb = cfg.newblk()
        nb.name = '%s.%s' % (name, tag)
        nb.id = fn.nblk
        fn.nblk += 1
        blks.append(nb)
    cont = cfg.newblk()
    cont.name = 'ret.%s' % tag
    cont.id = fn.nblk
    fn.nblk += 1

    allocs = []
    rets = []
    for nb, (_, phis, ins, jt, ja, s1, s2) in zip(blks, body.blocks):
        out = []
        for i in ins:
            o = i.op
            if D.ispar(o):
                a = args[_parno(body, i)]
                x = D.Ins(_EXT.get(o, D.Ocopy), i.cls, ref(i.to), a.arg[0], R)
                out.append(x)
                continue
            x = D.Ins(o, i.cls, ref(i.to), ref(i.arg[0]), ref(i.arg[1]))
            if D.isalloc(o):
                allocs.append(x)
            else:
                out.append(x)
        p = None
        for to, cls, pa, pb in phis:
            q = D.Phi()
            q.to = ref(to)
            q.cls = cls
            q.arg = [ref(a) for a in pa]
            q.blk = [blks[x] for x in pb]
            q.narg = len(q.arg)
            if p is None:
                nb.phi = q
            else:
                p.link = q
            p = q
        if D.isret(jt):
            if call.to != R:
                # each return's value in a temporary of its own, joined by a phi in cont
                # (the result may feed a phi, and ssa leaves those temporaries alone)
                if jt == D.Jret0:
                    v = CON_Z
                elif jt in _RETEXT:
                    v = newtmp('inl', call.cls, fn)
                    out.append(D.Ins(_RETEXT[jt], call.cls, v, ref(ja), R))
                else:
                    v = ref(ja)
                rets.append((nb, v))
            nb.jmp.type = D.Jjmp
            nb.s1 = cont
        else:
            nb.jmp.type = jt
            nb.jmp.arg = ref(ja)
            nb.s1 = blks[s1] if s1 is not None else None
            nb.s2 = blks[s2] if s2 is not None else None
        nb.ins = out
        nb.nins = len(out)

    if rets:
        q = D.Phi()
        q.to = call.to
        q.cls = call.cls
        q.arg = [v for _, v in rets]
        q.blk = [x for x, _ in rets]
        q.narg = len(rets)
        cont.phi = q
    # the instructions after the call, and the jump, move to cont
    cont.ins = b.ins[j + 1:]
    cont.nins = len(cont.ins)
    cont.jmp.type = b.jmp.type
    cont.jmp.arg = b.jmp.arg
    cont.s1 = b.s1
    cont.s2 = b.s2
    for s in (b.s1, b.s2):
        p = s.phi if s else None
        while p:
            p.blk = [cont if x is b else x for x in p.blk]
            p = p.link
    b.ins = b.ins[:k]
    b.nins = len(b.ins)
    b.jmp.type = D.Jjmp
    b.jmp.arg = R
    b.s1 = blks[0]
    b.s2 = None
    cont.link = b.link
    prev = b
    for nb in blks:
        prev.link = nb
        prev = nb
    prev.link = cont

    # stack slots: static allocations in the caller's start block, after its parameters
    st = fn.start
    n = 0
    while n < len(st.ins) and D.ispar(st.ins[n].op):
        n += 1
    st.ins[n:n] = allocs
    st.nins = len(st.ins)
    return cont


def _parno(body, i):
    for n, (_, _, to) in enumerate(body.pars):
        if to == i.to:
            return n
    raise AssertionError('parameter not found')


def expand(fn, bodies):
    """Inline the calls of fn to the functions in bodies; returns the number inlined."""
    if not bodies:
        return 0
    size = _count(fn)
    n = 0
    b = fn.start
    while b:
        j = 0
        while j < len(b.ins):
            i = b.ins[j]
            if i.op == D.Ocall:
                s = _symname(fn, i.arg[0])
                body = bodies.get(s) if s is not None else None
                if body is not None and body.name != fn.name and size + body.size <= CALLER_LIMIT:
                    k = _args(b, j, body)
                    if k is not None:
                        b = _expand1(fn, b, k, j, body)
                        size += body.size
                        n += 1
                        j = 0
                        continue
            j += 1
        b = b.link
    if n:
        fn.rpo = [None] * fn.nblk
    return n


def _callees(fn):
    out = []
    b = fn.start
    while b:
        for i in b.ins:
            if i.op == D.Ocall:
                s = _symname(fn, i.arg[0])
                if s is not None:
                    out.append(s)
        b = b.link
    return out


def expand_all(fns, limit):
    """Inline small functions of the module fns into their callers, callees first (a cycle of
    calls is broken at an arbitrary function: its call stays a call)."""
    from .util import intern
    byid = {intern(fn.name): fn for fn in fns}
    order = []
    state = {}
    for root in fns:  # post-order over the call graph, iteratively
        rid = intern(root.name)
        if rid in state:
            continue
        state[rid] = 1
        stack = [(root, iter(_callees(root)))]
        while stack:
            fn, it = stack[-1]
            for s in it:
                if s in byid and s not in state:
                    state[s] = 1
                    stack.append((byid[s], iter(_callees(byid[s]))))
                    break
            else:
                stack.pop()
                order.append(fn)
    bodies = {}
    for fn in order:
        expand(fn, bodies)
        body = snapshot(fn, limit)
        if body is not None:
            bodies[intern(fn.name)] = body
