"""cimo: inlining of small functions (ext.inline; not in QBE).

Runs on the parsed functions of a module before abi0, so the IR is still target independent
and not yet in SSA form (the later passes accept temporaries defined more than once). A
function is copied into its callers when it is small: at most ext.inline instructions, not
counting what the cleanup after inlining removes (LLVM's InlineCost does the same): parameters,
stack slots, debug locations, loads and stores of slots only ever loaded and stored (promoted to
temporaries), their constant offsets, and copies and extensions of constants. Callees are processed before their
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
from .defs import R, Kw, Kl, RTmp, RCon, Tmp0, CAddr, CON_Z, rtype, rval
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


def _free(fn):
    """The instructions of fn that cost nothing once inlined and cleaned up: loads and stores
    through a stack slot (or a constant offset into one) whose address goes nowhere else, as
    mem.promote and sroa turn those into temporaries; the offsets themselves; and copies and
    extensions of constants, which fold. Returns their ids."""
    slots = set()
    b = fn.start
    for i in b.ins if b else ():
        if D.isalloc(i.op) and rtype(i.to) == RTmp:
            slots.add(i.to)
    derived = {}  # offset temporary -> its slot
    if slots:
        grew = True
        while grew:
            grew = False
            b = fn.start
            while b:
                for i in b.ins:
                    if i.op == D.Oadd and rtype(i.to) == RTmp and i.to not in derived and i.to not in slots:
                        for a, c in ((i.arg[0], i.arg[1]), (i.arg[1], i.arg[0])):
                            base = a if a in slots else derived.get(a)
                            if base is not None and rtype(c) == RCon and fn.con[rval(c)].type != CAddr:
                                derived[i.to] = base
                                grew = True
                                break
                b = b.link
    escaped = set()
    if slots:
        def use(r):
            base = r if r in slots else derived.get(r)
            if base is not None:
                escaped.add(base)
        b = fn.start
        while b:
            p = b.phi
            while p:
                for a in p.arg:
                    use(a)
                p = p.link
            for i in b.ins:
                o = i.op
                if D.isalloc(o) or D.ispar(o) or o == D.Odbgloc:
                    continue
                if D.isload(o):
                    continue                    # arg[0] is the address
                if D.isstore(o):
                    use(i.arg[0])               # the value stored; arg[1] is the address
                    continue
                if o == D.Oadd and i.to in derived:
                    continue
                use(i.arg[0])
                use(i.arg[1])
            use(b.jmp.arg)
            b = b.link
    free = set()
    b = fn.start
    while b:
        for i in b.ins:
            o = i.op
            if D.isload(o):
                base = i.arg[0] if i.arg[0] in slots else derived.get(i.arg[0])
                if base is not None and base not in escaped:
                    free.add(id(i))
            elif D.isstore(o):
                base = i.arg[1] if i.arg[1] in slots else derived.get(i.arg[1])
                if base is not None and base not in escaped:
                    free.add(id(i))
            elif o == D.Oadd and i.to in derived and derived[i.to] not in escaped:
                free.add(id(i))
            elif o in (D.Ocopy, D.Oextsw, D.Oextuw) and rtype(i.arg[0]) == RCon or o == D.Ofma0:
                free.add(id(i))                 # fma0 only carries fma1's addend
        b = b.link
    return free


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
    free = _free(fn)
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
                if id(i) not in free:
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
            x.nsw = i.nsw
            x.contract = i.contract
            x.reassoc = i.reassoc
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


def cost(fn):
    """fn's size as snapshot counts it: what inlining copies and the cleanup keeps"""
    free = _free(fn)
    n = 0
    b = fn.start
    while b:
        for i in b.ins:
            if not (D.ispar(i.op) or D.isalloc(i.op) or i.op == D.Odbgloc or id(i) in free):
                n += 1
        b = b.link
    return n


def expand(fn, bodies, cap=None, once=None):
    """Inline the calls of fn to the functions in bodies; returns the number inlined. once:
    {symbol: its one caller's symbol} for the bodies taken at any size because one call refers to
    them; they inline only there, never with a copy of that caller. cap: fn's own inline limit
    when fn is small enough to be inlined itself; a once body that would take fn past it stays a
    call, so that a cold path doesn't keep a hot function out of its callers (sj_truediv and its
    long division)."""
    from .util import intern
    once = once or {}
    me = intern(fn.name)
    if not bodies:
        return 0
    size = _count(fn)
    small = None
    if cap is not None and once:
        small = cost(fn)
        if small > cap:
            small = None
    n = 0
    b = fn.start
    while b:
        j = 0
        while j < len(b.ins):
            i = b.ins[j]
            if i.op == D.Ocall:
                s = _symname(fn, i.arg[0])
                body = bodies.get(s) if s is not None else None
                if body is not None and body.name != fn.name and size + body.size <= CALLER_LIMIT and \
                        not (s in once and (once[s] != me or small is not None and small + body.size > cap)):
                    k = _args(b, j, body)
                    if k is not None:
                        b = _expand1(fn, b, k, j, body)
                        size += body.size
                        if small is not None:
                            small += body.size
                        n += 1
                        j = 0
                        continue
            j += 1
        b = b.link
    if n:
        fn.rpo = [None] * fn.nblk
    return n


def _noescape(fn):
    """No stack slot's address can outlive a call: each is only loaded from or stored through
    directly (before promotion, how cproc reads and writes locals). A self call that becomes a jump
    reuses the slots, so a pointer to one must not reach the callee (LLVM's TailCallElim: AllCallsAreTailCalls)."""
    slots = set()
    b = fn.start
    while b:
        for i in b.ins:
            if D.isalloc(i.op):
                slots.add(i.to)
        b = b.link
    if not slots:
        return True
    b = fn.start
    while b:
        p = b.phi
        while p:
            if any(a in slots for a in p.arg[:p.narg]):
                return False
            p = p.link
        if b.jmp.arg in slots:
            return False
        for i in b.ins:
            if D.isalloc(i.op):
                continue
            if D.isload(i.op):
                if i.arg[1] in slots:
                    return False
            elif D.isstore(i.op):
                if i.arg[0] in slots:
                    return False
            elif i.arg[0] in slots or i.arg[1] in slots:
                return False
        b = b.link
    return True


def selftail(fn):
    """LLVM's TailCallElim: `return f(args)` inside f is a jump to f's start, with the parameters
    assigned, so it runs in constant stack and the loop passes see a loop. Returns True if changed."""
    from .util import intern
    if fn.vararg or fn.dynalloc or fn.retty != D.Kx:
        return False
    me = intern(fn.name)
    st = fn.start
    pars = []
    for i in st.ins:
        if not D.ispar(i.op):
            break
        if i.op in (D.Oparc, D.Opare):
            return False
        pars.append(i)
    sites = []
    b = fn.start
    while b:
        j = len(b.ins) - 1
        while j >= 0 and b.ins[j].op == D.Odbgloc:
            j -= 1
        if j >= 0 and b.ins[j].op == D.Ocall and _symname(fn, b.ins[j].arg[0]) == me \
                and b.ins[j].arg[1] == R and D.isret(b.jmp.type):
            call = b.ins[j]
            t = b.jmp.type
            if (t == D.Jret0 or (D.Jretw <= t <= D.Jretd and call.to == b.jmp.arg and call.to != R
                                 and call.cls == t - D.Jretw)):
                k = j
                while k > 0 and D.isarg(b.ins[k - 1].op):
                    k -= 1
                args = b.ins[k:j]
                if len(args) == len(pars) and all(a.op - D.Oarg == p.op - D.Opar and a.cls == p.cls
                                                  for a, p in zip(args, pars)):
                    sites.append((b, k, args))
        b = b.link
    if not sites or not _noescape(fn):
        return False
    # a parameter that a phi reads stays out of ssa's reach when it has several definitions
    ptmps = {p.to for p in pars}
    b = fn.start
    while b:
        p = b.phi
        while p:
            if p.to in ptmps or any(a in ptmps for a in p.arg[:p.narg]):
                return False
            p = p.link
        b = b.link
    # the start block keeps the parameters and slots; the rest moves to the block jumped back to
    keep = [i for i in st.ins if D.ispar(i.op) or D.isalloc(i.op)]
    if any(D.isalloc(i.op) and rtype(i.arg[0]) != RCon for i in st.ins):
        return False
    head = cfg.newblk()
    head.name = 'tail.%s' % fn.name
    head.id = fn.nblk
    fn.nblk += 1
    head.ins = [i for i in st.ins if not (D.ispar(i.op) or D.isalloc(i.op))]
    head.nins = len(head.ins)
    head.jmp.type = st.jmp.type
    head.jmp.arg = st.jmp.arg
    head.s1 = st.s1
    head.s2 = st.s2
    head.link = st.link
    for s in (st.s1, st.s2):
        p = s.phi if s else None
        while p:
            p.blk = [head if x is st else x for x in p.blk]
            p = p.link
    st.ins = keep
    st.nins = len(keep)
    st.jmp.type = D.Jjmp
    st.jmp.arg = R
    st.s1 = head
    st.s2 = None
    st.link = head
    for b, k, args in sites:
        if b is st:
            b = head
            k = k - len(keep)
        # in parallel: an argument may read a parameter assigned just before it
        out = b.ins[:k]
        temps = []
        for a, p in zip(args, pars):
            t = newtmp('tail', p.cls, fn)
            out.append(D.Ins(_EXT.get(p.op, D.Ocopy), p.cls, t, a.arg[0], R))
            temps.append(t)
        for t, p in zip(temps, pars):
            out.append(D.Ins(D.Ocopy, p.cls, p.to, t, R))
        b.ins = out
        b.nins = len(out)
        b.jmp.type = D.Jjmp
        b.jmp.arg = R
        b.s1 = head
        b.s2 = None
    fn.rpo = [None] * fn.nblk
    return True


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


# the C library's math: no errno here, so they read and write no memory (requirements.md)
_LIBM = """sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh exp exp2 expm1 log log2 log10 log1p
    pow sqrt cbrt hypot fmod remainder floor ceil trunc round rint nearbyint fabs copysign fmin fmax fdim fma
    erf erfc tgamma ldexp scalbn logb""".split()
_LIBM = frozenset(_LIBM + [n + 'f' for n in _LIBM])
# C library functions that only read (LLVM's TargetLibraryInfo): loads survive a call of them
_READONLY = frozenset("""strlen strnlen strcmp strncmp strcasecmp strncasecmp strchr strrchr strstr strspn strcspn
    strpbrk memcmp memchr abs labs llabs isalpha isdigit isalnum isspace isupper islower toupper tolower""".split())
ONCE_LIMIT = 800  # LLVM's last-call bonus: a static function called once goes away when inlined
# a function declared `inline` may be this many times ext.inline (LLVM's inlinehint-threshold is 1.44x its
# default, gcc's max-inline-insns-single at -O2 about 4.7x max-inline-insns-auto)
HINT = 2


def _symrefs(fn):
    """{symbol id: times called directly}, with every other reference counted as many."""
    out = {}

    def ref(r, n):
        if rtype(r) == RCon:
            c = fn.con[rval(r)]
            if c.type == CAddr:
                out[c.sym.id] = out.get(c.sym.id, 0) + n
    b = fn.start
    while b:
        p = b.phi
        while p:
            for a in p.arg[:p.narg]:
                ref(a, 1000)
            p = p.link
        ref(b.jmp.arg, 1000)
        for i in b.ins:
            if i.op == D.Ocall:
                ref(i.arg[0], 1)
                ref(i.arg[1], 1000)
            else:
                ref(i.arg[0], 1000)
                ref(i.arg[1], 1000)
        b = b.link
    return out


def _writesmemory(fn, nowrite):
    """fn may write memory that anything else can see: LLVM's FunctionAttrs, for readonly. Stores to its
    own stack slots don't count (nothing promoted them yet). Volatile reads count, as they must happen."""
    own = set()
    b = fn.start
    while b:
        for i in b.ins:
            o = i.op
            if D.isalloc(o):
                own.add(i.to)
            elif o == D.Oadd and i.arg[0] in own and rtype(i.arg[1]) == RCon:
                own.add(i.to)
            elif D.isstore(o):
                if i.arg[1] not in own:
                    return True
            elif D.isvload(o) or D.isvstore(o) or o in (D.Oblit0, D.Oblit1, D.Ovastart, D.Ovaarg, D.Osalloc):
                return True
            elif o == D.Ocall:
                r = i.arg[0]
                c = fn.con[rval(r)] if rtype(r) == RCon else None
                if c is None or c.type != CAddr or c.i != 0 or c.sym.id not in nowrite:
                    return True
        b = b.link
    return False


def _unquote(s):
    """The bytes of a string as the IR text has it, in quotes with octal escapes; or None."""
    if isinstance(s, (bytes, bytearray)):
        s = bytes(s).decode('latin-1')
    if len(s) < 2 or s[0] != '"' or s[-1] != '"':
        return None
    out = bytearray()
    i = 1
    n = len(s) - 1
    while i < n:
        c = s[i]
        if c != '\\':
            out.append(ord(c))
            i += 1
            continue
        i += 1
        if i < n and s[i] in '01234567':
            j = i
            while j < n and j < i + 3 and s[j] in '01234567':
                j += 1
            out.append(int(s[i:j], 8) & 255)
            i = j
        elif i < n and s[i] in '\\"':
            out.append(ord(s[i]))
            i += 1
        else:
            return None  # an escape this doesn't know: leave the data alone
    return out


def recorddata(items, d):
    """Collect d, an item of the data stream: static data of plain numbers, as bytes, by name. None for
    what can't be read back that way."""
    if d.type == D.DStart:
        bad = d.lnk.export or d.lnk.thread or d.lnk.common or d.lnk.sec
        items[d.name] = None if bad else bytearray()
        items['#cur'] = d.name
        return
    if d.type == D.DEnd:
        return
    cur = items.get('#cur')
    b = items.get(cur)
    if b is None:
        return
    if d.isref:
        items[cur] = None
    elif d.type == D.DZ:
        b += bytes(d.num)
    elif d.isstr:
        v = _unquote(d.str)
        if v is None:
            items[cur] = None
        else:
            b += v
    else:
        n = {D.DB: 1, D.DH: 2, D.DW: 4, D.DL: 8}[d.type]
        b += (d.num & ((1 << (8 * n)) - 1)).to_bytes(n, 'little')


def constglobals(fns, items, datarefs):
    """LLVM's GlobalOpt: static data only ever loaded from keeps its initial value. Each use must be the
    address of a load, directly or through an add whose only uses are loads' addresses; anything else
    might store or lets the address go. Leaves {symbol id: bytes} for simplify, which folds the loads."""
    from . import simplify
    from .util import intern
    names = {n: b for n, b in items.items() if b is not None and n != '#cur' and n not in datarefs}
    bad = set()
    for fn in fns:
        addtmp = {}
        useful = {}

        def sym(r):
            if rtype(r) != RCon:
                return None
            c = fn.con[rval(r)]
            return c.sym.id if c.type == CAddr else None
        b = fn.start
        while b:
            p = b.phi
            while p:
                for a in p.arg[:p.narg]:
                    if sym(a) is not None:
                        bad.add(sym(a))
                    if a in addtmp or rtype(a) == RTmp:
                        useful.setdefault(a, []).append(None)
                p = p.link
            if sym(b.jmp.arg) is not None:
                bad.add(sym(b.jmp.arg))
            if rtype(b.jmp.arg) == RTmp:
                useful.setdefault(b.jmp.arg, []).append(None)
            for i in b.ins:
                for n in (0, 1):
                    a = i.arg[n]
                    s = sym(a)
                    if s is not None:
                        if D.isload(i.op) and n == 0:
                            continue
                        if i.op == D.Oadd and i.cls == Kl:
                            addtmp[i.to] = s
                            continue
                        if i.op == D.Ocall and n == 0:
                            continue  # calling it: code, not data
                        bad.add(s)
                    elif rtype(a) == RTmp:
                        useful.setdefault(a, []).append(n == 0 and D.isload(i.op))
                if i.to in addtmp and i.op != D.Oadd:
                    bad.add(addtmp[i.to])
            b = b.link
        for t, s in addtmp.items():
            if not all(useful.get(t, [])):
                bad.add(s)
    simplify.CONSTANTS = {intern(n): bytes(b) for n, b in names.items() if intern(n) not in bad}


def _plaincalls(fn):
    """(block, position of the call, [its arg instructions]) for every call; plain args or not."""
    b = fn.start
    while b:
        args = []
        for n, i in enumerate(b.ins):
            if D.isarg(i.op):
                args.append(i)
            elif i.op == D.Ocall:
                yield b, n, i, args
                args = []
            elif i.op not in (D.Onop, D.Odbgloc):
                args = []
        b = b.link


def _params(fn):
    """fn's parameter instructions, if all are plain."""
    out = []
    b = fn.start
    while b:
        for i in b.ins:
            if i.op == D.Ovastart or (D.ispar(i.op) and i.op != D.Opar):
                return None
            if i.op == D.Opar:
                out.append(i)
        b = b.link
    return out


def _deadparam(fn, par, uses, slots):
    """The parameter is never read: no uses, or only stored to a slot nothing else touches. Returns the
    instructions that go with it, or None."""
    n = uses.get(par.to, 0)
    if n == 0:
        return [par]
    if n != 1:
        return None
    b = fn.start
    while b:
        for i in b.ins:
            if D.isstore(i.op) and i.arg[0] == par.to:
                return [par, i] if i.arg[1] in slots and uses.get(i.arg[1], 0) == 1 else None
        b = b.link
    return None


def _inlinehint():
    """cimo: the functions declared `inline` (the frontend's paraminfo, through the middle end's params)"""
    from .. import middle
    return middle.params.get('#inline', ()) if middle.params else ()


def _noinline():
    """cimo: the functions declared noinline or noipa (the frontend's paraminfo, through the middle end's params):
    their calls stay calls, their parameters and results unknown to the callers"""
    from .. import middle
    return middle.params.get('#noinline', ()) if middle.params else ()


def ipo(fns, datarefs):
    """LLVM's IPSCCP and DeadArgumentElimination, on static functions whose every use is a direct call: a
    parameter that all calls give the same constant is that constant, one nothing reads goes, and a result
    that every return gives as the same constant is known at the calls."""
    from .util import getcon, intern
    byid = {intern(fn.name): fn for fn in fns}
    refs = {}
    for fn in fns:
        for s, n in _symrefs(fn).items():
            refs[s] = refs.get(s, 0) + n
    data = {intern(n) for n in datarefs}
    calls = {}
    for fn in fns:
        for b, n, i, args in _plaincalls(fn):
            r = i.arg[0]
            c = fn.con[rval(r)] if rtype(r) == RCon else None
            if c is not None and c.type == CAddr and c.i == 0 and c.sym.id in byid:
                calls.setdefault(c.sym.id, []).append((fn, b, i, args))
    for s, fn in byid.items():
        sites = calls.get(s)
        if fn.lnk.export or s in data or not sites or refs.get(s, 0) != len(sites) or fn.name in _noinline():
            continue
        pars = _params(fn)
        if pars is None or fn.vararg or any(len(a) != len(pars) or any(x.op != D.Oarg for x in a) for _, _, _, a in sites):
            continue
        uses = {}
        slots = set()
        b = fn.start
        while b:
            p = b.phi
            while p:
                for a in p.arg[:p.narg]:
                    uses[a] = uses.get(a, 0) + 1
                p = p.link
            for i in b.ins:
                if D.isalloc(i.op):
                    slots.add(i.to)
                for a in i.arg:
                    uses[a] = uses.get(a, 0) + 1
            uses[b.jmp.arg] = uses.get(b.jmp.arg, 0) + 1
            b = b.link
        gone = []
        for n, par in enumerate(pars):
            dead = _deadparam(fn, par, uses, slots)
            if dead is not None:
                for i in dead:
                    i.set(D.Ins(D.Onop))
                gone.append(n)
                continue
            vals = set()
            for f, _, _, a in sites:
                r = a[n].arg[0]
                c = f.con[rval(r)] if rtype(r) == RCon else None
                vals.add(c.i if c is not None and c.type == D.CBits else None)
            if len(vals) == 1 and None not in vals and D.KBASE(par.cls) == 0:
                par.op, par.arg[0], par.arg[1] = D.Ocopy, getcon(vals.pop(), fn), R
                gone.append(n)
        for f, cb, _, a in sites:
            for n in gone:
                a[n].set(D.Ins(D.Onop))
            cb.ins = [x for x in cb.ins if x.op != D.Onop]  # args stay next to their call
            cb.nins = len(cb.ins)
        s0 = fn.start  # parameters first, as the ABI reads them
        s0.ins = [x for x in s0.ins if D.ispar(x.op)] + [x for x in s0.ins if not D.ispar(x.op) and x.op != D.Onop]
        s0.nins = len(s0.ins)
        # the result
        v = None
        b = fn.start
        any_ret = False
        while b:
            t = b.jmp.type
            if D.isret(t):
                any_ret = True
                r = b.jmp.arg
                c = fn.con[rval(r)] if rtype(r) == RCon else None
                if t not in (D.Jretw, D.Jretl) or c is None or c.type != D.CBits or (v is not None and v != c.i):
                    v = None
                    break
                v = c.i
            b = b.link
        if v is None or not any_ret:
            continue
        for f, cb, call, _ in sites:
            if call.to != R and D.KBASE(call.cls) == 0:
                at = next(k for k, x in enumerate(cb.ins) if x is call)
                cb.ins[at + 1:at + 1] = [D.Ins(D.Ocopy, call.cls, call.to, getcon(v, f), R)]
                cb.nins = len(cb.ins)
                call.to = R


def expand_all(fns, limit, datarefs=()):
    """Inline small functions of the module fns into their callers, callees first (a cycle of
    calls is broken at an arbitrary function: its call stays a call), and any size of a static
    function called once. Returns the functions still needed: exported ones and what they, or
    data (datarefs: names), refer to."""
    from .util import intern
    from . import load
    for fn in fns:
        selftail(fn)
    byid = {intern(fn.name): fn for fn in fns}
    refs = {}
    for fn in fns:
        for s, n in _symrefs(fn).items():
            refs[s] = refs.get(s, 0) + n
    data = {intern(n) for n in datarefs}
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
    noinl = _noinline()
    hint = _inlinehint()
    nomem = {intern(n) for n in _LIBM | _READONLY if intern(n) not in byid}
    owner = {}  # symbol -> the function whose one reference it is
    for fn in fns:
        for sym in _symrefs(fn):
            if refs.get(sym) == 1:
                owner[sym] = intern(fn.name)
    oncebodies = {}
    for fn in order:
        s = intern(fn.name)
        once = not fn.lnk.export and refs.get(s, 0) == 1 and s not in data
        flimit = ONCE_LIMIT if once else limit * HINT if fn.name in hint else limit
        cap = None if once or fn.name in noinl or not refs.get(s) else flimit
        expand(fn, bodies, cap, oncebodies)
        body = None if fn.name in noinl else snapshot(fn, flimit)
        if body is not None and once and body.size > limit and s in owner:
            oncebodies[s] = owner[s]
        if body is not None:
            bodies[s] = body
        if not _writesmemory(fn, nomem):
            nomem.add(s)
    load.nomemcalls = frozenset(nomem)
    # what's still reachable (LLVM's GlobalDCE)
    work = [fn for fn in fns if fn.lnk.export or intern(fn.name) in data]
    live = {intern(fn.name) for fn in work}
    while work:
        fn = work.pop()
        for s in _symrefs(fn):
            if s in byid and s not in live:
                live.add(s)
                work.append(byid[s])
    return [fn for fn in fns if intern(fn.name) in live]
