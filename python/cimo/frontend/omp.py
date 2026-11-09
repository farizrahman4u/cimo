"""cimo: OpenMP directives, as gcc -fopenmp compiles them, for the ones cimo has.

`#pragma omp parallel for` runs a loop's iterations on a team of threads (runtime/omp.c, cimo/parallel.py). The
frontend parses the loop into the function it is in, as a loop over the part of the iterations that one thread
runs, and when the function is complete (emitfunc) moves that code into a function of its own,
`<function>._omp_fn.<n>(data, thread, team)` as gcc names them, and calls `__cimo_par(fn, data, nthreads)` in its
place. What the loop uses from the function goes in `data`, one 8-byte word each: a variable only read in the loop
and whose address the function never lets out goes by value (the loop's copy is a local of its own, which the
optimizer keeps in a register), anything else by its address, so the loop shares it.

- The loop must be OpenMP's canonical form: `for (var = lb; var OP b; incr)` with an integer var (declared there or
  before), OP one of < <= > >= != (!= with a step of 1 or -1), incr `var++`, `++var`, `var--`, `--var`,
  `var += s`, `var -= s`, `var = var + s`, `var = s + var` or `var = var - s`. lb, b and s are evaluated once, before
  the loop; the trip count is computed from them, so the loop runs as gcc's does.
- Each thread's part: `schedule(static)` (the default) gives thread t of n the t-th of n nearly equal runs of
  iterations, as libgomp does; `schedule(static, c)` chunks of c in turn; `dynamic` and `guided` take chunks from a
  counter in the team (__cimo_par_next); `auto` and `runtime` are static. Within its part a thread runs
  `for (var = first; var < end; var += s)` (> for a loop that counts down; != for an unsigned var with a step other
  than 1 or -1), so the optimizer sees an ordinary counted loop.
- The loop variable, `private`, `firstprivate`, `lastprivate` and `reduction` variables are declared again in the
  loop's scope (shadowing the originals), so the loop's code uses the thread's own; the variables the loop body
  declares are the thread's too. firstprivate copies the original in when a thread starts; lastprivate copies out
  from the thread that ran the last iteration; reductions start at the operator's identity and are combined into
  the original under a lock (+ - * & | ^ && || max min).
- `num_threads(n)` and `if(c)` set the team's size (1 when c is false); without them, omp_set_num_threads, then
  cimo.set_num_threads (OMP_NUM_THREADS, the CPU count). `#pragma omp for` outside a parallel loop is the same loop
  run by a team of one, so its clauses mean what they do under gcc.
- `critical [(name)]` and `atomic` (any form) hold a lock (one per name, one for atomic) around their statement.
  barrier, flush, taskwait, taskyield, single, master, sections, section, ordered, scope and taskgroup outside a
  parallel loop act as for a team of one (with no data-sharing clauses); inside one, the ones OpenMP doesn't
  allow there are errors. The rest of OpenMP (`parallel` without `for`, tasks, threadprivate, teams, target,
  collapse(n > 1), linear, ordered, copyin, ...) is a compile error, never ignored; `simd` and `declare` keep their
  meaning for the vectorizer (simd reductions) or none.
- Jumps into or out of the loop (goto, a case label of an enclosing switch, break, return) are errors.
"""

from .cc import (Token, TIDENT, TLPAREN, TRPAREN, TCOMMA, TCOLON, TSEMICOLON, TASSIGN, TINC, TDEC,
                 TADDASSIGN, TSUBASSIGN, TADD, TSUB, TMUL, TBAND, TBOR, TXOR, TLAND, TLOR, TLESS, TGREATER,
                 TLEQ, TGEQ, TNEQ, TSHL, TFOR, EXPRIDENT, EXPRCONST, DECLOBJECT, SDAUTO, LINKNONE, QUALCONST,
                 PROPINT, PROPARITH, PROPREAL, PROPFLOAT, PROPVM, TYPEBOOL, TYPEARRAY)
from .token import tok, tokenstr, error
from . import util as U
from . import qbe as Q
from . import type as TY
from . import expr as E
from . import eval as EV
from . import pp as PP
from .scope import mkscope, delscope, scopegetdecl, scopeputdecl

_RELOPS = {TLESS: '<', TLEQ: '<=', TGREATER: '>', TGEQ: '>=', TNEQ: '!='}
_FLIP = {TLESS: TGREATER, TLEQ: TGEQ, TGREATER: TLESS, TGEQ: TLEQ, TNEQ: TNEQ}
_REDOPS = {'+': TADD, '-': TADD, '*': TMUL, '&': TBAND, '|': TBOR, '^': TXOR, '&&': TLAND, '||': TLOR,
           'max': None, 'min': None}
_LOOP = ('private', 'firstprivate', 'lastprivate', 'shared', 'default', 'reduction', 'schedule', 'collapse',
         'order')
_PARALLEL = ('num_threads', 'if', 'proc_bind')
_SIMD = ('safelen', 'simdlen', 'aligned', 'nontemporal')
_DATA = ('private', 'firstprivate', 'lastprivate', 'reduction', 'copyprivate', 'linear', 'copyin', 'allocate',
         'in_reduction', 'task_reduction', 'filter')
# constructs that run their statement as a team of one does; inside a parallel loop the ones OpenMP forbids there
_BLOCKS = {'single': True, 'master': True, 'masked': True, 'sections': True, 'section': True, 'ordered': True,
           'scope': True, 'taskgroup': False}
_STANDALONE = ('barrier', 'flush', 'taskwait', 'taskyield')
_IGNORED = ('declare', 'end', 'begin', 'requires', 'nothing', 'assume', 'assumes', 'scan', 'tile', 'unroll')
_UNSUPPORTED = ('task', 'taskloop', 'threadprivate', 'target', 'teams', 'distribute', 'cancel', 'cancellation',
                'allocate', 'depobj', 'interop', 'dispatch', 'metadirective', 'workshare', 'atomic_default_mem_order')

# a break's target in a parallel loop's body: never labeled, so stmt reports it (see stmt's TBREAK)
NOBREAK = Q.mkblock('omp_nobreak')


class Directive(object):
    __slots__ = ('kind', 'simd', 'loc', 'name', 'private', 'firstprivate', 'lastprivate', 'shared', 'reduction',
                 'schedule', 'chunk', 'num_threads', 'cond', 'reassoc')

    def __init__(self, kind, loc):
        self.kind, self.loc, self.simd, self.name = kind, loc, False, None
        self.private, self.firstprivate, self.lastprivate, self.shared, self.reduction = [], [], [], [], []
        self.schedule, self.chunk, self.num_threads, self.cond = 'static', None, None, None
        self.reassoc = []  # a simd's reductions (pp._reductions): the shadows' ops for expr.assignexpr


class Region(object):
    __slots__ = ('rcall', 'r0', 'rexit', 'data', 'tid', 'team', 'nthreads', 'startinsts', 'retwice', 'loc')


def _word(t):
    return t.lit if t.lit is not None else tokenstr(t.kind)


def _clauses(toks):
    """[(name, argument tokens or None, loc)]"""
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        w = _word(t)
        if t.kind == TCOMMA:
            i += 1
            continue
        if not w.isidentifier():
            error(t.loc, "expected an OpenMP clause, not '%s'", w)
        args = None
        if i + 1 < len(toks) and toks[i + 1].kind == TLPAREN:
            depth, j = 0, i + 1
            while j < len(toks):
                if toks[j].kind == TLPAREN:
                    depth += 1
                elif toks[j].kind == TRPAREN:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j == len(toks):
                error(t.loc, "missing ')' in the '%s' clause", w)
            args = toks[i + 2:j]
            i = j + 1
        else:
            i += 1
        out.append((w, args, t.loc))
    return out


def _split(args):
    """Argument tokens split at the commas outside parentheses."""
    out, cur, depth = [], [], 0
    for t in args:
        if t.kind == TLPAREN:
            depth += 1
        elif t.kind == TRPAREN:
            depth -= 1
        if t.kind == TCOMMA and depth == 0:
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    out.append(cur)
    return out


def _names(name, args, loc):
    if not args:
        error(loc, "the '%s' clause needs a list of variables", name)
    out = []
    for item in _split(args):
        if len(item) != 1 or item[0].kind < TIDENT:
            error(item[0].loc if item else loc, "the '%s' clause takes variables (not array sections, members or "
                  "expressions)", name)
        out.append((tokenstr(item[0].kind), item[0].loc))
    return out


def _modifier(args):
    """args with a leading `modifier:` dropped, and the modifier's words."""
    for k, t in enumerate(args):
        if t.kind == TCOLON:
            mod = [_word(x) for x in args[:k] if x.kind != TCOMMA]
            return args[k + 1:], mod
    return args, []


def pragma(toks, loc):
    """`#pragma omp ...` (the tokens after omp): None (nothing to do), a simd loop's reductions (as before) or a
    Directive for the statement that follows (stmt) or, standalone, for where the parser is (pending)."""
    words = [_word(t) for t in toks]
    if not U.diagnostics or words[:1] == ['simd']:  # cproc's own mode, and simd loops: their reductions only
        return PP._reductions(['omp'] + words) if 'simd' in words else None
    if not words:
        error(loc, "expected a directive after '#pragma omp'")
    w0 = words[0]
    i = 1
    if w0 == 'parallel':
        if words[1:2] not in (['for'], ['loop']):
            error(loc, "'#pragma omp parallel' is supported as 'parallel for' (a loop whose iterations run on several "
                  "threads) only")
        d = Directive('parallel for', loc)
        i = 2
    elif w0 in ('for', 'loop'):
        d = Directive('for', loc)
    elif w0 == 'critical':
        d = Directive('critical', loc)
        if words[1:2] == ['(']:
            if len(words) < 4 or words[3] != ')' or not words[2].isidentifier():
                error(loc, "expected '#pragma omp critical (name)'")
            d.name = words[2]
            i = 4
        _clauses(toks[i:])  # hint(...): nothing to do
        return d
    elif w0 == 'atomic':
        for name, args, cloc in _clauses(toks[1:]):
            if name not in ('read', 'write', 'update', 'capture', 'compare', 'weak', 'seq_cst', 'acq_rel', 'release',
                            'acquire', 'relaxed', 'hint', 'fail'):
                error(cloc, "unknown clause '%s' on '#pragma omp atomic'", name)
        return Directive('atomic', loc)
    elif w0 in _STANDALONE or w0 in _BLOCKS:
        d = Directive(w0, loc)
        for name, args, cloc in _clauses(toks[1:]):
            if name in _DATA or name == 'depend' or name == 'doacross':
                error(cloc, "the '%s' clause on '#pragma omp %s' is not supported", name, w0)
        return d
    elif w0 in _IGNORED:
        return None
    elif w0 == 'error':
        sev, msg = 'fatal', None
        for name, args, cloc in _clauses(toks[1:]):
            if name == 'severity' and args:
                sev = _word(args[0])
            elif name == 'message' and args:
                msg = ''.join(_word(t) for t in args)
            elif name == 'at' and args and _word(args[0]) == 'execution':
                error(cloc, "'#pragma omp error at(execution)' is not supported")
        if sev != 'warning':
            error(loc, "#pragma omp error%s", ': ' + msg if msg else '')
        return None
    elif w0 in _UNSUPPORTED:
        error(loc, "'#pragma omp %s' is not supported (cimo has 'parallel for', 'for', 'critical', 'atomic' and "
              "'simd')", w0)
    else:
        error(loc, "unknown directive '#pragma omp %s'", w0)
    if words[i:i + 1] == ['simd']:
        d.simd = True
        d.reassoc = PP._reductions(['omp'] + words)
        i += 1
    allowed = _LOOP + (_PARALLEL if d.kind == 'parallel for' else ('nowait',)) + (_SIMD if d.simd else ())
    seen = {}
    for name, args, cloc in _clauses(toks[i:]):
        if name not in allowed:
            if name in ('nowait', 'num_threads', 'proc_bind', 'copyin', 'ordered', 'linear', 'allocate', 'if'):
                error(cloc, "the '%s' clause on '#pragma omp %s' is not supported", name, d.kind)
            error(cloc, "unknown clause '%s' on '#pragma omp %s'", name, d.kind)
        if name in ('private', 'firstprivate', 'lastprivate', 'shared'):
            if name == 'lastprivate' and args:
                args, mod = _modifier(args)
                if mod:
                    error(cloc, "'lastprivate(%s: ...)' is not supported", mod[0])
            getattr(d, name).extend(_names(name, args, cloc))
        elif name == 'reduction':
            if not args:
                error(cloc, "expected 'reduction(op: list)'")
            colon = next((k for k, t in enumerate(args) if t.kind == TCOLON), None)
            if colon is None:
                error(cloc, "expected 'reduction(op: list)'")
            head = [_word(t) for t in args[:colon] if t.kind != TCOMMA]
            if len(head) == 2 and head[0] == 'default':
                head = head[1:]
            elif len(head) == 2:
                error(cloc, "'reduction(%s, ...)' is not supported", head[0])
            op = ''.join(head)
            if op not in _REDOPS:
                error(cloc, "the reduction operator '%s' is not supported (+ - * & | ^ && || max min)", op)
            d.reduction.extend((op, n, l) for n, l in _names(name, args[colon + 1:], cloc))
        elif name == 'schedule':
            args, mod = _modifier(args or [])
            if any(m not in ('monotonic', 'nonmonotonic', 'simd') for m in mod):
                error(cloc, "unknown schedule modifier in 'schedule(%s: ...)'", ', '.join(mod))
            parts = _split(args)
            kind = _word(parts[0][0]) if parts and parts[0] else ''
            if kind not in ('static', 'dynamic', 'guided', 'auto', 'runtime') or len(parts[0]) != 1 or len(parts) > 2:
                error(cloc, "expected 'schedule(static|dynamic|guided|auto|runtime[, chunk])'")
            d.schedule = 'static' if kind in ('auto', 'runtime') else kind
            d.chunk = parts[1] if len(parts) == 2 else None
            if d.chunk is not None and not d.chunk:
                error(cloc, "missing chunk size in 'schedule'")
        elif name == 'collapse':
            if not args or len(args) != 1 or _word(args[0]) not in ('1', '1u', '1U'):
                error(cloc, "'collapse(%s)' is not supported: parallel loops split their outer loop only",
                      ''.join(_word(t) for t in args or ()))
        elif name == 'default':
            v = [_word(t) for t in args or ()]
            if v not in (['shared'], ['none']):
                error(cloc, "'default(%s)' is not supported", ' '.join(v))
        elif name == 'num_threads':
            if not args:
                error(cloc, "expected 'num_threads(n)'")
            d.num_threads = args
        elif name == 'if':
            args, mod = _modifier(args or [])
            if mod and mod[0] not in ('parallel', 'simd'):
                error(cloc, "'if(%s: ...)' does not apply to '#pragma omp %s'", mod[0], d.kind)
            if not args:
                error(cloc, "expected 'if(condition)'")
            if not mod or mod[0] == 'parallel':
                d.cond = args
        if name in ('num_threads', 'if', 'schedule', 'collapse', 'default', 'proc_bind', 'order') and name in seen:
            error(cloc, "more than one '%s' clause", name)
        seen[name] = True
    return d


def pending(f):
    """A standalone directive (barrier, flush, ...) read before the token the parser is at: what it does here."""
    d = U.ompnext
    if not isinstance(d, Directive) or d.kind not in _STANDALONE:
        return
    U.ompnext = None
    if d.kind == 'barrier' and U.ompregion:
        error(d.loc, "'#pragma omp barrier' inside a parallel loop is not allowed")
    if d.kind == 'flush':
        _call(f, '__cimo_flush_mem')


def unfinished():
    """At a block's end: a directive with no statement after it."""
    d = U.ompnext
    if isinstance(d, Directive):
        error(d.loc, "'#pragma omp %s' must be followed by a statement", d.kind)


def _sym(name, extern=True):
    v = Q.Value(Q.VALUE_GLOBAL | (Q.VALUE_EXTERN if extern else 0))
    v.name = name
    v.id = 0
    return v


def _call(f, name, *args):
    """A call of a runtime function: args are (class, value)."""
    v = Q.funcinst(f, Q.ICALL, 0, _sym(name), None)
    for cls, a in args:
        Q.funcinst(f, Q.IARG, cls, a, None)
    return v


def _ptr():
    return Q.abiclass(TY.mkpointertype(TY.typevoid, 0))


def statement(f, s, d, labelstmt):
    """The statement after directive d (stmt)."""
    if d.kind in ('parallel for', 'for'):
        return _loop(f, s, d, labelstmt)
    if d.kind in ('critical', 'atomic'):
        if d.kind == 'atomic':
            lock = _sym('__cimo_atomic')
        elif d.name is None:
            lock = _sym('__cimo_crit')
        else:
            lock = _sym('__cimo_crit_' + d.name, False)
            if d.name not in U.ompcrit:
                U.ompcrit.add(d.name)
                Q.w('data $__cimo_crit_%s = align 8 { z 8 }\n' % d.name)
        _call(f, '__cimo_lock', (_ptr(), lock))
        U.ompblock += 1
        labelstmt(f, s)
        U.ompblock -= 1
        _call(f, '__cimo_unlock', (_ptr(), lock))
        return
    if _BLOCKS.get(d.kind) and U.ompregion:
        error(d.loc, "'#pragma omp %s' inside a parallel loop is not allowed", d.kind)
    labelstmt(f, s)


def _isvar(s, var):
    return tok.kind >= TIDENT and scopegetdecl(s, tokenstr(tok.kind), 1) is var


def _ident(d):
    e = E.mkexpr(EXPRIDENT, d.type, None)
    e.qual = d.qual
    e.lvalue = True
    e.decl = d
    return E.decay(e)


def _canon(loc, what):
    error(loc, "a parallel loop must be 'for (var = lb; var < b; var++)' or another of OpenMP's forms: %s", what)


def _parse(toks, s, parse):
    """A clause's expression, parsed from its tokens in scope s: the token stream is resumed after."""
    old = tok.copy()
    end = (toks[-1] if toks else old).copy()
    end.kind, end.lit = TSEMICOLON, None
    seq = [t.copy() for t in toks] + [end]
    PP.ctxpush(seq, 0, len(seq), None, False)
    PP.next()
    e = parse(s)
    if tok.kind != TSEMICOLON:
        error(tok.loc, "unexpected '%s' in a clause", tokenstr(tok.kind))
    tok.set(old)
    return e


def _long(f, e, t=None):
    """Integer expression e's value as a 64-bit l, extended as its type says."""
    t = t or e.type
    v = Q.funcexpr(f, e)
    return Q.convert(f, TY.typelong if t.issigned else TY.typeulong, t, v)


def _int(e, what):
    if not (e.type.prop & PROPINT) or TY.isi128(e.type):
        error(tok.loc, "%s must have an integer type", what)
    return e


def _shadow(f, scope, orig, init_from=None):
    """orig declared again in scope, for the thread's own copy."""
    t = orig.type
    from . import decl as D
    nd = D.mkdecl(orig.name, DECLOBJECT, t, orig.qual & ~QUALCONST if init_from is None else orig.qual, LINKNONE)
    nd.storage = SDAUTO
    nd.align = max(orig.align, t.align)
    Q.funcalloc(f, nd)
    scopeputdecl(scope, nd)
    return nd


def _identity(op, t):
    if op in ('+', '-', '|', '^', '||'):
        return E.mkconstexpr(TY.typeint, 0)
    if op in ('*', '&&'):
        return E.mkconstexpr(TY.typeint, 1)
    if op == '&':
        return E.mkconstexpr(TY.typelong, -1)
    big = op == 'min'
    if t.prop & PROPFLOAT:
        e = E.mkexpr(EXPRCONST, TY.typedouble, None)
        e.f = float('inf') if big else float('-inf')
        return e
    if t.kind == TYPEBOOL:
        return E.mkconstexpr(TY.typeint, 1 if big else 0)
    bits = 8 * t.size
    if t.issigned:
        v = (1 << (bits - 1)) - 1 if big else -(1 << (bits - 1))
    else:
        v = (1 << bits) - 1 if big else 0
    return E.mkconstexpr(TY.typeulong if v > 0 else TY.typelong, v)


def _resolve(s, items, what):
    out = []
    for name, loc in items:
        dd = scopegetdecl(s, name, 1)
        if dd is None or dd.kind != DECLOBJECT:
            error(loc, "'%s' in a '%s' clause is not a variable", name, what)
        if dd.type.prop & PROPVM:
            error(loc, "'%s' in a '%s' clause has a variably modified type, which is not supported", name, what)
        out.append((dd, loc))
    return out


def _loop(f, s, d, labelstmt):
    if tok.kind != TFOR:
        error(tok.loc, "'#pragma omp %s' must be followed by a 'for' loop", d.kind)
    if d.kind == 'for' and U.ompregion:
        error(d.loc, "'#pragma omp for' inside a parallel loop is not allowed")
    outer = s
    priv = _resolve(outer, d.private, 'private')
    first = _resolve(outer, d.firstprivate, 'firstprivate')
    last = _resolve(outer, d.lastprivate, 'lastprivate')
    _resolve(outer, d.shared, 'shared')
    rvars = _resolve(outer, [(n, l) for _o, n, l in d.reduction], 'reduction')
    red = [(op, dd, loc) for (op, _n, _l), (dd, loc) in zip(d.reduction, rvars)]
    for op, dd, loc in red:
        t = dd.type
        if dd.qual & QUALCONST or not (t.prop & PROPARITH) or TY.isi128(t) or TY.isvec(t) or (
                op in ('&', '|', '^') and not (t.prop & PROPINT)) or (op in ('max', 'min') and not (t.prop & PROPREAL)):
            error(loc, "'%s' can't be a '%s' reduction's variable", dd.name, op)
    listed = {}
    for what, items in (('private', priv), ('firstprivate', first), ('lastprivate', last),
                        ('reduction', [(dd, loc) for _o, dd, loc in red])):
        for dd, loc in items:
            prev = listed.get(id(dd))
            if prev and {prev, what} != {'firstprivate', 'lastprivate'}:
                error(loc, "'%s' is in both a '%s' and a '%s' clause", dd.name, prev, what)
            listed[id(dd)] = what
    for dd, loc in first:
        if dd.type.kind == TYPEARRAY and dd.type.incomplete:
            error(loc, "'%s' in a 'firstprivate' clause has an incomplete type", dd.name)

    PP.next()
    PP.expect(TLPAREN, "after 'for'")
    s = mkscope(s)
    nvla = len(f.vlas)
    from . import decl as D
    loc = tok.loc
    if D.decl(s, f):
        objs = [x for x in s.decls.values() if x.kind == DECLOBJECT]
        if len(objs) != 1 or len(s.decls) != 1:
            _canon(loc, "one loop variable")
        var, lb = objs[0], None
        if var.storage != SDAUTO:
            _canon(loc, "the loop variable has automatic storage")
    else:
        if tok.kind < TIDENT:
            _canon(tok.loc, "'var = lb' first")
        var = scopegetdecl(s, tokenstr(tok.kind), 1)
        if var is None or var.kind != DECLOBJECT:
            _canon(tok.loc, "'var = lb' first, with var a variable")
        PP.next()
        PP.expect(TASSIGN, "after the loop variable")
        lb = E.assignexpr(s)
        PP.expect(TSEMICOLON, "after the loop's initialization")
    vt = var.type
    if not (vt.prop & PROPINT) or vt.kind == TYPEBOOL or TY.isi128(vt) or var.qual & QUALCONST:
        error(loc, "the loop variable of a parallel loop must be an integer variable (not '%s')", var.name)
    if id(var) in listed and listed[id(var)] != 'lastprivate' and listed[id(var)] != 'private':
        error(loc, "the loop variable '%s' can't be in a '%s' clause", var.name, listed[id(var)])
    # the condition: var OP b, or b OP var
    loc = tok.loc
    if _isvar(s, var):
        PP.next()
        op = tok.kind
        if op not in _RELOPS:
            _canon(tok.loc, "'var < b', '<=', '>', '>=' or '!='")
        PP.next()
        ub = E.binaryexpr(s, None, E.precedence(TSHL))
    else:
        ub = E.binaryexpr(s, None, E.precedence(TSHL))
        op = tok.kind
        if op not in _RELOPS:
            _canon(tok.loc, "'var < b', '<=', '>', '>=' or '!='")
        PP.next()
        if not _isvar(s, var):
            _canon(tok.loc, "the condition compares the loop variable")
        PP.next()
        op = _FLIP[op]
    if tok.kind != TSEMICOLON:
        _canon(tok.loc, "the condition is one comparison")
    PP.next()
    cmp = E.mkbinaryexpr(loc, op, _ident(var), _int(ub, "the loop's bound"))
    ct = cmp.l.type
    if not (ct.prop & PROPINT):
        error(loc, "the loop's bound must have an integer type")
    # the increment
    loc = tok.loc
    step, sign = None, 1
    if tok.kind in (TINC, TDEC):
        sign = 1 if tok.kind == TINC else -1
        PP.next()
        if not _isvar(s, var):
            _canon(tok.loc, "the increment changes the loop variable")
        PP.next()
        step = 1
    elif _isvar(s, var):
        PP.next()
        k = tok.kind
        PP.next()
        if k in (TINC, TDEC):
            step, sign = 1, 1 if k == TINC else -1
        elif k in (TADDASSIGN, TSUBASSIGN):
            step, sign = _int(E.assignexpr(s), "the loop's step"), 1 if k == TADDASSIGN else -1
        elif k == TASSIGN:
            if _isvar(s, var):
                PP.next()
                if tok.kind not in (TADD, TSUB):
                    _canon(tok.loc, "'var = var + s' or 'var = var - s'")
                sign = 1 if tok.kind == TADD else -1
                PP.next()
                step = _int(E.binaryexpr(s, None, E.precedence(TMUL)), "the loop's step")
            else:
                step = _int(E.binaryexpr(s, None, E.precedence(TMUL)), "the loop's step")
                if tok.kind != TADD:
                    _canon(tok.loc, "'var = s + var'")
                PP.next()
                if not _isvar(s, var):
                    _canon(tok.loc, "'var = s + var'")
                PP.next()
        else:
            _canon(loc, "var++, var--, var += s, var -= s or var = var + s")
    else:
        _canon(loc, "var++, var--, var += s, var -= s or var = var + s")
    if tok.kind != TRPAREN:
        _canon(tok.loc, "the increment is one expression")
    PP.next()
    if not isinstance(step, int):
        c = EV.eval(step)
        if c.kind == EXPRCONST:
            step = c.i if c.type.issigned else c.u
    if isinstance(step, int):
        step *= sign
        if step == 0:
            error(loc, "the loop's step is 0")
    up = op in (TLESS, TLEQ) or op == TNEQ and isinstance(step, int) and step > 0
    if op == TNEQ and step not in (1, -1):
        error(loc, "a parallel loop with '!=' must step by 1 or -1")
    if isinstance(step, int) and (step > 0) != up:
        error(loc, "the loop's step goes the wrong way for its condition")

    # once, before the loop: the bounds, the step and the trip count, all 64-bit
    signed = 'signed' if ct.issigned else 'unsigned'
    if lb is None:
        lbv = Q.funcload(f, vt, Q.LValue(var.value))
    else:
        lbv = Q.funcexpr(f, E.exprassign(lb, vt))
    lc = Q.convert(f, TY.typelong if ct.issigned else TY.typeulong, ct, Q.convert(f, ct, vt, lbv))
    uc = _long(f, cmp.r, ct)
    if isinstance(step, int):
        S = Q.mkintconst(step)
    else:
        S = _long(f, step)
        if sign < 0:
            S = Q.funcinst(f, Q.ISUB, 'l', Q.mkintconst(0), S)
    L = Q.convert(f, TY.typelong if vt.issigned else TY.typeulong, vt, lbv)
    I = Q.funcinst
    lt = {('signed', '<'): Q.ICSLTL, ('signed', '<='): Q.ICSLEL, ('unsigned', '<'): Q.ICULTL,
          ('unsigned', '<='): Q.ICULEL}
    if op == TNEQ:
        diff = I(f, Q.ISUB, 'l', uc, lc) if up else I(f, Q.ISUB, 'l', lc, uc)
        if ct.size < 8:
            diff = I(f, Q.IAND, 'l', diff, Q.mkintconst((1 << (8 * ct.size)) - 1))
        N = diff
    else:
        lo, hi = (lc, uc) if up else (uc, lc)
        strict = op in (TLESS, TGREATER)
        ok = I(f, Q.IEXTUW, 'l', I(f, lt[signed, '<' if strict else '<='], 'w', lo, hi), None)
        diff = I(f, Q.ISUB, 'l', hi, lo)
        if strict:
            diff = I(f, Q.ISUB, 'l', diff, Q.mkintconst(1))
        mag = S if up else (Q.mkintconst(-step) if isinstance(step, int) else I(f, Q.ISUB, 'l', Q.mkintconst(0), S))
        N = I(f, Q.IMUL, 'l', I(f, Q.IADD, 'l', I(f, Q.IUDIV, 'l', diff, mag), Q.mkintconst(1)), ok)
    # the team's size
    nthreads = Q.mkintconst(1 if d.kind == 'for' else 0)
    if d.num_threads is not None:
        nthreads = _long(f, _int(_parse(d.num_threads, outer, E.assignexpr), "'num_threads'"))
    if d.cond is not None:
        c = _parse(d.cond, outer, E.assignexpr)
        c = E.scalarcond(c) or c
        if not (c.type.prop & PROPARITH or c.type.kind == TY.TYPEPOINTER):
            error(d.loc, "the 'if' clause's condition must have scalar type")
        b = I(f, Q.IEXTUW, 'l', Q.convert(f, TY.typebool, c.type, Q.funcexpr(f, c)), None)
        nthreads = I(f, Q.IADD, 'l', Q.mkintconst(1), I(f, Q.IMUL, 'l', b,
                                                       I(f, Q.ISUB, 'l', nthreads, Q.mkintconst(1))))
    chunk = None
    if d.chunk is not None:
        chunk = _long(f, _int(_parse(d.chunk, outer, E.assignexpr), "the chunk size"))
        small = I(f, Q.IEXTUW, 'l', I(f, Q.ICSLTL, 'w', chunk, Q.mkintconst(1)), None)
        chunk = I(f, Q.IADD, 'l', chunk, I(f, Q.IMUL, 'l', small, I(f, Q.ISUB, 'l', Q.mkintconst(1), chunk)))
    elif d.schedule in ('dynamic', 'guided'):
        chunk = Q.mkintconst(1)

    # the call (filled in by outline), then the region: what one thread runs
    r = Region()
    r.loc, r.nthreads = d.loc, nthreads
    end = f.end
    f.end = f.start
    r.data = Q.mkinst(f, Q.IALLOC8, Q.ptrclass, Q.mkintconst(8), None)
    f.start.insts.append(r.data)
    f.end = end
    r.rcall, rafter, r.r0 = Q.mkblock('omp_call'), Q.mkblock('omp_join'), Q.mkblock('omp_part')
    Q.funcjmp(f, r.rcall)
    Q.funclabel(f, r.rcall)
    Q.funcjmp(f, rafter)
    Q.funclabel(f, r.r0)
    n0 = len(f.start.insts)
    retwice, vlalog, U.ompvla = f.retwice, U.ompvla, []
    reduce = U.reduce
    r.tid, r.team = Q.Value(), Q.Value()
    Q.functemp(f, r.tid)
    Q.functemp(f, r.team)
    U.ompregion += 1
    bs = mkscope(s)
    bs.breaklabel, bs.switchcases = NOBREAK, None
    iv = _shadow(f, bs, var)
    shadows = {}
    for dd, loc in priv + last:
        if dd is not var and id(dd) not in shadows:
            shadows[id(dd)] = _shadow(f, bs, dd)
    for dd, loc in first:
        nd = shadows.get(id(dd)) or _shadow(f, bs, dd, dd)
        shadows[id(dd)] = nd
        Q.funcstore(f, dd.type, nd.qual & ~QUALCONST, Q.LValue(nd.value), Q.funcload(f, dd.type, Q.LValue(dd.value),
                                                                                    dd.qual))
    reds = []
    for op, dd, loc in red:
        nd = _shadow(f, bs, dd)
        reds.append((op, dd, nd))
        Q.funcexpr(f, E.mkassignexpr(_ident(nd), _identity(op, dd.type)))
    U.reduce = dict(U.reduce)
    for op, name in d.reassoc:  # parallel for simd reduction(+:s): the thread's s adds in any order
        for rop, dd, nd in reds:
            if dd.name == name:
                U.reduce[id(nd)] = op

    team, tid = r.team, r.tid
    nthr = I(f, Q.ILOADL, 'l', I(f, Q.IADD, 'l', team, Q.mkintconst(24)), None)
    rexit = Q.mkblock('omp_end')
    lastslot = None
    if last and d.schedule != 'static' or last and chunk is not None:
        lastslot = Q._tempalloc(f, 8, 8)
        I(f, Q.ISTOREL, 0, Q.mkintconst(0), lastslot)

    def part(k0, k1, out):
        """The iterations k0 to k1 (exclusive) as an ordinary loop over var, then to block out."""
        if lastslot is not None:
            I(f, Q.ISTOREL, 0, I(f, Q.IEXTUW, 'l', I(f, Q.ICEQL, 'w', k1, N), None), lastslot)
        ivt = TY.typelong if vt.issigned else TY.typeulong
        v0 = Q.convert(f, vt, ivt, I(f, Q.IADD, 'l', L, I(f, Q.IMUL, 'l', k0, S)))
        stop = Q.convert(f, vt, ivt, I(f, Q.IADD, 'l', L, I(f, Q.IMUL, 'l', k1, S)))
        Q.funcstore(f, vt, 0, Q.LValue(iv.value), v0)
        wide = vt.size == 8
        if not wide:
            stop = Q.convert(f, TY.typeint, vt, stop)
        # < (> counting down) as gcc's loop has it; != where the end may wrap around: a char or short, or an
        # unsigned var stepping by more than 1
        unit = isinstance(step, int) and step in (1, -1)
        if vt.size < 4:
            cop = Q.ICNEW
        elif vt.issigned:
            cop = (Q.ICSLTL if up else Q.ICSGTL) if wide else (Q.ICSLTW if up else Q.ICSGTW)
        elif unit:
            cop = (Q.ICULTL if up else Q.ICUGTL) if wide else (Q.ICULTW if up else Q.ICUGTW)
        else:
            cop = Q.ICNEL if wide else Q.ICNEW
        bcond, bbody, bcont = Q.mkblock('omp_cond'), Q.mkblock('omp_body'), Q.mkblock('omp_cont')
        Q.funclabel(f, bcond)
        v = Q.funcload(f, vt, Q.LValue(iv.value))
        if not wide:
            v = Q.convert(f, TY.typeint, vt, v)
        Q.funcjnz(f, I(f, cop, 'w', v, stop), None, bbody, out)
        Q.funclabel(f, bbody)
        bs.continuelabel = bcont
        bs.breakvla = bs.contvla = len(f.vlas)
        labelstmt(f, bs)
        Q.funclabel(f, bcont)
        v = Q.funcload(f, vt, Q.LValue(iv.value))
        cls = 'l' if wide else 'w'
        sv = (Q.mkintconst(step) if isinstance(step, int) else S if wide else Q.convert(f, vt, ivt, S))
        v = I(f, Q.IADD, cls, v, sv)
        Q._nowrapinst(f, vt)
        Q.funcstore(f, vt, 0, Q.LValue(iv.value), v)
        Q.funcjmp(f, bcond)

    if d.schedule == 'static' and chunk is None:
        q = I(f, Q.IUDIV, 'l', N, nthr)
        rem = I(f, Q.IUREM, 'l', N, nthr)
        c = I(f, Q.IEXTUW, 'l', I(f, Q.ICULTL, 'w', tid, rem), None)
        q1 = I(f, Q.IADD, 'l', q, c)
        k0 = I(f, Q.IMUL, 'l', I(f, Q.IXOR, 'l', c, Q.mkintconst(1)), rem)  # libgomp's: q * t + (t < r ? 0 : r)
        k0 = I(f, Q.IADD, 'l', I(f, Q.IMUL, 'l', q1, tid), k0)
        k1 = I(f, Q.IADD, 'l', k0, q1)
        part(k0, k1, rexit)
        Q.funclabel(f, rexit)
        if last:
            lastcond = I(f, Q.IAND, 'w', I(f, Q.ICEQL, 'w', k1, N), I(f, Q.ICULTL, 'w', k0, k1))
    else:
        bnext, binit, bstep = Q.mkblock('omp_chunk'), Q.mkblock('omp_chunk_start'), Q.mkblock('omp_chunk_next')
        if d.schedule == 'static':
            cslot = Q._tempalloc(f, 8, 8)
            I(f, Q.ISTOREL, 0, I(f, Q.IMUL, 'l', tid, chunk), cslot)
            Q.funclabel(f, bnext)
            k0 = I(f, Q.ILOADL, 'l', cslot, None)
            Q.funcjnz(f, I(f, Q.ICULTL, 'w', k0, N), None, binit, rexit)
            Q.funclabel(f, binit)
            e = I(f, Q.IADD, 'l', k0, chunk)
            over = I(f, Q.IEXTUW, 'l', I(f, Q.IOR, 'w', I(f, Q.ICULTL, 'w', N, e), I(f, Q.ICULTL, 'w', e, k0)), None)
            k1 = I(f, Q.IADD, 'l', e, I(f, Q.IMUL, 'l', over, I(f, Q.ISUB, 'l', N, e)))
            part(k0, k1, bstep)
            Q.funclabel(f, bstep)
            k0 = I(f, Q.ILOADL, 'l', cslot, None)
            I(f, Q.ISTOREL, 0, I(f, Q.IADD, 'l', k0, I(f, Q.IMUL, 'l', nthr, chunk)), cslot)
            Q.funcjmp(f, bnext)
        else:
            hislot = Q._tempalloc(f, 8, 8)
            Q.funclabel(f, bnext)
            k0 = Q.funcinst(f, Q.ICALL, 'l', _sym('__cimo_par_next'), None)
            for cls, a in ((_ptr(), team), ('l', N), ('l', chunk), ('l', Q.mkintconst(d.schedule == 'guided')),
                           (_ptr(), hislot)):
                I(f, Q.IARG, cls, a, None)
            Q.funcjnz(f, I(f, Q.ICULTL, 'w', k0, N), None, binit, rexit)
            Q.funclabel(f, binit)
            part(k0, I(f, Q.ILOADL, 'l', hislot, None), bnext)
        Q.funclabel(f, rexit)
        if last:
            lastcond = I(f, Q.ICNEL, 'w', I(f, Q.ILOADL, 'l', lastslot, None), Q.mkintconst(0))
    if last:
        bcopy, bdone = Q.mkblock('omp_last'), Q.mkblock('omp_last_done')
        Q.funcjnz(f, lastcond, None, bcopy, bdone)
        Q.funclabel(f, bcopy)
        for dd, loc in last:
            if dd is var:  # the value the loop leaves it with: lb + N * step
                v = Q.convert(f, vt, TY.typelong if vt.issigned else TY.typeulong,
                              I(f, Q.IADD, 'l', L, I(f, Q.IMUL, 'l', N, S)))
                Q.funcstore(f, vt, 0, Q.LValue(var.value), v)
            else:
                nd = shadows[id(dd)]
                Q.funcstore(f, dd.type, dd.qual, Q.LValue(dd.value), Q.funcload(f, dd.type, Q.LValue(nd.value)))
        Q.funclabel(f, bdone)
    if reds:
        lock = _sym('__cimo_reduce')
        _call(f, '__cimo_lock', (_ptr(), lock))
        for op, dd, nd in reds:
            if op in ('max', 'min'):
                take, skip = Q.mkblock('omp_reduce'), Q.mkblock('omp_reduce_done')
                Q.funcbranch(f, E.mkbinaryexpr(d.loc, TGREATER if op == 'max' else TLESS, _ident(nd), _ident(dd)),
                             take, skip)
                Q.funclabel(f, take)
                Q.funcexpr(f, E.mkassignexpr(_ident(dd), _ident(nd)))
                Q.funclabel(f, skip)
            else:
                Q.funcexpr(f, E.mkassignexpr(_ident(dd), E.mkbinaryexpr(d.loc, _REDOPS[op], _ident(dd), _ident(nd))))
        _call(f, '__cimo_unlock', (_ptr(), lock))
    Q.funcret(f, None)
    r.rexit = f.end
    U.ompregion -= 1
    U.reduce = reduce
    for t in U.ompvla:  # VLA sizes first computed in the loop are computed there: again after it, if used
        t.asize = None
    U.ompvla = vlalog
    del f.vlas[nvla:]
    r.startinsts = f.start.insts[n0:]
    r.retwice = f.retwice and not retwice
    f.omp.append(r)
    Q.funclabel(f, rafter)
    delscope(s)


# ---------------------------------------------------------------------------------------------
# outlining (emitfunc)

_LOADS = {'w': Q.ILOADW, 'l': Q.ILOADL, 's': Q.ILOADS, 'd': Q.ILOADD}
_STORES = {'w': Q.ISTOREW, 'l': Q.ISTOREL, 's': Q.ISTORES, 'd': Q.ISTORED}
_ALLOCS = (Q.IALLOC4, Q.IALLOC8, Q.IALLOC16)


def _temp(v):
    return v is not None and v.kind == Q.VALUE_TEMP


def _operands(inst):
    """(position, value) of inst's operands that may be temporaries."""
    out = [(0, inst.arg0), (1, inst.arg1)]
    if inst.kind == Q.IFMA:
        out.append((2, inst.arg2))
    elif inst.kind == Q.IQ and isinstance(inst.arg2, tuple) and inst.arg2[1] is not None:
        out.append((3, inst.arg2[1]))
    return [(k, v) for k, v in out if _temp(v)]


def _set(inst, k, v):
    if k == 0:
        inst.arg0 = v
    elif k == 1:
        inst.arg1 = v
    elif k == 2:
        inst.arg2 = v
    else:
        inst.arg2 = (inst.arg2[0], v)


def _targets(b):
    if b.jump_kind == Q.JUMP_JMP:
        return [b.jump_blk[0]]
    if b.jump_kind == Q.JUMP_JNZ:
        return [b.jump_blk[0], b.jump_blk[1]]
    return []


def _chain(start):
    out = []
    b = start
    while b:
        out.append(b)
        b = b.next
    return out


def _isload(k):
    return Q.ILOADD <= k <= Q.ILOADUB


def _isstore(k):
    return Q.ISTORED <= k <= Q.ISTOREB


def outline(f):
    """Each parallel loop of f (innermost first) into a function of its own, emitted now; then f's checks."""
    fns = []
    for r in f.omp:
        fns.append(_outline(f, r))
    f.omp = []
    extra = {id(t) for t, _v, _g in f.vlagotos}
    for fn, blocks, params in [(f, _chain(f.start), f.paramtemps)] + fns:
        _check(fn, blocks, params, extra if fn is f else set())
    fenv = Q.paraminfo is not None and f.decl.value.name in Q.paraminfo.get('#fenv', ())
    for fn, blocks, params in fns:
        _emit(fn, blocks, params)
        if fenv:
            Q.paraminfo['#fenv'].add(fn.name)


def _check(fn, blocks, params, extra):
    ids = {id(b) for b in blocks} | extra
    defs = {v.id for v in params}
    for b in blocks:
        if b.phi_res.kind == Q.VALUE_TEMP:
            defs.add(b.phi_res.id)
        for inst in b.insts:
            if inst.res.kind == Q.VALUE_TEMP:
                defs.add(inst.res.id)
    for b in blocks:
        for t in _targets(b):
            if id(t) not in ids:
                error(tok.loc, "a jump into or out of a parallel loop (goto, or a case label of a switch around it) "
                      "is not allowed")
        uses = [v for inst in b.insts for _k, v in _operands(inst)]
        uses += [v for v in (b.jump_arg,) + tuple(b.phi_val) if _temp(v)]
        for v in uses:
            if v.id not in defs:
                U.fatal("internal error: a parallel loop's value %%.%d is used outside it" % v.id)


def _outline(f, r):
    blocks = _chain(r.r0)
    blocks = blocks[:blocks.index(r.rexit) + 1]
    assert r.rcall.next is r.r0
    r.rcall.next = r.rexit.next
    ids = {id(b) for b in blocks}
    # forward gotos out of VLA scopes inside the loop (vlafree): their blocks are the loop's
    keep = []
    for t, vlas, g in f.vlagotos:
        if any(t is x for b in blocks for x in _targets(b)):
            if len(vlas) > len(g.vlas):
                t.insts.append(Q.mkinst(f, Q.ISETSP, 0, vlas[len(g.vlas)], None))
            t.jump_kind, t.jump_blk[0] = Q.JUMP_JMP, g.label
            blocks.append(t)
            ids.add(id(t))
        else:
            keep.append((t, vlas, g))
    f.vlagotos = keep
    for a, b in zip(blocks, blocks[1:]):
        a.next = b
    blocks[-1].next = None
    # the instructions the loop added to f's start block (its variables' slots) are the outlined function's
    mine = {id(i) for i in r.startinsts}
    start = [i for i in f.start.insts if id(i) in mine]
    f.start.insts = [i for i in f.start.insts if id(i) not in mine]
    insts = start + [i for b in blocks for i in b.insts]
    defs = {r.tid.id, r.team.id}
    for i in insts:
        if i.res.kind == Q.VALUE_TEMP:
            defs.add(i.res.id)
    for b in blocks:
        if b.phi_res.kind == Q.VALUE_TEMP:
            defs.add(b.phi_res.id)
    livein = {}
    for i in insts:
        for _k, v in _operands(i):
            if v.id not in defs:
                livein.setdefault(v.id, v)
    for b in blocks:
        for v in (b.jump_arg,) + tuple(b.phi_val):
            if _temp(v) and v.id not in defs:
                livein.setdefault(v.id, v)
    # classes of f's temporaries, and which slots of f are only loaded and stored at (their address kept in)
    cls = {}
    fblocks = _chain(f.start)
    for b in fblocks + blocks:
        if b.phi_res.kind == Q.VALUE_TEMP:
            cls[b.phi_res.id] = b.phi_class
        for i in b.insts:
            if i.res.kind == Q.VALUE_TEMP:
                cls[i.res.id] = i.cls
    p = f.type.params
    for v in f.paramtemps:
        c = Q.abiclass(p.type)
        cls[v.id] = 'w' if c in ('sb', 'ub', 'sh', 'uh') else 'l' if c == 'p' else c
        p = p.next
    slots = {}
    for i in f.start.insts:
        if i.kind in _ALLOCS and i.res.id in livein and i.arg0.kind == Q.VALUE_INTCONST and 0 < i.arg0.i <= 8:
            slots[i.res.id] = None  # None: no load seen in the loop yet; False: not by value
    if slots:
        for b in fblocks:
            for i in b.insts:
                for k, v in _operands(i):
                    if v.id in slots and not (k == 0 and _isload(i.kind) or k == 1 and _isstore(i.kind)):
                        slots[v.id] = False
            for v in (b.jump_arg,) + tuple(b.phi_val):
                if _temp(v) and v.id in slots:
                    slots[v.id] = False
        for i in insts:
            for k, v in _operands(i):
                if v.id in slots and slots[v.id] is not False:
                    if k == 0 and _isload(i.kind) and slots[v.id] in (None, (i.kind, i.cls)):
                        slots[v.id] = (i.kind, i.cls)
                    else:
                        slots[v.id] = False
        for b in blocks:
            for v in (b.jump_arg,) + tuple(b.phi_val):
                if _temp(v) and v.id in slots:
                    slots[v.id] = False
    # the outlined function: its start loads what the loop uses
    f2 = Q.Func()
    f2.lastid = f.lastid
    data = Q.Value()
    Q.functemp(f2, data)
    s2 = Q.mkblock('start')
    s2.insts = list(start)
    caps = []
    fill = []
    for j, (vid, v) in enumerate(livein.items()):
        how = slots.get(vid)
        c = how[1] if how else cls.get(vid)
        if c not in _LOADS:
            U.fatal("internal error: a parallel loop uses a value of class %s" % c)
        a = data
        if j:
            ai = Q.mkinst(f2, Q.IADD, 'l', data, Q.mkintconst(8 * j))
            s2.insts.append(ai)
            a = ai.res
        li = Q.mkinst(f2, _LOADS[c], c, a, None)
        s2.insts.append(li)
        caps.append((v, how, c))
        fill.append((vid, how, li.res))
    byval = {vid: nv for vid, how, nv in fill if how}
    remap = {vid: nv for vid, how, nv in fill if not how}
    for i in insts:
        if _isload(i.kind) and _temp(i.arg0) and i.arg0.id in byval:
            i.kind, i.arg0, i.arg1 = Q.ICOPY, byval[i.arg0.id], None
            continue
        for k, v in _operands(i):
            if v.id in remap:
                _set(i, k, remap[v.id])
    for b in blocks:
        if _temp(b.jump_arg) and b.jump_arg.id in remap:
            b.jump_arg = remap[b.jump_arg.id]
        for k in (0, 1):
            v = b.phi_val[k]
            if _temp(v) and v.id in remap:
                b.phi_val[k] = remap[v.id]
    s2.jump_kind, s2.jump_blk[0] = Q.JUMP_JMP, r.r0
    s2.next = r.r0
    f2.start, f2.end = s2, blocks[-1]
    f2.retwice = r.retwice
    U.ompcount += 1
    name = '%s._omp_fn.%d' % (f.decl.value.name if f.decl.value.kind & Q.VALUE_QUOTE else f.name, U.ompcount - 1)
    f2.name = name
    # the call in f
    r.data.arg0 = Q.mkintconst(max(8, 8 * len(caps)))
    out = []
    for j, (v, how, c) in enumerate(caps):
        a = r.data.res
        if j:
            ai = Q.mkinst(f, Q.IADD, 'l', a, Q.mkintconst(8 * j))
            out.append(ai)
            a = ai.res
        if how:
            li = Q.mkinst(f, how[0], how[1], v, None)
            out.append(li)
            v = li.res
        out.append(Q.mkinst(f, _STORES[c], 0, v, a))
    out.append(Q.mkinst(f, Q.ICALL, 0, _sym('__cimo_par'), None))
    for c, a in ((_ptr(), _sym(name, False)), (_ptr(), r.data.res), ('l', r.nthreads)):
        out.append(Q.mkinst(f, Q.IARG, c, a, None))
    r.rcall.insts.extend(out)
    return f2, [s2] + blocks, [data, r.tid, r.team]


def _emit(f2, blocks, params):
    pc = _ptr()
    fn = _sym(f2.name, False)
    if f2.retwice and Q.volatile:  # as emitfunc does for a function that calls setjmp
        for b in blocks:
            for i in b.insts:
                if Q.ISTORED <= i.kind <= Q.ILOADUB:
                    i.kind = Q.IVOL + i.kind - Q.ISTORED
        if Q.paraminfo is not None:
            Q.paraminfo.setdefault('#noinline', set()).add(f2.name)
    Q.w('function ')
    Q.emitname(fn)
    Q.w('(')
    for k, (c, v) in enumerate(zip((pc, 'l', pc), params)):
        Q.w(', ' if k else '')
        Q.w(c + ' ')
        Q.emitname(v)
    Q.w(') {\n')
    Q.emitblocks(blocks[0])
    Q.w('}\n')
    if Q.paraminfo is not None:
        Q.paraminfo[f2.name] = [('%.' + str(v.id), False) for v in params]
