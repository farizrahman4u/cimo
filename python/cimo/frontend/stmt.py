"""Port of stmt.c."""

import re
from builtins import next as next_

from .cc import (SwitchCases, TCASE, TDEFAULT, TIDENT, TCOLON, TLBRACE, TRBRACE, TSEMICOLON,
                 TIF, TLPAREN, TRPAREN, TELSE, TSWITCH, TWHILE, TDO, TFOR, TGOTO, TCONTINUE,
                 TBREAK, TRETURN, T__ASM__, TVOLATILE, TINLINE, TSTRINGLIT, TCOMMA,
                 TLBRACK, TRBRACK, TCOLONCOLON, EXPRSTMT, TYPEARRAY, TYPEFUNC, TYPEVOID, PROPSCALAR, PROPINT, TELLIPSIS, TMUL, DECLOBJECT, SDAUTO)
from .token import tok, tokenstr, error
from .pp import next, expect, consume, peek
from . import type as TY
from .scope import mkscope, delscope
from . import attr as A
from . import decl as D
from . import expr as E
from . import qbe as Q
from . import util as U
from . import diag as DG


def label(f, s):
    """6.8.1 Labeled statements"""
    A.attr(None, 0)
    k = tok.kind
    if k == TCASE:
        next()
        if not s.switchcases:
            error(tok.loc, "'case' label must be in switch")
        if len(f.vlas) > s.switchcases.nvla:  # cimo: C11 6.8.4.2p2
            error(tok.loc, "jump into the scope of a variable length array")
        b = Q.mkblock("switch_case")
        Q.funclabel(f, b)
        i = E.intconstexpr(s, True)
        if U.diagnostics and consume(TELLIPSIS):  # cimo: `case a ... b:`, a GNU case range
            Q.switchrange(s.switchcases, i, E.intconstexpr(s, True), b)
        else:
            Q.switchcase(s.switchcases, i, b)
        expect(TCOLON, "after case expression")
    elif k == TDEFAULT:
        next()
        if not s.switchcases:
            error(tok.loc, "'default' label must be in switch")
        if s.switchcases.defaultlabel:
            error(tok.loc, "multiple 'default' labels")
        expect(TCOLON, "after 'default'")
        if len(f.vlas) > s.switchcases.nvla:  # cimo: C11 6.8.4.2p2
            error(tok.loc, "jump into the scope of a variable length array")
        s.switchcases.defaultlabel = Q.mkblock("switch_default")
        Q.funclabel(f, s.switchcases.defaultlabel)
    else:
        if tok.kind < TIDENT:
            return False
        name = tokenstr(tok.kind)
        if not peek(TCOLON):
            return False
        g = Q.funcgoto(f, name)
        if g.defined:  # cimo: cproc linked the block in twice, a cycle (endless output)
            error(tok.loc, "redefinition of label '%s'", name)
        g.defined = True
        g.vlas = tuple(f.vlas)
        Q.funclabel(f, g.label)
    return True


def _vlaleave(f, n):
    """cimo (vlafree): control leaves the scope of the VLAs after the first n: free them."""
    if Q.vlafree and not f.end.jump_kind:  # else unreachable (after a return, break, ...)
        Q.funcvlaleave(f, n)


def labelstmt(f, s):
    while label(f, s):
        pass
    stmt(f, s)


def stmt(f, s):
    """6.8 Statements and blocks"""
    A.attr(None, 0)
    k = tok.kind
    # 6.8.2 Compound statement
    if k == TLBRACE:
        next()
        s = mkscope(s)
        nvla = len(f.vlas)
        local = _locallabels(f)
        while tok.kind != TRBRACE:
            if not label(f, s) and not D.decl(s, f):
                stmt(f, s)
        _endlocallabels(f, local)
        _vlaleave(f, nvla)
        del f.vlas[nvla:]
        s = delscope(s)
        next()

    # 6.8.3 Expression statement
    elif k == TSEMICOLON:
        next()

    # 6.8.4 Selection statement
    elif k == TIF:
        next()
        s = mkscope(s)
        expect(TLPAREN, "after 'if'")
        e = E.expr(s)
        t = e.type
        if not (t.prop & PROPSCALAR):
            error(tok.loc, "controlling expression of if statement must have scalar type")
        b0 = Q.mkblock("if_true")
        b1 = Q.mkblock("if_false")
        Q.funcbranch(f, e, b0, b1)
        E.delexpr(e)
        expect(TRPAREN, "after expression")

        Q.funclabel(f, b0)
        s = mkscope(s)
        labelstmt(f, s)
        s = delscope(s)

        if consume(TELSE):
            b2 = Q.mkblock("if_join")
            Q.funcjmp(f, b2)
            Q.funclabel(f, b1)
            s = mkscope(s)
            labelstmt(f, s)
            s = delscope(s)
            Q.funclabel(f, b2)
        else:
            Q.funclabel(f, b1)
        s = delscope(s)
    elif k == TSWITCH:
        next()

        s = mkscope(s)
        expect(TLPAREN, "after 'switch'")
        e = E.expr(s)
        expect(TRPAREN, "after expression")

        if not (e.type.prop & PROPINT):
            error(tok.loc, "controlling expression of switch statement must have integer type")
        e = E.exprpromote(e)

        swtch = SwitchCases()
        swtch.root = None
        swtch.type = e.type
        swtch.defaultlabel = None
        swtch.nvla = len(f.vlas)

        b0 = Q.mkblock("switch_cond")
        b1 = Q.mkblock("switch_join")

        v = Q.funcexpr(f, e)
        Q.funcjmp(f, b0)
        s = mkscope(s)
        s.breaklabel = b1
        s.breakvla = len(f.vlas)
        s.switchcases = swtch
        labelstmt(f, s)
        Q.funcjmp(f, b1)

        Q.funclabel(f, b0)
        Q.funcswitch(f, v, swtch, swtch.defaultlabel if swtch.defaultlabel else b1)
        s = delscope(s)

        Q.funclabel(f, b1)
        s = delscope(s)

    # 6.8.5 Iteration statements
    elif k == TWHILE:
        next()
        s = mkscope(s)
        expect(TLPAREN, "after 'while'")
        e = E.expr(s)
        t = e.type
        if not (t.prop & PROPSCALAR):
            error(tok.loc, "controlling expression of loop must have scalar type")
        expect(TRPAREN, "after expression")

        b0 = Q.mkblock("while_cond")
        b1 = Q.mkblock("while_body")
        b2 = Q.mkblock("while_join")

        Q.funclabel(f, b0)
        Q.funcbranch(f, e, b1, b2)
        Q.funclabel(f, b1)
        s = mkscope(s)
        s.continuelabel = b0
        s.breaklabel = b2
        s.breakvla = s.contvla = len(f.vlas)
        labelstmt(f, s)
        s = delscope(s)
        Q.funcjmp(f, b0)
        Q.funclabel(f, b2)
        s = delscope(s)
    elif k == TDO:
        next()

        b0 = Q.mkblock("do_body")
        b1 = Q.mkblock("do_cond")
        b2 = Q.mkblock("do_join")

        s = mkscope(s)
        s = mkscope(s)
        s.continuelabel = b1
        s.breaklabel = b2
        s.breakvla = s.contvla = len(f.vlas)
        Q.funclabel(f, b0)
        labelstmt(f, s)
        s = delscope(s)

        expect(TWHILE, "after 'do' statement")
        expect(TLPAREN, "after 'while'")
        Q.funclabel(f, b1)
        e = E.expr(s)
        t = e.type
        if not (t.prop & PROPSCALAR):
            error(tok.loc, "controlling expression of loop must have scalar type")
        expect(TRPAREN, "after expression")

        Q.funcbranch(f, e, b0, b2)
        Q.funclabel(f, b2)
        s = delscope(s)
        expect(TSEMICOLON, "after 'do' statement")
    elif k == TFOR:
        next()
        expect(TLPAREN, "after 'for'")
        s = mkscope(s)
        nvla = len(f.vlas)
        if D.decl(s, f):
            if U.diagnostics:  # cimo: C11 6.8.5p3, only auto and register objects
                for d in s.decls.values():
                    if d.kind != DECLOBJECT or d.storage != SDAUTO:
                        error(tok.loc, "declaration of '%s' in a 'for' loop is not an "
                              "'auto' or 'register' object", d.name)
        else:
            if tok.kind != TSEMICOLON:
                e = E.expr(s)
                Q.funcexpr(f, e)
                E.delexpr(e)
            expect(TSEMICOLON, None)

        b0 = Q.mkblock("for_cond")
        b1 = Q.mkblock("for_body")
        b2 = Q.mkblock("for_cont")
        b3 = Q.mkblock("for_join")

        Q.funclabel(f, b0)
        if tok.kind != TSEMICOLON:
            e = E.expr(s)
            t = e.type
            if not (t.prop & PROPSCALAR):
                error(tok.loc, "controlling expression of loop must have scalar type")
            Q.funcbranch(f, e, b1, b3)
            E.delexpr(e)
        expect(TSEMICOLON, None)
        e = None if tok.kind == TRPAREN else E.expr(s)
        expect(TRPAREN, None)

        Q.funclabel(f, b1)
        s = mkscope(s)
        s.breaklabel = b3
        s.continuelabel = b2
        s.breakvla = s.contvla = len(f.vlas)
        labelstmt(f, s)
        s = delscope(s)

        Q.funclabel(f, b2)
        if e:
            Q.funcexpr(f, e)
            E.delexpr(e)
        Q.funcjmp(f, b0)
        Q.funclabel(f, b3)
        _vlaleave(f, nvla)
        del f.vlas[nvla:]
        s = delscope(s)

    # 6.8.6 Jump statements
    elif k == TGOTO:
        next()
        if U.diagnostics and tok.kind == TMUL:
            error(tok.loc, "computed goto ('goto *p', a GNU extension) is not supported")
        name = expect(TIDENT, "after 'goto'")
        if Q.vlafree:
            Q.funcvlagoto(f, Q.funcgoto(f, name))
        else:
            Q.funcjmp(f, Q.funcgoto(f, name).label)
        expect(TSEMICOLON, "after 'goto' statement")
    elif k == TCONTINUE:
        next()
        if not s.continuelabel:
            error(tok.loc, "'continue' statement must be in loop")
        _vlaleave(f, s.contvla)
        Q.funcjmp(f, s.continuelabel)
        expect(TSEMICOLON, "after 'continue' statement")
    elif k == TBREAK:
        next()
        if not s.breaklabel:
            error(tok.loc, "'break' statement must be in loop or switch")
        _vlaleave(f, s.breakvla)
        Q.funcjmp(f, s.breaklabel)
        expect(TSEMICOLON, "after 'break' statement")
    elif k == TRETURN:
        next()
        t = Q.functype(f)
        if t.base is not TY.typevoid:
            e = E.exprassign(E.expr(s), t.base)
            v = Q.funcexpr(f, e)
            E.delexpr(e)
        else:
            v = None
        Q.funcret(f, v)
        expect(TSEMICOLON, "after 'return' statement")

    elif k == T__ASM__:
        _asm(f, s)
    else:
        e = E.expr(s)
        v = Q.funcexpr(f, e)
        E.delexpr(e)
        expect(TSEMICOLON, "after expression statement")


_EMPTYASM = re.compile(r'(?:\s|\\[nt])*')


def _asmstrings():
    """Adjacent string literals: their text between the quotes."""
    parts = []
    while tok.kind == TSTRINGLIT:
        parts.append(tok.lit[tok.lit.index('"') + 1:-1])
        next()
    if not parts:
        error(tok.loc, "expected string literal in 'asm' statement")
    return ''.join(parts)


def _colon():
    """Consume one ':' (half of a `::` token: `asm ("" ::: "memory")`)."""
    if tok.kind == TCOLONCOLON:
        tok.kind = TCOLON
        return True
    return consume(TCOLON)


def _asmoperands(s, out):
    """`[name] "constraint" (expression)`, comma-separated: (name, constraint, expression)s."""
    ops = []
    if tok.kind in (TCOLON, TCOLONCOLON, TRPAREN):
        return ops
    while True:
        name = None
        if consume(TLBRACK):
            name = expect(TIDENT, "in 'asm' operand name")
            expect(TRBRACK, "after 'asm' operand name")
        c = _asmstrings()
        expect(TLPAREN, "before 'asm' operand")
        e = E.expr(s)
        expect(TRPAREN, "after 'asm' operand")
        if out and not e.lvalue:
            error(tok.loc, "'asm' output operand is not an lvalue")
        ops.append((name, c, e))
        if not consume(TCOMMA):
            return ops


def _asm(f, s):
    """cimo: a GNU asm statement whose template is empty (`asm volatile ("" ::: "memory")`, gcc's tests'
    and benchmarks' optimization barrier) does nothing, so it is compiled as nothing: its inputs are
    evaluated, an output tied to an input ("0", "[name]") is assigned that input, other outputs are
    evaluated (once, for their side effects) and keep their values. Any other template is an error (cimo has no inline assembly)."""
    loc = tok.loc
    next()
    while tok.kind in (TVOLATILE, TINLINE, TGOTO):
        next()
    expect(TLPAREN, "after 'asm'")
    if tok.kind != TSTRINGLIT or not _EMPTYASM.fullmatch(_asmstrings()):
        error(loc, DG.INLINEASM if U.diagnostics else "inline assembly is not yet supported")
    outs, ins = [], []
    if _colon():
        outs = _asmoperands(s, True)
        if _colon():
            ins = _asmoperands(s, False)
            if _colon():
                if tok.kind == TSTRINGLIT:
                    while True:  # clobbers
                        _asmstrings()
                        if not consume(TCOMMA):
                            break
                if _colon():  # asm goto's labels: never jumped to
                    while tok.kind == TIDENT:
                        next()
                        if not consume(TCOMMA):
                            break
    expect(TRPAREN, "after 'asm' operands")
    expect(TSEMICOLON, "after 'asm' statement")
    tiedouts = set()
    for name, c, e in ins:
        c = c.lstrip('=+&%')
        if c.isdigit():
            tiedouts.add(int(c))
        elif c.startswith('['):
            tiedouts.update(n for n, o in enumerate(outs) if o[0] == c[1:-1])
    for n, (name, c, e) in enumerate(outs):  # an output's side effects (`"+r" (*bar())`), once
        if n not in tiedouts:
            Q.funcexpr(f, e)
    for name, c, e in ins:
        c = c.lstrip('=+&%')
        tied = None
        if c.isdigit() and int(c) < len(outs):
            tied = outs[int(c)][2]
        elif c.startswith('[') and c.endswith(']'):
            tied = next_((o[2] for o in outs if o[0] == c[1:-1]), None)
        if tied is not None:
            e = E.mkassignexpr(tied, e)
        Q.funcexpr(f, e)
        E.delexpr(e)


_STMTSTART = frozenset((TLBRACE, TSEMICOLON, TIF, TSWITCH, TWHILE, TDO, TFOR, TGOTO, TCONTINUE, TBREAK, TRETURN,
                        T__ASM__))


def stmtexpr(s):
    """cimo: a GNU statement expression `({ ... })`, at its '{'. Its body is emitted now, into blocks of their own
    (cproc emits statements as it parses them, an expression's code only after the whole expression is parsed); the
    expression (EXPRSTMT) splices them in where its code goes, so the body runs in the expression's evaluation
    order (`a && ({ ... })`, a for loop's third expression). Its value: the last statement's, if an expression."""
    f = Q.curfunc
    if f is None:
        error(tok.loc, "statement expression outside a function")
    saved = f.end
    head = Q.mkblock("stmt_expr")
    f.end = head
    next()
    s = mkscope(s)
    nvla = len(f.vlas)
    local = _locallabels(f)
    t, v = TY.typevoid, None
    while tok.kind != TRBRACE:
        t, v = TY.typevoid, None
        if label(f, s) or D.decl(s, f):
            continue
        if tok.kind in _STMTSTART:
            stmt(f, s)
            continue
        e = E.expr(s)
        expect(TSEMICOLON, "after expression statement")
        if tok.kind == TRBRACE:
            if e.decayed:
                pass  # an array or function: already its pointer
            t = e.type
            if t.kind == TYPEARRAY or t.kind == TYPEFUNC:
                t = TY.typevoid
            v = Q.funcexpr(f, e)
            if t.kind == TYPEVOID:
                v = None
        else:
            Q.funcexpr(f, e)
        E.delexpr(e)
    _endlocallabels(f, local)
    _vlaleave(f, nvla)
    del f.vlas[nvla:]
    s = delscope(s)
    next()
    last = f.end
    f.end = saved
    r = E.Expr(EXPRSTMT, t, None)
    r.ct, r.cf, r.temp, r.post = head, last, v, False
    return r


def _locallabels(f):
    """cimo: GNU local labels, `__label__ a, b;` at a block's start: labels of this block only (a macro's
    statement expression expanded twice in one function). Returns {name: the label it hides, or None}."""
    hidden = {}
    while U.diagnostics and tok.kind >= TIDENT and tokenstr(tok.kind) == '__label__':
        next()
        while True:
            if tok.kind < TIDENT:
                error(tok.loc, "expected identifier after '__label__'")
            name = tokenstr(tok.kind)
            next()
            if name not in hidden:
                hidden[name] = f.gotos.pop(name, None)
            Q.funcgoto(f, name)
            if not consume(TCOMMA):
                break
        expect(TSEMICOLON, "after '__label__' declaration")
    return hidden


def _endlocallabels(f, hidden):
    for name, old in hidden.items():
        g = f.gotos.pop(name)
        if not g.defined and any(c[1] is g for c in f.gotochecks):
            error(tok.loc, "label '%s' used but not defined", name)
        if old is not None:
            f.gotos[name] = old
