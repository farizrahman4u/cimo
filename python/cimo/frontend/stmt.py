"""Port of stmt.c."""

from .cc import (SwitchCases, TCASE, TDEFAULT, TIDENT, TCOLON, TLBRACE, TRBRACE, TSEMICOLON,
                 TIF, TLPAREN, TRPAREN, TELSE, TSWITCH, TWHILE, TDO, TFOR, TGOTO, TCONTINUE,
                 TBREAK, TRETURN, T__ASM__, PROPSCALAR, PROPINT)
from .token import tok, tokenstr, error
from .pp import next, expect, consume, peek
from . import type as TY
from .scope import mkscope, delscope
from . import attr as A
from . import decl as D
from . import expr as E
from . import qbe as Q


def label(f, s):
    """6.8.1 Labeled statements"""
    A.attr(None, 0)
    k = tok.kind
    if k == TCASE:
        next()
        if not s.switchcases:
            error(tok.loc, "'case' label must be in switch")
        b = Q.mkblock("switch_case")
        Q.funclabel(f, b)
        i = E.intconstexpr(s, True)
        Q.switchcase(s.switchcases, i, b)
        expect(TCOLON, "after case expression")
    elif k == TDEFAULT:
        next()
        if not s.switchcases:
            error(tok.loc, "'default' label must be in switch")
        if s.switchcases.defaultlabel:
            error(tok.loc, "multiple 'default' labels")
        expect(TCOLON, "after 'default'")
        s.switchcases.defaultlabel = Q.mkblock("switch_default")
        Q.funclabel(f, s.switchcases.defaultlabel)
    else:
        if tok.kind < TIDENT:
            return False
        name = tokenstr(tok.kind)
        if not peek(TCOLON):
            return False
        g = Q.funcgoto(f, name)
        g.defined = True
        Q.funclabel(f, g.label)
    return True


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
        while tok.kind != TRBRACE:
            if not label(f, s) and not D.decl(s, f):
                stmt(f, s)
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

        b0 = Q.mkblock("switch_cond")
        b1 = Q.mkblock("switch_join")

        v = Q.funcexpr(f, e)
        Q.funcjmp(f, b0)
        s = mkscope(s)
        s.breaklabel = b1
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
        if not D.decl(s, f):
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
        labelstmt(f, s)
        s = delscope(s)

        Q.funclabel(f, b2)
        if e:
            Q.funcexpr(f, e)
            E.delexpr(e)
        Q.funcjmp(f, b0)
        Q.funclabel(f, b3)
        s = delscope(s)

    # 6.8.6 Jump statements
    elif k == TGOTO:
        next()
        name = expect(TIDENT, "after 'goto'")
        Q.funcjmp(f, Q.funcgoto(f, name).label)
        expect(TSEMICOLON, "after 'goto' statement")
    elif k == TCONTINUE:
        next()
        if not s.continuelabel:
            error(tok.loc, "'continue' statement must be in loop")
        Q.funcjmp(f, s.continuelabel)
        expect(TSEMICOLON, "after 'continue' statement")
    elif k == TBREAK:
        next()
        if not s.breaklabel:
            error(tok.loc, "'break' statement must be in loop or switch")
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
        error(tok.loc, "inline assembly is not yet supported")
    else:
        e = E.expr(s)
        v = Q.funcexpr(f, e)
        E.delexpr(e)
        expect(TSEMICOLON, "after expression statement")
