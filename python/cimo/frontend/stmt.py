"""Port of stmt.c."""

from .cc import (SwitchCases, TCASE, TDEFAULT, TIDENT, TCOLON, TLBRACE, TRBRACE, TSEMICOLON,
                 TIF, TLPAREN, TRPAREN, TELSE, TSWITCH, TWHILE, TDO, TFOR, TGOTO, TCONTINUE,
                 TBREAK, TRETURN, T__ASM__, PROPSCALAR, PROPINT, TELLIPSIS, TMUL, DECLOBJECT, SDAUTO)
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
        if U.diagnostics and tok.kind == TELLIPSIS:
            error(tok.loc, "case ranges ('case a ... b', a GNU extension) are not supported")
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
        while tok.kind != TRBRACE:
            if not label(f, s) and not D.decl(s, f):
                stmt(f, s)
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
        if U.diagnostics:
            error(tok.loc, DG.INLINEASM)
        error(tok.loc, "inline assembly is not yet supported")
    else:
        e = E.expr(s)
        v = Q.funcexpr(f, e)
        E.delexpr(e)
        expect(TSEMICOLON, "after expression statement")
