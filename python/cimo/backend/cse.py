"""cimo: common subexpressions QBE leaves behind (ext.cse; not in QBE).

gvn merges pure instructions, but QBE pins integer divisions (they may trap), and load
elimination runs before gvn, when two computations of an address are still two temporaries (and
gcm's sink copies them again). With this flag:

- main runs loadopt again after gvn, on one temporary per address (domcse);
- on arm64, where sdiv and udiv never trap, divisions are pure (gcm.isdivwl), so they merge across
  sibling branches and leave loops, and divrem computes a remainder from the division of the same
  operands: a - (a / b) * b is what the rem instruction's own sdiv + msub computes, also for
  b == 0 and INT_MIN / -1;
- on wasm, where they trap, gvn merges a division with an identical one that dominates it (if the
  first did not trap, neither does the second);
- gvn merges phis that join the two ways of the same condition with the same values, and removes
  a jnz that a dominating jnz on the same temporary has decided: a bounds check repeated on an
  index computed the same way (gvn.dedupselect, gvn.knownjmp).
"""

from . import defs as D
from .defs import KBASE
from .cfg import dom
from .gcm import pinned
from .gvn import replaceuses
from .util import newtmp

_DIV = {D.Odiv: D.Odiv, D.Orem: D.Odiv, D.Oudiv: D.Oudiv, D.Ourem: D.Oudiv}


def nload(fn):
    """How many loads fn has: load elimination removed some if it went down."""
    n = 0
    b = fn.start
    while b:
        n += sum(1 for i in b.ins[:b.nins] if D.isload(i.op))
        b = b.link
    return n


def domcse(fn):
    """Merge a pure instruction with an identical one that dominates it, which undoes gcm's sink
    (it copies cheap address arithmetic to each load and store): loads can only match on one
    temporary. Requires rpo, dom and use, maintains them. Returns True if anything changed."""
    seen = {}
    changed = False
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if i.op == D.Onop or pinned(i) or i.to == D.R:
                continue
            for b1, r in seen.setdefault((i.op, i.cls, i.arg[0], i.arg[1]), []):
                if dom(b1, b):
                    replaceuses(fn, i.to, r)
                    i.set(D.Ins(D.Onop))
                    changed = True
                    break
            else:
                seen[(i.op, i.cls, i.arg[0], i.arg[1])].append((b, i.to))
    return changed


def divrem(fn):
    """arm64: a remainder whose operands also have a division becomes a - (a / b) * b, and gvn
    then merges the divisions. Requires args as gvn left them; breaks use. Returns True if
    anything changed."""
    divs = set()
    rems = False
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if i.op in (D.Odiv, D.Oudiv) and KBASE(i.cls) == 0:
                divs.add((i.op, i.cls, i.arg[0], i.arg[1]))
            elif i.op in (D.Orem, D.Ourem) and KBASE(i.cls) == 0:
                rems = True
    if not (divs and rems):
        return False
    changed = False
    for b in fn.rpo[:fn.nblk]:
        out = []
        for i in b.ins[:b.nins]:
            op = D.Odiv if i.op == D.Orem else D.Oudiv if i.op == D.Ourem else None
            if op is None or (op, i.cls, i.arg[0], i.arg[1]) not in divs:
                out.append(i)
                continue
            d = newtmp('div', i.cls, fn)
            t = newtmp('mul', i.cls, fn)
            out.append(D.Ins(op, i.cls, d, i.arg[0], i.arg[1]))
            out.append(D.Ins(D.Omul, i.cls, t, d, i.arg[1]))
            out.append(D.Ins(D.Osub, i.cls, i.to, i.arg[0], t))
            changed = True
        b.ins = out
        b.nins = len(out)
    return changed
