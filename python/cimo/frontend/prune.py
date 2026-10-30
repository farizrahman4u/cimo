"""cimo: drop unreferenced static functions and data from the frontend's QBE IR.

cproc emits every function and object it defines, so a `static inline` helper from a header,
or a static table nothing uses, is compiled, assembled and loaded anyway. Exported definitions
are the roots; anything not reachable from them through `$name` references goes. Types are
kept (they cost nothing in the output).

The IR is cproc's: `data` definitions on one line (string bytes are escaped, so a line is
always complete), functions from `function ...{` to a line `}`, `export`/`thread` linkage on
the definition's line or the line before.
"""

import re

_NAME = re.compile(r'\$("(?:[^"\\]|\\.)*"|[A-Za-z_.$][\w.$]*)')


def _defname(line):
    m = _NAME.search(line)
    return m.group(1) if m else None


def prune(ir):
    """IR text with the definitions that no exported definition reaches removed."""
    lines = ir.split('\n')
    items = []  # [name or None (always kept), exported, first line, last line + 1]
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        start = i
        head = line
        if line == 'export' and i + 1 < n:  # cproc puts `export` alone before a function
            i += 1
            head = 'export ' + lines[i]
        words = head.split(' ', 3)
        kind = next((w for w in words if w in ('function', 'data')), None)
        if kind == 'function':
            while i < n and lines[i] != '}':
                i += 1
        if kind is None:
            items.append([None, True, start, i + 1])
        else:
            items.append([_defname(head), 'export' in words, start, i + 1])
        i += 1
    defs = {it[0]: it for it in items if it[0] is not None}
    keep = set()
    work = [it for it in items if it[1] or it[0] is None]
    while work:
        it = work.pop()
        if it[0] is not None:
            if it[0] in keep:
                continue
            keep.add(it[0])
        for j in range(it[2], it[3]):
            for m in _NAME.finditer(lines[j]):
                d = defs.get(m.group(1))
                if d is not None and d[0] not in keep:
                    work.append(d)
    out = []
    for it in items:
        if it[0] is None or it[0] in keep:
            out.extend(lines[it[2]:it[3]])
    return '\n'.join(out)
