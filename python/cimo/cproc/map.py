"""Port of map.c.

cproc's maps (string -> pointer/int hash tables with FNV-1a and linear probing)
are only used for lookup, never iterated in an order that reaches the output, so
the port uses Python dicts keyed by str/bytes everywhere (token table, scopes,
string literals, goto labels). `hash` is kept for reference.
"""


def hash(s):
    """FNV-1a over a bytes-like object, as unsigned long (64-bit)."""
    h = 0x811c9dc5
    for c in s:
        h = ((h ^ c) * 0x1000193) & 0xffffffffffffffff
    return h
