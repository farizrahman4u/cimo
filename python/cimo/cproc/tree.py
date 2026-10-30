"""Port of tree.c: the AVL tree used for switch cases.

A node is any object with `key`, `child` (a 2-list), `height` and `new`. The
`void **` slots of the C code are (container, index) pairs: the root slot is
(holder, None) with the root in holder.root.
"""


def height(n):
    return n.height if n else 0


def _get(slot):
    c, i = slot
    return c.root if i is None else c.child[i]


def _set(slot, v):
    c, i = slot
    if i is None:
        c.root = v
    else:
        c.child[i] = v


def rot(p, x, dir):
    y = x.child[dir]
    z = y.child[1 - dir]
    hx = x.height
    hz = height(z)
    if hz > height(y.child[dir]):
        x.child[dir] = z.child[1 - dir]
        y.child[1 - dir] = z.child[dir]
        z.child[1 - dir] = x
        z.child[dir] = y
        x.height = hz
        y.height = hz
        z.height = hz + 1
    else:
        x.child[dir] = z
        y.child[1 - dir] = x
        x.height = hz + 1
        y.height = hz + 2
        z = y
    _set(p, z)
    return z.height - hx


def balance(p):
    n = _get(p)
    h0 = height(n.child[0])
    h1 = height(n.child[1])
    if ((h0 - h1 + 1) & 0xffffffff) < 3:
        old = n.height
        n.height = h1 + 1 if h0 < h1 else h0 + 1
        return n.height - old
    return rot(p, n, 1 if h0 < h1 else 0)


def treeinsert(holder, key, mknode):
    """Insert key into the tree rooted at holder.root; mknode() allocates a node."""
    a = []
    n = holder.root
    a.append((holder, None))
    while n:
        if key == n.key:
            n.new = False
            return n
        d = 1 if key > n.key else 0
        a.append((n, d))
        n = n.child[d]
    n = mknode()
    n.key = key
    n.child = [None, None]
    n.height = 1
    i = len(a) - 1
    _set(a[i], n)
    while i and balance(a[i - 1]):
        i -= 1
    n.new = True
    return n
