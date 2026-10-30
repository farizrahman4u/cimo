"""Package files: the bundled headers (`include/`), the runtime sources (`runtime/`) and the
compiler's own sources (hashed into cache keys).

Installed or in the repo, they are files next to the package. In the single-file bundle
(cimo.bundle) they are a table embedded in `cimo.py`, and ROOT is a name like
`<cimo.py>/cimo` that exists only there. Paths elsewhere (user and host headers, the cache)
are read from the filesystem, so callers can mix both kinds of path freely.
"""

import os

# The bundle's loader carries the embedded files: {path relative to the package: text}.
_FILES = getattr(__loader__, 'bundle_files', None)
ROOT = __loader__.bundle_root if _FILES is not None else os.path.dirname(
    os.path.abspath(__file__))
_PREFIX = ROOT + os.sep


def path(rel):
    """The path of a package file, from its '/'-separated name ('include/stdio.h')."""
    return os.path.join(ROOT, *rel.split('/'))


def _embedded(p):
    """The table key of path `p` if the bundle embeds it, else None."""
    if _FILES is None or not p.startswith(_PREFIX):
        return None
    return p[len(_PREFIX):].replace(os.sep, '/')


def abspath(p):
    """os.path.abspath, keeping the bundle's paths as they are."""
    return p if _embedded(p + os.sep) is not None else os.path.abspath(p)


def isfile(p):
    k = _embedded(p)
    return os.path.isfile(p) if k is None else k in _FILES


def read(p):
    """The contents of file `p` as bytes (UTF-8 for embedded files)."""
    k = _embedded(p)
    if k is None:
        with open(p, 'rb') as f:
            return f.read()
    try:
        return _FILES[k].encode('utf-8', 'surrogateescape')
    except KeyError:
        raise FileNotFoundError('No such file in the bundle: %r' % p) from None


def read_text(p):
    return read(p).decode('utf-8', 'surrogateescape')


def _walk_order(k):
    parts = k.split('/')
    return [(1, d) for d in parts[:-1]] + [(0, parts[-1])]


def files(top, exts):
    """Sorted paths of the files under directory `top` (recursively) whose names end with one
    of `exts`, skipping __pycache__."""
    pre = _embedded(top + os.sep)
    if pre is not None:  # in os.walk's order: a directory's files, then its subdirectories
        return [path(k) for k in sorted(_FILES, key=_walk_order)
                if k.startswith(pre) and k.endswith(exts) and '/__pycache__/' not in k]
    out = []
    for d, dirs, names in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x != '__pycache__')
        for f in sorted(names):
            if f.endswith(exts):
                out.append(os.path.join(d, f))
    return out
