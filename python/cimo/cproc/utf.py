"""Port of utf.c. Byte sequences are lists/bytearrays of ints; functions return
the encoded units (or (c, n) for utf8dec, with n == -1 on error)."""

M32 = 0xffffffff


def utf8enc(c):
    """Returns the list of bytes encoding c."""
    if c < 0x80:
        return [c]
    if c < 0x800:
        return [0xc0 | c >> 6, 0x80 | c & 0x3f]
    if c < 0xd800 or (c - 0xe000) & M32 < 0x2000:
        return [0xe0 | c >> 12, 0x80 | c >> 6 & 0x3f, 0x80 | c & 0x3f]
    if (c - 0x10000) & M32 < 0x100000:
        return [0xf0 | c >> 18, 0x80 | c >> 12 & 0x3f, 0x80 | c >> 6 & 0x3f, 0x80 | c & 0x3f]
    assert 0
    return []


def utf8dec(s, pos, n):
    """Decodes from the str s (one char per byte) at pos, reading at most n bytes.
    Returns (c, length); length is -1 on error."""
    b = ord(s[pos])
    if b < 0x80:
        return b, 1
    if (b & 0xe0) == 0xc0:
        x = b & 0x1f
        ln = 2
    elif (b & 0xf0) == 0xe0:
        x = b & 0x0f
        ln = 3
    elif (b & 0xf8) == 0xf0:
        x = b & 0x07
        ln = 4
    else:
        return 0, -1
    if n < ln:
        return 0, -1
    for i in range(1, ln):
        b = ord(s[pos + i]) if pos + i < len(s) else 0
        if (b & 0xc0) != 0x80:
            return 0, -1
        x = x << 6 | b & 0x3f
    if x >= 0x110000 or (x - 0xd800) & M32 < 0x0200:
        return 0, -1
    return x, ln


def utf16enc(c):
    """Returns the list of 16-bit units encoding c."""
    if c < 0xd800 or (c - 0xe000) & M32 < 0x2000:
        return [c & 0xffff]
    c = (c - 0x10000) & M32
    if c < 0x100000:
        return [0xd800 | c >> 10 & 0x3ff, 0xdc00 | c & 0x3ff]
    assert 0
    return []
