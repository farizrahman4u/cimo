"""Port of QBE abi.c."""

from .defs import Oarg, Opar, Jretw, isargbh, isparbh, isretbh


def elimsb(fn):
    """eliminate sub-word abi op variants for targets that treat char/short/...
    as words with arbitrary high bits"""
    b = fn.start
    while b:
        for i in b.ins[:b.nins]:
            if isargbh(i.op):
                i.op = Oarg
            if isparbh(i.op):
                i.op = Opar
        if isretbh(b.jmp.type):
            b.jmp.type = Jretw
        b = b.link
