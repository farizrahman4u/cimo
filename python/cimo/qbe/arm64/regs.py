"""Port of QBE arm64/all.h."""

from ..defs import RXX, Tmp0

_gp = ('R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 IP0 IP1 R18 R19 R20 R21 '
       'R22 R23 R24 R25 R26 R27 R28 FP LR SP').split()
_fp = ['V%d' % i for i in range(31)]
for _i, _r in enumerate(_gp + _fp):
    globals()[_r] = RXX + 1 + _i
del _i, _r

NFPR = V30 - V0 + 1
NGPR = SP - R0 + 1
NGPS = R18 - R0 + 1 + 1  # + LR
NFPS = (V7 - V0 + 1) + (V30 - V16 + 1)
NCLR = (R28 - R19 + 1) + (V15 - V8 + 1)
assert V30 < Tmp0
