"""The wasm target: QBE's target-independent passes, then cimo.backend.wasm.emit (no register
allocation or instruction selection; the wasm engine does both)."""

from ..defs import Target


def _elimsb(fn):
    from ..abi import elimsb
    elimsb(fn)


def _emitfn(fn, obj):
    from .emit import emitfn
    emitfn(fn, obj)


T_wasm = Target(
    name="wasm",
    gpr0=0, ngpr=0, fpr0=0, nfpr=0, rglob=0, nrglob=0, rsave=[-1], nrsave=[0, 0],
    memargs=lambda op: 0,
    abi0=_elimsb,
    emitfn=_emitfn,
    emitfin=lambda obj: None,
    cansel=0,
)
