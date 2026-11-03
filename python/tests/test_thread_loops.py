"""Jump threading (backend/thread.py) leaves each block's loop depth right for the passes after it: loadopt
asserts on them. Csmith found a stale one on x86-64, through a function with much inlined into it."""

import pytest

import middle_gen
from cimo import driver
from cimo.backend import cfg, thread


@pytest.mark.parametrize('target', ['arm64', 'amd64_sysv'])
def test_loop_depths_after_threading(monkeypatch, target):
    stale = []
    run = thread.run

    def checked(fn):
        r = run(fn)
        had = [(b.name, b.loop) for b in fn.rpo[:fn.nblk]]
        cfg.fillloop(fn)
        now = [(b.name, b.loop) for b in fn.rpo[:fn.nblk]]
        if had != now:
            stale.append((fn.name, [(a, b) for a, b in zip(had, now) if a != b][:3]))
        return r

    monkeypatch.setattr(thread, 'run', checked)
    # gcc-torture's loop-12.c: the condition's || threads into the loop's latch
    srcs = ['char *p;\nstatic int end(void) { return *p == \'\\n\' || *p == \';\' || *p == \'!\'; }\n'
            'void foo(void) { while (!end()) p++; }\n']
    srcs += [middle_gen.program(seed)[0] for seed in range(10)]
    for src in srcs:
        pp = driver._preprocess(src, 'r.c', (), False, (), target=target)
        ssa, params = driver._frontend(pp, 'r.c', target=target, fma=False, nowrap=True, builtins=True,
                                       diagnostics=True, prune=True)
        driver._qbe(ssa, params, target=target)
    assert not stale, stale[:5]
