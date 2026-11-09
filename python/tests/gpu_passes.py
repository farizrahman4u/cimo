"""Which of cimo's passes are worth running on GPU code: each group off in turn, against all on, for every kernel in
tests/gpu/: cimo's compile time (source to PTX) and the kernel's time on this machine's GPU (best of interleaved
rounds). The numbers go to requirements.md ("PTX: measured").

    python tests/gpu_passes.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cimo.gpu as G  # noqa: E402
import gpu_bench  # noqa: E402

VARIANTS = [('all', {}), ('no middle (unrolling)', {'middle': False}), ('no cse', {'cse': False}),
            ('no inline', {'inline': 0}), ('no nowrap', {'nowrap': False})]


def main():
    gpu = G.device()
    print('%s, sm_%d' % (gpu.name, gpu.sm))
    for fname, kname, mk, launch, oi, check in gpu_bench.cases(gpu):
        src = open(os.path.join(gpu_bench.KDIR, fname)).read()
        res = {}
        mods = {}
        for vname, passes in VARIANTS:
            G.ptx(src, sm=gpu.sm, passes=passes)
            t = time.perf_counter()
            G._ptx1(src, gpu.sm, (), (), '<string>', None, passes)
            res[vname] = [time.perf_counter() - t, None]
            mods[vname] = gpu.compile(src, passes=passes, cache=False)
        args = mk()
        for _ in range(5):
            for vname, m in mods.items():
                k = getattr(m, kname)
                ms = gpu.time(lambda: k(*args, **launch), 20)
                res[vname][1] = ms if res[vname][1] is None else min(res[vname][1], ms)
        print(kname)
        for vname, (ct, ms) in res.items():
            print('    %-22s compile %6.3f s   kernel %7.3f ms' % (vname, ct, ms))


if __name__ == '__main__':
    main()
