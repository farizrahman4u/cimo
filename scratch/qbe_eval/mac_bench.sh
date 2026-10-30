#!/bin/bash
# Architecture check on macOS: patched QBE (Apple target) vs Apple clang.
# Run from a bundle dir containing qbe/ (patched QBE source), kernels.c, harness.c, gen_opt.py, edge.c.
set -e
cd "$(dirname "$0")"
make -C qbe -j8 >/dev/null 2>&1
grep -h "Deftgt\|T_" qbe/config.h | head -2
clang --version | head -1
mkdir -p out
python3 gen_opt.py > out/opt.ssa
./qbe/qbe -o out/opt.s out/opt.ssa
clang -c out/opt.s -o out/opt.o
# the other 11 kernels come from clang; strip the 4 hand-optimized ones out of kernels.c
python3 - <<'PY'
import re
src = open("kernels.c").read()
for k in ("saxpy", "gemm", "jacobi2d", "conv3x3"):
    src = re.sub(r"\nvoid " + k + r"\(.*?\n}\n", "\n", src, flags=re.S)
open("out/rest.c", "w").write(src)
PY
SC="-O3 -mcpu=native -fno-vectorize -fno-slp-vectorize"
clang $SC -ffp-contract=off -c out/rest.c -o out/rest.o
clang -O0 -ffp-contract=off edge.c out/opt.o out/rest.o -o out/edge && ./out/edge | tail -1
build() { clang $2 -c kernels.c -o out/k-$1.o && clang -O2 harness.c out/k-$1.o -o out/bench-$1; }
build clang-O3-scalar-nofma "$SC -ffp-contract=off"
build clang-O3-scalar "$SC"
build clang-O3 "-O3 -mcpu=native"
clang -O2 harness.c out/opt.o out/rest.o -o out/bench-qbe-opt
for r in 1 2 3; do for b in clang-O3-scalar-nofma clang-O3-scalar clang-O3 qbe-opt; do
  ./out/bench-$b | grep -E "^(saxpy|gemm|jacobi2d|conv3x3) " | sed "s/^/$b /"; done; done > out/times.txt
python3 - <<'PY'
best = {}
for line in open("out/times.txt"):
    cfg, k, ms, cs = line.split()
    key = (k, cfg)
    best[key] = min(best.get(key, 1e30), float(ms))
cfgs = ["clang-O3-scalar-nofma", "clang-O3-scalar", "clang-O3", "qbe-opt"]
print(f"{'kernel':<10}" + "".join(f"{c:>23}" for c in cfgs) + f"{'opt/scalar-nofma':>18}{'opt/scalar':>12}")
for k in ("saxpy", "gemm", "jacobi2d", "conv3x3"):
    t = [best[(k, c)] for c in cfgs]
    print(f"{k:<10}" + "".join(f"{x:>21.1f}ms" for x in t) + f"{t[3]/t[0]:>17.2f}x{t[3]/t[1]:>11.2f}x")
PY
