#!/usr/bin/env bash
# npu_postproc verification vs reference/qmath_ref.py rq(). N = 4, 8, 16, 10,000 vectors each (x N columns).
# Usage: bash scripts/sim_postproc.sh [rtl dir]   (default hw/rtl/core; mutate_postproc.sh passes another dir)
cd "$(dirname "$0")/.."
RTL=${1:-hw/rtl/core}
mkdir -p build
PY=$(command -v python || command -v python3 || command -v py)
fail=0
for n in 4 8 16; do
  $PY hw/tb/core/gen_pp_vectors.py $n 400 25 build/pp_N$n > /dev/null || exit 2
  iverilog -g2012 -P tb_postproc.N=$n -P "tb_postproc.PFX=\"build/pp_N$n\"" -o build/tb_postproc_$n hw/tb/core/tb_postproc.sv $RTL/npu_postproc.sv || exit 2
  out=$(vvp -n build/tb_postproc_$n | grep -E "RESULT|MISMATCH|tb_postproc:")
  echo "$out" | sed "s/^/[N=$n] /"
  echo "$out" | grep -q "RESULT: PASS" || fail=1
done
if [ $fail -eq 0 ]; then echo "ALL PASS"; else echo "SOME FAILED"; fi
exit $fail
