#!/usr/bin/env bash
# npu_top shell + K-tile accumulator + X replay buffer (v1.3): replay generated APB scripts at N = 4 and 16 against the RTL.
# Usage: bash scripts/sim_shell.sh
cd "$(dirname "$0")/.."
mkdir -p build
PY=$(command -v python || command -v python3 || command -v py)
RTL="hw/rtl/core/sa_pe.sv hw/rtl/core/sa_core.sv hw/rtl/core/npu_postproc.sv hw/rtl/shell/npu_fifo.sv hw/rtl/shell/npu_top.sv"
fail=0
for cfg in "4 32 64" "16 64 256" "16 1024 8192"; do
  set -- $cfg; n=$1; d=$2; xr=$3
  $PY hw/tb/shell/gen_shell_tests.py $n $d build/shell_N${n}_D${d}.cmd $n $xr > /dev/null || exit 2
  iverilog -g2012 -P tb_npu_top.N=$n -P tb_npu_top.ACC_DEPTH=$d -P tb_npu_top.XR_DEPTH=$xr -P "tb_npu_top.SCRIPT=\"build/shell_N${n}_D${d}.cmd\"" \
    -o build/tb_npu_top_${n}_${d} hw/tb/shell/tb_npu_top.sv $RTL || exit 2
  out=$(vvp -n build/tb_npu_top_${n}_${d})
  echo "$out" | grep -E "RESULT|MISMATCH|TIMEOUT|APB_acc|tb_npu_top|misaligned" | sed "s/^/[N=$n D=$d] /"
  echo "$out" | grep -q "RESULT: PASS" || fail=1
done
# the first NPU layers of the depth model + Canny tiles (ml/make_hw_vectors.py): real data, expected = golden layer outputs
iverilog -g2012 -P tb_npu_top.N=16 -P tb_npu_top.ACC_DEPTH=1024 -P "tb_npu_top.SCRIPT=\"build/vec.cmd\"" -o build/tb_npu_top_vec hw/tb/shell/tb_npu_top.sv $RTL || exit 2
for c in rand canny_blur canny_sobel enc0 enc1pw enc3pw; do
  cp hw/tb/vectors/$c.cmd build/vec.cmd
  out=$(vvp -n build/tb_npu_top_vec)
  echo "$out" | grep -E "RESULT|MISMATCH|TIMEOUT" | sed "s/^/[vectors $c] /"
  echo "$out" | grep -q "RESULT: PASS" || fail=1
done
if [ $fail -eq 0 ]; then echo "ALL PASS"; else echo "SOME FAILED"; fi
exit $fail
