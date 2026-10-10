#!/usr/bin/env bash
# Mutation check: every injected bug in npu_postproc must make sim_postproc.sh FAIL.
# On this laptop run it from PowerShell:  bash scripts/mutate_postproc.sh
cd "$(dirname "$0")/.."
SRC=hw/rtl/core/npu_postproc.sv
mkdir -p build/mut_pp
caught=0; total=0
run_mut() {  # name, sed expression
  total=$((total+1))
  sed "$2" $SRC > build/mut_pp/npu_postproc.sv
  if cmp -s $SRC build/mut_pp/npu_postproc.sv; then echo "[$1] MUTANT DID NOT APPLY"; return; fi
  if bash scripts/sim_postproc.sh build/mut_pp > /dev/null 2>&1; then echo "[$1] SURVIVED (bad)"; else echo "[$1] caught"; caught=$((caught+1)); fi
}
run_mut "no rounding"        's/(prod2 + (49.sd1 <<< (s_c - 5.d1)))/prod2/'
run_mut "logical shift"      's/assign shifted = rnd >>> s_c;/assign shifted = $signed($unsigned(rnd) >> s_c);/'
run_mut "relu floor -128"    's/relu_en ? zp_x : -49.sd128/-49'"'"'sd128/'
run_mut "sat high 128"       's/withzp > 49.sd127/withzp > 49'"'"'sd128/'
run_mut "bias ignored"       's/ + \$signed(bias\[c\*32 +: 32\]);/;/'
run_mut "sum zero-extended" 's/\$signed({{16{sum1\[32\]}}, sum1})/$signed({16'"'"'d0, sum1})/'
run_mut "zp not added"       's/assign withzp  = shifted + zp_x;/assign withzp  = shifted;/'
run_mut "round at S=0"       's/(s_c == 5.d0) ? prod2 : (prod2 + (49.sd1 <<< (s_c - 5.d1)))/(prod2 + (49'"'"'sd1 <<< (s_c - 5'"'"'d1)))/'
run_mut "out_raw has bias"   's/raw1    <= in_acc;/raw1    <= in_acc + bias;/'
echo "mutants caught: $caught / $total"
[ $caught -eq $total ]
