#!/usr/bin/env bash
# Mutation check: break the RTL on purpose in build/mut/ and make sure scripts/sim_core.sh FAILS for each mutant.
# A mutant that still passes means the testbench is blind to that bug. Run from PowerShell: bash scripts/mutate_core.sh
cd "$(dirname "$0")/.."
SRC=hw/rtl/core
bad=0
run_mutant () {   # name, file, sed expression
  rm -rf build/mut; mkdir -p build/mut; cp $SRC/sa_pe.sv $SRC/sa_core.sv build/mut/
  sed -i "$3" build/mut/$2
  if cmp -s $SRC/$2 build/mut/$2; then echo "[$1] MUTATION DID NOT APPLY"; bad=1; return; fi
  if bash scripts/sim_core.sh build/mut >build/mut/log.txt 2>&1; then echo "[$1] SURVIVED (tests did not notice) -> BAD"; bad=1
  else echo "[$1] caught (sim_core FAILED as it must)"; fi
}
run_mutant no_deskew       sa_core.sv 's/assign out_vec\[c\*ACC_W +: ACC_W\] = o_dly\[D-1\];/assign out_vec[c*ACC_W +: ACC_W] = ps_link[N][c];/'
run_mutant no_input_skew   sa_core.sv 's/assign a_skew\[r\] = a_dly\[D-1\];/assign a_skew[r] = a_head;/'
run_mutant ignore_flag     sa_pe.sv   's/assign w_eff = flag_in ? sw_out : w_act;/assign w_eff = w_act;/'
run_mutant late_latency    sa_core.sv 's/assign out_valid = vpipe\[2\*N-2\];/assign out_valid = vpipe[2*N-3];/'
run_mutant skip_drops_swap sa_pe.sv   's/w_act    <= w_eff;/if (!skip) w_act <= w_eff;/'
run_mutant skip_wrong_ps   sa_pe.sv   's/ps_out   <= skip ? ps_in : /ps_out   <= skip ? ps_out : /'
rm -rf build/mut
[ $bad -eq 0 ] && echo "MUTATION CHECK OK" || echo "MUTATION CHECK FAILED"
exit $bad
