#!/usr/bin/env bash
# Mutation check of the K-tile accumulator: each mutant of npu_top.sv must make the shell testbench FAIL.
# Usage: bash scripts/mutate_shell.sh   (run from PowerShell on this laptop, like sim_shell.sh)
cd "$(dirname "$0")/.."
mkdir -p build/mut
PY=$(command -v python || command -v python3 || command -v py)
$PY hw/tb/shell/gen_shell_tests.py 16 64 build/mut/shell.cmd 16 > /dev/null || exit 2
CORE="hw/rtl/core/sa_pe.sv hw/rtl/core/sa_core.sv hw/rtl/core/npu_postproc.sv hw/rtl/shell/npu_fifo.sv"
caught=0; total=0
run() {   # $1 name, $2 sed expression
  total=$((total+1))
  sed "$2" hw/rtl/shell/npu_top.sv > build/mut/npu_top_m.sv
  if cmp -s hw/rtl/shell/npu_top.sv build/mut/npu_top_m.sv; then echo "[$1] mutation did not apply"; return; fi
  iverilog -g2012 -P tb_npu_top.N=16 -P tb_npu_top.ACC_DEPTH=64 -P 'tb_npu_top.SCRIPT="build/mut/shell.cmd"' -o build/mut/tb hw/tb/shell/tb_npu_top.sv $CORE build/mut/npu_top_m.sv 2>/dev/null || { echo "[$1] compile error"; return; }
  if vvp -n build/mut/tb | grep -q "RESULT: PASS"; then echo "[$1] NOT caught"; else echo "[$1] caught"; caught=$((caught+1)); fi
}
run "accumulator never cleared"  's/(s1_first ? 32.d0 : s1_old\[32 \* c +: 32\])/s1_old[32 * c +: 32]/'
run "LAST flag ignored"          's/.in_valid(s1_valid \&\& s1_last)/.in_valid(s1_valid)/'
run "no read-after-write bypass" 's/assign s1_old = (wq_en \&\& wq_addr == s1_slot) ? wq_data : acc_rd;/assign s1_old = acc_rd;/'
run "slot counter not restarted" 's/        slot_ctr   <= .0;/        slot_ctr   <= slot_ctr;/'
run "FIRST forced on"            "s/(acc_en ? pass_first : 1'b1)/1'b1/"
echo "mutants caught: $caught / $total"
[ $caught -eq $total ]
