#!/usr/bin/env bash
# Full sa_pe / sa_core verification.
#   1. tb_pe        — PE unit test (exhaustive 256x256 products + random stress, ZERO_SKIP 0 and 1)
#   2. sim_ref      — the shared acceptance test (reference/tb/tb_sa.v) at N = 4, 8, 16
#   3. tb_core      — N = 4/8/16 x ZERO_SKIP 0/1 x ZERO_PCT 0/30 x GAP_PCT 0/30 (exact latency, in-order, no invented outputs)
# Usage: scripts/sim_core.sh [rtl dir]   (rtl dir default hw/rtl/core; scripts/mutate_core.sh uses another dir)
cd "$(dirname "$0")/.."
RTL=${1:-hw/rtl/core}
SRC="$RTL/sa_pe.sv $RTL/sa_core.sv"
mkdir -p build
fail=0

iverilog -g2012 -o build/tb_pe hw/tb/core/tb_pe.sv $RTL/sa_pe.sv || exit 2
vvp -n build/tb_pe | grep -E "RESULT|MISMATCH|tb_pe:" | sed 's/^/[tb_pe] /'
vvp -n build/tb_pe | grep -q "RESULT: PASS" || fail=1

for n in 4 8 16; do
  iverilog -g2012 -P tb_sa.N=$n -o build/tb_sa_$n reference/tb/tb_sa.v $SRC || exit 2
  out=$(vvp -n build/tb_sa_$n | grep -E "RESULT|MISMATCH")
  echo "$out" | sed "s/^/[tb_sa N=$n] /"
  echo "$out" | grep -q "RESULT: PASS" || fail=1
done

for n in 4 8 16; do for zs in 0 1; do for zp in 0 30; do for gp in 0 30; do
  iverilog -g2012 -P tb_core.N=$n -P tb_core.ZERO_SKIP=$zs -P tb_core.ZERO_PCT=$zp -P tb_core.GAP_PCT=$gp \
    -o build/tb_core hw/tb/core/tb_core.sv $SRC || exit 2
  out=$(vvp -n build/tb_core | grep -E "RESULT|MISMATCH|LATENCY|EXTRA|N=")
  echo "$out" | grep -q "RESULT: PASS" || { fail=1; echo "$out" | sed 's/^/[tb_core] /'; }
  echo "$out" | grep -E "^N=" | sed 's/^/[tb_core] /' | tr '\n' ' '; echo "$out" | grep -E "RESULT"
done; done; done; done

if [ $fail -eq 0 ]; then echo "ALL PASS"; else echo "SOME FAILED"; fi
exit $fail
