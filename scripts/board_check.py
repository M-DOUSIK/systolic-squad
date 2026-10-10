"""On-board check: weights, SELFTEST, golden frames bit-exact through the NPU path, BENCH CPU vs NPU.
  python scripts/board_check.py [COM8] [export dir name, default export; export_midas for MiDaS]"""
import os, struct, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))
import proto as P
from app import Board

port = sys.argv[1] if len(sys.argv) > 1 else "COM8"
EXP = sys.argv[2] if len(sys.argv) > 2 else "export"
b = Board(port, 921600)
pong = b.ping()
print("PONG firmware %08X, NPU id %08X" % pong)
print("model", EXP, "-> slot", b.select_model(1 if EXP == "export_midas" else 0))
t = time.time()
print("weights ok:", b.load_weights(open(os.path.join(ROOT, "ml", EXP, "model_blob.bin"), "rb").read()), "(%.0f s)" % (time.time() - t))
b.send(P.cmd(P.CMD_SELFTEST))
g, _ = b.wait([P.T_SELFTEST], 30)
if P.T_SELFTEST in g:
    ok, n, nf = struct.unpack("<BHH", g[P.T_SELFTEST])
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'} ({n - nf}/{n} NPU results equal the CPU)")
else:
    print("SELFTEST: no answer")
b.send(P.cmd(P.CMD_SET_GATE, 0))
for fr in (0, 5, 9):
    rgb = np.fromfile(os.path.join(ROOT, "ml", EXP, "vectors", f"frame_{fr:03d}.bin"), np.uint8).reshape(96, 128, 3)
    gold = np.fromfile(os.path.join(ROOT, "ml", EXP, "vectors", f"frame_{fr:03d}_depth.bin"), np.uint8)
    t = time.time()
    res, dm, tel, _ = b.run(rgb)
    rt = time.time() - t
    if res is None:
        print(f"frame {fr}: no RESULT"); continue
    same = dm is not None and np.array_equal(np.frombuffer(dm, np.uint8), gold)
    diff = int((np.frombuffer(dm, np.uint8) != gold).sum()) if dm is not None else -1
    print(f"frame {fr}: depth == golden: {same} ({diff} px differ) | depth net {res['t_cnn_us'] / 1e3:.0f} ms "
          f"(array layers {res['logits'][4] / 1e3:.0f} ms, CPU layers {res['logits'][5] / 1e3:.0f} ms), frame total {res['t_total_us'] / 1e3:.0f} ms, "
          f"flags {res['flags']}, NPU errors {res['logits'][7]}, array busy {res['npu_active']}/{res['npu_cycles']} clk, round trip {rt * 1000:.0f} ms")
b.send(P.cmd(P.CMD_BENCH))
g, _ = b.wait([P.T_BENCH], 300)
if P.T_BENCH in g:
    c, n, m, mt = struct.unpack("<IIIB", g[P.T_BENCH])
    print(f"BENCH: CPU only {c / 1e3:.0f} ms, NPU {n / 1e3:.0f} ms, speed-up {c / max(n, 1):.2f}x, {m / 1e6:.1f} M MACs, outputs match: {mt}")
else:
    print("BENCH: no answer")
b.send(P.cmd(P.CMD_SET_GATE, 1))
