"""One command from a trained checkpoint to everything the board, the tests and the laptop app need.
  python ml/export_all.py ml/export/fastdepth_distill_128x96.pth [--no-board-build]
Runs: convert -> export_c -> make_vectors -> make_hw_vectors -> check_exports -> accuracy_report -> firmware host tests (WSL)
      -> board ELFs (fw/board/build_app.ps1). Stops at the first failure. The laptop app re-streams the new blob automatically (CRC differs)."""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ck = sys.argv[1]
steps = [
    ("convert", ["python", "ml/convert.py", "--ckpt", ck]),
    ("export_c", ["python", "ml/export_c.py"]),
    ("golden vectors", ["python", "ml/make_vectors.py", "4"]),
    ("hw vectors", ["python", "ml/make_hw_vectors.py", "16"]),
    ("check_exports", ["python", "scripts/check_exports.py", "2"]),
    ("accuracy report", ["python", "ml/accuracy_report.py", "--ckpt", ck]),
    ("firmware host tests (WSL)", ["wsl", "-e", "bash", "-lc", "cd \"$(wslpath '" + os.path.join(ROOT, "fw").replace(os.sep, "/") + "')\" && make test 2>&1 | tail -4"]),
]
if "--no-board-build" not in sys.argv:
    steps.append(("board ELFs", ["powershell", "-ExecutionPolicy", "Bypass", "-File", "fw/board/build_app.ps1"]))
for name, cmd in steps:
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    ok = r.returncode == 0 and not any("FAIL" in l and "PASS" not in l for l in out[-6:])
    print(f"[{'OK' if ok else 'FAIL'}] {name} ({time.time() - t:.0f} s)")
    for l in out[-4:]:
        if "xFormers" not in l:
            print("      " + l)
    if not ok:
        sys.exit(1)
print("export_all: done - flash fw/board/out/depth_*.elf, the laptop app sends the new weights by itself")
