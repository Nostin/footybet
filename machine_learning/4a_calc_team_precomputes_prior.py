from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]  # repo root
PY = sys.executable

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))

script = ROOT / "import" / "5_build_team_precompute.py"
args = ["--stats-mode", "pre", "--exclude-finals-from-rolling"]

cmd = [PY, "-W", "ignore", str(script), *args]
rel = script.relative_to(ROOT)

print(f"🚀 Running {rel} {' '.join(args)} using {PY} ...")
try:
    # Streams output live; raises on non-zero exit
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ {rel} failed (exit {e.returncode})")
    sys.exit(e.returncode)

print("✅ All scripts executed successfully!")
