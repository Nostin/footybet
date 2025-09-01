from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Ensure child processes can import from repo root
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))

# Each entry: (path, [args...])
scripts = [
    (ROOT / "machine_learning" / "1_split_data.py", []),
    (ROOT / "machine_learning" / "2_create_disposal_features.py", []),
    (ROOT / "machine_learning" / "3_train_disposal_models.py", []),
    (ROOT / "machine_learning" / "4a_calc_team_precomputes_prior.py", []),
    (ROOT / "machine_learning" / "4b_train_match_winner.py", []),
    (ROOT / "machine_learning" / "4c_reset_team_precomputes.py", []),
]

for path, args in scripts:
    cmd = [PY, "-W", "ignore", str(path), *args]
    rel = path.relative_to(ROOT)
    print(f"🚀 Running {' '.join([str(rel), *args])} using {PY} ...")

    # stream output live; fail on non-zero exit
    try:
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {rel} failed (exit {e.returncode})")
        sys.exit(e.returncode)

print("✅ All scripts executed successfully!")
