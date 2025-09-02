from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Ensure child processes can import from repo root
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))

scripts = [
    ROOT / "import" / "1_player_aliases.py",
    ROOT / "import" / "2_import_csvs.py",
    ROOT / "import" / "3_create_player_precomputes.py",
    ROOT / "import" / "4_build_team_ratings.py",
    ROOT / "import" / "5_build_team_precompute.py",
    ROOT / "import" / "6_disposals_precomputes.py",
    ROOT / "import" / "7_goals_precomputes.py",
    ROOT / "import" / "8_marks_precomputes.py",
    ROOT / "import" / "9_clearances.py",
    ROOT / "import" / "10_tackles.py",
    ROOT / "import" / "11_kicks.py",
    ROOT / "import" / "12_handballs.py",
    ROOT / "import" / "13_predict_disposals.py",
    ROOT / "import" / "14_predict_team_wins.py",
]

for script in scripts:
    rel = script.relative_to(ROOT)
    print(f"🚀 Running {rel} using {PY} ...")
    # stream output live; fail hard on non-zero exit
    try:
        subprocess.run([PY, "-W", "ignore", str(script)], cwd=str(ROOT), env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {rel} failed (exit {e.returncode})")
        sys.exit(e.returncode)

print("✅ All scripts executed successfully!")
