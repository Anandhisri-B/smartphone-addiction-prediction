"""
run.py
======
Master launcher — runs the full pipeline in order:
  1. Extract features from screenshots → data/features.csv
  2. Expand dataset to 1700+ records  → data/features.csv (updated)
  3. Train Random Forest + Decision Tree → models/
  4. Start Flask web server            → http://localhost:5000

Usage:
    python run.py
"""
import subprocess, sys, os
from pathlib import Path

def run(cmd, desc):
    print(f"\n{'─'*55}\n  ▶  {desc}\n{'─'*55}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print("❌  Step failed. Fix errors above and retry.")
        sys.exit(1)

def main():
    os.chdir(Path(__file__).parent)

    # Create all necessary folders
    for d in ['data', 'models', 'static/charts', 'dataset']:
        Path(d).mkdir(parents=True, exist_ok=True)

    # STEP 1 — Build base CSV from screenshots (runs once)
    if not Path("data/features.csv").exists():
        run(f"{sys.executable} build_dataset.py",
            "Step 1: Extracting features from screenshots...")
    else:
        print("\n✅  Step 1 skipped — data/features.csv already exists.")

    # STEP 2 — Expand dataset to 1700+ records (runs once)
    import pandas as pd
    df = pd.read_csv("data/features.csv")
    if len(df) < 200:
        run(f"{sys.executable} expand_dataset.py --n 1700",
            "Step 2: Expanding dataset to 1700+ records...")
    else:
        print(f"✅  Step 2 skipped — dataset already has {len(df)} records.")

    # STEP 3 — Train models (runs once)
    if not Path("models/random_forest.pkl").exists():
        run(f"{sys.executable} train_model.py",
            "Step 3: Training Random Forest + Decision Tree...")
    else:
        print("✅  Step 3 skipped — trained models already exist.")

    # STEP 4 — Start Flask
    print(f"\n{'='*55}")
    print("  🚀  Starting Flask → http://localhost:5000")
    print(f"{'='*55}\n")
    os.system(f"{sys.executable} app.py")

if __name__ == "__main__":
    main()