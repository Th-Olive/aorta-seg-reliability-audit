"""
Baseline nnU-Net training (no dropout) for AortaSeg24.

This script is a thin wrapper that sets up the environment and launches
nnUNetv2_train with the correct arguments. Use it instead of calling
nnunet_run.py directly so training flags are recorded in one place.

Usage:
  python src/train_baseline.py [--resume]

Produces:
  nnUNet_results/Dataset001_AortaSeg24/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
    checkpoint_best.pth
    checkpoint_latest.pth
    training_log_*.txt
    validation_raw/          <- per-case Dice on the 10 val cases
    progress.png             <- live training curve (loss + pseudo-Dice)
"""
import os
import pathlib
import subprocess
import sys

_env = pathlib.Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ[k.strip()] = v.strip()

PYTHON_DIR = pathlib.Path(sys.executable).parent
DATASET_ID  = 1
CONFIG      = "3d_fullres"
FOLD        = 0
TRAINER     = "nnUNetTrainer"          # baseline: no dropout
PLANS       = "nnUNetPlans"

def run(resume: bool):
    cmd_name = PYTHON_DIR / "Scripts" / "nnUNetv2_train"
    cmd = [
        str(cmd_name),
        str(DATASET_ID),
        CONFIG,
        str(FOLD),
        "-tr", TRAINER,
        "-p", PLANS,
    ]
    if resume:
        cmd.append("--c")              # nnU-Net resume flag

    print("Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", help="Resume interrupted training")
    args = p.parse_args()
    run(args.resume)
