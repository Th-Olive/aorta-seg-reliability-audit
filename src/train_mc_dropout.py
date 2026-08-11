"""
MC Dropout nnU-Net training for AortaSeg24.

Run AFTER baseline training (src/train_baseline.py) has completed and
in-distribution Dice has been verified not to degrade meaningfully.

Usage:
  python src/train_mc_dropout.py [--resume]

Produces:
  nnUNet_results/Dataset001_AortaSeg24/
    nnUNetTrainerMCDropout__nnUNetPlans__3d_fullres/fold_0/
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
DATASET_ID = 1
CONFIG     = "3d_fullres"
FOLD       = 0
TRAINER    = "nnUNetTrainerMCDropout"
PLANS      = "nnUNetPlans"


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
        cmd.append("--c")

    print("Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args.resume)
