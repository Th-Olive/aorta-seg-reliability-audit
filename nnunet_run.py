"""
Thin wrapper that loads .env before running any nnUNet CLI command.
Usage: python nnunet_run.py <nnunet_command> [args...]

Examples:
  python nnunet_run.py nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
  python nnunet_run.py nnUNetv2_train 001 3d_fullres 0
  python nnunet_run.py nnUNetv2_predict -i input/ -o output/ -d 001 -c 3d_fullres -f 0
"""
import os
import sys
import subprocess
import pathlib

env_file = pathlib.Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

python_dir = pathlib.Path(sys.executable).parent
cmd = python_dir / "Scripts" / sys.argv[1]
result = subprocess.run([str(cmd)] + sys.argv[2:], env=os.environ)
sys.exit(result.returncode)
