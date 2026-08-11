"""
Run T MC-Dropout forward passes per case and save each pass's argmax as a
separate NIfTI.  Outputs feed `src/analyze_pairwise.py` which computes mean
pairwise multi-class Dice between the T passes as an alternative uncertainty
signal (Zenk et al. 2025 MedIA — outperforms per-voxel variance aggregation
when augmentation-invariance suppresses softmax variance).

Output layout:
  {output_dir}/{case_id}/pass_00.nii.gz
  {output_dir}/{case_id}/pass_01.nii.gz
  ...
  {output_dir}/{case_id}/pass_{T-1}.nii.gz

Resume support: a case is considered complete when {output_dir}/{case_id}/pass_{T-1}.nii.gz
exists; in-progress cases are detected by counting pass_*.nii.gz files and
resumed from the first missing pass index.

Usage:
  python src/inference_mc_passes.py \\
    --input_dir  nnUNet_raw/Dataset001_AortaSeg24/imagesTs \\
    --output_dir results/mc_passes/aortaseg24_test \\
    --T 20

  python src/inference_mc_passes.py --input_dir  data/avt_prepared/images --output_dir results/mc_passes/avt --T 20 --cases_first "D15,D16,D17,D18,K1,K2,K3,K4,K5,K6,K8,K9,K10,K11,K12,K13,K14,K15,K16,K17,K18,K20,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R17,R18,R16,R15"

  # Smoke test (1 case, T=2)
  python src/inference_mc_passes.py --input_dir ... --output_dir ... --T 2 --limit 1
"""
import argparse
import gc
import os
import pathlib
import sys

import torch

_env = pathlib.Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = pathlib.Path(__file__).parent.parent
MODEL_DIR = (
    ROOT / "nnUNet_results" / "Dataset001_AortaSeg24"
    / "nnUNetTrainerMCDropout__nnUNetPlans__3d_fullres"
)


class MCPassPredictor:
    """
    Wraps nnUNetPredictor to run a single deterministic-seed dropout-active
    forward pass.  Each call saves one argmax NIfTI to disk.  Dropout layers
    are kept active (network.train()) so the prediction varies between calls
    with the same input — the source of MC variability.
    """

    def __init__(self, model_dir: pathlib.Path,
                 checkpoint: str = "checkpoint_final.pth",
                 device: torch.device | None = None):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,   # we are running MC samples, not TTA
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=(0,),
            checkpoint_name=checkpoint,
        )

        # Force network into train() so Dropout3d stays active during inference.
        # nnUNet internally calls .eval() on the network; we no-op that.
        self._predictor.network.eval = lambda: self._predictor.network
        self._predictor.network.train()

    @torch.inference_mode()
    def predict_pass(self, input_files: list[str], output_stem: pathlib.Path,
                     seed: int) -> None:
        """Run one MC pass with a fixed seed; save argmax to f'{output_stem}.nii.gz'."""
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._predictor.predict_from_files_sequential(
            [input_files],
            [str(output_stem)],
            save_probabilities=False,
            overwrite=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_cases(input_dir: pathlib.Path, limit: int | None,
                  cases_first: list[str] | None = None):
    files = sorted(input_dir.glob("*_0000.nii.gz"))
    if cases_first:
        priority = []
        for cid in cases_first:
            f = input_dir / f"{cid}_0000.nii.gz"
            if f.exists():
                priority.append(f)
        priority_set = set(priority)
        files = priority + [f for f in files if f not in priority_set]
    if limit:
        files = files[:limit]
    return [(f.name.replace("_0000.nii.gz", ""), [str(f)]) for f in files]


def main():
    p = argparse.ArgumentParser(description="MC Dropout: save per-pass argmax")
    p.add_argument("--input_dir",  required=True, type=pathlib.Path)
    p.add_argument("--output_dir", required=True, type=pathlib.Path)
    p.add_argument("--T",          type=int, default=20)
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--seed_offset", type=int, default=42,
                   help="Pass i uses seed = seed_offset + i; reproducible across re-runs")
    p.add_argument("--cases_first", type=str, default=None,
                   help="Comma-separated case IDs to process first (others follow alphabetically)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    priority = [c.strip() for c in args.cases_first.split(",")] if args.cases_first else None
    cases = collect_cases(args.input_dir, args.limit, cases_first=priority)
    if not cases:
        print(f"No *_0000.nii.gz files in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(cases)} cases  |  T={args.T}  output={args.output_dir}")

    predictor = MCPassPredictor(model_dir=MODEL_DIR, checkpoint=args.checkpoint)

    for i, (case_id, input_files) in enumerate(cases):
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        # Count existing passes for resume
        done = sum(1 for k in range(args.T)
                   if (case_dir / f"pass_{k:02d}.nii.gz").exists())
        if done >= args.T:
            print(f"  [{i+1}/{len(cases)}] {case_id}  already complete  ({args.T} passes)")
            continue
        print(f"  [{i+1}/{len(cases)}] {case_id}  resuming from pass {done}/{args.T}")

        for pass_idx in range(done, args.T):
            output_stem = case_dir / f"pass_{pass_idx:02d}"
            predictor.predict_pass(input_files, output_stem,
                                   seed=args.seed_offset + pass_idx)
            print(f"     pass {pass_idx:02d} done")

    print(f"\nDone. Per-pass argmax -> {args.output_dir}")


if __name__ == "__main__":
    main()
