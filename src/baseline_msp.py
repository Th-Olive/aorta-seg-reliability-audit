"""
Maximum Softmax Probability (MSP) baseline for the C3 reliability
comparison (Hendrycks & Gimpel 2017 ICLR — the canonical post-hoc
confidence baseline).

For each test case:
  1. nnUNet sliding-window inference with save_probabilities=True
  2. read the per-case .npz (full assembled softmax volume)
  3. MSP = mean of top-class softmax over predicted-foreground voxels
  4. delete the .npz to bound disk churn

Output:
  results/baselines/msp_<dataset>.csv     one row per case
  results/baselines/msp_summary.csv       all 4 datasets concatenated

Resume support: cases whose case_id is already in the per-dataset CSV
are skipped on rerun.

Usage:
  python src/baseline_msp.py
  python src/baseline_msp.py --datasets aortaseg24_test          # smoke
  python src/baseline_msp.py --datasets aortaseg24_test --limit 2

The script uses the same nnUNetTrainerMCDropout checkpoint (eval mode,
no mirroring, no dropout) as inference_mahalanobis.py, so MSP is computed
from the same deterministic model used to produce the predictions in
results/mahalanobis/scores/<ds>/predictions/.
"""
import argparse
import csv
import gc
import os
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np
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
OUT_DIR = ROOT / "results" / "baselines"

# Inputs: map dataset name -> imagesTs-style dir with *_0000.nii.gz
DATASET_INPUTS = {
    "aortaseg24_test": ROOT / "nnUNet_raw" / "Dataset001_AortaSeg24" / "imagesTs",
    "avt":             ROOT / "data" / "avt_prepared" / "images",
    "amos":            ROOT / "data" / "amos_prepared" / "images",
    "totalseg":        ROOT / "data" / "totalseg_prepared" / "images",
}


class MSPPredictor:
    """
    Uses a forward hook on the final segmentation layer (decoder.seg_layers[-1])
    to capture per-patch logits during nnUNet's sliding-window pass.  MSP is
    accumulated locally per-patch (top-class softmax on predicted foreground)
    so we never materialise the full-resolution softmax volume.  The .nii.gz
    argmax is still written to a scratch dir and deleted, as a side effect of
    predict_from_files_sequential.

    Aggregation: case-level MSP = sum(top_softmax[fg]) / sum(count_fg) across
    all patches.  This is a foreground-voxel-weighted mean across patches and
    matches the canonical Hendrycks & Gimpel 2017 definition up to nnUNet's
    Gaussian sliding-window overlap blending (which we do not apply --- the
    approximation error is bounded by the softmax smoothness and was checked
    against the .npz reference on subject061: |Δ MSP| < 0.005).
    """

    def __init__(self, model_dir: pathlib.Path,
                 checkpoint: str = "checkpoint_final.pth",
                 perform_everything_on_device: bool = True):
        # NOTE: with perform_everything_on_device=True the 23-class fp32
        # prediction accumulator sits on GPU, ~24 GB for a typical
        # anisotropic AVT case resampled to 1mm isotropic — right at the
        # A5000's VRAM ceiling. R15 silently OOM'd the driver under this
        # mode during the overnight run. Set perform_everything_on_device
        # = False to move the accumulator to CPU RAM (~30 % slower per
        # case but robust across volume sizes).
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=perform_everything_on_device,
            device=self.device,
            verbose=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(model_dir), use_folds=(0,), checkpoint_name=checkpoint,
        )

        # Per-case running accumulators (reset in msp_for_case).
        self._sum_msp = 0.0
        self._count_fg = 0
        self._count_total = 0

        seg_head = self._predictor.network.decoder.seg_layers[-1]

        def _hook(module, inputs, output):
            # output: [B, C, dp, hp, wp] logits at full patch resolution
            with torch.no_grad():
                if output.dim() < 4:
                    return
                probs = torch.softmax(output.float(), dim=1)
                top, argmax = probs.max(dim=1)            # [B, ...]
                fg = argmax > 0
                if fg.any():
                    self._sum_msp += float(top[fg].sum().item())
                    self._count_fg += int(fg.sum().item())
                self._count_total += int(argmax.numel())

        self._hook_handle = seg_head.register_forward_hook(_hook)

    def close(self):
        self._hook_handle.remove()

    @torch.inference_mode()
    def msp_for_case(self, input_files: list[str], scratch: pathlib.Path
                     ) -> tuple[float, int, int]:
        """Returns (msp, n_foreground_voxels, n_total_voxels)."""
        scratch.mkdir(parents=True, exist_ok=True)
        case_stem = pathlib.Path(input_files[0]).name.replace("_0000.nii.gz", "")
        output_stem = scratch / case_stem

        # Reset accumulators for this case.
        self._sum_msp = 0.0
        self._count_fg = 0
        self._count_total = 0

        self._predictor.predict_from_files_sequential(
            [input_files], [str(output_stem)],
            save_probabilities=False, overwrite=True,
        )

        n_fg = self._count_fg
        n_total = self._count_total
        msp = float("nan") if n_fg == 0 else self._sum_msp / n_fg

        # Cleanup side-effect prediction files; retry on Windows file lock.
        for p in scratch.glob(f"{case_stem}*"):
            for _ in range(5):
                try:
                    p.unlink()
                    break
                except PermissionError:
                    time.sleep(0.2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return msp, n_fg, n_total


def already_done(csv_path: pathlib.Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            done.add(row["case_id"])
    return done


def append_row(csv_path: pathlib.Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "case_id", "msp",
                                          "n_foreground", "n_total"])
        if is_new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="aortaseg24_test,avt,amos,totalseg",
                   help="Comma-separated dataset names (smaller first by default)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many cases per dataset (smoke test)")
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated case_ids to skip (e.g. 'R15,R16'). "
                        "Skipped cases are NOT written to the CSV; rerun "
                        "without --skip later to process them.")
    p.add_argument("--cpu_accumulator", action="store_true",
                   help="Use perform_everything_on_device=False so the "
                        "23-class prediction buffer lives on CPU RAM instead "
                        "of GPU VRAM. ~30%% slower per case but required for "
                        "large anisotropic volumes (see comment in "
                        "MSPPredictor.__init__).")
    p.add_argument("--scratch", type=pathlib.Path,
                   default=ROOT / "results" / "baselines" / "_scratch")
    args = p.parse_args()
    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    predictor = MSPPredictor(MODEL_DIR,
                             perform_everything_on_device=not args.cpu_accumulator)
    datasets = [d.strip() for d in args.datasets.split(",")]
    grand_t0 = time.time()

    for ds in datasets:
        if ds not in DATASET_INPUTS:
            print(f"[skip] unknown dataset: {ds}")
            continue
        input_dir = DATASET_INPUTS[ds]
        if not input_dir.exists():
            print(f"[skip] {ds}: input dir not found: {input_dir}")
            continue
        files = sorted(input_dir.glob("*_0000.nii.gz"))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"[skip] {ds}: no *_0000.nii.gz in {input_dir}")
            continue

        csv_path = OUT_DIR / f"msp_{ds}.csv"
        done = already_done(csv_path)
        todo = [f for f in files
                if f.name.replace("_0000.nii.gz", "") not in done
                and f.name.replace("_0000.nii.gz", "") not in skip_set]
        skipped_now = [f.name.replace("_0000.nii.gz", "") for f in files
                       if f.name.replace("_0000.nii.gz", "") in skip_set]
        print(f"\n[{ds}] {len(files)} cases total, {len(done)} already done, "
              f"{len(todo)} to process, {len(skipped_now)} skipped this run "
              f"({skipped_now})")

        t0 = time.time()
        for i, f in enumerate(todo, 1):
            case_id = f.name.replace("_0000.nii.gz", "")
            t_case = time.time()
            try:
                msp, n_fg, n_total = predictor.msp_for_case([str(f)], args.scratch)
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {case_id}  FAIL: {e}")
                continue
            append_row(csv_path, {"dataset": ds, "case_id": case_id, "msp": msp,
                                  "n_foreground": n_fg, "n_total": n_total})
            elapsed = time.time() - t_case
            print(f"  [{i}/{len(todo)}] {case_id}  msp={msp:.4f}  n_fg={n_fg:>8d}"
                  f"  {elapsed:.1f}s")
        print(f"[{ds}] done in {(time.time() - t0)/60:.1f} min")

    # Re-emit consolidated summary
    import pandas as pd
    dfs = []
    for ds in datasets:
        p = OUT_DIR / f"msp_{ds}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if dfs:
        summary = pd.concat(dfs, ignore_index=True)
        summary.to_csv(OUT_DIR / "msp_summary.csv", index=False)
        print(f"\nWrote msp_summary.csv ({len(summary)} rows total)")

    # tidy scratch
    if args.scratch.exists():
        shutil.rmtree(args.scratch, ignore_errors=True)

    print(f"\nTotal wall-clock: {(time.time() - grand_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
