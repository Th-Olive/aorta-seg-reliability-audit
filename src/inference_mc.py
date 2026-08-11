"""
MC Dropout inference: T stochastic forward passes per case.

Outputs per case:
  {case_id}.nii.gz       -- prediction mask (mean of T passes, argmax)
  {case_id}_unc.nii.gz   -- per-voxel uncertainty map (mean class variance, original space)

Logs per-case uncertainty scalar to {output_dir}/uncertainty_summary.csv.

Usage (from repo root):
  python src/inference_mc.py \\
    --input_dir  nnUNet_raw/Dataset001_AortaSeg24/imagesTs \\
    --output_dir results/mc_dropout/predictions/aortaseg24_test \\
    --dataset_name aortaseg24_test \\
    --T 20

  python src/inference_mc.py \\
    --input_dir  data/avt_prepared/images \\
    --output_dir results/mc_dropout/predictions/avt \\
    --dataset_name avt

  python src/inference_mc.py \\
    --input_dir  data/amos_prepared/images \\
    --output_dir results/mc_dropout/predictions/amos \\
    --dataset_name amos

Smoke test (1 case, T=2):
  python src/inference_mc.py --input_dir ... --output_dir ... --T 2 --limit 1
"""
import argparse
import csv
import gc
import os
import pathlib
import sys
from typing import Union

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

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


# ---------------------------------------------------------------------------
# MC Dropout predictor
# ---------------------------------------------------------------------------

class MCDropoutPredictor:
    """
    Wraps nnUNetPredictor to run T stochastic forward passes per sliding-window
    sweep.  Overrides predict_sliding_window_return_logits to:
      1. Keep network in train() mode so Dropout3d is active.
      2. Accumulate T passes with Welford online mean/variance (GPU-efficient).
      3. Return mean logits; store per-voxel uncertainty in _last_uncertainty.
    """

    def __init__(self, model_dir: pathlib.Path, T: int = 20,
                 checkpoint: str = "checkpoint_final.pth",
                 device: torch.device | None = None):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.T = T
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_uncertainty: np.ndarray | None = None

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=(0,),
            checkpoint_name=checkpoint,
        )

        # Patch predict_sliding_window_return_logits with a closure that runs
        # T stochastic passes instead of one deterministic pass.
        # A closure is used (not __get__ / types.MethodType) because the target
        # attribute is looked up as an instance attribute — Python does NOT call
        # the descriptor protocol, so self is not passed automatically.
        predictor = self._predictor
        mc_ref = self

        @torch.inference_mode()
        def _mc_sliding_window(input_image: torch.Tensor):
            from acvl_utils.cropping_and_padding.padding import pad_nd_image
            from nnunetv2.utilities.helpers import empty_cache, dummy_context

            assert isinstance(input_image, torch.Tensor) and input_image.ndim == 4

            # Move network to device; keep dropout active (do NOT call .eval())
            predictor.network = predictor.network.to(predictor.device)
            _orig_eval = predictor.network.eval
            predictor.network.eval = lambda: predictor.network  # no-op
            predictor.network.train()

            empty_cache(predictor.device)

            ctx = (torch.autocast(predictor.device.type, enabled=True)
                   if predictor.device.type == "cuda" else dummy_context())

            with ctx:
                data, slicer_revert_padding = pad_nd_image(
                    input_image,
                    predictor.configuration_manager.patch_size,
                    "constant", {"value": 0}, True, None,
                )
                slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])

            # Float32 Welford, unbiased M2/(T-1) — identical formula and
            # precision to the already-completed runs.
            #
            # OOM root cause A (GPU): the old code called logits.float() on the
            # GPU, converting 11.5 GB float16 → 23 GB float32 and overflowing
            # 24 GB VRAM into Windows shared GPU memory (~22 GB of system RAM).
            # Fix: softmax stays float16 on the GPU (PyTorch's CUDA softmax
            # kernel uses fp32 accumulators internally, so the result is
            # numerically identical to computing on float32 input).  The
            # float32 conversion is deferred to after .cpu(), so no float32
            # tensor ever lands on the GPU.
            #
            # OOM root cause B (CPU): by pass 2, mean_probs (23 GB) + M2
            # (23 GB) + GPU-shared-mem (22 GB) = 68 GB were already pinned;
            # the 23 GB .cpu() allocation then exhausted system RAM (~91 GB
            # total demanded).  Fix: Welford is updated one z-slab at a time
            # (target ≤2 GB transient, adaptive to volume H×W) instead of from
            # the full-volume probs tensor (23 GB).  The per-slab update is
            # bitwise-identical to a full-volume update: softmax over the class
            # dimension (dim=0) is spatially independent, so slicing in z
            # commutes with it.  Peak CPU drops from ~91 GB to ~51 GB.
            mean_probs: torch.Tensor | None = None   # float32 CPU [C, H', W', D']
            M2: torch.Tensor | None = None           # float32 CPU
            count = 0
            chunk_z: int | None = None  # computed from logits shape on first pass

            for _ in range(mc_ref.T):
                with ctx:
                    if predictor.perform_everything_on_device and predictor.device.type != "cpu":
                        try:
                            logits = predictor._internal_predict_sliding_window_return_logits(
                                data, slicers, True)
                        except RuntimeError:
                            empty_cache(predictor.device)
                            logits = predictor._internal_predict_sliding_window_return_logits(
                                data, slicers, False)
                    else:
                        logits = predictor._internal_predict_sliding_window_return_logits(
                            data, slicers, predictor.perform_everything_on_device)

                count += 1
                D = logits.shape[-1]

                if mean_probs is None:
                    mean_probs = torch.zeros(logits.shape, dtype=torch.float32)
                    M2 = torch.zeros_like(mean_probs)
                    # Adaptive chunk size: target ≤2 GB transient slab (float32).
                    # Chunk size is pure memory policy — softmax(dim=0) is spatially
                    # independent, so slicing in z commutes with it; the Welford
                    # update is bitwise-identical regardless of chunk size.
                    n_cls, H_pad, W_pad = logits.shape[:3]
                    bytes_per_slice = n_cls * H_pad * W_pad * 4  # float32
                    chunk_z = max(1, int(2 * 1024 ** 3 // bytes_per_slice))

                for z0 in range(0, D, chunk_z):  # type: ignore[arg-type]
                    z1 = min(z0 + chunk_z, D)  # type: ignore[operator]
                    # float16 on GPU → CPU → float32; avoids float32 on GPU
                    slab = torch.softmax(logits[..., z0:z1], dim=0).cpu().float()
                    # Reuse slab buffer as delta = (p_t − mean_{t-1}), in-place
                    slab.sub_(mean_probs[..., z0:z1])
                    mean_probs[..., z0:z1].add_(slab, alpha=1.0 / count)
                    # M2 += delta·delta2 = delta² · (count−1)/count
                    M2[..., z0:z1].addcmul_(slab, slab, value=(count - 1.0) / count)
                    del slab

                del logits
                empty_cache(predictor.device)
                gc.collect()

            empty_cache(predictor.device)
            predictor.network.eval = _orig_eval

            M2.div_(max(count - 1, 1))   # unbiased variance in-place: M2/(T-1)
            spatial_slicer = tuple(slicer_revert_padding[1:])

            mc_ref._last_uncertainty = M2.mean(0)[spatial_slicer].numpy()
            del M2

            log_mean = mean_probs.clamp_(1e-7, 1 - 1e-7).log_()   # in-place
            result = log_mean[(slice(None), *spatial_slicer)]
            del mean_probs, log_mean
            gc.collect()
            return result

        self._predictor.predict_sliding_window_return_logits = _mc_sliding_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_case(self, input_files: list[str], output_path: pathlib.Path) -> float:
        """
        Run MC Dropout inference on one case.

        input_files : list of modality file paths (single-modality -> 1 element)
        output_path : destination for prediction NIfTI (without .nii.gz suffix)

        Returns per-case uncertainty scalar (mean over all foreground voxels).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._last_uncertainty = None

        # predict_from_files_sequential avoids Windows multiprocessing pipe
        # size limits that occur with large logit tensors (24 classes x full volume)
        self._predictor.predict_from_files_sequential(
            [input_files],
            [str(output_path)],
            save_probabilities=False,
            overwrite=True,
        )

        pred_file = pathlib.Path(str(output_path) + ".nii.gz")
        unc_scalar = self._save_uncertainty(pred_file)
        return unc_scalar

    def _save_uncertainty(self, pred_nifti: pathlib.Path) -> float:
        """Resample _last_uncertainty to prediction space and save _unc.nii.gz."""
        if self._last_uncertainty is None:
            return float("nan")

        unc_map = self._last_uncertainty   # [H_proc, W_proc, D_proc]

        if pred_nifti.exists():
            pred_nib = nib.load(pred_nifti)
            target_shape = pred_nib.shape  # (H_orig, W_orig, D_orig)

            scale = tuple(t / p for t, p in zip(target_shape, unc_map.shape))
            unc_resampled = zoom(unc_map, scale, order=1).astype(np.float32)

            unc_nib = nib.Nifti1Image(unc_resampled, pred_nib.affine, pred_nib.header)
            unc_path = pathlib.Path(str(pred_nifti).replace(".nii.gz", "_unc.nii.gz"))
            nib.save(unc_nib, unc_path)

            # Scalar: mean uncertainty over foreground voxels (pred != 0)
            fg_mask = pred_nib.get_fdata(dtype=np.float32) > 0
            if fg_mask.any():
                scalar = float(unc_resampled[fg_mask].mean())
            else:
                scalar = float(unc_map.mean())
        else:
            scalar = float(unc_map.mean())

        self._last_uncertainty = None
        return scalar


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_cases(input_dir: pathlib.Path, limit: int | None) -> list[tuple[str, list[str]]]:
    """Return [(case_id, [file_path]), ...] from a folder of *_0000.nii.gz files."""
    files = sorted(input_dir.glob("*_0000.nii.gz"))
    if limit:
        files = files[:limit]
    cases = []
    for f in files:
        case_id = f.name.replace("_0000.nii.gz", "")
        cases.append((case_id, [str(f)]))
    return cases


def main():
    parser = argparse.ArgumentParser(description="MC Dropout inference")
    parser.add_argument("--input_dir",   required=True, type=pathlib.Path)
    parser.add_argument("--output_dir",  required=True, type=pathlib.Path)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--T",  type=int, default=20, help="MC Dropout passes")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N cases (smoke test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.input_dir, args.limit)
    if not cases:
        print(f"No *_0000.nii.gz files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(cases)} cases in {args.input_dir}  |  T={args.T}")

    csv_path = args.output_dir / "uncertainty_summary.csv"

    # Resume: read already-completed case IDs from an existing CSV
    done_ids: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="") as _f:
            for row in csv.DictReader(_f):
                done_ids.add(row["case_id"])

    pending = [(cid, files) for cid, files in cases if cid not in done_ids]
    n_skip = len(cases) - len(pending)
    if n_skip:
        print(f"Resuming: {n_skip} already done, {len(pending)} remaining.")
    if not pending:
        print("All cases already done.")
        return

    predictor = MCDropoutPredictor(
        model_dir=MODEL_DIR,
        T=args.T,
        checkpoint=args.checkpoint,
    )

    csv_mode = "a" if csv_path.exists() else "w"
    with open(csv_path, csv_mode, newline="") as csv_f:
        writer = csv.DictWriter(csv_f,
                                fieldnames=["dataset", "case_id", "mean_uncertainty_fg"])
        if csv_mode == "w":
            writer.writeheader()

        for i, (case_id, input_files) in enumerate(pending):
            out_path = args.output_dir / case_id
            print(f"  [{i+1}/{len(pending)}] {case_id} ...", end=" ", flush=True)
            scalar = predictor.predict_case(input_files, out_path)
            print(f"unc={scalar:.5f}")
            writer.writerow({
                "dataset":            args.dataset_name,
                "case_id":            case_id,
                "mean_uncertainty_fg": f"{scalar:.6f}",
            })
            csv_f.flush()

    print(f"\nDone. Predictions -> {args.output_dir}")
    print(f"Uncertainty CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
