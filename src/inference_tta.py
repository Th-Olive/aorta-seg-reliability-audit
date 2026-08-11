"""
TTA (Test-Time Augmentation) inference: K=7 flip variants, deterministic model.

Outputs per case:
  {case_id}.nii.gz       -- prediction mask (mean of K variants, argmax)
  {case_id}_unc.nii.gz   -- per-voxel uncertainty map (mean class variance, original space)

Logs per-case uncertainty scalar to {output_dir}/uncertainty_summary.csv.

Usage (from repo root):
  python src/inference_tta.py \\
    --input_dir  nnUNet_raw/Dataset001_AortaSeg24/imagesTs \\
    --output_dir results/tta/predictions/aortaseg24_test \\
    --dataset_name aortaseg24_test

  python src/inference_tta.py \\
    --input_dir  data/avt_prepared/images \\
    --output_dir results/tta/predictions/avt \\
    --dataset_name avt

  python src/inference_tta.py \\
    --input_dir  data/amos_prepared/images \\
    --output_dir results/tta/predictions/amos \\
    --dataset_name amos

Smoke test (1 case, identity-only):
  python src/inference_tta.py --input_dir ... --output_dir ... --dataset_name avt --K 1 --limit 1
"""
import argparse
import csv
import gc
import os
import pathlib
import sys

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

# K=7 TTA flip configurations: tuples of spatial axes (0=L-R, 1=A-P, 2=I-S)
TTA_FLIPS = [
    (),        # identity
    (0,),      # L-R
    (1,),      # A-P
    (2,),      # I-S
    (0, 1),    # L-R + A-P
    (0, 2),    # L-R + I-S
    (1, 2),    # A-P + I-S
]

# AortaSeg24: channel pairs to swap for L-R flips (0-indexed class channels)
_LR_SWAP_PAIRS = ((18, 19), (20, 21), (22, 23))


def _swap_lr_channels(logits: torch.Tensor) -> torch.Tensor:
    """Swap left/right paired label channels in-place for AortaSeg24 L-R flips."""
    for a, b in _LR_SWAP_PAIRS:
        if b < logits.shape[0]:
            logits[[a, b]] = logits[[b, a]]
    return logits


# ---------------------------------------------------------------------------
# TTA predictor
# ---------------------------------------------------------------------------

class TTAPredictor:
    """
    Wraps nnUNetPredictor to run K deterministic TTA flip variants per case.
    Network is in .eval() mode — no dropout. Per-voxel variance across the K
    un-flipped softmax maps is stored as uncertainty.
    """

    def __init__(self, model_dir: pathlib.Path, dataset_name: str,
                 K: int = 7,
                 checkpoint: str = "checkpoint_final.pth",
                 device: torch.device | None = None):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.K = K
        self.dataset_name = dataset_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_uncertainty: np.ndarray | None = None

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,   # we handle TTA manually
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=(0,),
            checkpoint_name=checkpoint,
        )

        predictor = self._predictor
        tta_ref = self
        flip_table = TTA_FLIPS[:K]
        is_aortaseg24 = "aortaseg24" in dataset_name.lower()

        @torch.inference_mode()
        def _tta_sliding_window(input_image: torch.Tensor):
            from acvl_utils.cropping_and_padding.padding import pad_nd_image
            from nnunetv2.utilities.helpers import empty_cache, dummy_context

            assert isinstance(input_image, torch.Tensor) and input_image.ndim == 4

            predictor.network = predictor.network.to(predictor.device)
            predictor.network.eval()

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

            mean_probs: torch.Tensor | None = None  # float32 CPU [C, H', W', D']
            M2: torch.Tensor | None = None
            count = 0
            chunk_z: int | None = None

            for flip_axes in flip_table:
                # Flip spatial dims (tensor dim = spatial axis + 1 for channel offset)
                tensor_dims = [ax + 1 for ax in flip_axes]
                data_flip = torch.flip(data, tensor_dims) if tensor_dims else data

                with ctx:
                    if predictor.perform_everything_on_device and predictor.device.type != "cpu":
                        try:
                            logits = predictor._internal_predict_sliding_window_return_logits(
                                data_flip, slicers, True)
                        except RuntimeError:
                            empty_cache(predictor.device)
                            logits = predictor._internal_predict_sliding_window_return_logits(
                                data_flip, slicers, False)
                    else:
                        logits = predictor._internal_predict_sliding_window_return_logits(
                            data_flip, slicers, predictor.perform_everything_on_device)

                # Un-flip logits spatially and apply label remapping before accumulation
                if tensor_dims:
                    logits = torch.flip(logits, tensor_dims)

                if is_aortaseg24 and 0 in flip_axes:
                    logits = _swap_lr_channels(logits)

                count += 1
                D = logits.shape[-1]

                if mean_probs is None:
                    mean_probs = torch.zeros(logits.shape, dtype=torch.float32)
                    M2 = torch.zeros_like(mean_probs)
                    # Adaptive chunk: target ≤2 GB transient slab (float32)
                    n_cls, H_pad, W_pad = logits.shape[:3]
                    bytes_per_slice = n_cls * H_pad * W_pad * 4
                    chunk_z = max(1, int(2 * 1024 ** 3 // bytes_per_slice))

                for z0 in range(0, D, chunk_z):   # type: ignore[arg-type]
                    z1 = min(z0 + chunk_z, D)     # type: ignore[operator]
                    slab = torch.softmax(logits[..., z0:z1], dim=0).cpu().float()
                    slab.sub_(mean_probs[..., z0:z1])
                    mean_probs[..., z0:z1].add_(slab, alpha=1.0 / count)
                    M2[..., z0:z1].addcmul_(slab, slab, value=(count - 1.0) / count)
                    del slab

                del logits
                empty_cache(predictor.device)
                gc.collect()

            empty_cache(predictor.device)

            M2.div_(max(count - 1, 1))    # unbiased variance: M2/(K-1)
            spatial_slicer = tuple(slicer_revert_padding[1:])

            tta_ref._last_uncertainty = M2.mean(0)[spatial_slicer].numpy()
            del M2

            log_mean = mean_probs.clamp_(1e-7, 1 - 1e-7).log_()
            result = log_mean[(slice(None), *spatial_slicer)]
            del mean_probs, log_mean
            gc.collect()
            return result

        self._predictor.predict_sliding_window_return_logits = _tta_sliding_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_case(self, input_files: list[str], output_path: pathlib.Path) -> float:
        """
        Run TTA inference on one case.

        input_files : list of modality file paths (single-modality -> 1 element)
        output_path : destination for prediction NIfTI (without .nii.gz suffix)

        Returns per-case uncertainty scalar (mean over all foreground voxels).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._last_uncertainty = None

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
            target_shape = pred_nib.shape

            scale = tuple(t / p for t, p in zip(target_shape, unc_map.shape))
            unc_resampled = zoom(unc_map, scale, order=1).astype(np.float32)

            unc_nib = nib.Nifti1Image(unc_resampled, pred_nib.affine, pred_nib.header)
            unc_path = pathlib.Path(str(pred_nifti).replace(".nii.gz", "_unc.nii.gz"))
            nib.save(unc_nib, unc_path)

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
    parser = argparse.ArgumentParser(description="TTA inference")
    parser.add_argument("--input_dir",    required=True, type=pathlib.Path)
    parser.add_argument("--output_dir",   required=True, type=pathlib.Path)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--K",  type=int, default=7,
                        help="TTA flip variants (default 7; use 1 for identity-only smoke test)")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N cases (smoke test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.input_dir, args.limit)
    if not cases:
        print(f"No *_0000.nii.gz files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(cases)} cases in {args.input_dir}  |  K={args.K}")

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

    predictor = TTAPredictor(
        model_dir=MODEL_DIR,
        dataset_name=args.dataset_name,
        K=args.K,
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
