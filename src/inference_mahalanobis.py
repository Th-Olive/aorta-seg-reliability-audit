"""
Mahalanobis-distance reliability signal — fit + score.

Fit phase:
  Run a single deterministic forward pass on the 50 AortaSeg24 training cases
  using the trained nnUNetTrainerMCDropout checkpoint in eval() mode (dropout off).
  A forward hook on encoder.stages[-1] (the bottleneck, 320 channels at /16
  resolution) captures the feature map for each sliding-window patch.  Each patch
  feature is global-average-pooled over its spatial dims (D, H, W); the per-patch
  vectors are averaged over all patches of one case → one [320]-dim feature per
  scan.  PCA reduces 320 → K (default 40); EmpiricalCovariance is fitted on the
  PCA-projected training features.

Score phase:
  For each test case, extract the same [320]-dim feature, project with the
  fitted PCA, then compute squared Mahalanobis distance to the training Gaussian.
  Output: per-case scalar in mahalanobis_summary.csv.

References:
  Gonzalez et al. 2021 MICCAI / 2022 Med Image Anal — Mahalanobis on nnUNet bottleneck.
  Woodland et al. 2024 (arXiv:2408.02761) — PCA + Mahalanobis recipe for 3D CT OOD.

Usage (from repo root):
  # Fit on 50 AortaSeg24 training cases
  python src/inference_mahalanobis.py fit \\
    --input_dir   nnUNet_raw/Dataset001_AortaSeg24/imagesTr \\
    --splits_file nnUNet_preprocessed/Dataset001_AortaSeg24/splits_final.json \\
    --output_dir  results/mahalanobis/fit

  # Score AortaSeg24 test (40 cases)
  python src/inference_mahalanobis.py score \\
    --fit_dir     results/mahalanobis/fit \\
    --input_dir   nnUNet_raw/Dataset001_AortaSeg24/imagesTs \\
    --output_dir  results/mahalanobis/scores/aortaseg24_test \\
    --dataset_name aortaseg24_test

  # Score AVT (56) / AMOS (300 prepared CT, only those with aorta label)
  python src/inference_mahalanobis.py score \\
    --fit_dir results/mahalanobis/fit \\
    --input_dir data/avt_prepared/images \\
    --output_dir results/mahalanobis/scores/avt \\
    --dataset_name avt

  # Smoke test (fit on first 5 training cases; score 1 case)
  python src/inference_mahalanobis.py fit  --input_dir ... --splits_file ... \\
      --output_dir results/mahalanobis/fit_smoke --limit 5
  python src/inference_mahalanobis.py score --fit_dir results/mahalanobis/fit_smoke \\
      --input_dir ... --output_dir ... --dataset_name aortaseg24_test --limit 1
"""
import argparse
import csv
import gc
import json
import os
import pathlib
import sys
from typing import Optional

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


# ---------------------------------------------------------------------------
# Feature extractor: nnUNetPredictor + forward hook on encoder bottleneck
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Wraps nnUNetPredictor with a forward hook on encoder.stages[-1] (bottleneck).
    Each sliding-window patch contributes one [C_bot]-dim vector via global
    average pooling over its spatial dims.  extract() averages over all patches
    of one case to produce a single feature vector for the scan.

    Network is held in eval() mode and use_mirroring=False — fully deterministic
    single forward pass per patch.
    """

    def __init__(self, model_dir: pathlib.Path,
                 checkpoint: str = "checkpoint_final.pth",
                 device: Optional[torch.device] = None):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._patch_features: list[torch.Tensor] = []

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=(0,),
            checkpoint_name=checkpoint,
        )

        # Bottleneck = deepest encoder stage (PlainConvUNet with return_skips=True)
        bottleneck = self._predictor.network.encoder.stages[-1]
        self.feature_dim = int(self._predictor.network.encoder.output_channels[-1])

        def _hook(module, inputs, output):
            with torch.no_grad():
                # output: [B, C_bot, D_bot, H_bot, W_bot] — GAP over spatial dims
                if output.dim() < 3:
                    return
                spatial_dims = tuple(range(2, output.dim()))
                gap = output.mean(dim=spatial_dims)            # [B, C_bot]
                self._patch_features.append(gap.detach().float().cpu())

        self._hook_handle = bottleneck.register_forward_hook(_hook)

    @torch.inference_mode()
    def extract(self, input_files: list[str], output_stem: pathlib.Path) -> np.ndarray:
        """
        Run one deterministic forward pass per case.  Predictions are written to
        f"{output_stem}.nii.gz" as a side effect (useful for downstream Dice).

        Returns the [feature_dim]-dim feature vector for the case (mean over
        all sliding-window patches).
        """
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        self._patch_features = []

        self._predictor.predict_from_files_sequential(
            [input_files],
            [str(output_stem)],
            save_probabilities=False,
            overwrite=True,
        )

        if not self._patch_features:
            raise RuntimeError(
                f"Hook captured no bottleneck features for {input_files}. "
                "Check that encoder.stages[-1] is the correct hook target."
            )

        # [n_patches, feature_dim] → mean over patches → [feature_dim]
        feats = torch.cat(self._patch_features, dim=0).mean(dim=0).numpy().astype(np.float32)
        self._patch_features = []

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return feats

    def close(self):
        self._hook_handle.remove()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def collect_cases(input_dir: pathlib.Path, limit: Optional[int]) -> list[tuple[str, list[str]]]:
    """Return [(case_id, [file_path]), ...] from *_0000.nii.gz files."""
    files = sorted(input_dir.glob("*_0000.nii.gz"))
    if limit:
        files = files[:limit]
    return [(f.name.replace("_0000.nii.gz", ""), [str(f)]) for f in files]


# ---------------------------------------------------------------------------
# Fit phase
# ---------------------------------------------------------------------------

def cmd_fit(args):
    from sklearn.decomposition import PCA
    from sklearn.covariance import EmpiricalCovariance
    import joblib

    splits = json.loads(pathlib.Path(args.splits_file).read_text())
    train_ids = splits[0]["train"]
    if args.limit:
        train_ids = train_ids[: args.limit]
    print(f"Fit phase: {len(train_ids)} training cases from {args.splits_file}")

    input_dir = pathlib.Path(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = args.output_dir / "features"
    pred_dir = args.output_dir / "predictions"
    feat_dir.mkdir(parents=True, exist_ok=True)

    case_files: list[tuple[str, list[str]]] = []
    for cid in train_ids:
        f = input_dir / f"{cid}_0000.nii.gz"
        if not f.exists():
            print(f"  WARN: {f} not found, skipping")
            continue
        case_files.append((cid, [str(f)]))

    pending = [(cid, fs) for cid, fs in case_files
               if not (feat_dir / f"{cid}.npy").exists()]
    n_skip = len(case_files) - len(pending)
    if n_skip:
        print(f"Resuming: {n_skip} cases already extracted, {len(pending)} remaining.")

    if pending:
        extractor = FeatureExtractor(MODEL_DIR, checkpoint=args.checkpoint)
        for i, (cid, files) in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {cid} ...", end=" ", flush=True)
            feat = extractor.extract(files, pred_dir / cid)
            np.save(feat_dir / f"{cid}.npy", feat)
            print(f"feat[:3]={np.round(feat[:3], 3).tolist()}  ||feat||={np.linalg.norm(feat):.2f}")
        extractor.close()
    else:
        print("All features already extracted; reusing.")

    # Assemble feature matrix in split order
    ids_done = [cid for cid, _ in case_files if (feat_dir / f"{cid}.npy").exists()]
    X = np.stack([np.load(feat_dir / f"{cid}.npy") for cid in ids_done], axis=0)
    print(f"\nFeature matrix: {X.shape}")
    np.save(args.output_dir / "train_features.npy", X)
    (args.output_dir / "train_ids.json").write_text(json.dumps(ids_done, indent=2))

    n_components = min(args.pca_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    Z = pca.fit_transform(X)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"PCA: {X.shape[1]} -> {n_components} dims; cumulative EVR = {evr:.3f}")

    cov = EmpiricalCovariance().fit(Z)
    md_train = cov.mahalanobis(Z)
    print(f"In-distribution Mahalanobis on training set: "
          f"mean={md_train.mean():.2f}  std={md_train.std():.2f}  "
          f"min={md_train.min():.2f}  max={md_train.max():.2f}")

    joblib.dump(
        {"pca": pca, "cov": cov, "train_ids": ids_done,
         "feature_dim": int(X.shape[1]), "n_components": n_components,
         "evr": evr},
        args.output_dir / "fit_artifacts.joblib",
    )
    np.save(args.output_dir / "train_mahalanobis.npy", md_train)

    print(f"\nFit complete. Artifacts -> {args.output_dir}")


# ---------------------------------------------------------------------------
# Score phase
# ---------------------------------------------------------------------------

def cmd_score(args):
    import joblib
    art = joblib.load(args.fit_dir / "fit_artifacts.joblib")
    pca = art["pca"]
    cov = art["cov"]
    print(f"Loaded fit: feature_dim={art['feature_dim']} -> "
          f"PCA dims={art['n_components']} (EVR={art['evr']:.3f}), "
          f"trained on {len(art['train_ids'])} cases.")

    cases = collect_cases(args.input_dir, args.limit)
    if not cases:
        print(f"No *_0000.nii.gz files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(cases)} cases in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = args.output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mahalanobis_summary.csv"

    done_ids: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="") as _f:
            for row in csv.DictReader(_f):
                done_ids.add(row["case_id"])

    pending = [(cid, fs) for cid, fs in cases if cid not in done_ids]
    n_skip = len(cases) - len(pending)
    if n_skip:
        print(f"Resuming: {n_skip} already scored, {len(pending)} remaining.")
    if not pending:
        print("All cases already scored.")
        return

    extractor = FeatureExtractor(MODEL_DIR, checkpoint=args.checkpoint)

    csv_mode = "a" if csv_path.exists() else "w"
    with open(csv_path, csv_mode, newline="") as csv_f:
        writer = csv.DictWriter(csv_f,
                                fieldnames=["dataset", "case_id", "mahalanobis"])
        if csv_mode == "w":
            writer.writeheader()

        for i, (cid, files) in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {cid} ...", end=" ", flush=True)
            feat = extractor.extract(files, args.output_dir / cid)
            np.save(feat_dir / f"{cid}.npy", feat)
            z = pca.transform(feat.reshape(1, -1))
            md = float(cov.mahalanobis(z)[0])
            writer.writerow({
                "dataset": args.dataset_name,
                "case_id": cid,
                "mahalanobis": f"{md:.4f}",
            })
            csv_f.flush()
            print(f"MD={md:.2f}")

    extractor.close()
    print(f"\nDone. Scores -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mahalanobis reliability signal — fit + score")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="Fit PCA + EmpiricalCovariance on AortaSeg24 training cases")
    pf.add_argument("--input_dir",    required=True, type=pathlib.Path,
                    help="nnUNet_raw/Dataset001_AortaSeg24/imagesTr")
    pf.add_argument("--splits_file",  required=True, type=pathlib.Path,
                    help="nnUNet_preprocessed/Dataset001_AortaSeg24/splits_final.json")
    pf.add_argument("--output_dir",   required=True, type=pathlib.Path)
    pf.add_argument("--pca_components", type=int, default=40)
    pf.add_argument("--checkpoint",   default="checkpoint_final.pth")
    pf.add_argument("--limit",        type=int, default=None,
                    help="Process only first N training cases (smoke test)")
    pf.set_defaults(func=cmd_fit)

    ps = sub.add_parser("score", help="Compute per-case Mahalanobis distance scalar")
    ps.add_argument("--fit_dir",      required=True, type=pathlib.Path,
                    help="Output of `fit` — contains fit_artifacts.joblib")
    ps.add_argument("--input_dir",    required=True, type=pathlib.Path)
    ps.add_argument("--output_dir",   required=True, type=pathlib.Path)
    ps.add_argument("--dataset_name", required=True)
    ps.add_argument("--checkpoint",   default="checkpoint_final.pth")
    ps.add_argument("--limit",        type=int, default=None,
                    help="Process only first N cases (smoke test)")
    ps.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
