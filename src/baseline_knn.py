"""
kNN feature-distance baseline for OOD / reliability scoring.

Non-parametric counterpart to Mahalanobis distance (Sun et al. 2022 ICML
"Out-of-distribution detection with deep nearest neighbors"). Operates on
the same encoder-bottleneck features that the Mahalanobis fit was built
on, after applying the SAME fitted PCA so that the only difference
between the two signals is parametric (Gaussian) vs non-parametric (kNN).

For each test case:
  d_i = || PCA(feat_i) - PCA(train_j) ||_2   for j in {1..50}
  knn_k = sorted(d)[k - 1]                   # k=1, 5, 10

Output:
  results/baselines/knn_<dataset>.csv     per-dataset rows
  results/baselines/knn_summary.csv       all 4 datasets concatenated

Usage:
  python src/baseline_knn.py
"""
import pathlib

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
MAH = ROOT / "results" / "mahalanobis"
OUT = ROOT / "results" / "baselines"
DATASETS = ["aortaseg24_test", "avt", "amos", "totalseg"]
KS = [1, 5, 10]


def kth_distances(test_feat: np.ndarray, train_feats: np.ndarray,
                  ks: list[int]) -> dict[int, float]:
    d = np.linalg.norm(train_feats - test_feat[None, :], axis=1)
    d.sort()
    return {k: float(d[k - 1]) for k in ks}


def main() -> None:
    art = joblib.load(MAH / "fit" / "fit_artifacts.joblib")
    pca = art["pca"]
    train_raw = np.load(MAH / "fit" / "train_features.npy")
    train_proj = pca.transform(train_raw)
    print(f"train: raw {train_raw.shape} -> PCA {train_proj.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # leave-one-out sanity on training set
    print("\n[sanity] leave-one-out kNN-5 distance on 50 training cases:")
    loo = []
    for i in range(len(train_proj)):
        others = np.delete(train_proj, i, axis=0)
        loo.append(kth_distances(train_proj[i], others, [5])[5])
    print(f"  mean = {np.mean(loo):.3f}  max = {np.max(loo):.3f}")

    for ds in DATASETS:
        feat_dir = MAH / "scores" / ds / "features"
        if not feat_dir.exists():
            print(f"[skip] {ds} has no features dir")
            continue
        rows = []
        for f in sorted(feat_dir.glob("*.npy")):
            case_id = f.stem
            raw = np.load(f)
            proj = pca.transform(raw[None, :])[0]
            d = kth_distances(proj, train_proj, KS)
            rows.append({"dataset": ds, "case_id": case_id,
                         **{f"knn_k{k}": d[k] for k in KS}})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"knn_{ds}.csv", index=False)
        all_rows.extend(rows)
        print(f"[ok] {ds}: n={len(df)}  knn5 median={df['knn_k5'].median():.3f}"
              f"  max={df['knn_k5'].max():.3f}")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT / "knn_summary.csv", index=False)
    print(f"\nWrote {OUT / 'knn_summary.csv'}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
