"""
Pairwise-Dice reliability signal from per-pass MC samples.

For each case, load T per-pass argmax NIfTIs (output of inference_mc_passes.py)
and compute mean of all C(T, 2) pairwise multi-class Dice scores between
passes; lower mean = more disagreement among samples = higher uncertainty.

Hot-path optimization: iterate by class rather than by pair.  For each class c,
extract the T boolean masks across passes, pre-compute mask voxel counts once,
then compute the 190 pairwise intersection counts.  This avoids recomputing the
.sum() of each pass T-1 times and lets numpy keep the per-class mask arrays
resident in cache.

Output:
  results/mc_passes/pairwise_summary.csv  (case_id, pairwise_dice_mean,
      min, max, n_pairs, gt_dice, mahalanobis) — written one row per case
      so progress is visible during the run.
  results/mc_passes/pairwise_metrics.json (overall correlations + AUROCs)

Usage:
  python src/analyze_pairwise.py \\
    --passes_dir  results/mc_passes/aortaseg24_test \\
    --dice_csv    results/mahalanobis/metrics_aortaseg24.csv \\
    --md_csv      results/mahalanobis/scores/aortaseg24_test/mahalanobis_summary.csv \\
    --output_dir  results/mc_passes
"""
import argparse
import csv
import json
import pathlib
import sys
import time

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _pairwise_intersect_via_matmul(M_flat: np.ndarray,
                                   chunk_size: int = 8_000_000) -> np.ndarray:
    """All T×T pair intersections for a (T, N) uint8 0/1 matrix.

    Each chunk casts to **float32** and uses BLAS sgemm via numpy matmul, which
    is 10-50× faster than the per-pair Python loop with np.logical_and+sum.
    (Integer matmul in numpy is NOT BLAS-accelerated and was actually slower
    than the loop — verified empirically.) The chunked accumulation keeps peak
    memory bounded (~T × chunk_size × 4 bytes ≈ 640 MB for T=20, chunk=8e6).
    Values are 0/1 so float32 exactly represents them; np.rint guards against
    any FP drift before casting to int64.
    """
    T, N = M_flat.shape
    inter = np.zeros((T, T), dtype=np.int64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = M_flat[:, start:end].astype(np.float32, copy=False)
        inter += np.rint(chunk @ chunk.T).astype(np.int64)
    return inter


def compute_case_pairwise(case_dir: pathlib.Path, T: int) -> dict | None:
    """Load T passes; return summary stats for this case (or None if missing).

    Per class, the T pair intersections are computed via one chunked BLAS
    matmul on the (T, N) bool→int32 mask matrix — much faster than 190
    separate np.logical_and+sum calls. Output is numerically identical.
    """
    pass_paths = [case_dir / f"pass_{k:02d}.nii.gz" for k in range(T)]
    if not all(p.exists() for p in pass_paths):
        return None

    # Load all passes as uint8 (24 classes fits) — 1 byte/voxel
    t0 = time.time()
    passes = [nib.load(p).get_fdata().astype(np.uint8) for p in pass_paths]
    t_load = time.time() - t0

    n_cls = int(max(p.max() for p in passes)) + 1
    n_pairs = T * (T - 1) // 2

    pair_dice_sum   = np.zeros(n_pairs, dtype=np.float64)
    pair_dice_count = np.zeros(n_pairs, dtype=np.int32)

    # Stack all passes as a single (T, N) uint8 matrix — one allocation, reused
    # across classes. With T=20 and N≈125 M voxels this is ~2.5 GB.
    N = passes[0].size
    passes_flat = np.empty((T, N), dtype=np.uint8)
    for k, p in enumerate(passes):
        passes_flat[k] = p.reshape(-1)
    del passes

    t1 = time.time()
    for c in range(1, n_cls):
        # Boolean class-c masks for all T passes, (T, N) uint8 (0/1)
        M_flat = (passes_flat == c).astype(np.uint8, copy=False)
        sums_c = M_flat.sum(axis=1, dtype=np.int64)
        if sums_c.sum() == 0:
            continue
        inter_mat = _pairwise_intersect_via_matmul(M_flat)
        idx = 0
        for i in range(T):
            si = int(sums_c[i])
            for j in range(i + 1, T):
                den = si + int(sums_c[j])
                if den > 0:
                    pair_dice_sum[idx]   += 2.0 * int(inter_mat[i, j]) / den
                    pair_dice_count[idx] += 1
                idx += 1
        del M_flat, inter_mat
    t_compute = time.time() - t1
    del passes_flat

    # Mean Dice per pair (over the classes where at least one mask was nonempty)
    pair_dice_mean = np.where(pair_dice_count > 0,
                              pair_dice_sum / np.maximum(pair_dice_count, 1),
                              np.nan)

    return {
        "pairwise_dice_mean":   float(np.nanmean(pair_dice_mean)),
        "pairwise_dice_median": float(np.nanmedian(pair_dice_mean)),
        "pairwise_dice_min":    float(np.nanmin(pair_dice_mean)),
        "pairwise_dice_max":    float(np.nanmax(pair_dice_mean)),
        "n_pairs":              int(n_pairs),
        "n_classes":            int(n_cls - 1),
        "t_load_s":             round(t_load, 1),
        "t_compute_s":          round(t_compute, 1),
    }


def read_csv_keyed(path: pathlib.Path, key: str = "case_id"):
    if not path.exists():
        return {}
    return {r[key]: r for r in csv.DictReader(open(path))}


def mean_dice_aortaseg24(row: dict) -> float | None:
    vals = [float(v) for k, v in row.items()
            if k.startswith("dice_cls") and v not in ("", "nan", None)]
    return float(np.mean(vals)) if vals else None


def binary_dice(row: dict) -> float | None:
    v = row.get("dice_binary")
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_gt_dice(row: dict, dice_format: str) -> float | None:
    """Extract ground-truth Dice scalar from a row in either CSV format."""
    if dice_format == "multiclass":
        return mean_dice_aortaseg24(row)
    if dice_format == "binary":
        return binary_dice(row)
    raise ValueError(f"unknown dice_format={dice_format}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--passes_dir", required=True, type=pathlib.Path)
    p.add_argument("--dice_csv",   required=True, type=pathlib.Path)
    p.add_argument("--md_csv",     required=True, type=pathlib.Path)
    p.add_argument("--T",          type=int, default=20)
    p.add_argument("--output_dir", required=True, type=pathlib.Path)
    p.add_argument("--dice_format", choices=["multiclass", "binary"],
                   default="multiclass",
                   help="multiclass = mean over dice_cls* columns (AortaSeg24); "
                        "binary = single dice_binary column (AVT, AMOS)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([d for d in args.passes_dir.iterdir() if d.is_dir()])
    print(f"Found {len(case_dirs)} case directories in {args.passes_dir}",
          flush=True)

    dice_map = read_csv_keyed(args.dice_csv)
    md_map   = read_csv_keyed(args.md_csv)

    out_csv = args.output_dir / "pairwise_summary.csv"
    # Resume: skip case_ids already in CSV
    done_ids: set[str] = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f):
                done_ids.add(r["case_id"])

    fieldnames = ["case_id", "pairwise_dice_mean", "pairwise_dice_median",
                  "pairwise_dice_min", "pairwise_dice_max",
                  "n_pairs", "n_classes", "ground_truth_dice", "mahalanobis"]
    mode = "a" if out_csv.exists() else "w"
    f_csv = open(out_csv, mode, newline="")
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()
        f_csv.flush()

    pending = [d for d in case_dirs if d.name not in done_ids]
    print(f"Resuming: {len(done_ids)} done, {len(pending)} remaining", flush=True)

    rows = []
    if done_ids:
        # Re-read previous rows for final metrics
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({k: r[k] for k in fieldnames})

    for i, d in enumerate(pending):
        cid = d.name
        t0 = time.time()
        r = compute_case_pairwise(d, args.T)
        if r is None:
            print(f"  [{i+1}/{len(pending)}] {cid}: SKIP (missing passes)",
                  flush=True)
            continue
        gt_dice = read_gt_dice(dice_map[cid], args.dice_format) if cid in dice_map else None
        md_val  = float(md_map[cid]["mahalanobis"]) if cid in md_map else None
        row = {
            "case_id":              cid,
            "pairwise_dice_mean":   f"{r['pairwise_dice_mean']:.6f}",
            "pairwise_dice_median": f"{r['pairwise_dice_median']:.6f}",
            "pairwise_dice_min":    f"{r['pairwise_dice_min']:.6f}",
            "pairwise_dice_max":    f"{r['pairwise_dice_max']:.6f}",
            "n_pairs":              r["n_pairs"],
            "n_classes":            r["n_classes"],
            "ground_truth_dice":    f"{gt_dice:.6f}" if gt_dice is not None else "",
            "mahalanobis":          f"{md_val:.4f}" if md_val is not None else "",
        }
        writer.writerow(row); f_csv.flush()
        rows.append(row)
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(pending)}] {cid}  "
              f"pairwise={r['pairwise_dice_mean']:.4f}  "
              f"gt_dice={(f'{gt_dice:.4f}' if gt_dice is not None else 'NA')}  "
              f"md={(f'{md_val:.1f}' if md_val is not None else 'NA')}  "
              f"(load={r['t_load_s']}s compute={r['t_compute_s']}s total={elapsed:.0f}s)",
              flush=True)

    f_csv.close()
    print(f"\nPer-case -> {out_csv} ({len(rows)} cases)", flush=True)

    # Aggregate metrics
    pw, gt, mds = [], [], []
    for r in rows:
        try:
            pw.append(float(r["pairwise_dice_mean"]))
            if r.get("ground_truth_dice"):
                gt.append((len(pw) - 1, float(r["ground_truth_dice"])))
            if r.get("mahalanobis"):
                mds.append((len(pw) - 1, float(r["mahalanobis"])))
        except (ValueError, TypeError):
            continue
    pw = np.array(pw)

    summary = {
        "n":                  len(pw),
        "pairwise_dice_mean": round(float(pw.mean()), 4) if len(pw) else None,
        "pairwise_dice_std":  round(float(pw.std()),  4) if len(pw) else None,
    }
    if gt and len(gt) >= 4:
        gt_idx, gt_vals = zip(*gt)
        gt_arr = np.array(gt_vals)
        pw_aligned = pw[list(gt_idx)]
        summary["gt_dice_mean"] = round(float(gt_arr.mean()), 4)
        rho, pval = spearmanr(pw_aligned, gt_arr)
        summary["spearman_pairwise_vs_gt_dice"] = {
            "rho": round(float(rho), 4),
            "p":   round(float(pval), 4),
        }
        unc = 1.0 - pw_aligned
        for thr in (0.5, 0.7, 0.8):
            y = (gt_arr < thr).astype(int)
            if 0 < y.sum() < len(y):
                summary[f"unc_auroc_gtdice_lt_{thr}"] = round(
                    float(roc_auc_score(y, unc)), 4)
    if mds and len(mds) >= 4:
        md_idx, md_vals = zip(*mds)
        md_arr = np.array(md_vals)
        pw_aligned = pw[list(md_idx)]
        rho_md, _ = spearmanr(md_arr, pw_aligned)
        summary["spearman_md_vs_pairwise"] = round(float(rho_md), 4)

    out_json = args.output_dir / "pairwise_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nMetrics -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
