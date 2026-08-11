"""MC-sample-count (T) ablation for the pairwise-Dice signal.

Recomputes the pairwise-Dice signal at T in {5, 10, 20} from the saved
per-pass argmaxes, so the sample budget can be varied without re-running
inference.  Each case's 20 passes are loaded once and reduced to a single
20x20 per-class intersection matrix; the T=5/10/20 values are then averaged
over the pairs drawn from the first T passes.

BLAS threads are capped at module top, before numpy is imported, since the
thread-pool size is bound at import time.  Override with TABLATION_THREADS
(default 4).

Output: results/mc_passes/tablation_summary.csv plus a stdout table.
Per-case progress goes to stderr, so `python src/run_tablation.py > table.txt`
captures only the table.

Optional cohort filter:  python src/run_tablation.py ID AVT
"""
import os

# --- cap BLAS threads BEFORE numpy is imported.
# os.environ here is process-local (NOT a global Windows env var); setdefault
# so an explicit OMP_NUM_THREADS from the launching shell still wins.
_THREADS = os.environ.get("TABLATION_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import contextlib  # noqa: E402
import csv         # noqa: E402
import pathlib     # noqa: E402
import sys         # noqa: E402
import time        # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np     # noqa: E402
from scipy.stats import spearmanr     # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

try:
    from threadpoolctl import threadpool_limits
except ImportError:                    # optional; env vars already cap the pool
    threadpool_limits = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# reuse the building blocks from the pairwise analysis module
from analyze_pairwise import (  # noqa: E402
    _pairwise_intersect_via_matmul, read_csv_keyed, read_gt_dice)

TS = (5, 10, 20)

CONFIGS = [
    ("ID",   ROOT / "results/mc_passes/aortaseg24_test",
     ROOT / "results/mahalanobis/metrics_aortaseg24.csv", "multiclass"),
    ("AVT",  ROOT / "results/mc_passes/avt",
     ROOT / "results/mahalanobis/metrics_avt.csv", "binary"),
    ("AMOS", ROOT / "results/mc_passes/amos",
     ROOT / "results/mahalanobis/metrics_amos.csv", "binary"),
]

# optional cohort filter: `python src/run_tablation.py ID AVT`
_sel = {a.upper() for a in sys.argv[1:]}
if _sel:
    CONFIGS = [c for c in CONFIGS if c[0] in _sel]


def compute_case_pairwise_multiT(case_dir: pathlib.Path, Ts=TS):
    """Load max(Ts) passes once; return {T: pairwise_dice_mean} (or None).

    Numerically identical to evaluating analyze_pairwise.compute_case_pairwise(
    case_dir, T) for each T separately: inter[i,j] and the per-pass voxel
    counts do not depend on T, so the per-class accumulation for every pair is
    the same.  We build pair_dice_mean for all C(Tmax,2) pairs, then for a
    given T average only the pairs whose larger index j < T (i.e. the pairs
    drawn from passes 0..T-1).
    """
    Tmax = max(Ts)
    pass_paths = [case_dir / f"pass_{k:02d}.nii.gz" for k in range(Tmax)]
    if not all(p.exists() for p in pass_paths):
        return None

    # native-dtype read: avoids the get_fdata() float64 blow-up
    passes_flat = None
    for k, p in enumerate(pass_paths):
        arr = np.asanyarray(nib.load(p).dataobj).astype(np.uint8, copy=False)
        if passes_flat is None:
            passes_flat = np.empty((Tmax, arr.size), dtype=np.uint8)
        passes_flat[k] = arr.reshape(-1)

    n_cls = int(passes_flat.max()) + 1
    n_pairs = Tmax * (Tmax - 1) // 2

    # canonical pair ordering (i<j); store the larger index j for T-slicing
    pj = np.empty(n_pairs, dtype=np.int32)
    idx = 0
    for i in range(Tmax):
        for j in range(i + 1, Tmax):
            pj[idx] = j
            idx += 1

    pair_dice_sum = np.zeros(n_pairs, dtype=np.float64)
    pair_dice_count = np.zeros(n_pairs, dtype=np.int32)

    for c in range(1, n_cls):
        M_flat = (passes_flat == c).astype(np.uint8, copy=False)
        sums_c = M_flat.sum(axis=1, dtype=np.int64)
        if sums_c.sum() == 0:
            continue
        inter = _pairwise_intersect_via_matmul(M_flat)  # one (Tmax,Tmax) matmul
        idx = 0
        for i in range(Tmax):
            si = int(sums_c[i])
            for j in range(i + 1, Tmax):
                den = si + int(sums_c[j])
                if den > 0:
                    pair_dice_sum[idx] += 2.0 * int(inter[i, j]) / den
                    pair_dice_count[idx] += 1
                idx += 1
        del M_flat, inter
    del passes_flat

    pair_dice_mean = np.where(pair_dice_count > 0,
                              pair_dice_sum / np.maximum(pair_dice_count, 1),
                              np.nan)
    return {T: float(np.nanmean(pair_dice_mean[pj < T])) for T in Ts}


def auc(gt: np.ndarray, unc: np.ndarray, thr: float) -> float:
    """Unc-AUROC for the event gt < thr; nan if the event never/always fires."""
    y = (gt < thr).astype(int)
    return roc_auc_score(y, unc) if 0 < y.sum() < len(y) else float("nan")


def main():
    rows = []
    print(f"{'cohort':5s} {'T':>3s} {'n':>4s} {'rho':>7s} "
          f"{'uncAUROC@0.5':>13s} {'uncAUROC@0.7':>13s}", flush=True)
    for name, passes_dir, dice_csv, fmt in CONFIGS:
        dice_map = read_csv_keyed(dice_csv)
        case_dirs = sorted(d for d in passes_dir.iterdir() if d.is_dir())
        pw = {T: [] for T in TS}
        gt = []
        t_cohort = time.time()
        for ci, d in enumerate(case_dirs):
            t0 = time.time()
            res = compute_case_pairwise_multiT(d, TS)
            if res is None:
                print(f"  . {name} [{ci+1}/{len(case_dirs)}] {d.name}: "
                      f"SKIP (missing passes)", file=sys.stderr, flush=True)
                continue
            g = read_gt_dice(dice_map[d.name], fmt) if d.name in dice_map else None
            if g is None:
                print(f"  . {name} [{ci+1}/{len(case_dirs)}] {d.name}: "
                      f"SKIP (no GT dice)", file=sys.stderr, flush=True)
                continue
            for T in TS:
                pw[T].append(res[T])
            gt.append(g)
            print(f"  . {name} [{ci+1}/{len(case_dirs)}] {d.name} "
                  f"({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)
        gt = np.array(gt)
        for T in TS:
            pwT = np.array(pw[T])
            rho, _ = spearmanr(pwT, gt)
            a5, a7 = auc(gt, 1.0 - pwT, 0.5), auc(gt, 1.0 - pwT, 0.7)
            print(f"{name:5s} {T:3d} {len(pwT):4d} {rho:7.3f} "
                  f"{a5:13.3f} {a7:13.3f}", flush=True)
            rows.append({"cohort": name, "T": T, "n": len(pwT),
                         "rho": round(float(rho), 4),
                         "unc_auroc_0.5": round(float(a5), 4),
                         "unc_auroc_0.7": round(float(a7), 4)})
        print(f"  --> {name}: {len(gt)} cases in {time.time()-t_cohort:.0f}s",
              file=sys.stderr, flush=True)

    out = ROOT / "results/mc_passes/tablation_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n--> wrote {out}", flush=True)


if __name__ == "__main__":
    cap = (threadpool_limits(int(_THREADS)) if threadpool_limits
           else contextlib.nullcontext())
    with cap:
        main()
