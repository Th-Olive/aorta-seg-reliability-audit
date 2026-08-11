"""
Reliability-signal analysis: combine Mahalanobis distance with deterministic Dice.

Outputs:
  results/mahalanobis/reliability_<dataset>.csv   per-case: dataset, case_id, dice, mahalanobis
  results/mahalanobis/reliability_summary.json    overall metrics:
      OOD-AUROC (AortaSeg24 test ID vs each external as OOD)
      Unc-AUROC (low-Dice detection, per dataset, multiple Dice thresholds)
      Spearman rho(MD, Dice)
      ESCE — Expected Squared Calibration Error of Mahalanobis as Dice predictor
      Risk-coverage AUC

Usage:
  python src/analyze_mahalanobis.py
"""
import csv
import json
import pathlib
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).parent.parent
MAHA_ROOT = ROOT / "results" / "mahalanobis"

# Dataset config: (CSV path for MD, CSV path for Dice, Dice column name)
DATASETS = {
    "aortaseg24_test": (
        MAHA_ROOT / "scores" / "aortaseg24_test" / "mahalanobis_summary.csv",
        MAHA_ROOT / "metrics_aortaseg24.csv",
        None,  # 23-class — compute mean over per-class columns
    ),
    "avt": (
        MAHA_ROOT / "scores" / "avt" / "mahalanobis_summary.csv",
        MAHA_ROOT / "metrics_avt.csv",
        "dice_binary",
    ),
    "amos": (
        MAHA_ROOT / "scores" / "amos" / "mahalanobis_summary.csv",
        MAHA_ROOT / "metrics_amos.csv",
        "dice_binary",
    ),
}


def read_md(path: pathlib.Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return {row["case_id"]: float(row["mahalanobis"])
            for row in csv.DictReader(open(path))}


def read_dice(path: pathlib.Path, col: str | None) -> dict[str, float]:
    """Return case_id -> mean Dice (over 23 classes for AortaSeg24, single col else)."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["case_id"]
            if col is None:
                vals = [float(v) for k, v in row.items()
                        if k.startswith("dice_cls") and v not in ("", "nan")]
                if vals:
                    out[cid] = float(np.mean(vals))
            else:
                v = row.get(col)
                if v not in (None, "", "nan"):
                    out[cid] = float(v)
    return out


def join_cases(md: dict[str, float], dice: dict[str, float]) -> tuple[list, np.ndarray, np.ndarray]:
    ids = sorted(set(md) & set(dice))
    return (ids,
            np.array([md[i]   for i in ids], dtype=np.float64),
            np.array([dice[i] for i in ids], dtype=np.float64))


def unc_auroc(scores: np.ndarray, dice: np.ndarray, threshold: float) -> float | None:
    """AUROC for predicting Dice < threshold from MD. Higher MD = more uncertain."""
    y = (dice < threshold).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, scores))


def risk_coverage_auc(scores: np.ndarray, errors: np.ndarray) -> float:
    """AUC for risk-vs-coverage curve. Lower = better.
    Order by increasing MD (most confident first); plot mean error over kept fraction."""
    order = np.argsort(scores)
    errs = errors[order]
    n = len(errs)
    coverage = np.arange(1, n + 1) / n
    cum_risk = np.cumsum(errs) / np.arange(1, n + 1)
    return float(np.trapezoid(cum_risk, coverage))


def expected_squared_calibration_error(scores: np.ndarray, dice: np.ndarray,
                                       n_bins: int = 10) -> tuple[float, list]:
    """
    ESCE of mahalanobis as a Dice *unreliability* predictor.
    Bin by MD percentile; for each bin compute mean(MD) and mean(1 - Dice).
    Map MD to [0,1] via percentile rank (since MD has no natural scale).
    Return ESCE plus bin diagnostics.
    """
    if len(scores) < n_bins:
        n_bins = max(2, len(scores) // 2)
    # Normalize MD to [0,1] via empirical percentile rank — gives a calibration in distribution units
    order = np.argsort(scores)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(scores))
    norm_md = (rank + 0.5) / len(scores)
    err = 1.0 - dice
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(norm_md, bin_edges[1:-1])
    diagnostics = []
    sq_err_weighted = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        conf = norm_md[mask].mean()
        observed_err = err[mask].mean()
        diagnostics.append({
            "bin": b, "n": int(mask.sum()),
            "mean_md_rank": round(float(conf), 4),
            "mean_dice": round(float(dice[mask].mean()), 4),
            "mean_error": round(float(observed_err), 4),
        })
        sq_err_weighted += (mask.sum() / len(scores)) * (conf - observed_err) ** 2
    return float(sq_err_weighted), diagnostics


def main():
    # ------------------------------------------------------------------
    # 1. Load per-case MD and Dice for each dataset
    # ------------------------------------------------------------------
    data = {}
    for name, (md_path, dice_path, dice_col) in DATASETS.items():
        md = read_md(md_path)
        dice = read_dice(dice_path, dice_col)
        ids, md_arr, d_arr = join_cases(md, dice)
        data[name] = {"ids": ids, "md": md_arr, "dice": d_arr,
                      "all_md": np.array(list(md.values())) if md else np.array([])}
        print(f"{name}: {len(md)} MD scored, {len(dice)} Dice evaluated, "
              f"{len(ids)} joined")

        # Per-case reliability CSV
        out_path = MAHA_ROOT / f"reliability_{name}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "case_id", "mahalanobis", "dice"])
            for cid, m, d in zip(ids, md_arr, d_arr):
                w.writerow([name, cid, f"{m:.4f}", f"{d:.6f}"])

    # ------------------------------------------------------------------
    # 2. OOD-AUROC: AortaSeg24 test (ID) vs each external (OOD)
    # ------------------------------------------------------------------
    summary: dict = {"ood_auroc": {}, "per_dataset": {}, "calibration": {}}

    id_md_all = data["aortaseg24_test"]["all_md"]
    for ood_name in ("avt", "amos"):
        ood_md = data[ood_name]["all_md"]
        if len(ood_md) == 0 or len(id_md_all) == 0:
            continue
        y = np.concatenate([np.zeros(len(id_md_all)), np.ones(len(ood_md))])
        s = np.concatenate([id_md_all,                 ood_md])
        auroc = float(roc_auc_score(y, s))
        summary["ood_auroc"][f"id=aortaseg24_test_vs_ood={ood_name}"] = {
            "auroc":   round(auroc, 4),
            "n_id":    len(id_md_all),
            "n_ood":   len(ood_md),
            "id_med":  round(float(np.median(id_md_all)), 2),
            "ood_med": round(float(np.median(ood_md)),    2),
        }

    # ------------------------------------------------------------------
    # 3. Per-dataset reliability metrics (need MD + Dice joined)
    # ------------------------------------------------------------------
    for name, d in data.items():
        if len(d["ids"]) == 0:
            continue
        md, dice = d["md"], d["dice"]

        rho, p = spearmanr(md, dice)
        ds_summary: dict = {
            "n": len(d["ids"]),
            "dice_mean":  round(float(dice.mean()), 4),
            "dice_std":   round(float(dice.std()),  4),
            "md_mean":    round(float(md.mean()),   2),
            "md_median":  round(float(np.median(md)), 2),
            "md_max":     round(float(md.max()),    2),
            "spearman_rho_md_dice": round(float(rho), 4),
            "spearman_p":           round(float(p),   4),
        }

        for thr in (0.50, 0.70, 0.80):
            auroc = unc_auroc(md, dice, thr)
            ds_summary[f"unc_auroc_dice_lt_{thr:.2f}"] = (
                round(auroc, 4) if auroc is not None else None)

        rcauc = risk_coverage_auc(md, 1.0 - dice)
        ds_summary["risk_coverage_auc_lower_better"] = round(rcauc, 4)

        esce, bins_diag = expected_squared_calibration_error(md, dice)
        ds_summary["esce"] = round(esce, 4)
        summary["calibration"][name] = bins_diag

        summary["per_dataset"][name] = ds_summary

    # ------------------------------------------------------------------
    # 4. Write summary JSON
    # ------------------------------------------------------------------
    out_json = MAHA_ROOT / "reliability_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
