"""
Bootstrap 95% CIs for every headline metric reported in the UNSURE 2026 draft.

Reads the existing per-case CSVs and emits one JSON with point estimate, 95% CI
(percentile bootstrap, B=2000), and n for each metric.  For Spearman, also
reports the Fisher-z analytic CI as a sanity check against the bootstrap CI.

Metrics covered:
  - OOD-AUROC: Mahalanobis on AortaSeg24-test (ID) vs AVT / AMOS / TotalSeg (OOD)
  - Unc-AUROC: Mahalanobis vs (Dice < tau) within AortaSeg24-test, AVT, AMOS
  - Spearman: MD vs Dice within each dataset
  - Spearman: pairwise-Dice vs GT-Dice (AortaSeg24-test); MD vs pairwise (same)
  - Unc-AUROC: pairwise-Dice vs (GT-Dice < tau) within AortaSeg24-test
  - MD-vs-centerline-failure AUROC on AVT and AMOS
  - MLD MAE per dataset (and per AVT subgroup for completeness)

Output:
  results/bootstrap_ci.json   one block per metric, alphabetised within groups

Run:
  python src/bootstrap_metrics.py
"""
import json
import math
import pathlib
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).parent.parent
RESULTS = ROOT / "results"

B = 2000          # bootstrap replicates
ALPHA = 0.05
SEED = 20260519   # today's date for reproducibility


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC.  labels in {0, 1}, scores higher => more likely class 1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def bootstrap_auroc(scores: Sequence[float], labels: Sequence[int],
                    rng: np.random.Generator, b: int = B) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    point = auroc(scores, labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    boot = np.empty(b)
    for k in range(b):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        boot[k] = auroc(scores[idx], labels[idx])
    lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])
    return point, float(lo), float(hi)


def bootstrap_spearman(x: Sequence[float], y: Sequence[float],
                       rng: np.random.Generator, b: int = B
                       ) -> tuple[float, float, float, float, float, float]:
    """Returns (rho, lo_boot, hi_boot, p, lo_fisher, hi_fisher)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = stats.spearmanr(x, y)
    rho = float(res.statistic)
    p_value = float(res.pvalue)
    n = len(x)
    boot = np.empty(b)
    idx_all = np.arange(n)
    for k in range(b):
        idx = rng.choice(idx_all, size=n, replace=True)
        if len(set(idx)) < 3:
            boot[k] = np.nan
            continue
        boot[k] = stats.spearmanr(x[idx], y[idx]).statistic
    lo_b, hi_b = np.nanquantile(boot, [ALPHA / 2, 1 - ALPHA / 2])
    # Fisher-z analytic CI
    if abs(rho) < 1.0 and n > 3:
        z = math.atanh(rho)
        se = 1.0 / math.sqrt(n - 3)
        zcrit = stats.norm.ppf(1 - ALPHA / 2)
        lo_f = math.tanh(z - zcrit * se)
        hi_f = math.tanh(z + zcrit * se)
    else:
        lo_f = hi_f = float("nan")
    return rho, float(lo_b), float(hi_b), p_value, float(lo_f), float(hi_f)


def bootstrap_mae(diffs: Sequence[float], rng: np.random.Generator,
                  b: int = B) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    point = float(np.mean(np.abs(diffs)))
    n = len(diffs)
    boot = np.empty(b)
    for k in range(b):
        idx = rng.integers(0, n, size=n)
        boot[k] = float(np.mean(np.abs(diffs[idx])))
    lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])
    return point, float(lo), float(hi)


def fmt(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"meta": {"B": B, "alpha": ALPHA, "seed": SEED}}

    # ---- load per-case data ----
    id_df = pd.read_csv(RESULTS / "integrated_per_case_aortaseg24_test.csv")
    avt_df = pd.read_csv(RESULTS / "integrated_per_case_avt.csv")
    amos_df = pd.read_csv(RESULTS / "integrated_per_case_amos.csv")
    ts_df = pd.read_csv(RESULTS / "mahalanobis" / "scores" / "totalseg"
                        / "mahalanobis_summary.csv")
    pair_df = pd.read_csv(RESULTS / "mc_passes" / "pairwise_summary.csv")
    pair_avt_df = pd.read_csv(RESULTS / "mc_passes" / "avt_analysis" / "pairwise_summary.csv")
    pair_amos_path = RESULTS / "mc_passes" / "amos_analysis" / "pairwise_summary.csv"
    pair_amos_df = pd.read_csv(pair_amos_path) if pair_amos_path.exists() else None
    knn_path = RESULTS / "baselines" / "knn_summary.csv"
    knn_df = pd.read_csv(knn_path) if knn_path.exists() else None
    msp_path = RESULTS / "baselines" / "msp_summary.csv"
    msp_df = pd.read_csv(msp_path) if msp_path.exists() else None

    def merge_baseline(base_df: pd.DataFrame | None, ds: str,
                       integ_df: pd.DataFrame, col: str) -> pd.DataFrame | None:
        if base_df is None:
            return None
        sub = base_df[base_df.dataset == ds][["case_id", col]]
        return integ_df[["case_id", "dice"]].merge(sub, on="case_id", how="inner")

    # ---- 1. OOD-AUROC: MD on ID vs each OOD ----
    out["ood_auroc_md"] = {}
    for name, ood in [("avt", avt_df), ("amos", amos_df), ("totalseg", ts_df)]:
        scores = np.concatenate([id_df["mahalanobis"].values,
                                 ood["mahalanobis"].values])
        labels = np.concatenate([np.zeros(len(id_df)), np.ones(len(ood))])
        p, lo, hi = bootstrap_auroc(scores, labels, rng)
        out["ood_auroc_md"][f"aortaseg24_test_vs_{name}"] = {
            "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
            "n_id": int(len(id_df)), "n_ood": int(len(ood)),
            "report": fmt(p, lo, hi),
        }

    # ---- 2. Unc-AUROC: MD vs (Dice < tau) within each dataset ----
    out["unc_auroc_md"] = {}
    for name, df in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
        for tau in (0.5, 0.7):
            labels = (df["dice"].values < tau).astype(int)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            p, lo, hi = bootstrap_auroc(df["mahalanobis"].values, labels, rng)
            out["unc_auroc_md"][f"{name}_dice_lt_{tau}"] = {
                "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                "n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                "report": fmt(p, lo, hi),
            }

    # ---- 3. Spearman: MD vs Dice within each dataset ----
    out["spearman_md_vs_dice"] = {}
    for name, df in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
        rho, lob, hib, pv, lof, hif = bootstrap_spearman(
            df["mahalanobis"].values, df["dice"].values, rng)
        out["spearman_md_vs_dice"][name] = {
            "rho": rho, "p": pv,
            "ci95_lo_bootstrap": lob, "ci95_hi_bootstrap": hib,
            "ci95_lo_fisher": lof, "ci95_hi_fisher": hif,
            "n": int(len(df)),
            "report": fmt(rho, lob, hib),
        }

    # ---- 4. Pairwise-Dice block (AortaSeg24-test + AVT + AMOS if available) ----
    out["pairwise"] = {}
    pairwise_inputs = [("id", pair_df), ("avt", pair_avt_df)]
    if pair_amos_df is not None:
        pairwise_inputs.append(("amos", pair_amos_df))
    for suffix, pdf in pairwise_inputs:
        rho, lob, hib, pv, lof, hif = bootstrap_spearman(
            pdf["pairwise_dice_mean"].values, pdf["ground_truth_dice"].values, rng)
        out["pairwise"][f"spearman_pairwise_vs_gt_dice_{suffix}"] = {
            "rho": rho, "p": pv,
            "ci95_lo_bootstrap": lob, "ci95_hi_bootstrap": hib,
            "ci95_lo_fisher": lof, "ci95_hi_fisher": hif,
            "n": int(len(pdf)),
            "report": fmt(rho, lob, hib),
        }
        rho, lob, hib, pv, lof, hif = bootstrap_spearman(
            pdf["mahalanobis"].values, pdf["pairwise_dice_mean"].values, rng)
        out["pairwise"][f"spearman_md_vs_pairwise_{suffix}"] = {
            "rho": rho, "p": pv,
            "ci95_lo_bootstrap": lob, "ci95_hi_bootstrap": hib,
            "ci95_lo_fisher": lof, "ci95_hi_fisher": hif,
            "n": int(len(pdf)),
            "report": fmt(rho, lob, hib),
        }
        # Higher pairwise => higher Dice => predicts "Dice < tau" with NEGATED score
        for tau in (0.5, 0.7, 0.8):
            labels = (pdf["ground_truth_dice"].values < tau).astype(int)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            p, lo, hi = bootstrap_auroc(-pdf["pairwise_dice_mean"].values, labels, rng)
            out["pairwise"][f"unc_auroc_{suffix}_dice_lt_{tau}"] = {
                "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                "n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                "report": fmt(p, lo, hi),
            }

    # ---- 5. MD vs centerline-failure AUROC (AVT, AMOS) ----
    # The integrated_per_case_*.csv files have a stale `centerline_success=True`
    # column; the load-bearing failure rule (path<100mm OR n_components>=20) is
    # recomputed here from results/measurements/<ds>_pred.csv, then joined to
    # the Mahalanobis column on case_id.
    out["md_vs_centerline_failure"] = {}
    for name, integ_df in [("avt", avt_df), ("amos", amos_df)]:
        meas = pd.read_csv(RESULTS / "measurements" / f"{name}_pred.csv")
        fail = ((meas["path_mm"] < 100) | (meas["n_components"] >= 20)).astype(int)
        meas = meas.assign(failure=fail.values)[["case_id", "failure"]]
        merged = integ_df[["case_id", "mahalanobis"]].merge(meas, on="case_id", how="inner")
        labels = merged["failure"].values
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        p, lo, hi = bootstrap_auroc(merged["mahalanobis"].values, labels, rng)
        out["md_vs_centerline_failure"][name] = {
            "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
            "n_fail": int(labels.sum()), "n_ok": int(len(labels) - labels.sum()),
            "report": fmt(p, lo, hi),
        }

    # ---- 6. MLD MAE per dataset ----
    out["mld_mae_mm"] = {}
    for name, df in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
        diffs = df["mld_diff_mm"].dropna().values
        p, lo, hi = bootstrap_mae(diffs, rng)
        out["mld_mae_mm"][name] = {
            "mae_mm": p, "ci95_lo": lo, "ci95_hi": hi,
            "n_paired": int(len(diffs)),
            "report": fmt(p, lo, hi),
        }

    # ---- 7. Baseline: kNN feature distance (Sun et al. 2022 ICML) ----
    if knn_df is not None:
        out["baseline_knn"] = {}
        # 7a. OOD-AUROC: kNN-5 on ID vs each OOD
        id_knn = knn_df[knn_df.dataset == "aortaseg24_test"]["knn_k5"].values
        for ood_name in ("avt", "amos", "totalseg"):
            ood_knn = knn_df[knn_df.dataset == ood_name]["knn_k5"].values
            if len(ood_knn) == 0:
                continue
            scores = np.concatenate([id_knn, ood_knn])
            labels = np.concatenate([np.zeros(len(id_knn)), np.ones(len(ood_knn))])
            p, lo, hi = bootstrap_auroc(scores, labels, rng)
            out["baseline_knn"][f"ood_auroc_aortaseg24_test_vs_{ood_name}"] = {
                "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                "n_id": int(len(id_knn)), "n_ood": int(len(ood_knn)),
                "report": fmt(p, lo, hi),
            }
        # 7b. Unc-AUROC: kNN-5 vs (Dice < tau) within each labelled dataset
        for ds, integ in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
            m = merge_baseline(knn_df, ds, integ, "knn_k5")
            if m is None or len(m) == 0:
                continue
            for tau in (0.5, 0.7):
                labels = (m["dice"].values < tau).astype(int)
                if labels.sum() == 0 or labels.sum() == len(labels):
                    continue
                p, lo, hi = bootstrap_auroc(m["knn_k5"].values, labels, rng)
                out["baseline_knn"][f"unc_auroc_{ds}_dice_lt_{tau}"] = {
                    "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                    "n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                    "report": fmt(p, lo, hi),
                }
        # 7c. Spearman kNN-5 vs Mahalanobis (redundancy check)
        for ds, integ in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
            sub_knn = knn_df[knn_df.dataset == ds][["case_id", "knn_k5"]]
            merged = integ[["case_id", "mahalanobis"]].merge(sub_knn, on="case_id")
            rho, lob, hib, pv, lof, hif = bootstrap_spearman(
                merged["knn_k5"].values, merged["mahalanobis"].values, rng)
            out["baseline_knn"][f"spearman_knn_vs_md_{ds}"] = {
                "rho": rho, "p": pv,
                "ci95_lo_bootstrap": lob, "ci95_hi_bootstrap": hib,
                "ci95_lo_fisher": lof, "ci95_hi_fisher": hif,
                "n": int(len(merged)),
                "report": fmt(rho, lob, hib),
            }

    # ---- 7c. Centerline as continuous score (path_mm; H1) ----
    # Convention: report Spearman(path_mm, dice) with the natural sign (higher
    # path = better centerline = higher Dice → positive correlation). For
    # Unc-AUROC where "higher score = more likely Dice<τ", use -path_mm.
    out["centerline_path_length"] = {}
    for name, integ_df in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
        meas = pd.read_csv(RESULTS / "measurements" / f"{name}_pred.csv")
        m = integ_df[["case_id", "dice"]].merge(
            meas[["case_id", "path_mm"]], on="case_id", how="inner")
        m = m.dropna(subset=["path_mm"])
        if len(m) < 5:
            continue
        rho, lob, hib, pv, lof, hif = bootstrap_spearman(
            m["path_mm"].values, m["dice"].values, rng)
        out["centerline_path_length"][f"spearman_{name}"] = {
            "rho": rho, "p": pv,
            "ci95_lo_bootstrap": lob, "ci95_hi_bootstrap": hib,
            "ci95_lo_fisher": lof, "ci95_hi_fisher": hif,
            "n": int(len(m)),
            "report": fmt(rho, lob, hib),
        }
        for tau in (0.5, 0.7):
            labels = (m["dice"].values < tau).astype(int)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            # -path_mm so that higher score => more likely Dice<τ
            p, lo, hi = bootstrap_auroc(-m["path_mm"].values, labels, rng)
            out["centerline_path_length"][f"unc_auroc_{name}_dice_lt_{tau}"] = {
                "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                "n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                "report": fmt(p, lo, hi),
            }

    # ---- 8. Baseline: Maximum softmax probability (Hendrycks & Gimpel 2017) ----
    if msp_df is not None:
        out["baseline_msp"] = {}
        # Higher MSP => higher confidence => NEGATE so that higher score => OOD/failure
        id_msp = msp_df[msp_df.dataset == "aortaseg24_test"]["msp"].values
        for ood_name in ("avt", "amos", "totalseg"):
            ood_msp = msp_df[msp_df.dataset == ood_name]["msp"].values
            if len(ood_msp) == 0:
                continue
            scores = np.concatenate([-id_msp, -ood_msp])
            labels = np.concatenate([np.zeros(len(id_msp)), np.ones(len(ood_msp))])
            p, lo, hi = bootstrap_auroc(scores, labels, rng)
            out["baseline_msp"][f"ood_auroc_aortaseg24_test_vs_{ood_name}"] = {
                "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                "n_id": int(len(id_msp)), "n_ood": int(len(ood_msp)),
                "report": fmt(p, lo, hi),
            }
        for ds, integ in [("aortaseg24_test", id_df), ("avt", avt_df), ("amos", amos_df)]:
            m = merge_baseline(msp_df, ds, integ, "msp")
            if m is None or len(m) == 0:
                continue
            for tau in (0.5, 0.7):
                labels = (m["dice"].values < tau).astype(int)
                if labels.sum() == 0 or labels.sum() == len(labels):
                    continue
                p, lo, hi = bootstrap_auroc(-m["msp"].values, labels, rng)
                out["baseline_msp"][f"unc_auroc_{ds}_dice_lt_{tau}"] = {
                    "auroc": p, "ci95_lo": lo, "ci95_hi": hi,
                    "n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum()),
                    "report": fmt(p, lo, hi),
                }

    # ---- write JSON ----
    out_path = RESULTS / "bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    # human-readable summary to stdout
    for group, block in out.items():
        if group == "meta":
            continue
        print(f"\n[{group}]")
        for key, val in block.items():
            if isinstance(val, dict) and "report" in val:
                extra = ""
                if "p" in val:
                    extra = f"   p={val['p']:.4f}"
                print(f"  {key:50s}  {val['report']}{extra}")


if __name__ == "__main__":
    main()
