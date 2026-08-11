"""
End-to-end integrated reliability analysis.

Joins, per dataset and per case:
  - Mahalanobis distance (results/mahalanobis/scores/<ds>/mahalanobis_summary.csv)
  - Dice                 (results/mahalanobis/metrics_<ds>.csv)
  - Centerline meas      (results/measurements/<ds>_pred.csv  [optional])

Then computes:
  1. MLD MAE on AVT      (pred vs ref measurements)
  2. Centerline failure  (per-dataset rate; AUROC of MD vs failure)
  3. Combined reliability table per dataset
  4. Reliability summary JSON

Outputs:
  results/integrated_per_case_<ds>.csv
  results/integrated_summary.json
"""
import csv
import json
import pathlib
from typing import Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).parent.parent
MAHA = ROOT / "results" / "mahalanobis"
MEAS = ROOT / "results" / "measurements"
OUT_DIR = ROOT / "results"


def read_md(path: pathlib.Path) -> dict[str, float]:
    return ({r["case_id"]: float(r["mahalanobis"])
             for r in csv.DictReader(open(path))} if path.exists() else {})


def read_dice_aortaseg24(path: pathlib.Path) -> dict[str, float]:
    """Mean over 23-class Dice columns."""
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        vals = [float(v) for k, v in r.items()
                if k.startswith("dice_cls") and v not in ("", "nan")]
        if vals:
            out[r["case_id"]] = float(np.mean(vals))
    return out


def read_dice_binary(path: pathlib.Path) -> dict[str, float]:
    return ({r["case_id"]: float(r["dice_binary"])
             for r in csv.DictReader(open(path)) if r.get("dice_binary") not in ("", "nan", None)}
            if path.exists() else {})


def read_measurements(path: pathlib.Path) -> dict[str, dict]:
    """Return case_id -> {mld_mm, mld_min_mm, tortuosity, success, ...}"""
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        def f(x):
            return float(x) if x not in ("", "nan", None) else float("nan")
        out[r["case_id"]] = {
            "mld_mm":     f(r.get("mld_mm")),
            "mld_min_mm": f(r.get("mld_min_mm")),
            "mld_med_mm": f(r.get("mld_med_mm")),
            "tortuosity": f(r.get("tortuosity")),
            "path_mm":    f(r.get("path_mm")),
            "n_skel_pts": int(r.get("n_skel_pts", 0) or 0),
            "n_components": int(r.get("n_components", 0) or 0),
            "success":    r.get("success") == "True",
            "failure_reason": r.get("failure_reason", "") or "",
        }
    return out


def analyze_dataset(name: str, md: dict, dice: dict,
                    meas: dict, meas_ref: dict | None) -> dict:
    ids = sorted(set(md) & set(dice))
    if not ids:
        return {"n": 0, "skipped": True}

    md_arr   = np.array([md[i]   for i in ids])
    dice_arr = np.array([dice[i] for i in ids])

    ds: dict = {
        "n": len(ids),
        "dice_mean":   round(float(dice_arr.mean()), 4),
        "dice_std":    round(float(dice_arr.std()),  4),
        "md_median":   round(float(np.median(md_arr)), 2),
        "md_max":      round(float(md_arr.max()),   2),
    }
    rho, p = spearmanr(md_arr, dice_arr)
    ds["spearman_md_dice"] = round(float(rho), 4)
    ds["spearman_p"]       = round(float(p),   4)
    for thr in (0.5, 0.7):
        y = (dice_arr < thr).astype(int)
        if y.sum() > 0 and y.sum() < len(y):
            ds[f"unc_auroc_dice_lt_{thr}"] = round(float(roc_auc_score(y, md_arr)), 4)

    # Centerline failure analysis
    # We treat a centerline as "degenerate" if kimimaro failed OR the resulting
    # path is unphysiologically short OR the mask broke into many CCs.
    # Threshold path_mm < 100 (aortic tree should be ~500-1000mm) and n_cc >= 20
    # (normal range observed is 1-15 CCs on real cases).
    def is_degenerate(m: dict) -> bool:
        if not m["success"]:
            return True
        if m["path_mm"] < 100:
            return True
        if m["n_components"] >= 20:
            return True
        return False

    if meas:
        fails = {cid for cid, m in meas.items() if is_degenerate(m)}
        n_fail = sum(1 for i in ids if i in fails)
        ds["centerline_failures"] = n_fail
        ds["centerline_failure_pct"] = round(100 * n_fail / len(ids), 1)
        if 0 < n_fail < len(ids):
            y = np.array([1 if i in fails else 0 for i in ids])
            ds["md_vs_failure_auroc"] = round(float(roc_auc_score(y, md_arr)), 4)
            # Spearman between MD and degeneracy flag (point-biserial)
            rhof, pf = spearmanr(md_arr, y)
            ds["md_vs_failure_spearman"] = round(float(rhof), 4)
            ds["md_vs_failure_p"] = round(float(pf), 4)

    # MLD MAE on AVT (need both pred and ref measurements)
    if meas and meas_ref:
        diffs = []
        for i in ids:
            mp = meas.get(i); mr = meas_ref.get(i)
            if not (mp and mr and mp["success"] and mr["success"]):
                continue
            if np.isnan(mp["mld_mm"]) or np.isnan(mr["mld_mm"]):
                continue
            diffs.append(mp["mld_mm"] - mr["mld_mm"])
        if diffs:
            d = np.array(diffs)
            ds["mld_mae_mm"]  = round(float(np.abs(d).mean()), 3)
            ds["mld_bias_mm"] = round(float(d.mean()),         3)
            ds["mld_sd_mm"]   = round(float(d.std()),          3)
            ds["mld_n_paired"] = len(diffs)

    return ds


def write_per_case(name: str, md: dict, dice: dict, meas: dict,
                   meas_ref: dict | None, out_path: pathlib.Path):
    ids = sorted(set(md) | set(dice))
    fieldnames = ["dataset", "case_id", "mahalanobis", "dice",
                  "mld_mm", "tortuosity", "centerline_success", "failure_reason",
                  "ref_mld_mm", "mld_diff_mm"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in ids:
            row = {"dataset": name, "case_id": i,
                   "mahalanobis": f"{md[i]:.4f}" if i in md else "",
                   "dice":        f"{dice[i]:.6f}" if i in dice else "",
                   "mld_mm": "", "tortuosity": "",
                   "centerline_success": "", "failure_reason": "",
                   "ref_mld_mm": "", "mld_diff_mm": ""}
            if i in meas:
                m = meas[i]
                row["mld_mm"]     = f"{m['mld_mm']:.4f}"    if not np.isnan(m["mld_mm"])    else ""
                row["tortuosity"] = f"{m['tortuosity']:.4f}" if not np.isnan(m["tortuosity"]) else ""
                row["centerline_success"] = "True" if m["success"] else "False"
                row["failure_reason"]     = m["failure_reason"]
            if meas_ref and i in meas_ref:
                mr = meas_ref[i]
                row["ref_mld_mm"] = f"{mr['mld_mm']:.4f}" if not np.isnan(mr["mld_mm"]) else ""
                if i in meas and not np.isnan(meas[i]["mld_mm"]) and not np.isnan(mr["mld_mm"]):
                    row["mld_diff_mm"] = f"{meas[i]['mld_mm'] - mr['mld_mm']:.4f}"
            w.writerow(row)


def main():
    datasets: dict[str, dict] = {}

    # AortaSeg24 test
    md = read_md(MAHA / "scores" / "aortaseg24_test" / "mahalanobis_summary.csv")
    dice = read_dice_aortaseg24(MAHA / "metrics_aortaseg24.csv")
    meas = read_measurements(MEAS / "aortaseg24_test_pred.csv")
    meas_ref = read_measurements(MEAS / "aortaseg24_test_ref.csv")
    datasets["aortaseg24_test"] = analyze_dataset(
        "aortaseg24_test", md, dice, meas, meas_ref if meas_ref else None)
    write_per_case("aortaseg24_test", md, dice, meas, meas_ref,
                   OUT_DIR / "integrated_per_case_aortaseg24_test.csv")

    # AVT — with pred + ref measurements for MLD MAE
    md = read_md(MAHA / "scores" / "avt" / "mahalanobis_summary.csv")
    dice = read_dice_binary(MAHA / "metrics_avt.csv")
    meas      = read_measurements(MEAS / "avt_pred.csv")
    meas_ref  = read_measurements(MEAS / "avt_ref.csv")
    datasets["avt"] = analyze_dataset("avt", md, dice, meas, meas_ref)
    write_per_case("avt", md, dice, meas, meas_ref,
                   OUT_DIR / "integrated_per_case_avt.csv")

    # AMOS
    md = read_md(MAHA / "scores" / "amos" / "mahalanobis_summary.csv")
    dice = read_dice_binary(MAHA / "metrics_amos.csv")
    meas = read_measurements(MEAS / "amos_pred.csv")
    meas_ref = read_measurements(MEAS / "amos_ref.csv")
    datasets["amos"] = analyze_dataset("amos", md, dice, meas,
                                       meas_ref if meas_ref else None)
    write_per_case("amos", md, dice, meas, meas_ref,
                   OUT_DIR / "integrated_per_case_amos.csv")

    # OOD-AUROC (re-compute on integrated MD pool, all cases — incl. no-Dice ones)
    ood_auroc = {}
    id_md = list(read_md(MAHA / "scores" / "aortaseg24_test" / "mahalanobis_summary.csv").values())
    for ood_name in ("avt", "amos", "totalseg"):
        ood_md = list(read_md(MAHA / "scores" / ood_name / "mahalanobis_summary.csv").values())
        if id_md and ood_md:
            y = np.concatenate([np.zeros(len(id_md)), np.ones(len(ood_md))])
            s = np.concatenate([id_md, ood_md])
            ood_auroc[f"id=aortaseg24_test_vs_ood={ood_name}"] = {
                "auroc": round(float(roc_auc_score(y, s)), 4),
                "n_id": len(id_md), "n_ood": len(ood_md),
            }

    out_json = OUT_DIR / "integrated_summary.json"
    out_json.write_text(json.dumps(
        {"ood_auroc": ood_auroc, "per_dataset": datasets},
        indent=2))
    print(f"\nWrote {out_json}")
    print(json.dumps({"ood_auroc": ood_auroc, "per_dataset": datasets},
                     indent=2))


if __name__ == "__main__":
    main()
