"""
AVT pathology-subgroup analysis (C1 robustness contribution).

Per AVT (Radl 2025 / SEG.A. paper): the Rider subset (R1–R18) includes labeled
pathologies — five aortic dissection cases (R1, R2, R5, R7, R8) and one AAA
(R6). Our training data (AortaSeg24) is entirely type-B dissections, so we
expect AD cases on AVT to score closer to the training distribution than
healthy / non-AD cases.

Splits:
  AD-like (5)   : R1, R2, R5, R7, R8
  AAA  (1)      : R6
  Other Rider (12): R3, R4, R9-R18
  Dongyang (18) : D1-D18
  KiTS (20)     : K1-K20

Reports per subgroup:
  n, mean Dice, mean MD, median MD, centerline failure rate, MLD MAE
Plus Mann-Whitney U tests AD vs non-AD on each metric.

Output: results/mahalanobis/avt_subgroups.json
"""
import csv
import json
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).parent.parent
MAHA = ROOT / "results" / "mahalanobis"
MEAS = ROOT / "results" / "measurements"

AD_CASES   = {"R1", "R2", "R5", "R7", "R8"}
AAA_CASES  = {"R6"}


def subgroup(case_id: str) -> str:
    if case_id in AD_CASES:    return "AD"
    if case_id in AAA_CASES:   return "AAA"
    if case_id.startswith("R"): return "Rider-other"
    if case_id.startswith("D"): return "Dongyang"
    if case_id.startswith("K"): return "KiTS"
    return "unknown"


def read_csv_keyed(path, key="case_id"):
    return {r[key]: r for r in csv.DictReader(open(path))}


def fl(x):
    try:
        return float(x) if x not in ("", "nan", None) else float("nan")
    except ValueError:
        return float("nan")


def main():
    md   = read_csv_keyed(MAHA / "scores" / "avt" / "mahalanobis_summary.csv")
    dice = read_csv_keyed(MAHA / "metrics_avt.csv")
    pred = read_csv_keyed(MEAS / "avt_pred.csv")
    ref  = read_csv_keyed(MEAS / "avt_ref.csv")

    rows = []
    for cid in sorted(md):
        sg = subgroup(cid)
        m = fl(md[cid]["mahalanobis"])
        d = fl(dice[cid]["dice_binary"]) if cid in dice else float("nan")
        mld_p = fl(pred[cid]["mld_mm"]) if cid in pred else float("nan")
        mld_r = fl(ref[cid]["mld_mm"])  if cid in ref  else float("nan")
        path_p = fl(pred[cid]["path_mm"]) if cid in pred else 0.0
        ncc_p  = int(pred[cid]["n_components"]) if cid in pred else 0
        fail = (path_p < 100) or (ncc_p >= 20)
        rows.append({"case_id": cid, "subgroup": sg,
                     "md": m, "dice": d,
                     "mld_pred": mld_p, "mld_ref": mld_r,
                     "mld_diff": mld_p - mld_r,
                     "centerline_fail": fail})

    by_sg = {}
    for sg in ("AD", "AAA", "Rider-other", "Dongyang", "KiTS"):
        sub = [r for r in rows if r["subgroup"] == sg]
        if not sub:
            continue
        diffs = np.array([r["mld_diff"] for r in sub if not np.isnan(r["mld_diff"])])
        by_sg[sg] = {
            "n":              len(sub),
            "dice_mean":      round(float(np.nanmean([r["dice"] for r in sub])), 4),
            "dice_std":       round(float(np.nanstd ([r["dice"] for r in sub])), 4),
            "md_median":      round(float(np.median ([r["md"]   for r in sub])), 2),
            "md_mean":        round(float(np.mean   ([r["md"]   for r in sub])), 2),
            "centerline_fail_pct": round(100 * sum(r["centerline_fail"] for r in sub) / len(sub), 1),
            "mld_mae_mm":     round(float(np.abs(diffs).mean()), 3) if len(diffs) else None,
            "mld_bias_mm":    round(float(diffs.mean()),         3) if len(diffs) else None,
        }

    # AD vs all non-AD Mann-Whitney
    ad   = [r for r in rows if r["subgroup"] == "AD"]
    rest = [r for r in rows if r["subgroup"] != "AD"]
    tests = {}
    for key, label in [("dice", "Dice"), ("md", "Mahalanobis"),
                       ("mld_diff", "MLD diff (pred-ref)")]:
        a = np.array([r[key] for r in ad   if not np.isnan(r[key])])
        b = np.array([r[key] for r in rest if not np.isnan(r[key])])
        if len(a) < 1 or len(b) < 1:
            continue
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            tests[label] = {
                "AD_median":    round(float(np.median(a)), 4),
                "nonAD_median": round(float(np.median(b)), 4),
                "U":            float(stat),
                "p_two_sided":  round(float(p),            4),
                "AD_n":         int(len(a)),
                "nonAD_n":      int(len(b)),
            }
        except ValueError as e:
            tests[label] = {"error": str(e)}

    out = {"by_subgroup": by_sg, "AD_vs_nonAD_mannwhitney": tests}
    json_path = MAHA / "avt_subgroups.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n-> {json_path}")


if __name__ == "__main__":
    main()
