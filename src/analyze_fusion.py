"""
Fusion rules and centerline-threshold sensitivity.  Pure CPU re-analysis of
existing per-case CSVs, no re-inference.

Fusion.  Demonstrates that the three reliability signals are
complementary by combining them into a single non-trained triage rule:

    flag a case  iff   MD  >  tau_MD          (feature distance: out-of-envelope)
                  OR   pairwise_dice < tau_pw (stochastic disagreement)
                  OR   centerline degenerate  (path < 100 mm or >=20 comps)

Thresholds are calibrated on the 40-case AortaSeg24 *in-distribution* test
split (NOT on the OOD cohorts), so the rule is non-circular.  We report two
calibrations:
  * "envelope": tau = max(ID) for MD, min(ID) for pairwise -> ID FPR = 0 by
    construction (flag only what falls outside everything seen in-distribution).
  * "p95/p05": tau = 95th pct ID MD, 5th pct ID pairwise -> ID FPR ~= 5%.
The centerline flag uses the paper's fixed convention (path < 100 mm OR
n_components >= 20 OR no extractable centerline).

For each cohort we report, per single signal and for the OR-rule:
  recall  = P(flagged | GT Dice < 0.5)      (catastrophic-case sensitivity)
  fpr     = P(flagged | GT Dice >= 0.7)      (false alarms on good cases)
and we list which catastrophic cases each signal catches/misses (this is where
R15 shows up: MD misses it, centerline catches it).

Threshold sensitivity.  Re-thresholds the existing path_mm
column at 50/100/150/200 mm and reports the AVT degenerate-flag set at each, to
show the 100 mm choice is not a fragile magic number.

Inputs (all already on disk):
  results/mc_passes/pairwise_summary.csv                 (ID:   pw, gt, md)
  results/mc_passes/avt_analysis/pairwise_summary.csv    (AVT:  pw, gt, md)
  results/mc_passes/amos_analysis/pairwise_summary.csv   (AMOS: pw, gt, md)
  results/measurements/{aortaseg24_test,avt,amos}_pred.csv  (path_mm, n_components, success)

Output:
  results/fusion_summary.json
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"

PAIRWISE = {
    "ID":   R / "mc_passes" / "pairwise_summary.csv",
    "AVT":  R / "mc_passes" / "avt_analysis" / "pairwise_summary.csv",
    "AMOS": R / "mc_passes" / "amos_analysis" / "pairwise_summary.csv",
}
MEAS = {
    "ID":   R / "measurements" / "aortaseg24_test_pred.csv",
    "AVT":  R / "measurements" / "avt_pred.csv",
    "AMOS": R / "measurements" / "amos_pred.csv",
}


def load_cohort(name: str) -> pd.DataFrame:
    pw = pd.read_csv(PAIRWISE[name])
    me = pd.read_csv(MEAS[name])[["case_id", "success", "path_mm", "n_components"]]
    df = pw.merge(me, on="case_id", how="left")
    df = df.rename(columns={"ground_truth_dice": "gt", "pairwise_dice_mean": "pw",
                            "mahalanobis": "md"})
    # coerce success to bool (csv may give str/NaN); NaN -> False (no centerline)
    if df["success"].dtype != bool:
        df["success"] = df["success"].map(
            {"True": True, "False": False, True: True, False: False})
    df["success"] = df["success"].fillna(False)
    return df


def path_degenerate(df: pd.DataFrame, thr_mm: float = 100.0) -> pd.Series:
    """Paper convention: degenerate iff path<thr OR >=20 comps OR no centerline."""
    nc = df["n_components"].fillna(999)
    pm = df["path_mm"].fillna(0.0)
    return (~df["success"]) | (pm < thr_mm) | (nc >= 20)


def rate(flag: pd.Series, mask: pd.Series) -> str:
    sub = flag[mask]
    if len(sub) == 0:
        return "n/a (0 cases)"
    return f"{int(sub.sum())}/{len(sub)} = {sub.mean():.2f}"


def evaluate(name, df, tau_md, tau_pw, label):
    md_flag = df["md"] > tau_md
    pw_flag = df["pw"] < tau_pw
    pa_flag = path_degenerate(df)
    or_flag = md_flag | pw_flag | pa_flag
    and_flag = md_flag & pw_flag & pa_flag          # intersection (all three fire)
    or_cheap = md_flag | pa_flag                     # deployable union (MD + centerline)
    and_cheap = md_flag & pa_flag                    # deployable intersection

    cata = df["gt"] < 0.5          # catastrophic
    good = df["gt"] >= 0.7         # clean

    print(f"\n  [{name}]  ({label} thresholds: tau_MD={tau_md:.1f}, "
          f"tau_pw={tau_pw:.3f}, path<100mm)   n={len(df)}, "
          f"catastrophic(Dice<0.5)={int(cata.sum())}, good(Dice>=0.7)={int(good.sum())}")
    res = {}
    for sig, fl in [("MD", md_flag), ("pairwise", pw_flag),
                    ("centerline", pa_flag), ("OR-rule", or_flag),
                    ("AND-rule", and_flag), ("OR-cheap(MD|path)", or_cheap),
                    ("AND-cheap(MD&path)", and_cheap)]:
        rec = fl[cata]
        fp = fl[good]
        rec_v = float(rec.mean()) if len(rec) else None
        fpr_v = float(fp.mean()) if len(fp) else None
        print(f"    {sig:11s}  recall@Dice<0.5 = {rate(fl, cata):16s}   "
              f"FPR@Dice>=0.7 = {rate(fl, good)}")
        res[sig] = {"recall_cata": rec_v, "fpr_good": fpr_v,
                    "n_flagged": int(fl.sum())}

    # which catastrophic cases each signal catches (surfaces R15 on AVT)
    if cata.sum() > 0:
        print(f"    catastrophic-case breakdown:")
        for _, row in df[cata].sort_values("gt").iterrows():
            caught = []
            if row["md"] > tau_md:
                caught.append("MD")
            if row["pw"] < tau_pw:
                caught.append("pw")
            if path_degenerate(df).loc[row.name]:
                caught.append("path")
            print(f"      {row['case_id']:12s} Dice={row['gt']:.3f}  "
                  f"MD={row['md']:7.1f}  pw={row['pw']:.3f}  "
                  f"path={row['path_mm'] if pd.notna(row['path_mm']) else float('nan'):7.1f}  "
                  f"-> caught by: {','.join(caught) if caught else 'NONE'}")
    return res


def auroc(scores: pd.Series, pos: pd.Series):
    """Unc-AUROC via the Mann-Whitney U estimator (dependency-free).

    `scores` oriented so higher => more likely to be a failure; `pos` is the
    boolean positive (catastrophic) mask. Ties handled via average ranks.
    """
    s = pd.Series(np.asarray(scores, dtype=float))
    ok = s.notna()
    s, p = s[ok], pos[ok]
    n_pos, n_neg = int(p.sum()), int((~p).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = s.rank()  # average ranks
    return float((r[p].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_fusion_auroc(cohorts, idf):
    """Standardized (ID-calibrated) score-level fusion vs. best single signal.

    Each signal is z-scored on the ID AortaSeg24 test split and oriented so
    higher => more failure-likely (MD as-is; pairwise and path sign-flipped).
    The unsupervised fused score is the equal-weight sum. We compare the fused
    Unc-AUROC (catastrophic Dice<0.5 vs. rest) to the best single signal so the
    paper can state whether *any* combiner beats the strongest lone signal.
    Also reports a 'cheap' fusion (MD + centerline only, no ~100 min pairwise).
    """
    mu_md, sd_md = idf["md"].mean(), idf["md"].std()
    mu_pw, sd_pw = idf["pw"].mean(), idf["pw"].std()
    id_path = idf["path_mm"].fillna(0.0)
    mu_pa, sd_pa = id_path.mean(), id_path.std()

    print("\n" + "=" * 78)
    print("Score-level fusion — ID-standardized, equal-weight sum vs best single")
    print("(Unc-AUROC for catastrophic Dice<0.5; higher score = more failure-likely)")
    print("=" * 78)

    res = {}
    for name in ("ID", "AVT", "AMOS"):
        df = cohorts[name]
        pos = df["gt"] < 0.5
        z_md = (df["md"] - mu_md) / sd_md
        z_pw = (df["pw"] - mu_pw) / sd_pw
        z_pa = (df["path_mm"].fillna(0.0) - mu_pa) / sd_pa
        fused_all = z_md - z_pw - z_pa          # equal-weight, failure-oriented
        fused_cheap = z_md - z_pa               # MD + centerline only

        a = {
            "MD":            auroc(df["md"], pos),
            "pairwise":      auroc(-df["pw"], pos),
            "centerline":    auroc(-df["path_mm"].fillna(0.0), pos),
            "fused_all":     auroc(fused_all, pos),
            "fused_cheap":   auroc(fused_cheap, pos),
        }
        res[name] = a
        if a["MD"] is None:
            print(f"\n  [{name}]  no catastrophic case -> AUROC undefined (skipped)")
            continue
        singles = {k: a[k] for k in ("MD", "pairwise", "centerline")}
        best_single = max(singles, key=lambda k: singles[k])
        print(f"\n  [{name}]  n={len(df)}, catastrophic={int(pos.sum())}")
        for k in ("MD", "pairwise", "centerline", "fused_all", "fused_cheap"):
            tag = ""
            if k == best_single:
                tag = "  <- best single"
            if k == "fused_all":
                tag = (f"  (best single = {best_single} {singles[best_single]:.3f}; "
                       f"fused {'BEATS' if a[k] > singles[best_single] else 'does NOT beat'} it)")
            print(f"    {k:12s} Unc-AUROC = {a[k]:.3f}{tag}")
    return res


def main():
    cohorts = {k: load_cohort(k) for k in PAIRWISE}
    idf = cohorts["ID"]

    tau_md_env = float(np.nanmax(idf["md"]))
    tau_pw_env = float(np.nanmin(idf["pw"]))
    tau_md_p95 = float(np.nanpercentile(idf["md"], 95))
    tau_pw_p05 = float(np.nanpercentile(idf["pw"], 5))

    print("=" * 78)
    print("OR-rule fusion (thresholds calibrated on ID AortaSeg24 test, n=40)")
    print("=" * 78)
    print(f"  ID MD:       min={idf['md'].min():.1f}  median={idf['md'].median():.1f}  "
          f"p95={tau_md_p95:.1f}  max={tau_md_env:.1f}")
    print(f"  ID pairwise: min={tau_pw_env:.3f}  median={idf['pw'].median():.3f}  "
          f"p05={tau_pw_p05:.3f}  max={idf['pw'].max():.3f}")

    out = {"thresholds": {
        "envelope": {"tau_md": tau_md_env, "tau_pw": tau_pw_env},
        "p95_p05":  {"tau_md": tau_md_p95, "tau_pw": tau_pw_p05},
        "path_mm": 100.0}, "envelope": {}, "p95_p05": {}}

    print("\n--- ENVELOPE calibration (tau = max/min ID; ID FPR = 0 by construction) ---")
    for name in ("ID", "AVT", "AMOS"):
        out["envelope"][name] = evaluate(name, cohorts[name],
                                         tau_md_env, tau_pw_env, "envelope")

    print("\n--- p95/p05 calibration (ID FPR ~= 5%) ---")
    for name in ("ID", "AVT", "AMOS"):
        out["p95_p05"][name] = evaluate(name, cohorts[name],
                                        tau_md_p95, tau_pw_p05, "p95/p05")

    # centerline threshold sensitivity
    print("\n" + "=" * 78)
    print("centerline path-length threshold sensitivity")
    print("=" * 78)
    out["threshold_sensitivity"] = {}
    for name in ("ID", "AVT", "AMOS"):
        df = cohorts[name]
        out["threshold_sensitivity"][name] = {}
        print(f"\n  [{name}]  n={len(df)}")
        sets = {}
        for thr in (50, 100, 150, 200):
            fl = path_degenerate(df, thr_mm=thr)
            flagged = sorted(df["case_id"][fl].tolist())
            sets[thr] = set(flagged)
            out["threshold_sensitivity"][name][thr] = {
                "n_flagged": int(fl.sum()),
                "rate": float(fl.mean()),
                "cases": flagged if len(flagged) <= 12 else f"{len(flagged)} cases",
            }
            print(f"    path<{thr:3d}mm: {int(fl.sum()):3d}/{len(df)} flagged "
                  f"({fl.mean():.3f})"
                  + (f"   cases={flagged}" if len(flagged) <= 12 else ""))
        # stability: does the flagged set change across thresholds?
        base = sets[100]
        identical = all(sets[t] == base for t in (50, 150, 200))
        print(f"    flag-set identical to 100mm across 50/150/200mm? {identical}")
        out["threshold_sensitivity"][name]["identical_50_200_vs_100"] = identical

    # Score-level fusion AUROC comparison (AND-rule numbers are already in the
    # envelope/p95 blocks above via the extended signal list).
    out["score_fusion_auroc"] = score_fusion_auroc(cohorts, idf)

    (R / "fusion_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n--> wrote {R / 'fusion_summary.json'}")


if __name__ == "__main__":
    main()
