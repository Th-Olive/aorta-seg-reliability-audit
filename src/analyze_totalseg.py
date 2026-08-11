"""
TotalSegmentator inter-tool agreement analysis (NOT ground-truth Dice).

For each TotalSeg case we have:
  - Our 23-class nnUNet prediction (binary aorta-trunk + binary right/left iliac trees)
  - TotalSegmentator's labels: aorta.nii.gz, iliac_artery_right.nii.gz,
    iliac_artery_left.nii.gz

We compute *volumetric agreement* (Bland-Altman) per structure, NOT Dice — both
tools have documented imperfections (Wasserthal 2023 reports missing-parts on
iliacs; our model is trained on dissection CTA, not routine CT). Dice between
two imperfect tools is meaningless as a quality metric.

AortaSeg24 → structure mapping (SVS/STS zones, AortaSeg24 Table 2):
  Aortic trunk     : labels {1, 3, 5, 7, 8, 9, 10, 12, 14, 17}   (Zones 0-9)
  Right iliac tree : labels {18, 20, 22}     (common R + internal R + external R)
  Left  iliac tree : labels {19, 21, 23}     (common L + internal L + external L)

Branches NOT included in trunk (innominate=2, LCC=4, LSA=6, celiac=11, SMA=13,
renal=15,16) are excluded because TotalSegmentator's `aorta.nii.gz` excludes them.

Outputs:
  results/totalseg_agreement/agreement.csv     per-case volumes (ml) and diffs
  results/totalseg_agreement/summary.json      Bland-Altman bias + LoA per structure
"""
import argparse
import csv
import json
import pathlib

import nibabel as nib
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
TS_ROOT = ROOT / "data" / "TotalSegmentator"
DEFAULT_PRED_DIR = ROOT / "results" / "mahalanobis" / "scores" / "totalseg"
DEFAULT_OUT_DIR = ROOT / "results" / "totalseg_agreement"

AORTIC_TRUNK_CLASSES = {1, 3, 5, 7, 8, 9, 10, 12, 14, 17}
RIGHT_ILIAC_CLASSES  = {18, 20, 22}
LEFT_ILIAC_CLASSES   = {19, 21, 23}


def voxel_volume_ml(nib_img) -> float:
    """Volume of one voxel in mL (1mL = 1000 mm³)."""
    sx, sy, sz = (float(z) for z in nib_img.header.get_zooms()[:3])
    return sx * sy * sz / 1000.0


def structure_volume_ml(mask_arr: np.ndarray, vox_ml: float) -> float:
    return float(mask_arr.sum()) * vox_ml


def per_case_agreement(case_id: str, pred_path: pathlib.Path,
                       ts_case_dir: pathlib.Path) -> dict | None:
    if not pred_path.exists():
        return None
    pred_nib = nib.load(pred_path)
    pred = pred_nib.get_fdata(dtype=np.float32).astype(np.int32)
    pvol = voxel_volume_ml(pred_nib)

    our_trunk = np.isin(pred, list(AORTIC_TRUNK_CLASSES))
    our_r_iliac = np.isin(pred, list(RIGHT_ILIAC_CLASSES))
    our_l_iliac = np.isin(pred, list(LEFT_ILIAC_CLASSES))

    ts_seg_dir = ts_case_dir / "segmentations"
    ts_paths = {
        "aorta":   ts_seg_dir / "aorta.nii.gz",
        "r_iliac": ts_seg_dir / "iliac_artery_right.nii.gz",
        "l_iliac": ts_seg_dir / "iliac_artery_left.nii.gz",
    }
    row = {"case_id": case_id}
    for name, our_mask in [("aorta", our_trunk),
                           ("r_iliac", our_r_iliac),
                           ("l_iliac", our_l_iliac)]:
        ts_path = ts_paths[name]
        if not ts_path.exists():
            row[f"our_{name}_ml"] = ""
            row[f"ts_{name}_ml"]  = ""
            row[f"diff_{name}_ml"] = ""
            continue
        ts_nib = nib.load(ts_path)
        ts_arr = (ts_nib.get_fdata(dtype=np.float32) > 0)
        our_v = structure_volume_ml(our_mask, pvol)
        ts_v  = structure_volume_ml(ts_arr, voxel_volume_ml(ts_nib))
        row[f"our_{name}_ml"]  = f"{our_v:.3f}"
        row[f"ts_{name}_ml"]   = f"{ts_v:.3f}"
        row[f"diff_{name}_ml"] = f"{our_v - ts_v:.3f}"
    return row


def bland_altman(diffs: np.ndarray, means: np.ndarray) -> dict:
    diffs = np.asarray(diffs, dtype=float)
    means = np.asarray(means, dtype=float)
    bias = float(diffs.mean())
    sd   = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    return {
        "n":         int(len(diffs)),
        "bias_ml":   round(bias, 3),
        "sd_ml":     round(sd, 3),
        "loa_lower": round(bias - 1.96 * sd, 3),
        "loa_upper": round(bias + 1.96 * sd, 3),
        "mean_of_means_ml": round(float(means.mean()), 3) if len(means) else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",   type=pathlib.Path, default=DEFAULT_PRED_DIR,
                   help="Directory of our predictions named {case_id}.nii.gz")
    p.add_argument("--output_dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(args.pred_dir.glob("*.nii.gz"))
    pred_files = [f for f in pred_files if not f.name.endswith("_unc.nii.gz")]
    print(f"Found {len(pred_files)} predictions in {args.pred_dir}")

    rows = []
    for f in pred_files:
        cid = f.name.replace(".nii.gz", "")
        ts_dir = TS_ROOT / cid
        if not ts_dir.exists():
            continue
        r = per_case_agreement(cid, f, ts_dir)
        if r:
            rows.append(r)

    out_csv = args.output_dir / "agreement.csv"
    fieldnames = ["case_id",
                  "our_aorta_ml",   "ts_aorta_ml",   "diff_aorta_ml",
                  "our_r_iliac_ml", "ts_r_iliac_ml", "diff_r_iliac_ml",
                  "our_l_iliac_ml", "ts_l_iliac_ml", "diff_l_iliac_ml"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Per-case agreement -> {out_csv}  ({len(rows)} cases)")

    summary = {}
    for struct in ("aorta", "r_iliac", "l_iliac"):
        diffs, means = [], []
        for r in rows:
            d, our, ts = r.get(f"diff_{struct}_ml"), r.get(f"our_{struct}_ml"), r.get(f"ts_{struct}_ml")
            if d in (None, "") or our in (None, "") or ts in (None, ""):
                continue
            diffs.append(float(d)); means.append((float(our) + float(ts)) / 2)
        if diffs:
            summary[struct] = bland_altman(np.array(diffs), np.array(means))
    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Bland-Altman summary -> {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
