"""
Segmentation evaluation: Dice per class per dataset.

Label harmonization:
  aortaseg24 -- 23-class per-class Dice (predictions vs labelsTs/)
  avt        -- binary whole-aorta Dice (all predicted non-bg -> 1)
  amos       -- binary aorta Dice (all predicted non-bg -> 1 vs label 8 from AMOS)

Outputs:
  results/<run>/metrics_<dataset>.csv  -- per-case, per-class Dice
  results/<run>/summary_<dataset>.json -- mean ± std per class + optional uncertainty merge

Usage (from repo root):
  python src/evaluation.py \\
    --pred_dir  results/mc_dropout/predictions/aortaseg24_test \\
    --label_dir nnUNet_raw/Dataset001_AortaSeg24/labelsTs \\
    --dataset   aortaseg24 \\
    --output_dir results/mc_dropout/

  python src/evaluation.py \\
    --pred_dir  results/mc_dropout/predictions/avt \\
    --label_dir data/avt_prepared/labels \\
    --dataset   avt \\
    --uncertainty_csv results/mc_dropout/predictions/avt/uncertainty_summary.csv \\
    --output_dir results/mc_dropout/

  python src/evaluation.py \\
    --pred_dir  results/mc_dropout/predictions/amos \\
    --label_dir data/amos_prepared/labels \\
    --dataset   amos \\
    --uncertainty_csv results/mc_dropout/predictions/amos/uncertainty_summary.csv \\
    --output_dir results/mc_dropout/
"""
import argparse
import csv
import json
import pathlib
import os

import nibabel as nib
import numpy as np

_env = pathlib.Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# AortaSeg24 class names (indices 1-23)
AORTASEG24_CLASSES = {
    1: "Zone 0",
    2: "Innominate",
    3: "Zone 1",
    4: "Left Common Carotid",
    5: "Zone 2",
    6: "Left Subclavian",
    7: "Zone 3",
    8: "Zone 4",
    9: "Zone 5",
    10: "Zone 6",
    11: "Celiac Artery",
    12: "Zone 7",
    13: "SMA",
    14: "Zone 8",
    15: "Right Renal Artery",
    16: "Left Renal Artery",
    17: "Zone 9",
    18: "Zone 10 R",
    19: "Zone 10 L",
    20: "Right Internal Iliac",
    21: "Left Internal Iliac",
    22: "Zone 11 R",
    23: "Zone 11 L",
}

AMOS_AORTA_LABEL = 8  # 'arota' typo in dataset.json; always use index


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Binary Dice between two boolean or integer arrays."""
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    intersection = (pred_b & ref_b).sum()
    denom = pred_b.sum() + ref_b.sum()
    if denom == 0:
        return float("nan")   # both empty -- undefined; treated as NaN in aggregation
    return float(2 * intersection / denom)


def per_class_dice_aortaseg24(pred: np.ndarray, ref: np.ndarray) -> dict[int, float]:
    """23-class Dice for AortaSeg24 predictions."""
    scores = {}
    for cls_idx in AORTASEG24_CLASSES:
        scores[cls_idx] = dice(pred == cls_idx, ref == cls_idx)
    return scores


def binary_dice(pred: np.ndarray, ref_binary: np.ndarray) -> float:
    """Whole-structure binary Dice: collapse all non-bg to 1."""
    return dice(pred > 0, ref_binary > 0)


# ---------------------------------------------------------------------------
# Dataset-specific evaluation
# ---------------------------------------------------------------------------

def evaluate_aortaseg24(pred_dir: pathlib.Path, label_dir: pathlib.Path) \
        -> list[dict]:
    rows = []
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = [f for f in pred_files if not f.name.endswith("_unc.nii.gz")]

    for pred_file in pred_files:
        case_id = pred_file.name.replace(".nii.gz", "")
        # labels may be subject061.nii.gz (no _0000 suffix)
        lbl_file = label_dir / f"{case_id}.nii.gz"
        if not lbl_file.exists():
            print(f"  MISSING label: {lbl_file}")
            continue

        pred = nib.load(pred_file).get_fdata(dtype=np.float32).astype(np.int32)
        ref = nib.load(lbl_file).get_fdata(dtype=np.float32).astype(np.int32)

        if pred.shape != ref.shape:
            print(f"  SHAPE MISMATCH {case_id}: pred={pred.shape} ref={ref.shape} -- skipping")
            continue

        scores = per_class_dice_aortaseg24(pred, ref)
        row = {"case_id": case_id}
        row.update({f"dice_cls{idx}": (f"{v:.6f}" if not np.isnan(v) else "nan")
                    for idx, v in scores.items()})
        rows.append(row)
        mean_dice = np.nanmean(list(scores.values()))
        print(f"  {case_id}  mean_dice={mean_dice:.4f}")

    return rows


def evaluate_avt(pred_dir: pathlib.Path, label_dir: pathlib.Path) -> list[dict]:
    rows = []
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = [f for f in pred_files if not f.name.endswith("_unc.nii.gz")]

    for pred_file in pred_files:
        case_id = pred_file.name.replace(".nii.gz", "")
        lbl_file = label_dir / f"{case_id}.nii.gz"
        if not lbl_file.exists():
            print(f"  MISSING label: {lbl_file}")
            continue

        pred = nib.load(pred_file).get_fdata(dtype=np.float32)
        ref = nib.load(lbl_file).get_fdata(dtype=np.float32)

        # Resample ref to pred space if shapes differ
        if pred.shape != ref.shape:
            from scipy.ndimage import zoom
            scale = tuple(p / r for p, r in zip(pred.shape, ref.shape))
            ref = zoom(ref, scale, order=0)  # nearest for labels

        d = binary_dice(pred, ref)
        rows.append({"case_id": case_id, "dice_binary": f"{d:.6f}" if not np.isnan(d) else "nan"})
        print(f"  {case_id}  dice_binary={d:.4f}")

    return rows


def evaluate_amos(pred_dir: pathlib.Path, label_dir: pathlib.Path) -> list[dict]:
    rows = []
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = [f for f in pred_files if not f.name.endswith("_unc.nii.gz")]

    for pred_file in pred_files:
        case_id = pred_file.name.replace(".nii.gz", "")
        lbl_file = label_dir / f"{case_id}.nii.gz"
        if not lbl_file.exists():
            print(f"  MISSING label: {lbl_file}")
            continue

        pred = nib.load(pred_file).get_fdata(dtype=np.float32)
        ref_full = nib.load(lbl_file).get_fdata(dtype=np.float32)
        ref = (ref_full == AMOS_AORTA_LABEL).astype(np.float32)

        if pred.shape != ref.shape:
            from scipy.ndimage import zoom
            scale = tuple(p / r for p, r in zip(pred.shape, ref.shape))
            ref = zoom(ref, scale, order=0)

        d = binary_dice(pred, ref)
        rows.append({"case_id": case_id, "dice_binary": f"{d:.6f}" if not np.isnan(d) else "nan"})
        print(f"  {case_id}  dice_binary={d:.4f}")

    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_aortaseg24(rows: list[dict]) -> dict:
    summary = {}
    for cls_idx, cls_name in AORTASEG24_CLASSES.items():
        key = f"dice_cls{cls_idx}"
        vals = []
        for r in rows:
            v = r.get(key)
            if v is not None and v != "nan":
                vals.append(float(v))
        if vals:
            summary[cls_name] = {
                "class_idx": cls_idx,
                "n": len(vals),
                "mean": round(float(np.mean(vals)), 5),
                "std":  round(float(np.std(vals)),  5),
                "median": round(float(np.median(vals)), 5),
            }
    if summary:
        all_means = [v["mean"] for v in summary.values()]
        summary["_overall_mean_dice"] = round(float(np.mean(all_means)), 5)
    return summary


def summarize_binary(rows: list[dict]) -> dict:
    vals = [float(r["dice_binary"]) for r in rows if r.get("dice_binary") not in (None, "nan")]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 5),
        "std":  round(float(np.std(vals)),  5),
        "median": round(float(np.median(vals)), 5),
    }


# ---------------------------------------------------------------------------
# Uncertainty merge
# ---------------------------------------------------------------------------

def merge_uncertainty(rows: list[dict], unc_csv: pathlib.Path) -> list[dict]:
    unc_map = {}
    with open(unc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            unc_map[row["case_id"]] = row.get("mean_uncertainty_fg", "")

    for row in rows:
        row["mean_uncertainty_fg"] = unc_map.get(row["case_id"], "")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Segmentation evaluation")
    parser.add_argument("--pred_dir",  required=True, type=pathlib.Path)
    parser.add_argument("--label_dir", required=True, type=pathlib.Path)
    parser.add_argument("--dataset",   required=True,
                        choices=["aortaseg24", "avt", "amos"])
    parser.add_argument("--output_dir", required=True, type=pathlib.Path)
    parser.add_argument("--uncertainty_csv", type=pathlib.Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nevaluation.py -- dataset={args.dataset}")
    print(f"  pred_dir : {args.pred_dir}")
    print(f"  label_dir: {args.label_dir}\n")

    if args.dataset == "aortaseg24":
        rows = evaluate_aortaseg24(args.pred_dir, args.label_dir)
        fieldnames = ["case_id"] + [f"dice_cls{i}" for i in AORTASEG24_CLASSES]
        summary = summarize_aortaseg24(rows)
    elif args.dataset == "avt":
        rows = evaluate_avt(args.pred_dir, args.label_dir)
        fieldnames = ["case_id", "dice_binary"]
        summary = summarize_binary(rows)
    elif args.dataset == "amos":
        rows = evaluate_amos(args.pred_dir, args.label_dir)
        fieldnames = ["case_id", "dice_binary"]
        summary = summarize_binary(rows)

    if args.uncertainty_csv and args.uncertainty_csv.exists():
        rows = merge_uncertainty(rows, args.uncertainty_csv)
        if "mean_uncertainty_fg" not in fieldnames:
            fieldnames.append("mean_uncertainty_fg")

    # Write per-case CSV
    csv_path = args.output_dir / f"metrics_{args.dataset}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Write summary JSON
    json_path = args.output_dir / f"summary_{args.dataset}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  -> {csv_path}  ({len(rows)} cases)")
    print(f"  -> {json_path}")

    if args.dataset == "aortaseg24":
        od = summary.get("_overall_mean_dice")
        print(f"\n  Overall mean Dice: {od}")
    else:
        print(f"\n  Binary Dice: {summary}")


if __name__ == "__main__":
    main()
