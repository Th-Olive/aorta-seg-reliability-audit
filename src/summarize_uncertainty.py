"""
Post-process MC Dropout uncertainty maps into per-case scalar summaries.

This script is intentionally separate from inference_mc.py so existing
long-running MC Dropout inference jobs do not need to be rerun.

Usage examples:
  python src/summarize_uncertainty.py \
    --pred_dir results/mc_dropout/predictions/avt \
    --label_dir data/avt_prepared/labels \
    --dataset avt \
    --metrics_csv results/mc_dropout/metrics_avt.csv \
    --output_csv results/mc_dropout/uncertainty_extended_avt.csv \
    --correlation_json results/mc_dropout/uncertainty_correlations_avt.json

  python src/summarize_uncertainty.py \
    --pred_dir results/mc_dropout/predictions/amos \
    --label_dir data/amos_prepared/labels \
    --dataset amos \
    --metrics_csv results/mc_dropout/metrics_amos.csv \
    --output_csv results/mc_dropout/uncertainty_extended_amos.csv
"""
import argparse
import csv
import json
import math
import pathlib
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, zoom
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


AMOS_AORTA_LABEL = 8
DECISION_COLUMNS = ["unc_auroc"]

UNCERTAINTY_COLUMNS = [
    "mean_uncertainty_fg",
    "p95_uncertainty_fg",
    "p99_uncertainty_fg",
    "top1pct_mean_uncertainty_fg",
    "max_uncertainty_fg",
    "nonzero_fraction_fg",
    "mean_uncertainty_boundary",
    "p95_uncertainty_boundary",
    "p99_uncertainty_boundary",
    "top1pct_mean_uncertainty_boundary",
    "mean_uncertainty_pred_or_label",
    "p95_uncertainty_pred_or_label",
    "p99_uncertainty_pred_or_label",
    "top1pct_mean_uncertainty_pred_or_label",
]


def load_float(path: pathlib.Path) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)


def load_int(path: pathlib.Path) -> np.ndarray:
    return load_float(path).astype(np.int32)


def resample_label_to_shape(label: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if label.shape == target_shape:
        return label
    scale = tuple(t / s for t, s in zip(target_shape, label.shape))
    return zoom(label, scale, order=0)


def reference_mask(dataset: str, label: np.ndarray) -> np.ndarray:
    if dataset == "amos":
        return label == AMOS_AORTA_LABEL
    return label > 0


def finite_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "top1pct_mean": float("nan"),
            "max": float("nan"),
            "nonzero_fraction": float("nan"),
        }
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "top1pct_mean": float("nan"),
            "max": float("nan"),
            "nonzero_fraction": float("nan"),
        }
    top_k = max(1, int(math.ceil(values.size * 0.01)))
    top_values = np.partition(values, values.size - top_k)[-top_k:]
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "top1pct_mean": float(np.mean(top_values)),
        "max": float(np.max(values)),
        "nonzero_fraction": float(np.count_nonzero(values) / values.size),
    }


def bounding_box(mask: np.ndarray, padding: int) -> tuple[slice, ...] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lower = np.maximum(coords.min(axis=0) - padding, 0)
    upper = np.minimum(coords.max(axis=0) + padding + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def case_unc_auroc(
    pred: np.ndarray,
    unc: np.ndarray,
    ref: np.ndarray,
    dataset: str,
    max_voxels: int = 500_000,
) -> float:
    """Voxel-level Unc-AUROC: uncertainty as binary classifier of prediction errors.

    Over the union of predicted and reference foreground, each voxel is labelled
    as an error or correct.  Uncertainty is the score.
    AUC = 0.5 means random; > 0.5 means uncertainty ranks error voxels higher.

    For multi-class datasets (aortaseg24): error = exact class mismatch — captures
    inter-class confusions that drive Dice loss, not just fg/bg boundary errors.
    For binary datasets (avt, amos): error = fg/bg mismatch (equivalent to exact).
    """
    pred_fg = pred > 0
    ref_fg = reference_mask(dataset, ref)
    union = pred_fg | ref_fg
    if not union.any():
        return float("nan")

    flat_unc = unc[union]
    if dataset == "aortaseg24":
        flat_err = (pred[union] != ref[union]).astype(np.int8)
    else:
        flat_err = (pred_fg[union] != ref_fg[union]).astype(np.int8)

    n_pos = int(flat_err.sum())
    n_neg = flat_err.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    if flat_unc.size > max_voxels:
        rng = np.random.default_rng(0)
        pos_idx = np.where(flat_err == 1)[0]
        neg_idx = np.where(flat_err == 0)[0]
        n_pos_keep = min(n_pos, max_voxels // 2)
        n_neg_keep = min(n_neg, max_voxels - n_pos_keep)
        idx = np.concatenate([
            rng.choice(pos_idx, n_pos_keep, replace=False),
            rng.choice(neg_idx, n_neg_keep, replace=False),
        ])
        flat_unc = flat_unc[idx]
        flat_err = flat_err[idx]

    if flat_unc.max() == flat_unc.min():
        return 0.5

    return float(roc_auc_score(flat_err, flat_unc))


def boundary_values(uncertainty: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    if not mask.any():
        return np.asarray([], dtype=uncertainty.dtype)
    bbox = bounding_box(mask, padding=iterations)
    if bbox is None:
        return np.asarray([], dtype=uncertainty.dtype)
    cropped_mask = mask[bbox]
    dilated = binary_dilation(cropped_mask, iterations=iterations)
    eroded = binary_erosion(cropped_mask, iterations=iterations)
    return uncertainty[bbox][dilated ^ eroded]


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def collect_prediction_files(pred_dir: pathlib.Path) -> list[pathlib.Path]:
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    return [path for path in pred_files if not path.name.endswith("_unc.nii.gz")]


def summarize_case(
    pred_file: pathlib.Path,
    label_dir: pathlib.Path,
    dataset: str,
    boundary_iterations: int,
) -> dict[str, str] | None:
    case_id = pred_file.name.replace(".nii.gz", "")
    unc_file = pred_file.with_name(f"{case_id}_unc.nii.gz")
    label_file = label_dir / f"{case_id}.nii.gz"

    if not unc_file.exists():
        print(f"  SKIP {case_id}: missing uncertainty map {unc_file}")
        return None
    if not label_file.exists():
        print(f"  SKIP {case_id}: missing label {label_file}")
        return None

    pred = load_int(pred_file)
    unc = load_float(unc_file)
    label = load_int(label_file)

    if unc.shape != pred.shape:
        scale = tuple(t / s for t, s in zip(pred.shape, unc.shape))
        unc = zoom(unc, scale, order=1).astype(np.float32)

    label = resample_label_to_shape(label, pred.shape)
    pred_fg = pred > 0
    ref_fg = reference_mask(dataset, label)
    union = pred_fg | ref_fg

    fg_summary = finite_summary(unc[pred_fg])
    boundary_summary = finite_summary(boundary_values(unc, pred_fg, boundary_iterations))
    union_summary = finite_summary(unc[union])
    auroc = case_unc_auroc(pred, unc, label, dataset)

    row = {
        "dataset": dataset,
        "case_id": case_id,
        "mean_uncertainty_fg": format_float(fg_summary["mean"]),
        "p95_uncertainty_fg": format_float(fg_summary["p95"]),
        "p99_uncertainty_fg": format_float(fg_summary["p99"]),
        "top1pct_mean_uncertainty_fg": format_float(fg_summary["top1pct_mean"]),
        "max_uncertainty_fg": format_float(fg_summary["max"]),
        "nonzero_fraction_fg": format_float(fg_summary["nonzero_fraction"]),
        "mean_uncertainty_boundary": format_float(boundary_summary["mean"]),
        "p95_uncertainty_boundary": format_float(boundary_summary["p95"]),
        "p99_uncertainty_boundary": format_float(boundary_summary["p99"]),
        "top1pct_mean_uncertainty_boundary": format_float(boundary_summary["top1pct_mean"]),
        "mean_uncertainty_pred_or_label": format_float(union_summary["mean"]),
        "p95_uncertainty_pred_or_label": format_float(union_summary["p95"]),
        "p99_uncertainty_pred_or_label": format_float(union_summary["p99"]),
        "top1pct_mean_uncertainty_pred_or_label": format_float(union_summary["top1pct_mean"]),
        "unc_auroc": format_float(auroc),
    }
    print(
        f"  {case_id}  mean_fg={row['mean_uncertainty_fg']}  "
        f"top1_fg={row['top1pct_mean_uncertainty_fg']}  auroc={row['unc_auroc']}"
    )
    return row


def write_rows(rows: list[dict[str, str]], output_csv: pathlib.Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "case_id"] + UNCERTAINTY_COLUMNS + DECISION_COLUMNS
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | None) -> float:
    if value in (None, "", "nan"):
        return float("nan")
    return float(value)


def mean_aortaseg24_dice(row: dict[str, str]) -> float:
    values = [
        parse_float(value)
        for key, value in row.items()
        if key.startswith("dice_cls")
    ]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def load_metrics(metrics_csv: pathlib.Path, dataset: str) -> dict[str, float]:
    scores = {}
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            if dataset == "aortaseg24":
                score = mean_aortaseg24_dice(row)
            else:
                score = parse_float(row.get("dice_binary"))
            if math.isfinite(score):
                scores[row["case_id"]] = score
    return scores


def compute_correlations(
    rows: Iterable[dict[str, str]],
    metrics_csv: pathlib.Path,
    dataset: str,
) -> dict[str, dict[str, float | int | None]]:
    dice_by_case = load_metrics(metrics_csv, dataset)
    output: dict[str, dict[str, float | int | None]] = {}

    for column in UNCERTAINTY_COLUMNS:
        dice_values = []
        uncertainty_values = []
        for row in rows:
            case_id = row["case_id"]
            if case_id not in dice_by_case:
                continue
            uncertainty = parse_float(row.get(column))
            if not math.isfinite(uncertainty):
                continue
            uncertainty_values.append(uncertainty)
            dice_values.append(dice_by_case[case_id])

        if len(dice_values) < 3 or len(set(uncertainty_values)) < 2:
            output[column] = {"n": len(dice_values), "spearman_rho": None, "p_value": None}
            continue

        result = spearmanr(uncertainty_values, dice_values)
        output[column] = {
            "n": len(dice_values),
            "spearman_rho": round(float(result.statistic), 6),
            "p_value": round(float(result.pvalue), 6),
        }
    return output


def compute_risk_coverage(
    rows: list[dict[str, str]],
    dice_by_case: dict[str, float],
    unc_column: str = "mean_uncertainty_fg",
    deferral_budgets: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> dict:
    """Case-level risk-coverage: selective Dice when deferring the N% most uncertain cases.

    Cases are ranked by uncertainty ascending (most confident first).  At each
    deferral budget, we retain the bottom (1 - budget) fraction and report the
    mean Dice of the retained subset and the gain over the full cohort.
    """
    pairs = []
    for row in rows:
        cid = row["case_id"]
        if cid not in dice_by_case:
            continue
        unc = parse_float(row.get(unc_column))
        if not math.isfinite(unc):
            continue
        pairs.append((unc, dice_by_case[cid]))

    if len(pairs) < 2:
        return {}

    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    all_dice = [d for _, d in pairs]

    result: dict = {
        "n_cases": n,
        "unc_column": unc_column,
        "mean_dice_all": round(float(np.mean(all_dice)), 5),
        "deferral": {},
    }
    for budget in deferral_budgets:
        n_retain = max(1, round(n * (1.0 - budget)))
        retained = [d for _, d in pairs[:n_retain]]
        key = f"defer_{int(budget * 100)}pct"
        result["deferral"][key] = {
            "n_retained": n_retain,
            "mean_dice_retained": round(float(np.mean(retained)), 5),
            "dice_gain": round(float(np.mean(retained)) - float(np.mean(all_dice)), 5),
        }
    return result


def compute_decision_summary(
    rows: list[dict[str, str]],
    dice_by_case: dict[str, float],
) -> dict:
    """Aggregate decision-aware metrics: mean Unc-AUROC and risk-coverage curves."""
    auroc_values = [
        parse_float(row.get("unc_auroc", "nan"))
        for row in rows
        if math.isfinite(parse_float(row.get("unc_auroc", "nan")))
    ]

    summary: dict = {}
    if auroc_values:
        summary["unc_auroc"] = {
            "n": len(auroc_values),
            "mean": round(float(np.mean(auroc_values)), 5),
            "std": round(float(np.std(auroc_values)), 5),
            "median": round(float(np.median(auroc_values)), 5),
        }

    for col in ("mean_uncertainty_fg", "nonzero_fraction_fg", "top1pct_mean_uncertainty_fg"):
        rc = compute_risk_coverage(rows, dice_by_case, unc_column=col)
        if rc:
            summary[f"risk_coverage_{col}"] = rc

    return summary


def print_correlations(correlations: dict[str, dict[str, float | int | None]]) -> None:
    print("\nSpearman correlation: uncertainty vs Dice")
    print("  Useful reliability direction is negative (higher uncertainty, lower Dice).")
    for column, values in correlations.items():
        rho = values["spearman_rho"]
        p_value = values["p_value"]
        print(f"  {column}: n={values['n']} rho={rho} p={p_value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MC Dropout uncertainty maps")
    parser.add_argument("--pred_dir", required=True, type=pathlib.Path)
    parser.add_argument("--label_dir", required=True, type=pathlib.Path)
    parser.add_argument("--dataset", required=True, choices=["aortaseg24", "avt", "amos"])
    parser.add_argument("--output_csv", required=True, type=pathlib.Path)
    parser.add_argument("--metrics_csv", type=pathlib.Path, default=None)
    parser.add_argument("--correlation_json", type=pathlib.Path, default=None)
    parser.add_argument("--decision_json", type=pathlib.Path, default=None,
                        help="Output JSON for Unc-AUROC and risk-coverage metrics")
    parser.add_argument("--boundary_iterations", type=int, default=1)
    args = parser.parse_args()

    print(f"summarize_uncertainty.py -- dataset={args.dataset}")
    print(f"  pred_dir : {args.pred_dir}")
    print(f"  label_dir: {args.label_dir}")

    rows = []
    for pred_file in collect_prediction_files(args.pred_dir):
        row = summarize_case(
            pred_file=pred_file,
            label_dir=args.label_dir,
            dataset=args.dataset,
            boundary_iterations=args.boundary_iterations,
        )
        if row is not None:
            rows.append(row)

    write_rows(rows, args.output_csv)
    print(f"\n  -> {args.output_csv} ({len(rows)} cases)")

    if args.metrics_csv and args.metrics_csv.exists():
        correlations = compute_correlations(rows, args.metrics_csv, args.dataset)
        print_correlations(correlations)
        if args.correlation_json:
            args.correlation_json.parent.mkdir(parents=True, exist_ok=True)
            args.correlation_json.write_text(json.dumps(correlations, indent=2))
            print(f"  -> {args.correlation_json}")

        dice_by_case = load_metrics(args.metrics_csv, args.dataset)
        decision = compute_decision_summary(rows, dice_by_case)
        if decision:
            auroc_info = decision.get("unc_auroc", {})
            print(f"\nUnc-AUROC (voxel-level error detection):  "
                  f"mean={auroc_info.get('mean')}  std={auroc_info.get('std')}  "
                  f"n={auroc_info.get('n')}")
            rc = decision.get("risk_coverage_mean_uncertainty_fg", {})
            if rc:
                print(f"Risk-coverage ({rc.get('unc_column')})  "
                      f"baseline Dice={rc.get('mean_dice_all')}")
                for key, val in rc.get("deferral", {}).items():
                    print(f"  {key}: retained={val['n_retained']}  "
                          f"Dice={val['mean_dice_retained']}  "
                          f"gain={val['dice_gain']:+.5f}")
        if args.decision_json and decision:
            args.decision_json.parent.mkdir(parents=True, exist_ok=True)
            args.decision_json.write_text(json.dumps(decision, indent=2))
            print(f"  -> {args.decision_json}")


if __name__ == "__main__":
    main()
