"""
AVT Mahalanobis distance vs centerline path length.

Output (in results/mahalanobis/figures/):
  fig_md_vs_failure.png   -- MD vs path length, coloured by Dice; degenerate cases marked
"""
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
MAHA = ROOT / "results" / "mahalanobis"
MEAS = ROOT / "results" / "measurements"
FIG_DIR = MAHA / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_measurements(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["case_id"]] = r
    return out


def get_float(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "nan", None) else float("nan")


# ---------------------------------------------------------------------------
# Figure: AVT MD vs path length, color by Dice, mark failed cases
# ---------------------------------------------------------------------------

def fig_md_vs_failure():
    # MD
    md_path = MAHA / "scores" / "avt" / "mahalanobis_summary.csv"
    md = {r["case_id"]: float(r["mahalanobis"])
          for r in csv.DictReader(open(md_path))}
    # Dice
    dice = {r["case_id"]: float(r["dice_binary"])
            for r in csv.DictReader(open(MAHA / "metrics_avt.csv"))}
    # Measurements
    meas = read_measurements(MEAS / "avt_pred.csv")

    ids = sorted(set(md) & set(dice) & set(meas))
    md_arr   = np.array([md[i]   for i in ids])
    dice_arr = np.array([dice[i] for i in ids])
    path_arr = np.array([get_float(meas[i], "path_mm") for i in ids])
    n_cc_arr = np.array([int(meas[i]["n_components"]) for i in ids])

    failed = (path_arr < 100) | (n_cc_arr >= 20)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(md_arr, path_arr, c=dice_arr, cmap="RdYlGn", s=70,
                    alpha=0.8, edgecolor="black", linewidth=0.5,
                    vmin=0, vmax=1)
    if failed.any():
        ax.scatter(md_arr[failed], path_arr[failed], s=180, marker="x",
                   color="black", linewidths=2, label=f"degenerate centerline (n={failed.sum()})")
    ax.set_xscale("log")
    ax.set_xlabel("Mahalanobis distance² (log)")
    ax.set_ylabel("Centerline path length (mm)")
    ax.set_title("AVT: Mahalanobis vs. centerline path (color = Dice)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Dice")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_md_vs_failure.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {FIG_DIR / 'fig_md_vs_failure.png'}  "
          f"degenerate={failed.sum()}/{len(failed)}")


# ---------------------------------------------------------------------------

def main():
    print(f"Writing figures to {FIG_DIR}")
    fig_md_vs_failure()


if __name__ == "__main__":
    main()
