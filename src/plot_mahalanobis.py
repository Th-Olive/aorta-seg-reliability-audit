"""
Paper-ready figures for the Mahalanobis reliability signal.

Output (in results/mahalanobis/figures/):
  fig_md_distribution.png  -- per-dataset MD violin + strip plot (ID vs OOD)

Usage:
  python src/plot_mahalanobis.py
"""
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
MAHA_ROOT = ROOT / "results" / "mahalanobis"
FIG_DIR = MAHA_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_reliability(name: str):
    p = MAHA_ROOT / f"reliability_{name}.csv"
    if not p.exists():
        return [], np.array([]), np.array([])
    ids, md, dice = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ids.append(r["case_id"])
            md.append(float(r["mahalanobis"]))
            dice.append(float(r["dice"]))
    return ids, np.array(md), np.array(dice)


def read_all_md(name: str) -> np.ndarray:
    """Read MD scores for all scored cases (including those without Dice)."""
    p = MAHA_ROOT / "scores" / name / "mahalanobis_summary.csv"
    if not p.exists():
        return np.array([])
    return np.array([float(r["mahalanobis"]) for r in csv.DictReader(open(p))])


# ---------------------------------------------------------------------------
# MD distribution per dataset (ID vs OOD)
# ---------------------------------------------------------------------------

def fig_md_distribution():
    datasets = [
        ("AortaSeg24 test\n(ID, n=40)", read_all_md("aortaseg24_test")),
        ("AVT\n(OOD, n=?)",             read_all_md("avt")),
        ("AMOS\n(OOD, n=?)",            read_all_md("amos")),
        ("TotalSeg\n(OOD, n=?)",        read_all_md("totalseg")),
    ]
    datasets = [(n, d) for n, d in datasets if len(d) > 0]
    # Fix N labels with actual counts
    datasets = [(n.replace("n=?", f"n={len(d)}"), d) for n, d in datasets]
    palette = ["#3a7bd5", "#d63d3d", "#d6943d", "#7a3dd6"]

    data = [d for _, d in datasets]
    labels = [n for n, _ in datasets]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Log scale to handle the heavy tails (linear panel dropped -- uninformative)
    parts = ax.violinplot([np.log10(d) for _, d in datasets], showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], palette):
        body.set_facecolor(color); body.set_alpha(0.5)
    parts["cmedians"].set_color("black")
    for i, d in enumerate(data, 1):
        x = np.full_like(d, i, dtype=float) + np.random.normal(0, 0.04, len(d))
        ax.scatter(x, np.log10(d), s=12, alpha=0.6, color="black", zorder=2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("log10(Mahalanobis distance²)")
    ax.set_title("Mahalanobis distribution per dataset (log10)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_md_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {FIG_DIR / 'fig_md_distribution.png'}")


def main():
    print(f"Writing figures to {FIG_DIR}")
    fig_md_distribution()
    print("Done.")


if __name__ == "__main__":
    main()
