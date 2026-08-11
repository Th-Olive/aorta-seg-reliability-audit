"""
Qualitative 3D surface rendering (marching cubes) of three AVT cases.

One correct case plus the two AVT cases that each fire only one signal, under the
thresholds used in the paper (MD > 64.9; path < 100 mm or >= 20 components):
  R11 - correct              (Dice 0.82, MD 28,  path 771)  both signals quiet
  K20 - spurious branch, gap (Dice 0.44, MD 189, path 495)  MD flags, path quiet
  R15 - collapsed prediction (Dice 0.08, MD 58,  path 49)   path flags, MD quiet

The mesh is built in physical mm (spacing taken from the affine) and drawn with a
true box aspect, so proportions are correct on anisotropic data. Each panel shows:
  - prediction surface : opaque, shaded
  - ground-truth ghost : translucent grey
  - centerline         : the kimimaro longest path, same code and parameters as
                         the measurement pipeline, computed in-frame so it aligns.

Output: results/figures/qualitative_avt_3d.png

Run (from repo root):
  python src/plot_qualitative_3d.py
"""
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measurements import _binary_largest_cc, _walk_longest_path  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB_DIR = ROOT / "data" / "avt_prepared" / "labels"
PRED_DIR = ROOT / "results" / "mahalanobis" / "scores" / "avt"
OUT_DIR = ROOT / "results" / "figures"

DUST_THRESHOLD = 1000  # identical to compute_measurements

# case -> two-line title.  "(flag)" = signal fires; "(quiet)" = signal silent.
CASES = [
    ("R11", "R11 — correct\nDice 0.82   MD 28 (quiet)   path 771 mm (quiet)"),
    ("K20", "K20 — spurious branch, upper gap\nDice 0.44   MD 189 (flag)   path 495 mm (quiet)"),
    ("R15", "R15 — collapsed prediction\nDice 0.08   MD 58 (quiet)   path 49 mm (flag)"),
]


def load_canonical(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(), np.asarray(img.header.get_zooms()[:3], float)


def drop_debris(mask, min_vox=60):
    """Remove tiny disconnected components so the surface reads cleanly.
    (Only cosmetic debris < min_vox voxels; the aortic tree is untouched.)"""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    keep = np.zeros_like(mask)
    for c in range(1, n + 1):
        comp = lab == c
        if comp.sum() >= min_vox:
            keep |= comp
    return keep


def crop_union(pred_bin, gt_bin, pad=6):
    roi = pred_bin | gt_bin
    xs, ys, zs = np.where(roi)
    sl = tuple(slice(max(v.min() - pad, 0), v.max() + pad + 1) for v in (xs, ys, zs))
    return pred_bin[sl], gt_bin[sl]


def mesh(mask, spacing, step=2):
    """Marching-cubes surface in physical mm. Returns (verts, faces) or None."""
    if mask.sum() < 10:
        return None
    # pad by 1 so surfaces close at the volume border
    m = np.pad(mask.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(m, level=0.5, spacing=tuple(spacing),
                                                step_size=step, allow_degenerate=False)
    verts -= np.asarray(spacing)  # undo the 1-voxel pad
    return verts, faces


def centerline(pred_bin, spacing):
    """Exact pipeline centerline in the SAME mm frame as the meshes."""
    cc, _ = _binary_largest_cc(pred_bin)
    if cc.sum() < 200:
        return None
    import kimimaro
    skels = kimimaro.skeletonize(
        cc.astype(np.uint8), anisotropy=tuple(spacing),
        dust_threshold=DUST_THRESHOLD, fix_branching=True, fix_borders=True,
        progress=False, parallel=1,
    )
    walked = _walk_longest_path(next(iter(skels.values())))
    if walked is None:
        return None
    return walked[0]  # pts_mm, same (i*sx, j*sy, k*sz) frame as marching_cubes


def gt_box_mm(gt_b, sp):
    """Physical centre and span (mm) of the ground-truth aorta."""
    idx = np.where(gt_b)
    lo = np.array([idx[k].min() * sp[k] for k in range(3)])
    hi = np.array([idx[k].max() * sp[k] for k in range(3)])
    return (lo + hi) / 2.0, (hi - lo)


def window_view(mask, sp, centre, span):
    """Crop `mask` to a physical box of size `span` centred on `centre`.

    Returns the cropped mask and the mm offset that maps its local coordinates
    into the shared [0, span] frame, so every panel ends up at one scale.
    Geometry outside the box is dropped, which matplotlib's 3D axes would
    otherwise still draw past the limits.
    """
    lo_mm = centre - span / 2.0
    sl, off = [], []
    for k in range(3):
        i0 = int(np.floor(lo_mm[k] / sp[k]))
        i1 = int(np.ceil((lo_mm[k] + span[k]) / sp[k]))
        c0, c1 = max(i0, 0), min(i1, mask.shape[k])
        sl.append(slice(c0, max(c1, c0 + 1)))
        off.append(c0 * sp[k] - lo_mm[k])
    return mask[tuple(sl)], np.asarray(off)


def render():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # pass 1: load everything and find one physical scale that suits all panels,
    # driven by the aorta itself so a distant prediction fragment cannot shrink it
    loaded, spans = [], []
    for case, title in CASES:
        pred, sp = load_canonical(PRED_DIR / f"{case}.nii.gz")
        gt, _ = load_canonical(LAB_DIR / f"{case}.nii.gz")
        pred_b, gt_b = crop_union(drop_debris(pred > 0), gt > 0)
        centre, span = gt_box_mm(gt_b, sp)
        loaded.append((case, title, pred_b, gt_b, sp, centre))
        spans.append(span)
    span = np.max(np.vstack(spans), axis=0) * 1.04   # common box + small margin

    fig = plt.figure(figsize=(12, 5.0))
    for i, (case, title, pred_b, gt_b, sp, centre) in enumerate(loaded, 1):
        cl = centerline(pred_b, sp)
        pred_w, off = window_view(pred_b, sp, centre, span)
        gt_w, _ = window_view(gt_b, sp, centre, span)
        if cl is not None:
            cl = cl - (centre - span / 2.0)
            inside = np.all((cl >= 0) & (cl <= span), axis=1)
            cl = cl[inside] if inside.any() else None

        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.set_facecolor("white")

        gm = mesh(gt_w, sp, step=3)
        if gm is not None:
            v, f = gm
            ax.plot_trisurf(v[:, 0] + off[0], v[:, 1] + off[1], f, v[:, 2] + off[2],
                            color=(0.50, 0.56, 0.66),
                            alpha=0.16, linewidth=0, shade=False)
        pm = mesh(pred_w, sp, step=2)
        if pm is not None:
            v, f = pm
            ax.plot_trisurf(v[:, 0] + off[0], v[:, 1] + off[1], f, v[:, 2] + off[2],
                            color=(0.83, 0.22, 0.18),
                            alpha=0.82, linewidth=0, shade=True)

        if cl is not None and len(cl):
            ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], color=(1.0, 0.80, 0.0),
                    lw=2.0, solid_capstyle="round", zorder=10)

        # one physical scale for all three panels, plus headroom for the title
        head = 0.10
        ax.set_box_aspect([span[0], span[1], span[2] * (1 + head)])
        ax.set_xlim(0, span[0])
        ax.set_ylim(0, span[1])
        ax.set_zlim(0, span[2] * (1 + head))
        ax.view_init(elev=12, azim=-72)
        ax.set_title(title, fontsize=10, pad=2, y=1.0)
        ax.set_axis_off()

    fig.text(0.5, 0.015,
             "3D surface of the prediction (red) over the ground truth (grey ghost); "
             "yellow = kimimaro centerline (longest path).   "
             "“flag/quiet” = whether each signal fires.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, wspace=0.02)
    out = OUT_DIR / "qualitative_avt_3d.png"
    fig.savefig(out, dpi=190)
    trim_vertical_margin(out)
    print(f"Wrote {out}")


def trim_vertical_margin(path, pad=8, max_gap=24):
    """Remove vertical dead space: blank rows at the edges, and any blank band
    inside the figure taller than `max_gap` (the 3D axes leave one above the
    titles and another between the meshes and the footer line).

    Only rows are touched. The width is untouched, so at a fixed \\linewidth the
    anatomy still prints at exactly the same scale.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    ink = (arr < 250).any(axis=2).any(axis=1)      # rows containing anything
    if not ink.any():
        return
    first, last = int(np.argmax(ink)), int(len(ink) - np.argmax(ink[::-1]))
    keep = np.zeros(len(ink), dtype=bool)
    keep[max(first - pad, 0):min(last + pad, len(ink))] = True

    run = 0                                         # collapse interior gaps
    for i in range(first, last):
        if ink[i]:
            run = 0
        else:
            run += 1
            if run > max_gap:
                keep[i] = False
    Image.fromarray(arr[keep]).save(path)


if __name__ == "__main__":
    render()
