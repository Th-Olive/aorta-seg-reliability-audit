"""
Centerline-based clinical measurements for aortic segmentations.

For each prediction mask we compute:
  - mld_mm        : robust minimum luminal diameter — 2 × 5th-percentile radius
                    along the trimmed centerline (mm).  Endpoint voxels are
                    dropped because they always sit one voxel away from the
                    background, biasing the absolute min downward.
  - mld_min_mm    : 2 × strict minimum radius along the trimmed centerline (mm)
  - mld_med_mm    : 2 × median radius along the trimmed centerline (mm)
  - mld_max_mm    : 2 × maximum radius along the trimmed centerline (mm)
  - tortuosity    : centerline arc length / chord length (unitless)
  - path_mm       : centerline arc length (mm)
  - chord_mm      : straight-line chord length between centerline endpoints (mm)
  - n_skel_pts    : number of skeleton points (after dust filtering)
  - n_components  : number of connected components in the (binarised) mask
  - success       : True if measurements are valid
  - failure_reason: short string when success=False

Pipeline:
  1. Load NIfTI prediction → binary mask (any non-bg label → foreground)
  2. Take the largest connected component (fail if multi-component is too lopsided)
  3. kimimaro.skeletonize at the image's voxel spacing → centerline graph
  4. SciPy EDT on the binary mask → per-voxel radius (mm)
  5. Walk longest skeleton path between two endpoints; MLD = 2 × min(radius)
  6. Tortuosity = path / chord

The VMTK surface→centerline pipeline is more sophisticated, but failure-prone on
imperfect (disconnected, holed) segmentations. kimimaro + EDT degrades gracefully
and returns a failure flag — which is itself one of our reliability signals.

Usage:
  # Whole-aorta binary measurements (AortaSeg24 test, AVT, AMOS)
  python src/measurements.py \\
    --pred_dir   results/mahalanobis/scores/aortaseg24_test \\
    --output_csv results/measurements/aortaseg24_test_whole_aorta.csv \\
    --dataset_name aortaseg24_test
"""
import argparse
import csv
import pathlib
from typing import Optional

import nibabel as nib
import numpy as np


def _binary_largest_cc(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (largest CC mask, total number of CCs)."""
    from scipy.ndimage import label
    lbl, n = label(mask > 0)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return lbl == keep, int(n)


def _walk_longest_path(skel) -> tuple[np.ndarray, float, float] | None:
    """
    Walk the longest geodesic path on a kimimaro Skeleton instance.
    Returns (path_points_xyz_mm, path_length_mm, chord_length_mm) or None on failure.
    """
    # Skeleton.vertices is [N, 3] in mm (we passed anisotropy=spacing)
    # Skeleton.edges is [M, 2] of vertex indices
    vertices = skel.vertices
    edges = skel.edges
    if len(vertices) < 2 or len(edges) == 0:
        return None

    n = len(vertices)
    # Build adjacency with euclidean edge weights
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b in edges:
        d = float(np.linalg.norm(vertices[a] - vertices[b]))
        adj[a].append((b, d))
        adj[b].append((a, d))

    # Endpoints = degree-1 vertices (skeleton terminals)
    endpoints = [i for i in range(n) if len(adj[i]) == 1]
    if not endpoints:
        # No clear terminals (cyclic skeleton); fall back to farthest-pair via BFS on any vertex
        endpoints = [0]

    import heapq

    def dijkstra(src: int):
        dist = [float("inf")] * n
        prev = [-1] * n
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    # Two-pass farthest-vertex search starting from the first endpoint
    src = endpoints[0]
    dist, _ = dijkstra(src)
    far1 = int(np.argmax([d if np.isfinite(d) else -1 for d in dist]))
    dist, prev = dijkstra(far1)
    far2 = int(np.argmax([d if np.isfinite(d) else -1 for d in dist]))
    if not np.isfinite(dist[far2]):
        return None

    # Reconstruct path
    path = [far2]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])
    path = path[::-1]
    pts = vertices[path]
    path_len = float(dist[far2])
    chord = float(np.linalg.norm(pts[0] - pts[-1]))
    return pts, path_len, chord


def compute_measurements(pred_nifti: pathlib.Path,
                         min_cc_voxels: int = 200,
                         dust_threshold: int = 1000,
                         endpoint_trim_frac: float = 0.05) -> dict:
    """Run full pipeline on one case. Returns dict with measurements + success flag.

    endpoint_trim_frac: fraction of skeleton points to drop from each end of the
        path before computing radius statistics — removes near-surface voxels
        that always read 1-voxel radius from the EDT.
    """
    out: dict = {
        "mld_mm": float("nan"), "mld_min_mm": float("nan"),
        "mld_med_mm": float("nan"), "mld_max_mm": float("nan"),
        "tortuosity": float("nan"),
        "path_mm": float("nan"), "chord_mm": float("nan"),
        "n_skel_pts": 0, "n_components": 0,
        "success": False, "failure_reason": None,
    }

    nib_img = nib.load(pred_nifti)
    arr = nib_img.get_fdata(dtype=np.float32)
    mask_bin = arr > 0

    if mask_bin.sum() == 0:
        out["failure_reason"] = "empty_mask"
        return out

    # voxel spacing in mm (zooms returns (sx, sy, sz))
    spacing = tuple(float(z) for z in nib_img.header.get_zooms()[:3])

    # Largest connected component
    cc_mask, n_cc = _binary_largest_cc(mask_bin)
    out["n_components"] = n_cc
    if cc_mask.sum() < min_cc_voxels:
        out["failure_reason"] = "cc_too_small"
        return out

    # Skeleton in mm
    try:
        import kimimaro
        skels = kimimaro.skeletonize(
            cc_mask.astype(np.uint8),
            anisotropy=spacing,
            dust_threshold=dust_threshold,
            fix_branching=True, fix_borders=True,
            progress=False, parallel=1,
        )
    except Exception as e:
        out["failure_reason"] = f"skeletonize_failed:{type(e).__name__}"
        return out

    if not skels:
        out["failure_reason"] = "no_skeleton"
        return out
    # kimimaro returns dict[label_id, Skeleton] for binary input keyed by 1
    skel = next(iter(skels.values()))
    out["n_skel_pts"] = int(len(skel.vertices))
    if out["n_skel_pts"] < 5:
        out["failure_reason"] = "skel_too_short"
        return out

    walked = _walk_longest_path(skel)
    if walked is None:
        out["failure_reason"] = "no_path"
        return out
    pts_mm, path_len, chord = walked

    # EDT on the largest-CC mask → per-voxel distance to background (mm)
    from scipy.ndimage import distance_transform_edt
    edt = distance_transform_edt(cc_mask, sampling=spacing)

    # Map skeleton points (mm) back to voxel indices to look up EDT
    inv_spacing = np.array([1.0 / s for s in spacing])
    vox_idx = np.round(pts_mm * inv_spacing).astype(int)
    vox_idx[:, 0] = np.clip(vox_idx[:, 0], 0, edt.shape[0] - 1)
    vox_idx[:, 1] = np.clip(vox_idx[:, 1], 0, edt.shape[1] - 1)
    vox_idx[:, 2] = np.clip(vox_idx[:, 2], 0, edt.shape[2] - 1)
    radii_full = edt[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]

    # Trim endpoints — they always lie one voxel from background and bias min down
    n_trim = max(1, int(endpoint_trim_frac * len(radii_full)))
    radii = radii_full[n_trim: len(radii_full) - n_trim]
    radii = radii[radii > 0]

    if radii.size < 3:
        out["failure_reason"] = "no_valid_radii"
        return out

    # MLD: 2 × 5th-percentile radius (robust to residual artifacts).
    # Also report absolute min, median, max for sensitivity analysis.
    out["mld_mm"]     = float(2.0 * np.percentile(radii, 5))
    out["mld_min_mm"] = float(2.0 * radii.min())
    out["mld_med_mm"] = float(2.0 * np.median(radii))
    out["mld_max_mm"] = float(2.0 * radii.max())
    out["path_mm"]    = float(path_len)
    out["chord_mm"]   = float(chord)
    out["tortuosity"] = float(path_len / chord) if chord > 1e-6 else float("nan")
    out["success"]    = True
    return out


def collect_predictions(pred_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Find all <case>.nii.gz (ignore _unc.nii.gz)."""
    files = sorted(pred_dir.glob("*.nii.gz"))
    return [(f.name.replace(".nii.gz", ""), f) for f in files
            if not f.name.endswith("_unc.nii.gz")]


def main():
    parser = argparse.ArgumentParser(description="Centerline + MLD measurements")
    parser.add_argument("--pred_dir",   required=True, type=pathlib.Path)
    parser.add_argument("--output_csv", required=True, type=pathlib.Path)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cases = collect_predictions(args.pred_dir)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Found {len(cases)} cases in {args.pred_dir}")

    fieldnames = ["dataset", "case_id", "success", "failure_reason",
                  "mld_mm", "mld_min_mm", "mld_med_mm", "mld_max_mm",
                  "tortuosity", "path_mm", "chord_mm",
                  "n_skel_pts", "n_components"]
    n_success = 0
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (cid, path) in enumerate(cases):
            r = compute_measurements(path)
            row = {"dataset": args.dataset_name, "case_id": cid}
            row.update(r)
            for k in ("mld_mm", "mld_min_mm", "mld_med_mm", "mld_max_mm",
                      "tortuosity", "path_mm", "chord_mm"):
                row[k] = f"{r[k]:.4f}" if not np.isnan(r[k]) else ""
            w.writerow(row)
            f.flush()
            marker = "OK " if r["success"] else "FAIL"
            print(f"  [{i+1}/{len(cases)}] {cid}  {marker}  "
                  f"mld={r['mld_mm']:.2f}mm  tort={r['tortuosity']:.3f}  "
                  f"path={r['path_mm']:.1f}  n_cc={r['n_components']}  "
                  f"reason={r['failure_reason'] or '-'}")
            if r["success"]:
                n_success += 1
    print(f"\n{n_success}/{len(cases)} successful "
          f"({100*n_success/len(cases):.0f}%)  -> {args.output_csv}")


if __name__ == "__main__":
    main()
