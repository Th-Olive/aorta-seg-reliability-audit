"""
Sanitize staged TotalSegmentator NIfTI headers so ITK accepts them.

TotalSeg CTs from clinical scanners with gantry tilt have direction cosines
that drift by ~1e-4 from orthonormal. SimpleITK rejects these. Fix: SVD-project
the 3x3 rotation block of the affine onto the nearest orthonormal matrix.
Voxel data is untouched.

If a staged file is a hard link to the source TotalSeg case, we unlink it first
and write a sanitized copy so the source is preserved.

Usage:
  python src/sanitize_totalseg.py
"""
import argparse
import os
import pathlib

import nibabel as nib
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
DST_IMG_DIR = ROOT / "data" / "totalseg_prepared" / "images"


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix to R via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        # Flip the sign of the last U column to preserve right-handedness
        U[:, -1] *= -1
        Q = U @ Vt
    return Q


def needs_fix(R: np.ndarray, tol: float = 1e-6) -> bool:
    return float(np.max(np.abs(R @ R.T - np.eye(3)))) > tol


def sanitize_one(p: pathlib.Path) -> str:
    img = nib.load(p)
    aff = img.affine.copy()
    zooms = np.array(img.header.get_zooms()[:3])
    R = aff[:3, :3] / zooms
    if not needs_fix(R):
        return "ok"
    Q = orthonormalize(R)
    new_aff = np.eye(4)
    new_aff[:3, :3] = Q * zooms
    new_aff[:3, 3] = aff[:3, 3]

    # Build a new image with the corrected affine; preserve dtype + zooms
    new_img = nib.Nifti1Image(np.asarray(img.dataobj), new_aff, img.header)
    new_img.header.set_zooms(tuple(float(z) for z in zooms))

    # Replace file (unlink first in case it's a hard link to source data)
    os.unlink(p)
    nib.save(new_img, p)
    return "fixed"


def main():
    files = sorted(DST_IMG_DIR.glob("*_0000.nii.gz"))
    if not files:
        print(f"No files in {DST_IMG_DIR}")
        return
    print(f"Scanning {len(files)} staged TotalSeg files for affine sanity...")
    counts = {"ok": 0, "fixed": 0, "error": 0}
    for i, f in enumerate(files):
        try:
            r = sanitize_one(f)
        except Exception as e:
            r = "error"
            print(f"  ERROR {f.name}: {e}")
        counts[r] += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}]  ok={counts['ok']}  fixed={counts['fixed']}  err={counts['error']}")
    print(f"\nDone: ok={counts['ok']}  fixed={counts['fixed']}  err={counts['error']}")


if __name__ == "__main__":
    main()
