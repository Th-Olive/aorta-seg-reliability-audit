"""
Stage TotalSegmentator pelvic-coverage cases for nnU-Net inference.

Reads configs/totalseg_pelvic_cases.csv (566 case IDs) and hard-links each
{case}/ct.nii.gz -> data/totalseg_prepared/images/{case}_0000.nii.gz.
Hard links are used to avoid duplicating GB-scale data; falls back to copy if
the source/dest are on different volumes.

Reference labels stay in place at data/TotalSegmentator/{case}/segmentations/
(aorta.nii.gz, iliac_artery_left.nii.gz, iliac_artery_right.nii.gz).

Usage (from repo root):
  python src/prepare_totalseg.py [--dry-run]
"""
import argparse
import csv
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).parent.parent
SRC_ROOT = ROOT / "data" / "TotalSegmentator"
DST_IMG_DIR = ROOT / "data" / "totalseg_prepared" / "images"
CASES_CSV = ROOT / "configs" / "totalseg_pelvic_cases.csv"


def link_or_copy(src: pathlib.Path, dst: pathlib.Path, dry_run: bool) -> str:
    if dst.exists():
        return "skip"
    if dry_run:
        return "would-link"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N cases (smoke test)")
    args = p.parse_args()

    if not CASES_CSV.exists():
        print(f"Missing {CASES_CSV}", file=sys.stderr)
        sys.exit(1)

    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    with open(CASES_CSV) as f:
        for row in csv.DictReader(f):
            case_ids.append(row["image_id"])
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"Staging {len(case_ids)} TotalSegmentator cases -> {DST_IMG_DIR}")

    counts = {"link": 0, "copy": 0, "skip": 0, "missing": 0, "would-link": 0}
    for cid in case_ids:
        src = SRC_ROOT / cid / "ct.nii.gz"
        dst = DST_IMG_DIR / f"{cid}_0000.nii.gz"
        if not src.exists():
            counts["missing"] += 1
            continue
        action = link_or_copy(src, dst, args.dry_run)
        counts[action] += 1

    print(f"\nResults: {counts}")
    print(f"  staged dir contents: {len(list(DST_IMG_DIR.glob('*_0000.nii.gz')))}")


if __name__ == "__main__":
    main()
