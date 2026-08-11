"""
Prepare AortaSeg24 for nnU-Net v2 training.

Creates:
  nnUNet_raw/Dataset001_AortaSeg24/
    imagesTr/   subject001_0000.nii.gz ... (60 train+val cases)
    labelsTr/   subject001.nii.gz ...
    imagesTs/   subject061_0000.nii.gz ... (40 test cases)
    dataset.json

  nnUNet_preprocessed/Dataset001_AortaSeg24/
    splits_final.json   (fold 0 = our exact 50/10 train/val split)

Usage (from repo root):
  python src/prepare_nnunet.py [--dry-run]
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import SimpleITK as sitk

# ------------------------------------------------------------------
# Paths (loaded from .env if present)
# ------------------------------------------------------------------
_env = pathlib.Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT       = pathlib.Path(__file__).parent.parent
DATA       = ROOT / "data" / "AortaSeg24"
SPLIT_FILE = ROOT / "configs" / "aortaseg24_split.json"
DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_AortaSeg24"

RAW_DIR  = pathlib.Path(os.environ["nnUNet_raw"])  / DATASET_NAME
PRE_DIR  = pathlib.Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME

# AortaSeg24 label mapping — source: AortaSeg24.pdf (challenge documentation)
# SVS/STS zone classification + named branches
LABELS = {
    "background":                        0,
    "Zone 0":                            1,
    "Innominate":                        2,
    "Zone 1":                            3,
    "Left Common Carotid":               4,
    "Zone 2":                            5,
    "Left Subclavian Artery":            6,
    "Zone 3":                            7,
    "Zone 4":                            8,
    "Zone 5":                            9,
    "Zone 6":                           10,
    "Celiac Artery":                    11,
    "Zone 7":                           12,
    "SMA":                              13,
    "Zone 8":                           14,
    "Right Renal Artery":               15,
    "Left Renal Artery":                16,
    "Zone 9":                           17,
    "Zone 10 R":                        18,   # Right Common Iliac Artery
    "Zone 10 L":                        19,   # Left Common Iliac Artery
    "Right Internal Iliac Artery":      20,
    "Left Internal Iliac Artery":       21,
    "Zone 11 R":                        22,   # Right External Iliac Artery
    "Zone 11 L":                        23,   # Left External Iliac Artery
}

# For cross-dataset evaluation: TotalSegmentator filename → AortaSeg24 label index
# Aorta: TotalSegmentator's single "aorta" label spans all aortic zones (1,3,5,7-10,12,14,17)
# Evaluated as a merged label — see src/evaluation.py
AORTIC_ZONE_LABELS = [1, 3, 5, 7, 8, 9, 10, 12, 14, 17]   # Zones 0-9 (whole aorta)
TOTALSEG_TO_AORTASEG = {
    "iliac_artery_right.nii.gz": 18,   # Zone 10 R
    "iliac_artery_left.nii.gz":  19,   # Zone 10 L
}


def load_split():
    split = json.loads(SPLIT_FILE.read_text())
    return split["train"], split["val"], split["test"]


def convert_case(subject_id: str, src_img: pathlib.Path, src_mask: pathlib.Path,
                 dst_img: pathlib.Path, dst_mask: pathlib.Path | None,
                 dry_run: bool):
    """Copy image (NIfTI→NIfTI rename) and convert mask (.seg.nrrd→.nii.gz)."""
    if not dry_run:
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        if dst_mask is not None:
            dst_mask.parent.mkdir(parents=True, exist_ok=True)
            mask = sitk.ReadImage(str(src_mask))
            sitk.WriteImage(mask, str(dst_mask))
    action = "DRY" if dry_run else "OK "
    label_note = "" if dst_mask else " (no mask — test)"
    print(f"  {action}  {subject_id}{label_note}")


def write_dataset_json(n_training: int, dry_run: bool):
    ds = {
        "channel_names": {"0": "CT"},
        "labels": LABELS,
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
        "description": (
            "AortaSeg24 — 23-class aortic tree segmentation. "
            "100 ECG-gated contrast CTA, all type B aortic dissection. "
            "Fold 0 split: 50 train / 10 val / 40 test (subject061-101 excl. 084)."
        ),
    }
    path = RAW_DIR / "dataset.json"
    if not dry_run:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ds, indent=2))
    print(f"  {'DRY' if dry_run else 'OK '}  dataset.json  ({n_training} training cases, {len(LABELS)-1} classes)")


def write_splits_final(train_ids: list, val_ids: list, dry_run: bool):
    # nnU-Net v2 expects a list of fold dicts; fold 0 = our fixed split
    splits = [{"train": train_ids, "val": val_ids}]
    path = PRE_DIR / "splits_final.json"
    if not dry_run:
        PRE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(splits, indent=2))
    print(f"  {'DRY' if dry_run else 'OK '}  splits_final.json  "
          f"(fold 0: {len(train_ids)} train / {len(val_ids)} val)")


def main(dry_run: bool):
    train_ids, val_ids, test_ids = load_split()
    trainval_ids = train_ids + val_ids

    print(f"\nDataset     : {DATASET_NAME}")
    print(f"Raw dir     : {RAW_DIR}")
    print(f"Pre dir     : {PRE_DIR}")
    print(f"Train+val   : {len(trainval_ids)} cases -> imagesTr/labelsTr")
    print(f"Test        : {len(test_ids)} cases  -> imagesTs")
    print(f"Dry run     : {dry_run}\n")

    # --- imagesTr / labelsTr ---
    print("Converting train+val cases ...")
    errors = []
    for sid in trainval_ids:
        src_img  = DATA / "images" / f"{sid}_CTA.nii.gz"
        src_mask = DATA / "masks"  / f"{sid}_label.seg.nrrd"
        dst_img  = RAW_DIR / "imagesTr" / f"{sid}_0000.nii.gz"
        dst_mask = RAW_DIR / "labelsTr" / f"{sid}.nii.gz"
        if not src_img.exists():
            print(f"  MISSING image: {src_img}")
            errors.append(sid)
            continue
        if not src_mask.exists():
            print(f"  MISSING mask: {src_mask}")
            errors.append(sid)
            continue
        convert_case(sid, src_img, src_mask, dst_img, dst_mask, dry_run)

    # --- imagesTs ---
    print("\nConverting test cases ...")
    for sid in test_ids:
        src_img = DATA / "images" / f"{sid}_CTA.nii.gz"
        dst_img = RAW_DIR / "imagesTs" / f"{sid}_0000.nii.gz"
        if not src_img.exists():
            print(f"  MISSING image: {src_img}")
            errors.append(sid)
            continue
        convert_case(sid, src_img, None, dst_img, None, dry_run)

    # --- dataset.json ---
    print("\nWriting dataset.json ...")
    write_dataset_json(len(trainval_ids), dry_run)

    # --- splits_final.json ---
    print("\nWriting splits_final.json ...")
    write_splits_final(train_ids, val_ids, dry_run)

    # --- Summary ---
    print()
    if errors:
        print(f"ERRORS: {len(errors)} cases could not be converted: {errors}")
        sys.exit(1)
    else:
        print(f"Done. {'(dry run — nothing written)' if dry_run else ''}")
        if not dry_run:
            print()
            print("Next steps:")
            print("  python nnunet_run.py nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -np 4")
            print("  python nnunet_run.py nnUNetv2_train 1 3d_fullres 0 --npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()
    main(args.dry_run)
