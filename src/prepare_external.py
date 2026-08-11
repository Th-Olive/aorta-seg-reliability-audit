"""
Stage external test datasets for nnU-Net inference.

Does three things:
  1. Converts AortaSeg24 test masks (seg.nrrd -> nii.gz) -> nnUNet_raw/…/labelsTs/
  2. Converts AVT (NRRD -> NIfTI) -> data/avt_prepared/{images,labels}/
  3. Copies AMOS CT cases (IDs < 500) -> data/amos_prepared/{images,labels}/
     with _0000 suffix on images

Usage (from repo root):
  python src/prepare_external.py [--dry-run]
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import SimpleITK as sitk

_env = pathlib.Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"

AVT_SRC = DATA / "AortaSeg24"                       # source for AortaSeg24 test masks
AVT_PREPARED = DATA / "avt_prepared"
AMOS_SRC = DATA / "amos22"
AMOS_PREPARED = DATA / "amos_prepared"

DATASET_NAME = "Dataset001_AortaSeg24"
RAW_DIR = pathlib.Path(os.environ.get("nnUNet_raw", ROOT / "nnUNet_raw")) / DATASET_NAME
SPLIT_FILE = CONFIGS / "aortaseg24_split.json"

# AVT pathological cases (from dataset paper)
AVT_AD_CASES = {"R1", "R2", "R5", "R7", "R8"}
AVT_AAA_CASES = {"R6"}


# ---------------------------------------------------------------------------
# 1. AortaSeg24 labelsTs
# ---------------------------------------------------------------------------

def prepare_aortaseg24_labels_ts(dry_run: bool) -> int:
    """Convert AortaSeg24 test masks (.seg.nrrd) to labelsTs/*.nii.gz."""
    split = json.loads(SPLIT_FILE.read_text())
    test_ids = split["test"]
    masks_dir = AVT_SRC / "masks"         # original AortaSeg24 masks
    labels_ts = RAW_DIR / "labelsTs"

    print(f"\n[1] AortaSeg24 labelsTs -- {len(test_ids)} cases -> {labels_ts}")
    ok = skip = err = 0
    for sid in test_ids:
        src = masks_dir / f"{sid}_label.seg.nrrd"
        dst = labels_ts / f"{sid}.nii.gz"
        if dst.exists():
            skip += 1
            continue
        if not src.exists():
            print(f"  MISSING  {src}")
            err += 1
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            mask = sitk.ReadImage(str(src))
            sitk.WriteImage(mask, str(dst))
        print(f"  {'DRY' if dry_run else 'OK '}  {sid}")
        ok += 1
    print(f"  -> wrote {ok}, skipped {skip}, errors {err}")
    return err


# ---------------------------------------------------------------------------
# 2. AVT
# ---------------------------------------------------------------------------

def _load_avt_image_with_hu_fix(src: pathlib.Path, collection: str) -> sitk.Image:
    """Read an AVT NRRD image, applying the missing −1024 HU intercept for KiTS/Rider.

    KiTS (non-K18): stored as uint16 raw DICOM pixels without the intercept.
    Rider: stored as int16 but shifted +1024 (intercept never applied).
    Both need: pixel − 1024 → correct HU. K18 is int16 and already correct.
    Dongyang: already correct HU, returned unchanged.
    """
    img = sitk.ReadImage(str(src))
    needs_fix = (
        collection == "KiTS" and img.GetPixelID() == sitk.sitkUInt16
    ) or collection == "Rider"
    if not needs_fix:
        return img
    arr = sitk.GetArrayFromImage(img).astype(np.int32) - 1024
    arr = np.clip(arr, -32768, 32767).astype(np.int16)
    assert arr.min() < -500, (
        f"{src.name}: min={arr.min()} after HU fix — expected < −500 (air near −1024 HU)"
    )
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def prepare_avt(dry_run: bool, force: bool = False) -> int:
    """Convert AVT NRRD cases to NIfTI in data/avt_prepared/."""
    avt_src = DATA / "AVT"
    img_out = AVT_PREPARED / "images"
    lbl_out = AVT_PREPARED / "labels"

    print(f"\n[2] AVT -> {AVT_PREPARED}")

    rows = []
    ok = skip = err = 0

    for collection in ("KiTS", "Rider", "Dongyang"):
        coll_dir = avt_src / collection
        if not coll_dir.exists():
            print(f"  MISSING collection dir: {coll_dir}")
            continue

        for case_dir in sorted(coll_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            # Directory may be "R1 (AD)" but file inside is "R1.nrrd"
            # Extract base case ID as the part before the first space
            cid = case_dir.name.split(" ")[0]
            src_img = case_dir / f"{cid}.nrrd"
            src_msk = case_dir / f"{cid}.seg.nrrd"

            dst_img = img_out / f"{cid}_0000.nii.gz"
            dst_msk = lbl_out / f"{cid}.nii.gz"

            pathology = (
                "AD" if cid in AVT_AD_CASES else
                "AAA" if cid in AVT_AAA_CASES else
                "normal"
            )
            rows.append({"case_id": cid, "source": collection, "pathology": pathology})

            if dst_img.exists() and dst_msk.exists() and not force:
                skip += 1
                continue
            if not src_img.exists():
                print(f"  MISSING  image: {src_img}")
                err += 1
                continue
            if not src_msk.exists():
                print(f"  MISSING  mask: {src_msk}")
                err += 1
                continue

            if not dry_run:
                img_out.mkdir(parents=True, exist_ok=True)
                lbl_out.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(_load_avt_image_with_hu_fix(src_img, collection), str(dst_img))
                sitk.WriteImage(sitk.ReadImage(str(src_msk)), str(dst_msk))
            print(f"  {'DRY' if dry_run else 'OK '}  {cid}  [{pathology}]")
            ok += 1

    # Write case list
    csv_path = CONFIGS / "avt_cases.csv"
    if not dry_run and rows:
        import csv
        CONFIGS.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "source", "pathology"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {csv_path}")

    print(f"  -> wrote {ok}, skipped {skip}, errors {err}, total {len(rows)}")
    return err


# ---------------------------------------------------------------------------
# 3. AMOS
# ---------------------------------------------------------------------------

def prepare_amos(dry_run: bool) -> int:
    """Copy AMOS CT cases (ID < 500) to data/amos_prepared/ with _0000 suffix."""
    img_out = AMOS_PREPARED / "images"
    lbl_out = AMOS_PREPARED / "labels"

    print(f"\n[3] AMOS -> {AMOS_PREPARED}")

    rows = []
    ok = skip = err = 0

    for split_name, img_dir, lbl_dir in [
        ("train", AMOS_SRC / "imagesTr", AMOS_SRC / "labelsTr"),
        ("val",   AMOS_SRC / "imagesVa", AMOS_SRC / "labelsVa"),
    ]:
        if not img_dir.exists():
            print(f"  MISSING dir: {img_dir}")
            continue

        for src_img in sorted(img_dir.glob("amos_*.nii.gz")):
            stem = src_img.name.replace(".nii.gz", "")         # e.g. amos_0001
            num = int(stem.split("_")[1])
            if num >= 500:
                continue                                         # MRI -- skip

            src_lbl = lbl_dir / f"{stem}.nii.gz"
            dst_img = img_out / f"{stem}_0000.nii.gz"
            dst_lbl = lbl_out / f"{stem}.nii.gz"

            rows.append({"case_id": stem, "split": split_name})

            if dst_img.exists() and dst_lbl.exists():
                skip += 1
                continue
            if not src_lbl.exists():
                print(f"  MISSING  label: {src_lbl}")
                err += 1
                continue

            if not dry_run:
                img_out.mkdir(parents=True, exist_ok=True)
                lbl_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
            print(f"  {'DRY' if dry_run else 'OK '}  {stem}  [{split_name}]")
            ok += 1

    csv_path = CONFIGS / "amos_cases.csv"
    if not dry_run and rows:
        import csv
        CONFIGS.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "split"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {csv_path}")

    print(f"  -> wrote {ok}, skipped {skip}, errors {err}, total {len(rows)}")
    return err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool, force_avt: bool = False):
    print(f"prepare_external.py  dry_run={dry_run}  force_avt={force_avt}")
    total_errors = 0
    total_errors += prepare_aortaseg24_labels_ts(dry_run)
    total_errors += prepare_avt(dry_run, force=force_avt)
    total_errors += prepare_amos(dry_run)

    print()
    if total_errors:
        print(f"DONE with {total_errors} errors -- check output above.")
        sys.exit(1)
    else:
        msg = "(dry run -- nothing written)" if dry_run else ""
        print(f"Done. {msg}")
        if not dry_run:
            print()
            print("Next step: run inference")
            print("  python src/inference_mc.py --input_dir nnUNet_raw/Dataset001_AortaSeg24/imagesTs "
                  "--output_dir results/mc_dropout/predictions/aortaseg24_test --dataset_name aortaseg24_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-avt", action="store_true",
                        help="Re-prepare KiTS/Rider even if output files already exist")
    args = parser.parse_args()
    main(args.dry_run, force_avt=args.force_avt)
