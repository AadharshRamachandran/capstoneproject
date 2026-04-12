import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LUNG_WW = 1400
LUNG_WC = -700
CLAHE_CLIP = 1.0
CLAHE_GRID = (16, 16)
IMG_SIZE = 256
TUMOR_PADDING = 20

def apply_lung_window(vol: np.ndarray) -> np.ndarray:
    lo = LUNG_WC-LUNG_WW/2
    hi = LUNG_WC+LUNG_WW/2
    return ((np.clip(vol, lo, hi)-lo)/(hi-lo)*255).astype(np.uint8)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(img)

def load_nifti(path: str) -> Tuple[np.ndarray, Dict]:
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()
    meta = {
        "pixel_spacing_x": float(sp[0]),
        "pixel_spacing_y": float(sp[1]),
        "slice_thickness": float(sp[2]),
        "shape": arr.shape,
    }
    return arr, meta

def inspect_msd_fold(msd_dir: str):
    msd_dir = Path(msd_dir)
    print(f"\nMSD folder: {msd_dir}")
    print(f"  Exists: {msd_dir.exists()}")
    for sub in ["imagesTr","labelsTr","imagesTs"]:
        p = msd_dir/sub
        if p.exists():
            files = list(p.glob("*.nii*"))
            junk = [f for f in files if f.name.startswith("._")]
            valid = [f for f in files if not f.name.startswith("._")]
            print(f"  {sub}: {len(valid)} valid, {len(junk)} junk files")
        else:
            print(f"  {sub}: NOT FOUND")
    dataset_json = msd_dir/"dataset.json"
    if dataset_json.exists():
        import json
        d = json.loads(dataset_json.read_text())
        print(f"  dataset.json: {d.get('name','')} â€” {d.get('numTraining',0)} training cases")

def scan_msd_folder(msd_dir: str) -> List[Dict]:
    msd_dir = Path(msd_dir)
    images_dir = msd_dir/"imagesTr"
    labels_dir = msd_dir/"labelsTr"
    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr not found in {msd_dir}")
    records = []
    for img_file in sorted(images_dir.glob("*.nii*")):
        if img_file.name.startswith("._"):
            continue
        stem = img_file.name.replace(".nii.gz","").replace(".nii","")
        lbl_file = labels_dir/f"{stem}.nii.gz"
        if not lbl_file.exists():
            lbl_file = labels_dir/f"{stem}.nii"
        if lbl_file.exists() and not lbl_file.name.startswith("._"):
            records.append({"image": str(img_file), "label": str(lbl_file), "stem": stem})
    return records

def extract_slices_from_volume(image_path: str, label_path: str, stem: str,
                                out_dir: Path, crop_roi: bool=True) -> List[Dict]:
    try:
        vol_hu, meta = load_nifti(image_path)
        lbl, _ = load_nifti(label_path)
        lbl_bin = (lbl>0).astype(np.uint8)
    except Exception as e:
        print(f"  ERROR loading {stem}: {e}")
        return []

    vol_win = apply_lung_window(vol_hu)
    records = []

    img_dir = out_dir/"images"
    msk_dir = out_dir/"masks"
    roi_img_dir = out_dir/"roi_images"
    roi_msk_dir = out_dir/"roi_masks"
    for d in [img_dir, msk_dir, roi_img_dir, roi_msk_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(vol_win.shape[0]):
        slc = vol_win[i]
        msk = lbl_bin[i]
        if msk.sum()==0:
            continue

        slc_clahe = apply_clahe(slc)
        slc_resized = cv2.resize(slc_clahe, (IMG_SIZE, IMG_SIZE))
        msk_resized = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img_path = img_dir/f"{stem}_s{i:04d}.png"
        msk_path = msk_dir/f"{stem}_s{i:04d}.png"
        cv2.imwrite(str(img_path), slc_resized)
        cv2.imwrite(str(msk_path), msk_resized*255)

        record = {
            "stem": stem,
            "slice_idx": i,
            "image_path": str(img_path),
            "mask_path": str(msk_path),
            "pixel_spacing_x": meta["pixel_spacing_x"],
            "pixel_spacing_y": meta["pixel_spacing_y"],
            "slice_thickness": meta["slice_thickness"],
        }

        if crop_roi:
            rows = np.where(np.any(msk>0, axis=1))[0]
            cols = np.where(np.any(msk>0, axis=0))[0]
            if len(rows)>0 and len(cols)>0:
                r0 = max(0, rows[0]-TUMOR_PADDING)
                r1 = min(slc.shape[0], rows[-1]+TUMOR_PADDING+1)
                c0 = max(0, cols[0]-TUMOR_PADDING)
                c1 = min(slc.shape[1], cols[-1]+TUMOR_PADDING+1)
                roi_img = cv2.resize(apply_clahe(slc[r0:r1, c0:c1]), (IMG_SIZE, IMG_SIZE))
                roi_msk = cv2.resize(msk[r0:r1, c0:c1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                roi_img_path = roi_img_dir/f"{stem}_s{i:04d}.png"
                roi_msk_path = roi_msk_dir/f"{stem}_s{i:04d}.png"
                cv2.imwrite(str(roi_img_path), roi_img)
                cv2.imwrite(str(roi_msk_path), roi_msk*255)
                record["roi_image_path"] = str(roi_img_path)
                record["roi_mask_path"] = str(roi_msk_path)

        records.append(record)

    return records

def process_msd_dataset(msd_dir: str, out_dir: str, crop_roi: bool=True) -> pd.DataFrame:
    msd_dir = Path(msd_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir/"msd_slice_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        print(f"Manifest already exists: {len(df)} slices â€” {manifest_path}")
        print("Delete it to reprocess.")
        return df

    records = scan_msd_folder(str(msd_dir))
    print(f"Found {len(records)} valid image-label pairs")

    all_slices = []
    for i, rec in enumerate(records, 1):
        print(f"  [{i:02d}/{len(records)}] {rec['stem']}", end=" ")
        slices = extract_slices_from_volume(rec["image"], rec["label"], rec["stem"], out_dir, crop_roi)
        print(f"â†’ {len(slices)} tumor slices")
        all_slices.extend(slices)

    df = pd.DataFrame(all_slices)
    df.to_csv(manifest_path, index=False)
    print(f"\nDone.")
    print(f"  Total tumor slices : {len(df)}")
    print(f"  Unique volumes     : {df['stem'].nunique()}")
    print(f"  Manifest saved     : {manifest_path}")
    print(f"Next step: pass '{manifest_path}' to CachedTumorSegDataset2D in main.py")
    return df

def main():
    parser = argparse.ArgumentParser(description="MSD Task06_Lung dataset loader")
    parser.add_argument("--msd_dir", required=True, help="Path to Task06_Lung folder")
    parser.add_argument("--out_dir", default="working/msd_processed", help="Output directory")
    parser.add_argument("--inspect", action="store_true", help="Inspect folder structure only")
    parser.add_argument("--no_crop", action="store_true", help="Skip ROI cropping")
    args = parser.parse_args()

    if args.inspect:
        inspect_msd_fold(args.msd_dir)
    else:
        process_msd_dataset(args.msd_dir, args.out_dir, crop_roi=not args.no_crop)

if __name__=="__main__":
    main()