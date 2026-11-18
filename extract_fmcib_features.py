import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


import torch
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", message=r"You are using a non-supported Python version.*")
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dicom_utils import read_hu_slice, select_series

PATCH = 50


def _valid_spacing(value):
    try:
        value = abs(float(value))
        return value if np.isfinite(value) and 0.05 <= value <= 50.0 else None
    except (TypeError, ValueError):
        return None


def _pixel_spacing(path):
    import pydicom
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    for name in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        values = getattr(ds, name, None)
        if values is None or len(values) < 2:
            continue
        psy, psx = _valid_spacing(values[0]), _valid_spacing(values[1])
        if psy is not None and psx is not None:
            return psx, psy, name
    return 1.0, 1.0, "fallback_1mm"


def _slice_spacing(info):
    zpos = info.get("z")
    if zpos is not None and len(zpos) > 1:
        diffs = np.abs(np.diff(np.asarray(zpos, dtype=float)))
        diffs = diffs[np.isfinite(diffs) & (diffs >= 0.05) & (diffs <= 50.0)]
        if diffs.size:
            return float(np.median(diffs)), "ImagePositionPatient"

    import pydicom
    candidates = []
    for path in info.get("files", [])[: min(16, len(info.get("files", [])))]:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            for name in ("SpacingBetweenSlices", "SliceThickness"):
                value = _valid_spacing(getattr(ds, name, None))
                if value is not None:
                    candidates.append((name, value))
                    break
        except Exception:
            continue
    if candidates:

        for name in ("SpacingBetweenSlices", "SliceThickness"):
            values = [v for n, v in candidates if n == name]
            if values:
                return float(np.median(values)), name
    return 1.0, "fallback_1mm"


def parse_boxes(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is not None:
            out.append((int(float(bb.findtext("xmin", "0"))), int(float(bb.findtext("ymin", "0"))),
                        int(float(bb.findtext("xmax", "512"))), int(float(bb.findtext("ymax", "512")))))
    return out


def find_roots(shard_dir):
    shard_dir = Path(shard_dir)
    img = next((c for c in [shard_dir / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx", shard_dir / "Lung-PET-CT-Dx", shard_dir]
                if c.is_dir() and any(p.is_dir() and p.name.startswith("Lung_Dx") for p in c.iterdir())), None)
    ann = next((c for c in [shard_dir / "Annotation" / "Annotation", shard_dir / "Annotation"] if c.is_dir()), None)
    if img is None or ann is None:
        sys.exit(f"cannot find Lung-PET-CT-Dx/ and Annotation/ under {shard_dir}")
    return img, ann


def make_patch(pdir, ann_dir):
    xmls = sorted(ann_dir.glob("*.xml")) if ann_dir.is_dir() else []
    by_uid = {x.stem: x for x in xmls}
    sdir, info, how = select_series(pdir, set(by_uid))
    if sdir is None:
        return None, "no_series"
    uid_to_idx = {u: i for i, u in enumerate(info["uids"])}
    items = []
    for u, xp in by_uid.items():
        if u in uid_to_idx:
            for b in parse_boxes(xp):
                items.append((uid_to_idx[u], b))
    if not items:
        return None, "no_box_matched"
    w = np.array([(b[2] - b[0]) * (b[3] - b[1]) for _, b in items], float)
    zc = float(np.average([i for i, _ in items], weights=w))
    xc = float(np.average([(b[0] + b[2]) / 2 for _, b in items], weights=w))
    yc = float(np.average([(b[1] + b[3]) / 2 for _, b in items], weights=w))
    psx, psy, spacing_source = _pixel_spacing(info["files"][int(round(zc))])
    dz, dz_source = _slice_spacing(info)
    half = PATCH / 2.0
    nz = int(np.ceil(half / dz)) + 2
    lo, hi = max(0, int(round(zc)) - nz), min(info["n"] - 1, int(round(zc)) + nz)
    sl = []
    for i in range(lo, hi + 1):
        s = read_hu_slice(info["files"][i])
        if s is None:
            return None, "non-2D pixel data (RGB or multi-frame)"
        sl.append(s)
    vol = np.stack(sl)
    off = np.arange(PATCH, dtype=np.float32) - (PATCH - 1) / 2.0
    zz = (zc - lo) + off / dz
    yy = yc + off / psy
    xx = xc + off / psx
    grid = np.meshgrid(zz, yy, xx, indexing="ij")
    patch = map_coordinates(vol, grid, order=1, mode="constant", cval=-1024.0).astype(np.float32)
    return patch, {"how": how, "psx": psx, "psy": psy, "dz": dz, "dz_source": dz_source,
                   "spacing_source": spacing_source, "zc": zc, "xc": xc, "yc": yc, "n_box": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--patch_dir", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3, help="parallel CT patch preparation workers")
    a = ap.parse_args()
    import nibabel as nib
    out = Path(a.out_csv)
    patch_dir = Path(a.patch_dir) if a.patch_dir else out.parent / "fmcib_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if out.exists():
        try:
            old = pd.read_csv(out)
            if "pid" in old and any(str(c).startswith("pred_") for c in old.columns):
                existing = old.drop_duplicates("pid", keep="first")
                print(f"Resuming from {len(existing)} existing CT embeddings in {out}")
        except Exception as e:
            print(f"Existing embedding CSV could not be reused ({e}); it will be replaced after extraction.")
            existing = pd.DataFrame()
    completed_pids = set(existing["pid"].astype(str)) if not existing.empty else set()

    rows, failed = [], []
    all_pids = set()
    for sd in a.shard_dirs:
        img_root, ann_root = find_roots(sd)
        pdirs = sorted(p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Dx"))
        if a.limit:
            pdirs = pdirs[:a.limit]
        print(f"\n{sd}: {len(pdirs)} patients")
        all_pids.update(p.name for p in pdirs)
        pending = [p for p in pdirs if p.name not in completed_pids]
        for pdir in pending:
            f = patch_dir / f"{pdir.name}.nii.gz"
            if f.exists():
                rows.append({"pid": pdir.name, "image_path": str(f)})
        pending = [p for p in pending if not (patch_dir / f"{p.name}.nii.gz").exists()]

        def prepare_patch(pdir):
            pid = pdir.name
            short = pid.replace("Lung_Dx-", "").replace("Lung_Dx_", "")
            try:
                patch, info = make_patch(pdir, ann_root / short)
            except Exception as exc:
                patch, info = None, repr(exc)[:160]
            if patch is None:
                return pid, None, info
            target = patch_dir / f"{pid}.nii.gz"
            nib.save(nib.Nifti1Image(patch, np.eye(4)), str(target))
            return pid, str(target), info

        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futures = [pool.submit(prepare_patch, pdir) for pdir in pending]
            for future in as_completed(futures):
                pid, image_path, info = future.result()
                if image_path is None:
                    failed.append((pid, info))
                    print(f"  {pid}: skipped ({info})")
                else:
                    rows.append({"pid": pid, "image_path": image_path})
                    print(f"  {pid}: patch saved (boxes {info['n_box']}, dz {info['dz']:.2f} mm; {info['dz_source']})")

    new_features = pd.DataFrame()
    if rows:
        csv_in = patch_dir / "fmcib_input.csv"
        pd.DataFrame(rows).to_csv(csv_in, index=False)
        print(f"\nExtracting FMCIB features for {len(rows)} new patches")
        from fmcib.run import get_features
        new_features = get_features(str(csv_in), precropped=True, batch_size=a.batch_size, num_workers=0)
        new_features = new_features.drop(columns=["image_path"], errors="ignore")
    elif existing.empty:
        sys.exit("no CT patches were created and no prior embedding CSV exists")

    feats = pd.concat([existing, new_features], ignore_index=True, sort=False)
    feats["pid"] = feats["pid"].astype(str)
    feats = feats.drop_duplicates("pid", keep="last")
    tmp_out = out.with_name(out.stem + ".partial.csv")
    feats.to_csv(tmp_out, index=False)
    tmp_out.replace(out)
    print(f"Saved {feats.shape[0]} x {feats.shape[1] - 1} CT features -> {out}")


    failed_pids = all_pids - set(feats["pid"].astype(str))
    prior_failed = out.with_name(out.stem + "_failed.csv")
    reason_by_pid = {pid: reason for pid, reason in failed}
    prior = {}
    if prior_failed.exists():
        try:
            old_failed = pd.read_csv(prior_failed)
            prior = dict(zip(old_failed["pid"].astype(str), old_failed["reason"].astype(str)))
        except Exception:
            pass
    for pid in failed_pids:
        reason_by_pid.setdefault(pid, prior.get(pid, "not_embedded"))
    if reason_by_pid:
        pd.DataFrame(sorted(reason_by_pid.items()), columns=["pid", "reason"]).to_csv(prior_failed, index=False)
        print(f"{len(failed_pids)} patients still lack embeddings (see {prior_failed})")
    elif prior_failed.exists():
        prior_failed.unlink()


if __name__ == "__main__":
    main()
