import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_seg_sizes import feret_mm, find_roots, largest_cc
from box_source import BoxSource, short_pid
from dicom_utils import select_series, to_2d

SOLID_HU, ANY_HU = -300.0, -600.0
OPEN_MM = 3.0


def _opened_cc(mask, psx, psy):
    r = max(1, int(round(OPEN_MM / max(0.25, 0.5 * (psx + psy)))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    op = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    return largest_cc(op) if op.any() else op


def measure_box(hu, box, psx, psy):
    x0, y0, x1, y1 = box
    patch = cv2.medianBlur(np.ascontiguousarray(hu[y0:y1, x0:x1], dtype=np.float32), 3)
    box_area = float(patch.size * psx * psy)
    solid_raw = largest_cc(patch > SOLID_HU) if (patch > SOLID_HU).any() else np.zeros(patch.shape, np.uint8)
    solid_op = _opened_cc(patch > SOLID_HU, psx, psy)
    any_op = _opened_cc(patch > ANY_HU, psx, psy)
    def rect_long(m):
        if not m.any():
            return 0.0
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) * np.array([psx, psy], np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        return float(max(w, h))
    a = lambda m: float(m.sum() * psx * psy)
    f = lambda m: feret_mm(m, psx, psy) if m.any() else 0.0
    return {"box_side_mm": float(max((x1 - x0) * psx, (y1 - y0) * psy)), "box_area_mm2": box_area,
            "solid_raw_mm": f(solid_raw), "solid_open_mm": f(solid_op), "any_open_mm": f(any_op),
            "solid_rect_mm": rect_long(solid_op), "solid_eq_mm": float(2.0 * np.sqrt(a(solid_op) / np.pi)),
            "solid_area_mm2": a(solid_op), "any_area_mm2": a(any_op),
            "hu_p50": float(np.median(patch)), "hu_solid_mean": float(patch[solid_op > 0].mean()) if solid_op.any() else np.nan}


def process_patient(pdir, boxes_by_uid, legacy=False):
    import pydicom
    sdir, info, how = select_series(pdir, set(boxes_by_uid))
    if sdir is None:
        return None
    zs = info["z"]
    z_of = {u: (zs[i] if zs else float(i)) for i, u in enumerate(info["uids"])}
    rows = []
    for f, uid in zip(info["files"], info["uids"]):
        if uid not in boxes_by_uid:
            continue
        ds = pydicom.dcmread(str(f), force=True)
        if legacy:
            px = ds.pixel_array
            if px.ndim != 2:
                continue
        else:
            px = to_2d(ds.pixel_array)
        hu = px.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        psy, psx = float(ps[0]), float(ps[1])
        H, W = hu.shape
        for (x0, y0, x1, y1) in boxes_by_uid[uid]:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            m = measure_box(hu, (x0, y0, x1, y1), psx, psy)
            m["z"] = z_of.get(uid, 0.0)
            rows.append(m)
    if not rows:
        return {"how": how, "n_box": 0}
    d = pd.DataFrame(rows)
    dz = abs(float(np.median(np.diff(zs)))) if zs and len(zs) > 1 else 1.0
    i = int(d["solid_open_mm"].values.argmax())
    out = {"how": how, "solid_n_box": len(d), "box_side_mm": float(d["box_side_mm"].max()), "solid_dz_mm": dz,
           "solid_open_mm": float(d["solid_open_mm"].max()), "solid_raw_mm": float(d["solid_raw_mm"].max()),
           "solid_any_mm": float(d["any_open_mm"].max()), "solid_rect_mm": float(d["solid_rect_mm"].max()),
           "solid_eq_mm": float(d["solid_eq_mm"].max()),
           "solid_frac": float(d["solid_area_mm2"].iloc[i] / max(d["box_area_mm2"].iloc[i], 1e-6)),
           "solid_ggo_frac": float(1.0 - d["solid_area_mm2"].iloc[i] / max(d["any_area_mm2"].iloc[i], 1e-6)),
           "solid_hu_p50": float(d["hu_p50"].iloc[i]),
           "solid_hu_mean": float(d["hu_solid_mean"].iloc[i]) if np.isfinite(d["hu_solid_mean"].iloc[i]) else np.nan,
           "solid_vol_cm3": float((d.groupby("z")["solid_area_mm2"].sum() * dz).sum() / 1000.0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--source", choices=["xml", "yolo"], default="xml", help="which boxes to measure (xml = radiologist, yolo = detector)")
    ap.add_argument("--legacy", action="store_true", help="reproduce the older v5 log 2 solid file (skips RGB / multi-frame slices, ~251 patients)")
    ap.add_argument("--yolo_csv", default="", help="yolo_boxes.csv (needed for --source yolo)")
    a = ap.parse_args()
    out = Path(a.out_csv)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done_pids = set(done["pid"].astype(str)) if len(done) else set()
    rows = done.to_dict("records")
    for sd in a.shard_dirs:
        img_root, ann_root = find_roots(sd)
        src = BoxSource("xml", ann_root=ann_root) if a.source == "xml" else BoxSource("yolo", yolo_csv=a.yolo_csv)
        pdirs = sorted(p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Dx"))
        print(f"\n{sd}: {len(pdirs)} patients")
        for pdir in pdirs:
            pid = pdir.name
            if pid in done_pids:
                continue
            try:
                r = process_patient(pdir, src.boxes(pid), legacy=a.legacy) or {"error": "no_boxes"}
            except Exception as e:
                r = {"error": repr(e)[:150]}
            r = r or {"error": "no_series"}
            r["pid"] = pid
            rows.append(r)
            print(f"  {pid}: box {r.get('box_side_mm', float('nan')):.0f} mm | solid {r.get('solid_open_mm', float('nan')):.0f} mm | "
                  f"rect {r.get('solid_rect_mm', float('nan')):.0f} mm | solid frac {r.get('solid_frac', float('nan')):.2f}"
                  + (f" | ERROR {r['error']}" if "error" in r else ""))
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} patients -> {out}")


if __name__ == "__main__":
    main()
