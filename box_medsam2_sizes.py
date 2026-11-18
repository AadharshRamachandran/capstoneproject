import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dicom_utils import pixel_spacing, select_series, to_2d


def read_hu_slice(path):
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    px = to_2d(ds.pixel_array)
    return px.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))

WINDOWS = {"lung": (-1350.0, 150.0), "soft": (-160.0, 240.0)}
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
BOX_DILATE = 6
REGION_DILATE = 12


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


def to_uint8(vol_hu, lo, hi):
    return (np.clip((vol_hu - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def to_model_input(vol_u8, device):
    import torch
    n = vol_u8.shape[0]
    out = np.zeros((n, 3, 512, 512), np.float32)
    for i in range(n):
        r = cv2.resize(vol_u8[i], (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        out[i] = r[None]
    t = torch.from_numpy(out).to(device)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32, device=device)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32, device=device)[:, None, None]
    return (t - mean) / std


def feret_mm(mask2d, psx, psy):
    cnts, _ = cv2.findContours(np.ascontiguousarray(mask2d.astype(np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = 0.0
    for c in cnts:
        h = cv2.convexHull(c).reshape(-1, 2).astype(np.float64) * np.array([psx, psy])
        d = (cdist(h, h).max() if len(h) >= 2 else 0.0) + 0.5 * (psx + psy)
        best = max(best, d)
    return float(best)


class MedSAM2:
    def __init__(self, cfg, checkpoint, device="cuda"):
        import torch
        from sam2.build_sam import build_sam2_video_predictor_npz
        self.torch, self.device = torch, device
        self.predictor = build_sam2_video_predictor_npz(cfg, checkpoint)
        print(f"MedSAM2 loaded ({Path(checkpoint).name}) on {device}")

    def segment(self, vol_u8, key_idx, bbox_xyxy):
        torch = self.torch
        n, H, W = vol_u8.shape
        seg = np.zeros((n, H, W), np.uint8)
        img = to_model_input(vol_u8, self.device)
        box = np.array(bbox_xyxy, dtype=np.float32)
        ctx = torch.autocast(self.device if self.device != "cpu" else "cpu", dtype=torch.bfloat16)
        with torch.inference_mode(), ctx:
            state = self.predictor.init_state(img, H, W)
            for reverse in (False, True):
                self.predictor.add_new_points_or_box(inference_state=state, frame_idx=key_idx, obj_id=1, box=box)
                for fi, _oid, logits in self.predictor.propagate_in_video(state, reverse=reverse):
                    seg[fi][(logits[0] > 0.0).cpu().numpy()[0]] = 1
                self.predictor.reset_state(state)
        del img
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return seg


def lesion_features(seg, boxes_by_local, psx, psy, dz, prefix):
    out = {}
    if not seg.any():
        return {f"{prefix}size_mm": 0.0, f"{prefix}size3d_mm": 0.0, f"{prefix}vol_cm3": 0.0, f"{prefix}zextent_mm": 0.0,
                f"{prefix}fill_mean": 0.0, f"{prefix}iou_box": 0.0}
    zs = np.where(seg.reshape(seg.shape[0], -1).any(1))[0]
    sizes = [feret_mm(seg[z], psx, psy) for z in zs]
    out[f"{prefix}size_mm"] = float(max(sizes))
    out[f"{prefix}zextent_mm"] = float((zs.max() - zs.min() + 1) * dz)
    out[f"{prefix}vol_cm3"] = float(seg.sum() * psx * psy * dz / 1000.0)
    vox = np.argwhere(seg).astype(np.float64) * np.array([dz, psy, psx])
    if len(vox) > 3000:
        try:
            vox = vox[ConvexHull(vox).vertices]
        except Exception:
            vox = vox[np.linspace(0, len(vox) - 1, 1500).astype(int)]
    out[f"{prefix}size3d_mm"] = float(cdist(vox, vox).max()) if len(vox) > 2 else out[f"{prefix}size_mm"]
    fills, ious = [], []
    for z, bl in boxes_by_local.items():
        if z >= seg.shape[0]:
            continue
        bm = np.zeros(seg.shape[1:], bool)
        for (x0, y0, x1, y1) in bl:
            bm[y0:y1, x0:x1] = True
        m = seg[z] > 0
        fills.append(m.sum() / max(bm.sum(), 1))
        if m.any():
            ys, xs = np.where(m)
            mb = np.zeros_like(bm)
            mb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            ious.append((mb & bm).sum() / max((mb | bm).sum(), 1))
        else:
            ious.append(0.0)
    out[f"{prefix}fill_mean"] = float(np.mean(fills)) if fills else np.nan
    out[f"{prefix}iou_box"] = float(np.mean(ious)) if ious else np.nan
    return out


def process_patient(pdir, ann_dir, seg_model, windows, z_margin):
    xmls = sorted(ann_dir.glob("*.xml")) if ann_dir.is_dir() else []
    by_uid = {x.stem: x for x in xmls}
    sdir, info, how = select_series(pdir, set(by_uid))
    if sdir is None:
        return {"error": "no_series"}
    uid_to_idx = {u: i for i, u in enumerate(info["uids"])}
    boxes = {}
    for u, xp in by_uid.items():
        if u in uid_to_idx:
            bl = parse_boxes(xp)
            if bl:
                boxes[uid_to_idx[u]] = bl
    if not boxes:
        return {"how": how, "n_annot": 0}
    D = info["n"]
    lo, hi = max(0, min(boxes) - z_margin), min(D - 1, max(boxes) + z_margin)
    slices = []
    for i in range(lo, hi + 1):
        s = read_hu_slice(info["files"][i])
        if s is None:
            return {"how": how, "error": "non-2D pixel data (RGB or multi-frame)"}
        slices.append(s)
    vol = np.stack(slices)
    n, H, W = vol.shape
    psx, psy = pixel_spacing(info["files"][lo])
    zpos = info["z"]
    dzs = np.abs(np.diff(zpos)) if zpos and len(zpos) > 1 else np.array([])
    dzs = dzs[dzs > 1e-3]
    dz = float(np.median(dzs)) if len(dzs) else 1.0
    boxes_local = {i - lo: bl for i, bl in boxes.items()}

    key_local, key_box, best = None, None, -1
    for z, bl in boxes_local.items():
        for b in bl:
            a = (b[2] - b[0]) * (b[3] - b[1])
            if a > best:
                best, key_local, key_box = a, z, b

    allb = [b for bl in boxes_local.values() for b in bl]
    X0, Y0 = min(b[0] for b in allb) - REGION_DILATE, min(b[1] for b in allb) - REGION_DILATE
    X1, Y1 = max(b[2] for b in allb) + REGION_DILATE, max(b[3] for b in allb) + REGION_DILATE
    region = np.zeros((n, H, W), bool)
    region[:, max(0, Y0):Y1, max(0, X0):X1] = True
    for z, bl in boxes_local.items():
        region[z] = False
        for (x0, y0, x1, y1) in bl:
            region[z, max(0, y0 - BOX_DILATE):y1 + BOX_DILATE, max(0, x0 - BOX_DILATE):x1 + BOX_DILATE] = True
    res = {"how": how, "n_annot": len(boxes), "key_slice": int(key_local + lo), "psx": psx, "psy": psy, "dz_mm": dz,
           "box_side_mm": float(max(max((b[2] - b[0]) * psx, (b[3] - b[1]) * psy) for b in allb))}
    for wname in windows:
        lo_hu, hi_hu = WINDOWS[wname]
        seg = seg_model.segment(to_uint8(vol, lo_hu, hi_hu), key_local, key_box)
        seg = (seg.astype(bool) & region)
        lab, k = ndi.label(seg)
        if k > 1:
            sz = np.bincount(lab.ravel())[1:]
            seg = lab == (int(sz.argmax()) + 1)
        res.update(lesion_features(seg.astype(np.uint8), boxes_local, psx, psy, dz, f"ms2{wname}_"))
    return res


def find_roots(shard_dir):
    shard_dir = Path(shard_dir)
    img = next((c for c in [shard_dir / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx", shard_dir / "Lung-PET-CT-Dx", shard_dir]
                if c.is_dir() and any(p.is_dir() and p.name.startswith("Lung_Dx") for p in c.iterdir())), None)
    ann = next((c for c in [shard_dir / "Annotation" / "Annotation", shard_dir / "Annotation"] if c.is_dir()), None)
    if img is None or ann is None:
        sys.exit(f"cannot find Lung-PET-CT-Dx/ and Annotation/ under {shard_dir}")
    return img, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True, help="e.g. MedSAM2/checkpoints/MedSAM2_CTLesion.pt (or MedSAM2_latest.pt)")
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--windows", default="lung,soft", help="comma list of: lung, soft")
    ap.add_argument("--z_margin", type=int, default=4, help="extra slices above/below the annotated range given to the model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only the first N patients per shard (quick test)")
    a = ap.parse_args()
    windows = [w for w in a.windows.split(",") if w in WINDOWS]
    seg_model = MedSAM2(a.cfg, a.checkpoint, a.device)
    out = Path(a.out_csv)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {str(r["pid"]) for r in rows}
    for sd in a.shard_dirs:
        img_root, ann_root = find_roots(sd)
        pdirs = sorted(p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Dx"))
        if a.limit:
            pdirs = pdirs[:a.limit]
        print(f"\n{sd}: {len(pdirs)} patients")
        for pdir in pdirs:
            pid = pdir.name
            if pid in done:
                continue
            short = pid.replace("Lung_Dx-", "").replace("Lung_Dx_", "")
            try:
                r = process_patient(pdir, ann_root / short, seg_model, windows, a.z_margin)
            except Exception as e:
                r = {"error": repr(e)[:160]}
            r["pid"] = pid
            rows.append(r)
            msg = " | ".join(f"{w}: {r.get(f'ms2{w}_size_mm', float('nan')):.0f} mm (fill {r.get(f'ms2{w}_fill_mean', float('nan')):.2f})" for w in windows)
            print(f"  {pid}: annotated {r.get('n_annot', 0)} | box side {r.get('box_side_mm', float('nan')):.0f} mm | {msg}"
                  + (f" | ERROR {r['error']}" if "error" in r else ""))
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} patients -> {out}")


if __name__ == "__main__":
    main()
