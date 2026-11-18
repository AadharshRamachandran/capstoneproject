import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dicom_utils import select_series, to_2d

LUNG_WC, LUNG_WW = -700.0, 1400.0
PAD, IMG = 20, 256
BOX_DILATE = 4


def window(hu):
    lo, hi = LUNG_WC - LUNG_WW / 2, LUNG_WC + LUNG_WW / 2
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def clahe(img):
    return cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16)).apply(img)


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


def feret_mm(mask, psx, psy):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = 0.0
    for c in cnts:
        h = cv2.convexHull(c).reshape(-1, 2).astype(np.float64) * np.array([psx, psy])
        if len(h) >= 2:
            d = np.sqrt(((h[:, None, :] - h[None, :, :]) ** 2).sum(-1)).max() + 0.5 * (psx + psy)
        else:
            d = max(psx, psy)
        best = max(best, d)
    return float(best)


def largest_cc(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


class Segmenter:
    def __init__(self, ckpt_dir, device, threshold=None, stem="TumorNet_2D"):
        import torch
        from models import get_segmentation_model
        self.torch, self.device = torch, device
        ckpt_dir = Path(ckpt_dir)
        ck = next(ckpt_dir.rglob(f"{stem}_fold0_best.pth"), None)
        if ck is None:
            sys.exit(f"{stem}_fold0_best.pth not found under {ckpt_dir}")
        th = next(ckpt_dir.rglob(f"{stem}_threshold.pth"), None)
        self.threshold = threshold if threshold else (float(torch.load(str(th), map_location="cpu", weights_only=True).get("threshold", 0.5)) if th else 0.5)
        self.model = get_segmentation_model("tumornet_2d", pretrained=False, mc_dropout=False).to(device)
        self.model.load_state_dict(torch.load(str(ck), map_location=device, weights_only=True))
        self.model.eval()
        print(f"TumorNet loaded; threshold {self.threshold:.2f}; device {device}")

    def probs(self, batch_u8):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(batch_u8), 16):
                x = torch.from_numpy(np.stack(batch_u8[i:i + 16])[:, None].astype(np.float32) / 255.0).to(self.device)
                lg = self.model(x)
                lg = lg[0] if isinstance(lg, tuple) else lg
                out.append(torch.sigmoid(lg).squeeze(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, IMG, IMG), np.float32)


def process_patient(pdir, ann_dir, seg, dilate=BOX_DILATE):
    import pydicom
    xmls = sorted(ann_dir.glob("*.xml")) if ann_dir.is_dir() else []
    uids = {x.stem: x for x in xmls}
    sdir, info, how = select_series(pdir, set(uids))
    if sdir is None:
        return None

    items = []
    for f in sorted(Path(sdir).glob("*.dcm"), key=lambda f: f.stem):
        h = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        uid = str(getattr(h, "SOPInstanceUID", ""))
        if uid in uids:
            items.append((f, uid))
    recs = []
    zs = info["z"]
    z_of = {u: (zs[i] if zs else float(i)) for i, u in enumerate(info["uids"])}
    for f, uid in items:
        ds = pydicom.dcmread(str(f), force=True)
        px = to_2d(ds.pixel_array).astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        win = window(px)
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        psy, psx = float(ps[0]), float(ps[1])
        H, W = win.shape
        for (x0, y0, x1, y1) in parse_boxes(uids[uid]):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            r0, r1, c0, c1 = max(0, y0 - PAD), min(H, y1 + PAD), max(0, x0 - PAD), min(W, x1 + PAD)
            roi = cv2.resize(clahe(np.ascontiguousarray(win[r0:r1, c0:c1])), (IMG, IMG))
            full = cv2.resize(clahe(win), (IMG, IMG))
            recs.append({"uid": uid, "z": z_of.get(uid, 0.0), "box": (x0, y0, x1, y1), "crop": (r0, r1, c0, c1),
                         "roi_in": roi, "full_in": full, "psx": psx, "psy": psy, "H": H, "W": W})
    if not recs:
        return {"how": how, "n_box": 0}
    p_roi = seg.probs([r["roi_in"] for r in recs])
    p_full = seg.probs([r["full_in"] for r in recs])
    rows = []
    for r, pr, pf in zip(recs, p_roi, p_full):
        x0, y0, x1, y1 = r["box"]
        r0, r1, c0, c1 = r["crop"]
        boxm = np.zeros((r["H"], r["W"]), np.uint8)
        boxm[max(0, y0 - dilate):y1 + dilate, max(0, x0 - dilate):x1 + dilate] = 1
        inner = np.zeros_like(boxm)
        inner[y0:y1, x0:x1] = 1

        m_roi = np.zeros_like(boxm)
        m_roi[r0:r1, c0:c1] = cv2.resize((pr > seg.threshold).astype(np.uint8), (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST)

        m_full = cv2.resize((pf > seg.threshold).astype(np.uint8), (r["W"], r["H"]), interpolation=cv2.INTER_NEAREST)
        row = {"z": r["z"], "box_area_mm2": float(inner.sum() * r["psx"] * r["psy"]),
               "box_side_mm": float(max((x1 - x0) * r["psx"], (y1 - y0) * r["psy"]))}
        for name, m in (("roi", m_roi), ("full", m_full)):
            m = largest_cc((m * boxm).astype(np.uint8))
            area = float(m.sum() * r["psx"] * r["psy"])
            row[f"{name}_size_mm"] = feret_mm(m, r["psx"], r["psy"]) if m.any() else 0.0
            row[f"{name}_area_mm2"] = area
            row[f"{name}_fill"] = area / max(row["box_area_mm2"], 1e-6)
        rows.append(row)
    d = pd.DataFrame(rows)
    zs_sorted = np.sort(d["z"].unique())
    dz = float(np.median(np.diff(info["z"]))) if info["z"] and len(info["z"]) > 1 else 1.0
    out = {"how": how, "n_box": len(d), "n_slices": int(len(zs_sorted)), "dz_mm": abs(dz),
           "box_side_mm": float(d["box_side_mm"].max()), "z_extent_mm": float(zs_sorted.max() - zs_sorted.min() + abs(dz))}
    for name in ("roi", "full"):
        out[f"seg_{name}_size_mm"] = float(d[f"{name}_size_mm"].max())
        out[f"seg_{name}_size_p75_mm"] = float(d[f"{name}_size_mm"].quantile(0.75))
        out[f"seg_{name}_fill_mean"] = float(d[f"{name}_fill"].mean())
        out[f"seg_{name}_fill_at_max"] = float(d.loc[d[f"{name}_size_mm"].idxmax(), f"{name}_fill"])
        out[f"seg_{name}_vol_cm3"] = float((d.groupby("z")[f"{name}_area_mm2"].sum() * abs(dz)).sum() / 1000.0)
        out[f"seg_{name}_empty_frac"] = float((d[f"{name}_size_mm"] <= 0).mean())
    return out


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
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    a = ap.parse_args()
    import torch
    seg = Segmenter(a.ckpt_dir, "cuda" if torch.cuda.is_available() else "cpu", a.threshold or None)
    out = Path(a.out_csv)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done_pids = set(done["pid"].astype(str)) if len(done) else set()
    rows = done.to_dict("records")
    for sd in a.shard_dirs:
        img_root, ann_root = find_roots(sd)
        pdirs = sorted(p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Dx"))
        print(f"\n{sd}: {len(pdirs)} patients")
        for pdir in pdirs:
            pid = pdir.name
            if pid in done_pids:
                continue
            short = pid.replace("Lung_Dx-", "").replace("Lung_Dx_", "")
            try:
                r = process_patient(pdir, ann_root / short, seg)
            except Exception as e:
                r = {"error": repr(e)[:150]}
            r = r or {"error": "no_series"}
            r["pid"] = pid
            rows.append(r)
            print(f"  {pid}: boxes {r.get('n_box', 0)} | box side {r.get('box_side_mm', float('nan')):.0f} mm | "
                  f"seg roi {r.get('seg_roi_size_mm', float('nan')):.0f} mm | seg full {r.get('seg_full_size_mm', float('nan')):.0f} mm"
                  + (f" | ERROR {r['error']}" if "error" in r else ""))
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} patients -> {out}")


if __name__ == "__main__":
    main()
