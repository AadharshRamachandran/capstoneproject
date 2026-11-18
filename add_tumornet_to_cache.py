import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_boxes_to_cache import find_roots, parse_boxes
from box_seg_sizes import IMG, Segmenter, clahe, window
from dicom_utils import select_series, to_2d


def cc_touching(mask, inner):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, np.uint8)
    for k in range(1, n):
        comp = lab == k
        if stats[k, cv2.CC_STAT_AREA] >= 12 and (comp & (inner > 0)).any():
            keep[comp] = 1
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--shard_num", type=int, default=0)
    ap.add_argument("--grow", type=int, default=8, help="pixels the XML box is grown before masking the prediction")
    ap.add_argument("--threshold", type=float, default=0.0, help="0 = use TumorNet_2D_threshold.pth")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import pydicom
    import torch
    seg = Segmenter(a.ckpt_dir, "cuda" if torch.cuda.is_available() else "cpu", a.threshold or None)
    img_root, ann_root = find_roots(a.shard_dir)
    cache = Path(a.cache_dir)
    rep = cache / "_reports"
    rep.mkdir(parents=True, exist_ok=True)
    rows = []
    for pdir in sorted(p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Dx")):
        pid = pdir.name
        cf = cache / f"{pid}.npz"
        rec = {"pid": pid, "status": "ok"}
        if not cf.exists():
            rec["status"] = "no_cache_file"
            rows.append(rec)
            continue
        try:
            z = np.load(str(cf))
            data = {k: z[k] for k in z.files}
            if "tumornet" in data and not a.overwrite:
                rec["status"] = "already_done"
                rows.append(rec)
                continue
            if "tumor_box" not in data:
                rec["status"] = "no_tumor_box (run add_boxes_to_cache.py first)"
                rows.append(rec)
                continue
            short = pid.replace("Lung_Dx-", "").replace("Lung_Dx_", "")
            xmls = {x.stem: x for x in sorted((ann_root / short).glob("*.xml"))} if (ann_root / short).is_dir() else {}
            sdir, info, how = select_series(pdir, set(xmls))
            D = data["tumor_box"].shape[0]
            if sdir is None or info["n"] != D:
                rec["status"] = f"series_mismatch ({info['n'] if sdir else None} vs {D})"
                rows.append(rec)
                continue
            u2i = {u: i for i, u in enumerate(info["uids"])}
            out = np.zeros(data["tumor_box"].shape, np.uint8)
            prob = np.zeros(data["tumor_box"].shape, np.uint8)
            items, ins = [], []
            for uid, xp in xmls.items():
                i = u2i.get(uid)
                if i is None:
                    continue
                ds = pydicom.dcmread(str(info["files"][i]), force=True)
                hu = to_2d(ds.pixel_array).astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
                H, W = hu.shape
                inner = np.zeros((H, W), np.uint8)
                for (x0, y0, x1, y1, _nm) in parse_boxes(xp):
                    inner[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 1
                items.append((i, H, W))
                ins.append((inner, cv2.resize(clahe(window(hu)), (IMG, IMG))))
            if not items:
                rec["status"] = "no_box_slice"
                rows.append(rec)
                continue
            batch = [x[1] for x in ins]
            p = seg.probs(batch)
            if not a.no_tta:
                p = 0.5 * (p + seg.probs([b[:, ::-1].copy() for b in batch])[:, :, ::-1])
            k = a.grow
            for (i, H, W), (inner, _), pr in zip(items, ins, p):
                pf = cv2.resize(pr.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                grown = cv2.dilate(inner, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
                prob[i] = (np.clip(pf, 0, 1) * 255 * grown).astype(np.uint8)
                out[i] = cc_touching(((pf > seg.threshold) & (grown > 0)).astype(np.uint8), inner)
            data["tumornet"], data["tumornet_p"] = out, prob
            np.savez_compressed(str(cf), **data)
            bx = (data["tumor_box"] > 0).sum()
            rec.update({"box_slices": len(items), "tn_vox": int(out.sum()), "tn_fill": float(out.sum() / max(bx, 1)),
                        "tn_empty_slices": int(sum(1 for (i, _, _) in items if out[i].sum() == 0))})
        except Exception as e:
            rec["status"] = f"error: {e!r}"[:200]
        rows.append(rec)
        print(f"  {pid}: {rec['status']} fill={rec.get('tn_fill', float('nan')):.2f} empty={rec.get('tn_empty_slices', '-')}/{rec.get('box_slices', '-')}")
    df = pd.DataFrame(rows)
    df.to_csv(rep / f"tumornet_report_shard{a.shard_num:03d}.csv", index=False)
    print("\nstatus counts:", df["status"].value_counts().to_dict())
    if "tn_fill" in df:
        print(f"median TumorNet fill of the box: {df['tn_fill'].median():.2f}  (box = whole lesion + margin, so 0.3-0.7 is expected)")


if __name__ == "__main__":
    main()
