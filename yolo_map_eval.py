import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def match(dets, gt, thr):
    dets = dets.sort_values("conf", ascending=False)
    used = {k: [False] * len(v) for k, v in gt.items()}
    tp = []
    for r in dets.itertuples():
        key = (r.pid, r.uid)
        best, bj = 0.0, -1
        for j, g in enumerate(gt.get(key, [])):
            if used[key][j]:
                continue
            o = iou((r.x0, r.y0, r.x1, r.y1), g)
            if o > best:
                best, bj = o, j
        if bj >= 0 and best >= thr:
            used[key][bj] = True
            tp.append(1)
        else:
            tp.append(0)
    return np.array(tp, dtype=int), dets["conf"].to_numpy()


def ap_from(tp, n_gt):
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    xs = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, xs, side="left")
    return float(np.mean([mpre[i] if i < len(mpre) else 0.0 for i in idx]))


def evaluate(dets, gt):
    n_gt = sum(len(v) for v in gt.values())
    res = {"n_patients": len({k[0] for k in gt}), "n_gt_boxes": n_gt, "n_detections": len(dets)}
    aps = {}
    for thr in np.arange(0.5, 0.96, 0.05):
        tp, _ = match(dets, gt, round(float(thr), 2))
        aps[round(float(thr), 2)] = ap_from(tp, n_gt)
    res["mAP50"] = aps[0.5]
    res["mAP50_95"] = float(np.mean(list(aps.values())))
    tp, conf = match(dets, gt, 0.5)
    for name, c in (("conf0.25", 0.25),):
        k = conf >= c
        t = int(tp[k].sum())
        p, r = t / max(k.sum(), 1), t / max(n_gt, 1)
        res.update({f"precision@{name}": p, f"recall@{name}": r, f"f1@{name}": 2 * p * r / max(p + r, 1e-9)})
    best = (0, 0, 0, 0)
    for c in np.unique(np.round(conf, 3)):
        k = conf >= c
        t = int(tp[k].sum())
        p, r = t / max(k.sum(), 1), t / max(n_gt, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        if f > best[0]:
            best = (f, p, r, c)
    res.update({"best_f1": best[0], "precision@best": best[1], "recall@best": best[2], "best_conf": best[3]})
    return res


def main():
    from box_source import BoxSource, norm_pid
    from yolo_boxes import discover
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--cache_dir", default="C:\\cache", help="used to pick the same CT series the detector ran on")
    ap.add_argument("--out_csv", default="yolo_map.csv")
    a = ap.parse_args()
    raw = pd.concat([pd.read_csv(f) for f in sorted(Path(a.raw_dir).glob("fold*.csv"))], ignore_index=True)
    raw["pid"] = raw["pid"].astype(str).map(norm_pid)
    pats = discover(a.shard_dirs)
    from yolo_boxes import cache_depth, chosen_series
    gt, dropped, kept_n = {}, 0, 0
    for pid in sorted(raw["pid"].unique()):
        if pid not in pats:
            continue
        pdir, ann = pats[pid]
        allb = BoxSource("xml", ann_root=ann).boxes(pid)
        if not allb:
            continue


        sdir, info, how = chosen_series(pdir, cache_depth(a.cache_dir, pid))
        if sdir is None:
            continue
        uids = set(info["uids"])
        for uid, boxes in allb.items():
            if uid in uids:
                gt[(pid, uid)] = [tuple(b) for b in boxes]
                kept_n += len(boxes)
            else:
                dropped += len(boxes)
    print(f"ground-truth boxes on the detector's CT series: {kept_n} | XML boxes on other series (PET/fused/other CT), ignored: {dropped}")
    have = {k[0] for k in gt}
    dets = raw[(raw["slice_idx"] >= 0) & raw["pid"].isin(have)][["pid", "uid", "x0", "y0", "x1", "y1", "conf"]]
    res = evaluate(dets, gt)
    print("\n=== YOLOv11 (cross-fitted), class-agnostic tumour detection over ALL slices ===")
    for k, v in res.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
    print("\nPaper (Wehbe 2024, YOLOv8-S, 3 subtype classes): mAP@0.5 0.971, precision 0.961, recall 0.945 (120 held-out studies)")
    pd.DataFrame([res]).to_csv(a.out_csv, index=False)
    print("saved", a.out_csv)


if __name__ == "__main__":
    main()
