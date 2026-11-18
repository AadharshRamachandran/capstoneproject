import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_seg_sizes import clahe, find_roots, window
from box_source import BoxSource, norm_pid
from sidecar import save_side
from dicom_utils import list_series_dirs, select_series, series_info, to_2d

IMG = 512


def discover(shard_dirs):
    out = {}
    for sd in shard_dirs:
        img, ann = find_roots(sd)
        for p in sorted(img.iterdir()):
            if p.is_dir() and p.name.startswith("Lung_Dx"):
                out[norm_pid(p.name)] = (p, ann)
    return out


def render(hu):
    return clahe(window(hu))


def read_slice(path):
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    px = to_2d(ds.pixel_array)
    return px.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))


def chosen_series(pdir, cache_D=None):
    sdir, info, how = select_series(pdir, None)
    if sdir is None or cache_D is None or info["n"] == cache_D:
        return sdir, info, how
    for d in list_series_dirs(pdir):
        i = series_info(d)
        if i["n"] == cache_D and i["modality"].upper() == "CT":
            return d, i, "matched_cache_D"
    return sdir, info, "cache_D_mismatch"


def cache_depth(cache_dir, pid):
    f = Path(cache_dir) / f"{pid}.npz" if cache_dir else None
    if f is None or not f.exists():
        return None
    z = np.load(str(f))
    for k in ("tumor_box", "tumor", "anat__lung"):
        if k in z.files:
            return int(z[k].shape[0])
    return None


def cmd_folds(a):
    from sklearn.model_selection import StratifiedKFold
    pats = discover(a.shard_dirs)
    df = pd.DataFrame({"pid": sorted(pats)})
    lab = pd.read_csv(a.labels_csv, dtype={"pid": str})[["pid", "T_stage"]]
    lab["pid"] = lab["pid"].map(norm_pid)
    lab["T"] = lab["T_stage"].astype(str).str.upper().str.extract(r"(T[1-4])")[0]
    df = df.merge(lab[["pid", "T"]], on="pid", how="left")
    df["fold"] = -1
    ok = df["T"].notna()
    skf = StratifiedKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
    idx = np.flatnonzero(ok)
    for f, (_, te) in enumerate(skf.split(idx, df.loc[ok, "T"])):
        df.loc[idx[te], "fold"] = f
    rest = np.flatnonzero(~ok)
    df.loc[rest, "fold"] = np.arange(len(rest)) % a.k
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f"{len(df)} patients -> {a.k} folds {df['fold'].value_counts().sort_index().to_dict()} | saved {a.out_csv}")


def _yolo_line(b, W, H):
    x0, y0, x1, y1 = b
    return f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {max(x1 - x0, 1) / W:.6f} {max(y1 - y0, 1) / H:.6f}"


def cmd_pool(a):
    folds = pd.read_csv(a.folds_csv)
    folds["pid"] = folds["pid"].map(norm_pid)
    pats = discover(a.shard_dirs)
    pool = Path(a.work) / "pool"
    for d in ("images", "labels", "done"):
        (pool / d).mkdir(parents=True, exist_ok=True)
    n_new = 0
    for pid in folds.loc[folds["fold"] >= 0, "pid"]:
        mark = pool / "done" / f"{pid}.txt"
        if pid not in pats or mark.exists():
            continue
        pdir, ann = pats[pid]
        boxes = BoxSource("xml", ann_root=ann).boxes(pid)
        if not boxes:
            mark.write_text("0")
            continue
        sdir, info, how = select_series(pdir, set(boxes))
        if sdir is None:
            continue
        rng = np.random.default_rng(sum(map(ord, pid)) + a.seed)
        pos = [i for i, u in enumerate(info["uids"]) if u in boxes]
        lo, hi = (min(pos), max(pos)) if pos else (0, 0)
        ps = set(pos)
        near = [i for i in range(max(0, lo - 15), min(info["n"], hi + 16)) if i not in ps]
        ns = set(near)
        far = [i for i in range(info["n"]) if i not in ps and i not in ns]
        n_neg = int(round(a.neg_ratio * len(pos)))
        neg = list(rng.choice(near, min(len(near), n_neg // 2), replace=False)) if near else []
        neg += list(rng.choice(far, min(len(far), n_neg - len(neg)), replace=False)) if far else []
        rest = [i for i in near + far if i not in set(int(x) for x in neg)]
        if len(neg) < n_neg and rest:
            neg += list(rng.choice(rest, min(len(rest), n_neg - len(neg)), replace=False))
        n = 0
        for i in pos + [int(x) for x in neg]:
            hu = read_slice(info["files"][i])
            if hu is None:
                continue
            H, W = hu.shape
            stem = f"{pid}_{i:04d}"
            cv2.imwrite(str(pool / "images" / f"{stem}.png"), render(hu))
            lines = [_yolo_line(b, W, H) for b in boxes.get(info["uids"][i], [])]
            (pool / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            n += 1
        mark.write_text(str(n))
        n_new += 1
        print(f"  pool {pid}: {n} images ({len(pos)} positive)", flush=True)
    print(f"pool ready ({n_new} patients rendered now) -> {pool}")


def cmd_prepare(a):
    cmd_pool(a)
    folds = pd.read_csv(a.folds_csv)
    folds["pid"] = folds["pid"].map(norm_pid)
    train = folds.loc[(folds["fold"] != a.fold) & (folds["fold"] >= 0), "pid"].tolist()
    rng = np.random.default_rng(a.seed + a.fold)
    val = set(rng.choice(train, max(1, int(0.1 * len(train))), replace=False))
    pool = Path(a.work) / "pool"
    root = Path(a.work) / f"fold{a.fold}"
    root.mkdir(parents=True, exist_ok=True)
    lists = {"train": [], "val": []}
    for pid in train:
        sp = "val" if pid in val else "train"
        lists[sp] += sorted(f.as_posix() for f in (pool / "images").glob(f"{pid}_*.png"))
    for sp, files in lists.items():
        (root / f"{sp}.txt").write_text("\n".join(files))
    (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: {(root / 'train.txt').as_posix()}\nval: {(root / 'val.txt').as_posix()}\nnames:\n  0: tumor\n")
    print(f"fold {a.fold}: train {len(lists['train'])} / val {len(lists['val'])} images (shared pool) -> {root}")


def cmd_train(a):
    from ultralytics import YOLO
    root = Path(a.work) / f"fold{a.fold}"
    kw = dict(data=str(root / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, project=str(root / "runs"),
              name="det", exist_ok=True, patience=a.patience, workers=a.workers, seed=a.seed, plots=False, amp=True, flipud=0.0, fliplr=0.5)
    last = root / "runs" / "det" / "weights" / "last.pt"
    if last.exists():
        try:
            print(f"fold {a.fold}: found {last.name} from an interrupted run -> resuming", flush=True)
            YOLO(str(last)).train(resume=True)
            print("best weights:", root / "runs" / "det" / "weights" / "best.pt")
            return
        except Exception as e:
            print(f"fold {a.fold}: resume not possible ({e!r:.120}) -> training from scratch", flush=True)
    YOLO(a.model).train(**kw)
    print("best weights:", root / "runs" / "det" / "weights" / "best.pt")


def cmd_predict(a):
    from ultralytics import YOLO
    folds = pd.read_csv(a.folds_csv)
    folds["pid"] = folds["pid"].map(norm_pid)
    pats = discover(a.shard_dirs)
    root = Path(a.work) / f"fold{a.fold}"
    w = root / "runs" / "det" / "weights" / "best.pt"
    model = YOLO(str(w))
    out_csv = Path(a.work) / "raw" / f"fold{a.fold}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    done_p = set(done["pid"]) if len(done) else set()
    rows = done.to_dict("records")
    for pid in folds.loc[folds["fold"] == a.fold, "pid"]:
        if pid in done_p or pid not in pats:
            continue
        pdir, _ = pats[pid]
        sdir, info, how = chosen_series(pdir, cache_depth(a.cache_dir, pid))
        if sdir is None:
            continue
        n_det = 0
        for s in range(0, info["n"], a.batch):
            idx = list(range(s, min(info["n"], s + a.batch)))
            imgs = []
            for i in idx:
                hu = read_slice(info["files"][i])
                g = render(hu) if hu is not None else np.zeros((IMG, IMG), np.uint8)
                imgs.append(np.repeat(g[:, :, None], 3, axis=2))
            res = model.predict(imgs, imgsz=a.imgsz, conf=a.raw_conf, iou=0.5, device=a.device, verbose=False)
            for i, r in zip(idx, res):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                H, W = r.orig_shape
                for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    rows.append({"pid": pid, "uid": info["uids"][i], "slice_idx": i, "x0": float(xyxy[0]), "y0": float(xyxy[1]),
                                 "x1": float(xyxy[2]), "y1": float(xyxy[3]), "conf": float(c), "H": int(H), "W": int(W), "series": how, "n": info["n"]})
                    n_det += 1
        rows.append({"pid": pid, "uid": "", "slice_idx": -1, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "conf": 0.0, "H": 0, "W": 0, "series": how, "n": info["n"]})
        print(f"  {pid}: {info['n']} slices, {n_det} raw detections ({how})", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("raw detections ->", out_csv)


def cmd_all(a):
    cmd_pool(a)
    for k in range(a.k):
        a.fold = k
        print(f"\n===== fold {k}/{a.k - 1}: image lists =====")
        cmd_prepare(a)
        ok = Path(a.work) / f"fold{k}" / "trained.ok"
        if ok.exists() and not a.retrain:
            print(f"===== fold {k}: already trained -> skipped =====")
        else:
            print(f"===== fold {k}: train =====")
            cmd_train(a)
            ok.write_text("ok")
        print(f"===== fold {k}: predict held-out patients =====")
        cmd_predict(a)


def postprocess(d: pd.DataFrame, conf=0.25, max_per_slice=3, gap=2, min_run=2, solo_conf=0.6) -> pd.DataFrame:
    d = d[(d["slice_idx"] >= 0) & (d["conf"] >= conf)].copy()
    if d.empty:
        return d.assign(run_id=pd.Series(dtype=int))
    d = d.sort_values(["slice_idx", "conf"], ascending=[True, False]).groupby("slice_idx").head(max_per_slice)
    sl = sorted(d["slice_idx"].unique())
    runs, cur = [], [sl[0]]
    for s in sl[1:]:
        if s - cur[-1] <= gap:
            cur.append(s)
        else:
            runs.append(cur)
            cur = [s]
    runs.append(cur)
    keep = []
    for rid, r in enumerate(runs):
        sub = d[d["slice_idx"].isin(r)]
        if len(r) >= min_run or sub["conf"].max() >= solo_conf:
            keep.append(sub.assign(run_id=rid))
    return pd.concat(keep) if keep else d.iloc[0:0].assign(run_id=pd.Series(dtype=int))


def _mask_from_boxes(df, D, H, W):
    m = np.zeros((D, H, W), np.uint8)
    for r in df.itertuples():
        if 0 <= int(r.slice_idx) < D:
            m[int(r.slice_idx), max(0, int(r.y0)):min(H, int(np.ceil(r.y1))), max(0, int(r.x0)):min(W, int(np.ceil(r.x1)))] = 1
    return m


def cmd_build(a):
    raw = pd.concat([pd.read_csv(f) for f in sorted((Path(a.work) / "raw").glob("fold*.csv"))], ignore_index=True)
    raw["pid"] = raw["pid"].map(norm_pid)
    cache = Path(a.cache_dir) if a.cache_dir else None
    kept, agree = [], []
    for pid, d in raw.groupby("pid"):
        k = postprocess(d, a.conf, a.max_per_slice, a.gap, a.min_run, a.solo_conf)
        k = k.assign(pid=pid)
        kept.append(k)
        rec = {"pid": pid, "yolo_n_boxes": len(k), "yolo_n_slices": int(k["slice_idx"].nunique()) if len(k) else 0,
               "yolo_n_lesions": int(k["run_id"].nunique()) if len(k) else 0,
               "yolo_conf_mean": float(k["conf"].mean()) if len(k) else np.nan, "yolo_conf_max": float(k["conf"].max()) if len(k) else np.nan}
        f = cache / f"{pid}.npz" if cache else None
        if f is not None and f.exists():
            z = np.load(str(f))
            tbox = z["tumor_box"] if "tumor_box" in z.files else None
            ref = tbox if tbox is not None else (z["tumor"] if "tumor" in z.files else None)
            if ref is not None:
                D, H, W = ref.shape
                my = _mask_from_boxes(k, D, H, W) if len(k) else np.zeros((D, H, W), np.uint8)
                save_side(cache, pid, "tumor_yolo", my)
                if tbox is not None and tbox.any():
                    mx = tbox > 0
                    inter, uni = (mx & (my > 0)).sum(), (mx | (my > 0)).sum()
                    sx, sy = mx.reshape(D, -1).any(1), (my > 0).reshape(D, -1).any(1)
                    rec.update({"yolo_iou3d": float(inter / max(uni, 1)), "yolo_slice_dice": float(2 * (sx & sy).sum() / max(sx.sum() + sy.sum(), 1)),
                                "yolo_slice_recall": float((sx & sy).sum() / max(sx.sum(), 1)), "yolo_slice_precision": float((sx & sy).sum() / max(sy.sum(), 1))})
        agree.append(rec)
    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    out.to_csv(a.out_csv, index=False)
    ag = pd.DataFrame(agree)
    ag.to_csv(Path(a.out_csv).with_name("yolo_agreement.csv"), index=False)
    print(f"{len(ag)} patients | with >=1 detected lesion: {int((ag['yolo_n_boxes'] > 0).sum())} | boxes -> {a.out_csv}")
    if "yolo_iou3d" in ag:
        print(f"agreement with radiologist boxes: median 3-D IoU {ag['yolo_iou3d'].median():.2f}, median slice Dice {ag['yolo_slice_dice'].median():.2f}, "
              f"slice recall {ag['yolo_slice_recall'].median():.2f}, slice precision {ag['yolo_slice_precision'].median():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["folds", "pool", "prepare", "train", "predict", "all", "build"])
    ap.add_argument("--shard_dirs", nargs="*", default=[])
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--folds_csv", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--retrain", action="store_true", help="retrain folds that already finished")
    ap.add_argument("--neg_ratio", type=float, default=1.0)
    ap.add_argument("--raw_conf", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_per_slice", type=int, default=3)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--min_run", type=int, default=2)
    ap.add_argument("--solo_conf", type=float, default=0.6)
    a = ap.parse_args()
    {"folds": cmd_folds, "pool": cmd_pool, "prepare": cmd_prepare, "train": cmd_train, "predict": cmd_predict, "all": cmd_all, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
