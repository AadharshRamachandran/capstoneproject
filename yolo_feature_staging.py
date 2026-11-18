import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LAYER = 10


def list_slices(work, folds):
    pool = Path(work) / "pool"
    lab = pool / "labels"
    info = folds.set_index("pid")
    rows = []
    for f in sorted((pool / "images").glob("*.png")):
        stem = f.stem
        pid, sl = stem.rsplit("_", 1)
        if pid not in info.index:
            continue
        lf = lab / f"{stem}.txt"
        if not lf.exists() or lf.stat().st_size == 0:
            continue
        t = info.loc[pid, "T"]
        if not isinstance(t, str) or not t.startswith("T"):
            continue
        rows.append((pid, int(sl), str(f), int(t[1]), int(info.loc[pid, "fold"])))
    return pd.DataFrame(rows, columns=["pid", "slice", "path", "T", "fold"])


def extract(model_path, paths, layer, imgsz, batch, device):
    import torch
    from ultralytics import YOLO
    dev = torch.device(f"cuda:{device}" if str(device).isdigit() and torch.cuda.is_available() else ("cpu" if device == "cpu" else str(device)))
    net = YOLO(str(model_path)).model.to(dev).eval().float()
    store = {}
    h = net.model[layer].register_forward_hook(lambda _, __, out: store.__setitem__("f", out.detach()))
    feats = []
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            arr = []
            for p in paths[s:s + batch]:
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    g = np.zeros((imgsz, imgsz), np.uint8)
                if g.shape != (imgsz, imgsz):
                    g = cv2.resize(g, (imgsz, imgsz))
                arr.append(g)
            x = torch.from_numpy(np.stack(arr)).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1).to(dev)
            net(x)
            feats.append(torch.nn.functional.adaptive_avg_pool2d(store["f"], 1).flatten(1).float().cpu().numpy())
            if (s // batch) % 50 == 0:
                print(f"    {s}/{len(paths)} slices", flush=True)
    h.remove()
    return np.concatenate(feats)


def build_features(a, sl):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"features_{a.layer}.npz"
    if cache.exists():
        z = np.load(cache)
        if len(z["held"]) == len(sl):
            print("features loaded from", cache)
            return z["held"], z["seen"]
    K = int(sl["fold"].max()) + 1
    wts = [Path(a.work) / f"fold{k}" / "runs" / "det" / "weights" / "best.pt" for k in range(K)]
    for w in wts:
        if not w.exists():
            sys.exit(f"missing {w}")

    print("detector fold0 on all slices (single-detector features)", flush=True)
    seen = extract(wts[0], sl["path"].values.tolist(), a.layer, a.imgsz, a.batch, a.device).astype(np.float32)

    held = np.zeros_like(seen)
    for k in range(K):
        own = np.flatnonzero(sl["fold"].values == k)
        print(f"detector {k}: {len(own)} held-out slices", flush=True)
        held[own] = extract(wts[k], sl["path"].values[own].tolist(), a.layer, a.imgsz, a.batch, a.device)
    np.savez_compressed(cache, held=held, seen=seen)
    return held, seen


def make_clf(name, seed):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "mlp":
        est = MLPClassifier((256, 128), max_iter=300, early_stopping=True, random_state=seed)
    else:
        est = RandomForestClassifier(300, n_jobs=-1, random_state=seed)
    return make_pipeline(StandardScaler(), PCA(50, random_state=seed), est)


def vote(pids, pred, y):
    d = pd.DataFrame({"pid": pids, "p": pred, "y": y})
    g = d.groupby("pid").agg(p=("p", lambda s: s.value_counts().idxmax()), y=("y", "first"))
    return g["y"].values, g["p"].values


def score(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def run(a, sl, X, tag, clf_name):
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    y = sl["T"].values
    pid = sl["pid"].values
    rows = []

    r = []
    for s in range(a.repeats):
        tr, te = train_test_split(np.arange(len(sl)), test_size=0.2, stratify=y, random_state=s)
        m = make_clf(clf_name, s).fit(X[tr], y[tr])
        r.append(score(y[te], m.predict(X[te])))
    r = np.array(r)
    rows.append((tag, clf_name, "random 80/20 split of slices (per slice)", r[:, 0].mean(), r[:, 1].mean(), r[:, 0].std()))

    for rep in range(a.repeats):
        skf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        pred = np.zeros(len(sl), int)
        for tr, te in skf.split(X, y, groups=pid):
            pred[te] = make_clf(clf_name, rep).fit(X[tr], y[tr]).predict(X[te])
        sa = score(y, pred)
        yy, pp = vote(pid, pred, y)
        pa = score(yy, pp)
        rows.append((tag, clf_name, f"patient-level 5-fold, rep {rep} (per slice)", sa[0], sa[1], np.nan))
        rows.append((tag, clf_name, f"patient-level 5-fold, rep {rep} (per patient, vote)", pa[0], pa[1], np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help=r"...\working\yolo  (has fold0..fold4 and pool)")
    ap.add_argument("--folds_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--clf", nargs="*", default=["mlp", "rf"])
    a = ap.parse_args()

    folds = pd.read_csv(a.folds_csv)
    sl = list_slices(a.work, folds)
    print(f"{len(sl)} annotated slices, {sl['pid'].nunique()} patients; T counts (patients): "
          f"{sl.drop_duplicates('pid')['T'].value_counts().sort_index().to_dict()}")
    share_s = 100 * (sl["T"] == 1).mean()
    share_p = 100 * (sl.drop_duplicates("pid")["T"] == 1).mean()
    print(f"always-T1 reference: {share_s:.1f}% of slices, {share_p:.1f}% of patients\n")

    held, seen = build_features(a, sl)
    all_rows = []
    for tag, X in (("single-detector features (fold 0 detector, saw ~80% of patients)", seen), ("heldout features (own-fold detector, never saw the patient)", held)):
        for c in a.clf:
            print(f"== {tag} | {c}", flush=True)
            rows = run(a, sl, X, tag, c)
            for r in rows:
                print(f"   {r[2]:62s} acc {r[3]*100:5.1f}  macroF1 {r[4]*100:5.1f}")
            all_rows += rows
    res = pd.DataFrame(all_rows, columns=["features", "classifier", "protocol", "acc", "macro_f1", "acc_sd"])
    res["always_T1_slice_%"] = round(share_s, 1)
    res["always_T1_patient_%"] = round(share_p, 1)
    f = Path(a.out_dir) / "yolo_feature_staging_results.csv"
    res.to_csv(f, index=False)
    print("\nsaved", f)
    print(res.groupby(["features", "classifier", res["protocol"].str.replace(r", rep \d+", "", regex=True)])[["acc", "macro_f1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
