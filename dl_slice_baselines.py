import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_feature_staging import list_slices

TIMM = {"resnet152": "resnet152", "densenet121": "densenet121", "swin_tiny": "swin_tiny_patch4_window7_224"}
SIZE = 224


def build(name, pretrained):
    import timm
    import torch.nn as nn
    b = timm.create_model(TIMM[name], pretrained=pretrained, num_classes=0)
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(b.num_features, 256), nn.ReLU(), nn.Linear(256, 4))
    return b, head


class DS:
    def __init__(self, paths, y, train):
        self.p, self.y, self.train = paths, y, train

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        import torch
        g = cv2.imread(self.p[i], cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (SIZE, SIZE)).astype(np.float32) / 255.0
        if self.train and np.random.rand() < 0.5:
            g = g[:, ::-1]
        x = np.repeat(g[None], 3, axis=0)
        x = (x - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
        return torch.tensor(x, dtype=torch.float32), int(self.y[i])


def focal(logits, y, w, gamma=2.0):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, y, weight=w, reduction="none")
    pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
    return ((1 - pt) ** gamma * ce).mean()


def train_predict(name, tr_p, tr_y, te_p, a, dev, pretrained):
    import torch
    from torch.utils.data import DataLoader
    b, h = build(name, pretrained)
    b, h = b.to(dev), h.to(dev)
    opt = torch.optim.AdamW([{"params": b.parameters(), "lr": 3e-5}, {"params": h.parameters(), "lr": 3e-4}], weight_decay=1e-2)
    cnt = np.bincount(tr_y, minlength=4).astype(float)
    w = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    dl = DataLoader(DS(tr_p, tr_y, True), batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr_p) > a.batch)
    for ep in range(a.epochs):
        b.train(); h.train()
        tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = focal(h(b(x)), y, w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(y)
        print(f"    epoch {ep + 1}/{a.epochs} loss {tot / len(tr_p):.3f}", flush=True)
    b.eval(); h.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(DS(te_p, np.zeros(len(te_p), int), False), batch_size=a.batch, num_workers=a.workers):
            out.append(h(b(x.to(dev))).argmax(1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                prec=precision_score(y, p, average="macro", zero_division=0), rec=recall_score(y, p, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--folds_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--models", nargs="*", default=["resnet152", "densenet121", "swin_tiny"])
    ap.add_argument("--eval", choices=["patient_folds", "full"], default="patient_folds")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="run only these folds")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--no_pretrained", action="store_true", help="skip the ImageNet weights download (tests only)")
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    folds = pd.read_csv(a.folds_csv)
    sl = list_slices(a.work, folds)
    y = (sl["T"].values - 1).astype(int)
    K = int(sl["fold"].max()) + 1
    print(f"{len(sl)} slices, {sl['pid'].nunique()} patients; device {dev}")
    rows = []
    for name in a.models:
        pred = np.full(len(sl), -1)
        if a.eval == "full":
            idx = np.arange(len(sl))
            pred = train_predict(name, sl["path"].values[idx].tolist(), y[idx], sl["path"].values.tolist(), a, dev, not a.no_pretrained)
        else:
            for k in (a.folds if a.folds is not None else range(K)):
                print(f"[{name}] fold {k}", flush=True)
                te = np.flatnonzero(sl["fold"].values == k)
                tr = np.flatnonzero(sl["fold"].values != k)
                pred[te] = train_predict(name, sl["path"].values[tr].tolist(), y[tr], sl["path"].values[te].tolist(), a, dev, not a.no_pretrained)
        ok = pred >= 0
        d = pd.DataFrame({"pid": sl["pid"].values[ok], "p": pred[ok], "y": y[ok]})
        g = d.groupby("pid").agg(p=("p", lambda s: s.value_counts().idxmax()), y=("y", "first"))
        ms, mp = metrics(d["y"], d["p"]), metrics(g["y"], g["p"])
        rows.append(dict(model=name, eval=a.eval, level="per slice", n=len(d), **ms))
        rows.append(dict(model=name, eval=a.eval, level="per patient (majority vote)", n=len(g), **mp))
        print(f"[{name}] slice acc {ms['acc']*100:.1f} F1 {ms['macro_f1']*100:.1f} | patient acc {mp['acc']*100:.1f} F1 {mp['macro_f1']*100:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("saved", a.out_csv)


if __name__ == "__main__":
    main()
