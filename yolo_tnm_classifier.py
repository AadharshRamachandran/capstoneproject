import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_feature_staging import extract, list_slices

try:
    from box_source import norm_pid
except Exception:
    def norm_pid(x):
        return str(x).strip()


def first_int(x, lo=None):
    m = re.search(r"\d", str(x))
    return int(m.group()) if m else np.nan


def load_nm(labels_csv, pids):
    if not labels_csv:
        return None
    lab = pd.read_csv(labels_csv)
    idc = next((c for c in lab.columns if c.lower() in ("pid", "patient", "patient_id", "patient id", "case")), lab.columns[0])
    cols = {}
    for k in ("N", "M"):
        c = next((c for c in lab.columns if c != idc and re.match(rf"^{k}(_?stage|_?label)?$", c, re.I)), None)
        if c is not None:
            cols[k] = c
    if not cols:
        print("no N / M columns found in the labels csv -> T only")
        return None
    out = pd.DataFrame({"pid": lab[idc].map(norm_pid)})
    for k, c in cols.items():
        out[k] = lab[c].map(first_int)
    print("N/M columns used:", cols)
    return out.drop_duplicates("pid")


def detector_features(a, sl, K):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    F = []
    for k in range(K):
        f = out / f"det_features_L{a.layer}_fold{k}.npy"
        if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(sl):
            F.append(np.load(f))
            continue
        w = Path(a.work) / f"fold{k}" / "runs" / "det" / "weights" / "best.pt"
        if not w.exists():
            sys.exit(f"missing {w}")
        print(f"detector {k}: features of all {len(sl)} slices", flush=True)
        x = extract(w, sl["path"].values.tolist(), a.layer, a.imgsz, a.batch, a.device).astype(np.float32)
        np.save(f, x)
        F.append(x)
    return F


class TNMClassifier:

    def __init__(self, d_in, heads, device):
        import torch
        import torch.nn as nn
        self.torch, self.dev = torch, device
        self.body = nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5)).to(device)
        self.heads = nn.ModuleDict({k: nn.Linear(128, n) for k, n in heads.items()}).to(device)

    def params(self):
        return list(self.body.parameters()) + list(self.heads.parameters())

    def logits(self, x):
        h = self.body(x)
        return {k: m(h) for k, m in self.heads.items()}

    def fit(self, X, Y, Xv=None, Yv=None, epochs=30, lr=1e-3, bs=64, seed=0):
        torch = self.torch
        torch.manual_seed(seed)
        opt = torch.optim.Adam(self.params(), lr=lr)
        ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        X = torch.tensor(X, dtype=torch.float32, device=self.dev)
        Y = {k: torch.tensor(v, dtype=torch.long, device=self.dev) for k, v in Y.items()}
        n = len(X)
        hist = []
        for ep in range(epochs):
            self.body.train()
            perm = torch.randperm(n, device=self.dev)
            for s in range(0, n, bs):
                i = perm[s:s + bs]
                lg = self.logits(X[i])
                loss = sum(ce(lg[k], Y[k][i]) for k in lg)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if Xv is not None and ep in (epochs - 1,):
                hist.append(self.accuracy(Xv, Yv))
        return hist[-1] if hist else None

    def predict(self, X):
        torch = self.torch
        self.body.eval()
        with torch.no_grad():
            lg = self.logits(torch.tensor(X, dtype=torch.float32, device=self.dev))
        return {k: v.argmax(1).cpu().numpy() for k, v in lg.items()}

    def accuracy(self, X, Y):
        p = self.predict(X)
        return {k: float((p[k] == Y[k])[Y[k] >= 0].mean()) for k in p}


def make_xy(sl, nm, use):
    Y = {"T": (sl["T"].values - 1).astype(int)}
    if nm is not None:
        m = sl[["pid"]].merge(nm, on="pid", how="left")
        for k in ("N", "M"):
            if k in m and k in use:
                v = m[k].values.astype(float)
                Y[k] = np.where(np.isnan(v), -1, v).astype(int)
    return Y


def score(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def vote(pids, pred, y):
    d = pd.DataFrame({"pid": pids, "p": pred, "y": y}).groupby("pid").agg(p=("p", lambda s: s.value_counts().idxmax()), y=("y", "first"))
    return d["y"].values, d["p"].values


def prep(Xtr, Xother, ncomp, seed):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    ncomp = min(ncomp, Xtr.shape[1], Xtr.shape[0])
    pca = PCA(ncomp, random_state=seed).fit(sc.transform(Xtr))
    sc2 = StandardScaler().fit(pca.transform(sc.transform(Xtr)))
    z = lambda X: sc2.transform(pca.transform(sc.transform(X)))
    return z(Xtr), [z(X) for X in Xother]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--folds_csv", required=True)
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--pca", type=int, nargs="*", default=[50, 128])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random_split", action="store_true", help="also score the same MLP on a random 80/20 split of slices")
    a = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    folds = pd.read_csv(a.folds_csv)
    folds["pid"] = folds["pid"].map(norm_pid)
    sl = list_slices(a.work, folds)
    K = int(sl["fold"].max()) + 1
    nm = load_nm(a.labels_csv, sl["pid"].unique())
    Y = make_xy(sl, nm, ("N", "M"))
    heads = {"T": 4, "N": 4, "M": 2}
    heads = {k: heads[k] for k in Y}
    print(f"{len(sl)} annotated slices, {sl['pid'].nunique()} patients, heads: {list(heads)}; always-T1: "
          f"{100 * (sl['T'] == 1).mean():.1f}% of slices, {100 * (sl.drop_duplicates('pid')['T'] == 1).mean():.1f}% of patients")
    FE = detector_features(a, sl, K)

    rows = []
    for ncomp in a.pca:
        for k in range(K):
            te = np.flatnonzero(sl["fold"].values == k)
            trp = sl.loc[sl["fold"] != k, "pid"].unique()
            rng = np.random.default_rng(a.seed + k)
            vp = set(rng.choice(trp, max(1, int(0.1 * len(trp))), replace=False))
            va = np.flatnonzero(sl["pid"].isin(vp).values & (sl["fold"].values != k))
            tr = np.flatnonzero(~sl["pid"].isin(vp).values & (sl["fold"].values != k))
            X = FE[k]
            Ztr, (Zva, Zte) = prep(X[tr], [X[va], X[te]], ncomp, a.seed)
            net = TNMClassifier(Ztr.shape[1], heads, dev)
            sub = lambda idx: {h: v[idx] for h, v in Y.items()}
            net.fit(Ztr, sub(tr), epochs=a.epochs, seed=a.seed + k)
            ptr, pva, pte = net.predict(Ztr), net.predict(Zva), net.predict(Zte)
            row = {"protocol": "patient-disjoint test (fold)", "pca": ncomp, "fold": k, "n_test_slices": len(te), "n_test_patients": sl.iloc[te]["pid"].nunique(),
                   "train_acc_T": float((ptr["T"] == Y["T"][tr]).mean()), "val_acc_T": float((pva["T"] == Y["T"][va]).mean())}
            acc, f1 = score(Y["T"][te], pte["T"])
            row.update(test_acc_T_slice=acc, test_f1_T_slice=f1)
            yy, pp = vote(sl["pid"].values[te], pte["T"], Y["T"][te])
            acc, f1 = score(yy, pp)
            row.update(test_acc_T_patient=acc, test_f1_T_patient=f1)
            for h in ("N", "M"):
                if h in Y:
                    ok = Y[h][te] >= 0
                    row[f"test_acc_{h}_slice"] = float((pte[h][ok] == Y[h][te][ok]).mean()) if ok.any() else np.nan
            rows.append(row)
            print(f"pca {ncomp} fold {k}: test slices {len(te)}: T acc {row['test_acc_T_slice']*100:5.1f} (patient vote {row['test_acc_T_patient']*100:5.1f}) | "
                  f"train {row['train_acc_T']*100:5.1f} val {row['val_acc_T']*100:5.1f}", flush=True)
        if a.random_split:
            from sklearn.model_selection import train_test_split
            X = FE[0]
            tr, te = train_test_split(np.arange(len(sl)), test_size=0.2, stratify=Y["T"], random_state=a.seed)
            Ztr, (Zte,) = prep(X[tr], [X[te]], ncomp, a.seed)
            net = TNMClassifier(Ztr.shape[1], heads, dev)
            net.fit(Ztr, {h: v[tr] for h, v in Y.items()}, epochs=a.epochs, seed=a.seed)
            p = net.predict(Zte)
            acc, f1 = score(Y["T"][te], p["T"])
            rows.append({"protocol": "random 80/20 slice split", "pca": ncomp, "fold": -1, "n_test_slices": len(te), "test_acc_T_slice": acc, "test_f1_T_slice": f1})
            print(f"pca {ncomp} random 80/20 slice split: T acc {acc*100:.1f}  macro F1 {f1*100:.1f}", flush=True)

    res = pd.DataFrame(rows)
    f = Path(a.out_dir) / "yolo_tnm_classifier_results.csv"
    res.to_csv(f, index=False)
    print("\nsaved", f)
    g = res.groupby(["protocol", "pca"]).mean(numeric_only=True).drop(columns=["fold"], errors="ignore")
    print((g.filter(regex="acc|f1")*100).round(1).to_string())


if __name__ == "__main__":
    main()
