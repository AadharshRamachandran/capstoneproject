import argparse
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def norm_t(v):
    s = str(v).strip().upper()
    if s and s[0] in "1234":
        s = "T" + s
    return s if s in {"T1", "T2", "T3", "T4"} else ""


def load_data(features_dir, labels_csv, fmcib_csv, box_features="", solid_features=""):
    frames = []
    for path in sorted(Path(features_dir).glob("patient_features_shard[0-9][0-9][0-9].zip")):
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if Path(n).name == "patient_features.csv")
            part = pd.read_csv(archive.open(name))
        part["source"] = path.stem.replace("patient_features_", "")
        frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No shard feature zips in {features_dir}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["pid"] = df.pid.astype(str)

    numeric = [c for c in df.select_dtypes(include=[np.number]).columns]
    df["_n"] = df[numeric].notna().sum(axis=1)
    df = df.sort_values(["_n", "source"], ascending=[False, False]).drop_duplicates("pid").drop(columns="_n")
    labels = pd.read_csv(labels_csv, dtype={"pid": str})[["pid", "T_stage"]]
    labels["T"] = labels.T_stage.map(norm_t)
    df = df.merge(labels[["pid", "T"]], on="pid", how="left", validate="one_to_one")
    emb = pd.read_csv(fmcib_csv)
    cols = [c for c in emb if re.fullmatch(r"pred_\d+", str(c))]
    emb = emb[["pid", *cols]].drop_duplicates("pid").rename(columns={c: "fmcib_" + c for c in cols})
    emb["pid"] = emb.pid.astype(str)
    df = df.merge(emb, on="pid", how="left", validate="one_to_one")
    if box_features:
        import tnm_staging as _ts
        df, n_seg = _ts.merge_box_features(df, box_features)
        print(f"Box-segmentation sizes: {n_seg}/{len(df)} patients")
    if solid_features:
        import tnm_staging as _ts
        df, n_sol = _ts.merge_box_features(df, solid_features)
        print(f"Solid-component sizes: {n_sol}/{len(df)} patients")
    df = df[df["T"].isin(["T1", "T2", "T3", "T4"])].reset_index(drop=True)
    return df, ["fmcib_" + c for c in cols]


def metrics(y, pred, proba):
    out = {"accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
           "within1": np.mean(np.abs(y - pred) <= 1)}
    out["auc_T34"] = roc_auc_score(y >= 2, proba[:, 2] + proba[:, 3]) if len(np.unique(y >= 2)) == 2 else np.nan
    return out


def fit_predict(kind, params, Xtr, ytr, Xte, seed):
    import tnm_staging as _ts
    tab_n = len(_ts.SIZE_FEATURES)
    if kind in ("ordinal_size", "ordinal_fusion", "ordinal_xgb"):

        m = _ts.OrdinalClassifier(base="xgb" if kind == "ordinal_xgb" else "lr", C=params.get("C", 0.1),
                                  n_tabular=tab_n, n_use=params.get("n_use"), n_components=params.get("components", 0) if kind == "ordinal_fusion" else 0,
                                  seed=seed).fit(Xtr, ytr)
        probs = m.predict_proba(Xte)
        return probs.argmax(1), probs
    tabtr, tabte = Xtr[:, :tab_n], Xte[:, :tab_n]
    imtr, imte = Xtr[:, tab_n:], Xte[:, tab_n:]
    ti = SimpleImputer(strategy="median", keep_empty_features=True)
    tsi = StandardScaler()
    tabtr = tsi.fit_transform(ti.fit_transform(tabtr))
    tabte = tsi.transform(ti.transform(tabte))
    ii = SimpleImputer(strategy="median", keep_empty_features=True)
    imtr = ii.fit_transform(imtr)
    imte = ii.transform(imte)
    iscale = StandardScaler()
    imtr = iscale.fit_transform(imtr)
    imte = iscale.transform(imte)
    ncomp = min(params["components"], imtr.shape[0] - 1, imtr.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=seed)
    imtr = pca.fit_transform(imtr)
    imte = pca.transform(imte)
    if kind == "pls_lda":

        from sklearn.cross_decomposition import PLSRegression
        pls = PLSRegression(n_components=min(params["components"], 3, Xtr.shape[0] - 2), scale=False)
        y_onehot = np.eye(4)[ytr]
        ztr = pls.fit_transform(np.column_stack([tabtr, imtr]), y_onehot)[0]
        zte = pls.transform(np.column_stack([tabte, imte]))
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(ztr, ytr)
        probs = np.zeros((len(Xte), 4)); raw = clf.predict_proba(zte)
        for j, c in enumerate(clf.classes_): probs[:, int(c)] = raw[:, j]
    else:
        xtr, xte = np.column_stack([tabtr, imtr]), np.column_stack([tabte, imte])
        if kind == "size_lr":
            clf = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000,
                                     random_state=seed).fit(tabtr, ytr)
            raw = clf.predict_proba(tabte)
        elif kind == "fusion_lr":
            clf = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=3000,
                                     random_state=seed).fit(xtr, ytr)
            raw = clf.predict_proba(xte)
        elif kind == "fusion_catboost":
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier(iterations=params["iterations"], depth=4,
                                     learning_rate=0.03, loss_function="MultiClass",
                                     class_weights=[1.0, 1.0, 1.0, 1.0],
                                     verbose=False, random_seed=seed, thread_count=1)
            weights = len(ytr) / (4.0 * np.maximum(np.bincount(ytr, minlength=4), 1))
            clf.fit(xtr, ytr, sample_weight=weights[ytr])
            raw = clf.predict_proba(xte)
        else:
            clf = SVC(C=params["C"], kernel="rbf", class_weight="balanced", probability=True,
                      random_state=seed).fit(xtr, ytr)
            raw = clf.predict_proba(xte)
        probs = np.zeros((len(Xte), 4))
        for j, c in enumerate(clf.classes_): probs[:, int(c)] = raw[:, j]
    return probs.argmax(1), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=r"C:\Users\HP\Downloads\working\shard_features_v3")
    ap.add_argument("--labels_csv", default=r"C:\Users\HP\Downloads\C__lung_labels.csv")
    ap.add_argument("--fmcib_csv", default=r"C:\Users\HP\Downloads\working\fmcib_features_ct.csv")
    ap.add_argument("--solid_features", default="", help="box_solid_features.csv (HU-based solid sizes)")
    ap.add_argument("--box_features", default="", help="box_seg_features.csv (tumour-tight sizes)")
    ap.add_argument("--out_csv", default=r"C:\Users\HP\Downloads\working\results_ct_image\ct_model_experiments.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df, image_cols = load_data(a.features_dir, a.labels_csv, a.fmcib_csv, a.box_features, a.solid_features)
    import tnm_staging as ts
    features = ts._make_features(df)
    X = features[ts.SIZE_FEATURES + image_cols].to_numpy(float)
    y = df["T"].map({"T1": 0, "T2": 1, "T3": 2, "T4": 3}).to_numpy()
    matched = np.isfinite(X[:, len(ts.SIZE_FEATURES):]).any(axis=1)
    specs = [("size_logistic_C0.3", "size_lr", {"components": 1, "C": 0.3}),
             ("ordinal_size_lr_C0.1", "ordinal_size", {"C": 0.1}),
             ("ordinal_size_boxonly_C0.1", "ordinal_size", {"C": 0.1, "n_use": len(ts.BOX_SIZE_FEATURES)}),
             ("ordinal_size_box+seg_C0.1", "ordinal_size", {"C": 0.1, "n_use": len(ts.BOX_SIZE_FEATURES) + len(ts.SEG_FEATURES)}),
             ("ordinal_size_xgb_monotone", "ordinal_xgb", {}),
             ("ordinal_fusion_pca16_C0.1", "ordinal_fusion", {"components": 16, "C": 0.1}),
             ("ordinal_fusion_pca24_C0.03", "ordinal_fusion", {"components": 24, "C": 0.03}),
             ("fusion_logistic_pca24_C0.3", "fusion_lr", {"components": 24, "C": 0.3}),
             ("fusion_svm_pca8_C0.1", "svm", {"components": 8, "C": 0.1}),
             ("fusion_svm_pca16_C0.1", "svm", {"components": 16, "C": 0.1}),
             ("fusion_svm_pca16_C1", "svm", {"components": 16, "C": 1.0}),
             ("fusion_pls_pca16", "pls_lda", {"components": 16, "C": 0.1}),
             ("fusion_catboost_pca16", "fusion_catboost", {"components": 16, "iterations": 300})]
    results = []
    for subset, mask in [("all_351", np.ones(len(df), dtype=bool)), ("FMCIB_matched", matched)]:
        ids = np.flatnonzero(mask); yy = y[mask]; xx = X[mask]
        counts = np.bincount(yy, minlength=4)
        k = min(5, int(counts[counts > 0].min()))
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=a.repeats, random_state=a.seed)
        for name, kind, params in specs:
            fold_metrics = []
            for fold, (tr, te) in enumerate(cv.split(xx, yy)):
                pred, prob = fit_predict(kind, params, xx[tr], yy[tr], xx[te], a.seed + fold)
                fold_metrics.append(metrics(yy[te], pred, prob))
            row = {"subset": subset, "model": name, "n": len(yy), "n_with_embedding": int(matched[mask].sum())}
            for key in fold_metrics[0]:
                vals = np.array([m[key] for m in fold_metrics], dtype=float)
                row[key + "_mean"] = np.nanmean(vals)
                row[key + "_sd_fold"] = np.nanstd(vals)
            results.append(row)
            print(row)
    out = Path(a.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved exploratory model comparison: {out}")


if __name__ == "__main__":
    main()
