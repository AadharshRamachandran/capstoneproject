import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUB_LEVEL = {"1A": 0, "1B": 1, "1C": 2, "1": 2, "2": 3, "2A": 3, "2B": 4, "3": 5, "4": 6}
SUB_TO_T = np.array([0, 0, 0, 1, 1, 2, 3])


class Ordinal:

    def __init__(self, n_levels, to_t, C=0.3):
        self.K, self.to_t, self.C = n_levels, np.asarray(to_t), C

    def fit(self, X, lev, w=None):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Z = self.imp.transform(X)
        self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        self.m = []
        for k in range(self.K - 1):
            yk = (lev > k).astype(int)
            if yk.sum() < 3 or (1 - yk).sum() < 3:
                self.m.append(float(yk.mean()))
                continue
            self.m.append(LogisticRegression(C=self.C, max_iter=2000).fit(Z, yk, sample_weight=w))
        return self

    def proba_levels(self, X):
        Z = self.sc.transform(self.imp.transform(X))
        gt = np.column_stack([np.full(len(Z), m) if isinstance(m, float) else m.predict_proba(Z)[:, 1] for m in self.m])
        gt = np.minimum.accumulate(gt, axis=1)
        full = np.column_stack([np.ones(len(Z)), gt, np.zeros(len(Z))])
        return np.clip(full[:, :-1] - full[:, 1:], 0, 1)

    def predict_proba(self, X):
        pl = self.proba_levels(X)
        out = np.zeros((len(pl), 4))
        for j, t in enumerate(self.to_t):
            out[:, t] += pl[:, j]
        return out / np.maximum(out.sum(1, keepdims=True), 1e-9)


class Gated:

    def __init__(self, ca, cb, gate_col, tau, C=0.3):
        self.ca, self.cb, self.g, self.tau, self.C = ca, cb, gate_col, tau, C

    def fit(self, X, t, w=None):
        self.a = Ordinal(4, np.arange(4), self.C).fit(X[:, self.ca], t, w)
        big = np.nan_to_num(X[:, self.g], nan=0.0) >= self.tau
        self.b = Ordinal(4, np.arange(4), self.C).fit(X[big][:, self.cb], t[big], None if w is None else w[big]) if big.sum() >= 40 and len(np.unique(t[big])) >= 3 else None
        return self

    def predict_proba(self, X):
        p = self.a.predict_proba(X[:, self.ca])
        if self.b is not None:
            big = np.nan_to_num(X[:, self.g], nan=0.0) >= self.tau
            if big.any():
                p[big] = self.b.predict_proba(X[big][:, self.cb])
        return p


def feature_sets(feats):
    import tnm_staging as ts
    have = lambda cols: [c for c in cols if c in feats.columns and feats[c].notna().mean() > 0.5]
    sets = {"legacy_box": have(["f_size", "f_side", "f_depth"]),
            "all_size": have(ts.SIZE_FEATURES),
            "lean_tight": have(["f_solid_eq", "f_solid_raw", "f_seg_mean", "f_depth", "f_solid_frac", "f_ggo_frac"])}
    sets["lean_tight+anat"] = sets["lean_tight"] + have(["f_dist_medi", "f_contact_medi", "f_contact2_medi", "f_pleural_contact2", "f_lung_depth"])
    return {k: v for k, v in sets.items() if v}


def coverage_acc(y, prob, cov):
    conf = prob.max(1)
    keep = conf >= np.quantile(conf, 1 - cov)
    return float((prob.argmax(1)[keep] == y[keep]).mean())


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0),
            "qwk": cohen_kappa_score(y, p, weights="quadratic"), "within1": float(np.mean(np.abs(y - p) <= 1))}


def fit_variant(X_tr, t_tr, s_tr, target, noise, thr=0.10, low=0.25, seed=0):
    use_sub = target == "sub" and s_tr is not None
    lev = s_tr if use_sub else t_tr
    K, to_t = (7, SUB_TO_T) if use_sub else (4, np.arange(4))
    w = None
    if noise:
        pt = np.zeros(len(t_tr))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for a, b in cv.split(X_tr, t_tr):
            pr = Ordinal(4, np.arange(4)).fit(X_tr[a], t_tr[a]).predict_proba(X_tr[b])
            pt[b] = pr[np.arange(len(b)), t_tr[b]]
        w = np.where(pt < thr, low, 1.0)
    return Ordinal(K, to_t).fit(X_tr, lev, w)


def run_variants(feats, T, sub, repeats=5, seed=42, k=5, taus=(35.0, 45.0, 55.0)):
    sets = feature_sets(feats)
    variants = [("legacy_box", "T", False, None)] + [(fs, tg, nz, None) for fs in sets if fs != "legacy_box" for tg in ("T", "sub") for nz in (False, True)]
    if "lean_tight+anat" in sets and "lean_tight" in sets:
        variants += [("gated", "T", nz, tau) for tau in taus for nz in (False, True)]
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
    rows, per_fold = [], {}
    for fs, tg, nz, tau in variants:
        if tg == "sub" and (sub < 0).any():
            continue
        if fs == "gated":
            cols = list(dict.fromkeys(sets["lean_tight"] + sets["lean_tight+anat"] + sets.get("all_size", [])))
            ca = [cols.index(c) for c in sets["lean_tight"]]
            cb = list(range(len(cols)))
            gc = cols.index("f_solid_eq") if "f_solid_eq" in cols else ca[0]
            X = feats[cols].to_numpy(float)
        else:
            X = feats[sets[fs]].to_numpy(float)
        ms = []
        for f, (tr, te) in enumerate(cv.split(X, T)):
            if fs == "gated":
                w = None
                if nz:
                    pt = np.zeros(len(tr)); inner = StratifiedKFold(5, shuffle=True, random_state=seed + f)
                    for a_, b_ in inner.split(X[tr][:, ca], T[tr]):
                        pr = Ordinal(4, np.arange(4)).fit(X[tr][a_][:, ca], T[tr][a_]).predict_proba(X[tr][b_][:, ca]); pt[b_] = pr[np.arange(len(b_)), T[tr][b_]]
                    w = np.where(pt < 0.10, 0.25, 1.0)
                m = Gated(ca, cb, gc, tau).fit(X[tr], T[tr], w)
            else:
                m = fit_variant(X[tr], T[tr], sub[tr] if tg == "sub" else None, tg, nz, seed=seed + f)
            prob = m.predict_proba(X[te])
            r = metrics(T[te], prob.argmax(1))
            r["acc_at_80pct_coverage"] = coverage_acc(T[te], prob, 0.8)
            r["acc_at_60pct_coverage"] = coverage_acc(T[te], prob, 0.6)
            ms.append(r)
        name = (f"gated tau={tau:.0f}mm (A=lean_tight, B=all+anat) | noise={'on' if nz else 'off'}" if fs == "gated" else f"{fs} | target={tg} | noise={'on' if nz else 'off'}")
        per_fold[name] = pd.DataFrame(ms)
        r = {"variant": name, **{c + "_mean": per_fold[name][c].mean() for c in ms[0]}, "accuracy_sd_fold": per_fold[name]["accuracy"].std()}
        rows.append(r)
        print(f"{name:62s} acc {r['accuracy_mean']:.3f}  F1 {r['macro_f1_mean']:.3f}  QWK {r['qwk_mean']:.3f}  within1 {r['within1_mean']:.3f}  acc@80%cov {r['acc_at_80pct_coverage_mean']:.3f}", flush=True)
    base = per_fold.get("all_size | target=T | noise=off")
    if base is not None:
        for r in rows:
            d = per_fold[r["variant"]]
            r["d_acc_vs_base"] = float((d["accuracy"] - base["accuracy"]).mean())
            r["d_qwk_vs_base"] = float((d["qwk"] - base["qwk"]).mean())
            r["folds_better_acc"] = float((d["accuracy"] > base["accuracy"]).mean())
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--fmcib_csv", required=True)
    ap.add_argument("--box_features", default="")
    ap.add_argument("--solid_features", default="")
    ap.add_argument("--clinical_xlsx", default="")
    ap.add_argument("--out_csv", default="experiment_v15.csv")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import tnm_staging as ts
    from experiment_ct_models import load_data
    df, _ = load_data(a.features_dir, a.labels_csv, a.fmcib_csv, a.box_features, a.solid_features)
    feats = ts._make_features(df)
    T = df["T"].map({"T1": 0, "T2": 1, "T3": 2, "T4": 3}).to_numpy()
    sub = np.full(len(df), -1)
    if a.clinical_xlsx:
        c = pd.read_excel(a.clinical_xlsx)
        c.columns = [str(x).strip() for x in c.columns]
        c["pid"] = "Lung_Dx-" + c["NewPatientID"].astype(str).str.strip()
        lv = c.set_index("pid")["T-Stage"].astype(str).str.strip().str.upper().map(SUB_LEVEL)
        sub = df["pid"].map(lv).fillna(-1).astype(int).to_numpy()
        bad = (sub >= 0) & (SUB_TO_T[np.clip(sub, 0, 6)] != T)
        sub[bad] = -1 if bad.sum() else sub[bad]
        print(f"sub-stage labels for {(sub >= 0).sum()}/{len(df)} patients ({int(bad.sum())} disagreed with T and were dropped)")
        if (sub < 0).any():
            print("target=sub skipped unless every patient has a sub-stage; filling the rest from T (T1->1c, T2->2a, T3->3, T4->4)")
            fill = np.array([2, 3, 5, 6])[T]
            sub = np.where(sub < 0, fill, sub)
    res = run_variants(feats, T, sub, a.repeats, a.seed)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out_csv, index=False)
    print("\nsaved", a.out_csv)
    print("Read the d_acc_vs_base / folds_better_acc columns: a gain smaller than accuracy_sd_fold (about 0.04) is noise.")


if __name__ == "__main__":
    main()
