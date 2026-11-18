import argparse
import contextlib
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = Path(__file__).resolve().parent
DEFAULT_FEATURES_DIR = r"C:\Users\HP\Downloads\working\shard_features"
DEFAULT_OUT = r"C:\Users\HP\Downloads\working\results_final"


def parse_args():
    p = argparse.ArgumentParser(description="Merge shard features and produce honest T-stage predictions")
    p.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--zips", nargs="*", default=None)
    p.add_argument("--labels_csv", default="", help="optional CSV with columns pid,T_stage")
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--code_dir", default=str(HERE), help="folder containing tnm_staging.py and evaluation.py")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", choices=["image", "size", "full"], default="image",
                   help="image uses frozen CT FMCIB embeddings plus CT-derived size measurements")
    p.add_argument("--box_features", default="",
                   help="box_seg_features.csv from box_seg_sizes.py (tumour-tight sizes; strongly recommended)")
    p.add_argument("--clean", choices=["0", "1"], default="0",
                   help="1 = also EXCLUDE training labels that are >=2 stages from a confident out-of-fold prediction. "
                        "Default 0: suspect labels are only REPORTED (label_issues.csv); try --clean 1 and compare the CV table")
    p.add_argument("--solid_features", default="",
                   help="box_solid_features.csv from box_solid_sizes.py (HU-based solid-component sizes)")
    p.add_argument("--yolo_summary", default="", help="yolo_summary.csv (mode 4: radiologist XML + YOLOv11 second reader)")
    p.add_argument("--fmcib_csv", default="", help="CT-only FMCIB embeddings CSV with pid and pred_* columns")
    p.add_argument("--override", choices=["auto", "none", "size", "full"], default="auto",
                   help="IASLC hard-override mode; auto = choose by repeated CV (default)")
    p.add_argument("--tabpfn", action="store_true", help="also try TabPFN as a model candidate (pip install tabpfn)")
    return p.parse_args()


def find_zips(args):
    if args.zips:
        zips = [Path(z) for z in args.zips]
    else:
        zips = sorted(Path(args.features_dir).glob("patient_features_shard[0-9][0-9][0-9].zip"))
    if not zips:
        sys.exit(f"ERROR: no patient_features_shardXXX.zip in {args.features_dir}. Run local_shard_features.py first or pass --zips.")
    nums = sorted(int(m.group(1)) for z in zips if (m := re.search(r"shard(\d+)", z.stem)))
    if nums and nums != list(range(nums[0], nums[-1] + 1)):
        print(f"WARNING: shard numbers {nums}; missing: {sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))}")
    return zips


def load_zip(zip_path):
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        if "patient_features.csv" not in names:
            sys.exit(f"ERROR: patient_features.csv not found inside {zip_path}")
        with z.open(names["patient_features.csv"]) as fh:
            f = pd.read_csv(fh)
        m = None
        if "patient_manifest.csv" in names:
            with z.open(names["patient_manifest.csv"]) as fh:
                m = pd.read_csv(fh)
    return zip_path.stem.replace("patient_features_", ""), f, m


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def sanity_report(feat_df, labelled_df):
    print("\n" + "=" * 78 + "\n  EXTRACTION QA (per shard)\n" + "=" * 78)
    rows = []
    for src, g in feat_df.groupby("source"):
        size = pd.to_numeric(g.get("tumor_size_2d_inplane_mm", pd.Series(0.0, index=g.index)), errors="coerce").fillna(0.0)
        r = {"source": src, "n": len(g), "no_tumor(%)": round(100 * (size <= 0).mean(), 1),
             "median_size_mm": round(float(size[size > 0].median()), 1) if (size > 0).any() else np.nan}
        if "tumor_in_lung_frac" in g:
            r["in_lung_frac(med)"] = round(float(pd.to_numeric(g["tumor_in_lung_frac"], errors="coerce").median()), 2)
        if "align_choice" in g:
            r["align"] = ",".join(f"{k}:{v}" for k, v in g["align_choice"].value_counts().items())
        for c in ["dist_lung_mm", "dist_mediastinum_mm", "dist_vertebrae_mm", "dist_heart_mm", "dist_aorta_mm",
                  "dist_trachea_mm", "dist_esophagus_mm"]:
            if c in g:
                r[c.replace("dist_", "").replace("_mm", "") + "_NaN(%)"] = round(100 * pd.to_numeric(g[c], errors="coerce").isna().mean(), 1)
        rows.append(r)
    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 40):
        print(pd.DataFrame(rows).to_string(index=False))
    if "tumor_in_lung_frac" in feat_df:
        bad = feat_df[pd.to_numeric(feat_df["tumor_in_lung_frac"], errors="coerce") < 0.5]
        if len(bad):
            print(f"\n  NOTE: {len(bad)} patient(s) have <50% of the tumor inside the lung mask. This is EXPECTED for bulky / central / "
                  f"solid masses (the lung mask excludes tumour tissue) and is NOT by itself evidence of misalignment; check only if small "
                  f"peripheral nodules appear here. Original wording: tumor/anatomy are probably "
                  f"misaligned or the box is wrong. First pids: {bad['pid'].head(8).tolist()}")
    if "tumor_mask_is_box" in feat_df:
        print(f"\n  Tumor masks from XML boxes: {int(pd.to_numeric(feat_df['tumor_mask_is_box'], errors='coerce').fillna(0).sum())}/{len(feat_df)}"
              f" (box masks are converted to inscribed ellipses)")
    if "T_true" in labelled_df:
        t = labelled_df.copy()
        print("\n  T distribution per shard (shard is NOT a model input):")
        print(pd.crosstab(t["source"], t["T_true"]).to_string())
        t["letter"] = t["pid"].str.extract(r"Lung_Dx-([A-Z])")[0]
        if t["letter"].notna().any():
            print("\n  T distribution per pid letter (in Lung-PET-CT-Dx the letter is the histology group - check the dataset docs):")
            print(pd.crosstab(t["letter"], t["T_true"]).to_string())


def metric_row(yt, yp, p_hi=None):
    yt, yp = np.asarray(yt), np.asarray(yp)
    r = {"acc": float((yt == yp).mean()), "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0, labels=sorted(set(yt)))),
         "bal_acc": float(balanced_accuracy_score(yt, yp)),
         "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic")) if len(set(yt)) > 1 else np.nan,
         "within1": float((np.abs(yt - yp) <= 1).mean())}
    hi = (yt >= 2).astype(int)
    r["auc_T34"] = float(roc_auc_score(hi, p_hi)) if p_hi is not None and 0 < hi.sum() < len(hi) else np.nan
    return r


def summarize(rows):
    df = pd.DataFrame(rows)
    return {c: (df[c].mean(), df[c].std(ddof=0)) for c in df.columns}


def print_table(title, table):
    print(f"\n  {title}")
    cols = ["acc", "macro_f1", "bal_acc", "qwk", "within1", "auc_T34"]
    print("  " + f"{'':34s}" + "".join(f"{c:>16s}" for c in cols))
    for name, s in table.items():
        line = f"  {name:34s}"
        for c in cols:
            m, sd = s.get(c, (np.nan, np.nan))
            line += f"{m:10.3f}±{sd:4.2f}" if not np.isnan(m) else f"{'n/a':>16s}"
        print(line)


def _pred_th(x, th):
    return (x > th[0]).astype(int) + (x > th[1]) + (x > th[2])


def learn_thresholds(x, y):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    xo, yo = x[ok], np.asarray(y)[ok]
    cum = np.cumsum(np.bincount(yo, minlength=4))[:3] / len(yo)
    th = [float(np.quantile(xo, c)) for c in cum]
    cand = np.unique(np.quantile(xo, np.linspace(0.03, 0.97, 95)))
    best = float((_pred_th(xo, th) == yo).mean())
    for _ in range(3):
        for k in range(3):
            lo = th[k - 1] if k > 0 else -np.inf
            hi = th[k + 1] if k < 2 else np.inf
            for c in cand:
                if lo < c < hi:
                    t2 = list(th)
                    t2[k] = float(c)
                    a = float((_pred_th(xo, t2) == yo).mean())
                    if a > best + 1e-9:
                        best, th = a, t2
    return th


def report_metrics(ev, y_true_raw, y_pred_raw, title, png_path):
    y_true = [ev._T_NORM4.get(ev.normalize_t(x), "") for x in y_true_raw]
    y_pred = [ev._T_NORM4.get(ev.normalize_t(x), "") for x in y_pred_raw]
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in ev.STAGE_LABELS and p in ev.STAGE_LABELS]
    if not pairs:
        return
    yt, yp = map(list, zip(*pairs))
    labels = [l for l in ev.STAGE_LABELS if l in set(yt) | set(yp)]
    m = ev.compute_classification_metrics(yt, yp, labels=labels)
    ev.print_metrics_table(m, f"{title} (n={len(yt)})")
    print(m["report"])
    ev.plot_confusion_matrix(yt, yp, labels, title, str(png_path))


def main():
    args = parse_args()
    os.environ["TNM_CLEAN"] = args.clean
    import importlib.util
    if importlib.util.find_spec("tabpfn") is None:
        print("TIP: pip install tabpfn  -> adds TabPFN v2 (Nature 2025 small-data foundation model) to the candidate set "
              "(weights download on first use; set TNM_USE_TABPFN=0 to disable)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(args.code_dir)))
    os.environ["TNM_RESULTS_DIR"] = str(out_dir)
    os.environ["TNM_FEATURE_PROFILE"] = args.profile
    if args.tabpfn:
        os.environ["TNM_USE_TABPFN"] = "1"
    try:
        import tnm_staging as ts
        import evaluation as ev
    except ImportError as e:
        sys.exit(f"ERROR: {e}\n  tnm_staging.py and evaluation.py must be in --code_dir")


    feats = []
    manis = []
    for z in find_zips(args):
        name, f, m = load_zip(z)
        f["source"] = name
        f["_shard_no"] = int(re.search(r"(\d+)", name).group(1)) if re.search(r"(\d+)", name) else -1
        feats.append(f)
        if m is not None:
            manis.append(m)
        print(f"Loaded {name}: {len(f)} patients, {f.shape[1] - 2} columns")
    feat_df = pd.concat(feats, ignore_index=True, sort=False)
    feat_df["pid"] = feat_df["pid"].astype(str)
    n_before = len(feat_df)
    dup = feat_df["pid"].duplicated(keep=False)
    if dup.any():
        num_cols = [c for c in feat_df.columns if c not in ("pid", "source", "_shard_no", "T_stage")]
        sz = pd.to_numeric(feat_df.get("tumor_size_2d_inplane_mm"), errors="coerce")
        d = feat_df.loc[dup].assign(_sz=sz[dup])
        spread = d.groupby("pid")["_sz"].agg(lambda s: (s.max() - s.min()) / max(s.max(), 1e-6))
        print(f"\nWARNING: {d['pid'].nunique()} pid(s) appear in more than one shard. "
              f"{int((spread > 0.25).sum())} of them differ >25% in tumor size between copies (shards were run with different settings).")
        print("         Keeping, per pid, the copy with the most populated features (ties -> the latest shard).")
        feat_df = (feat_df.assign(_n=feat_df[num_cols].notna().sum(axis=1))
                   .sort_values(["_n", "_shard_no"], ascending=[False, False])
                   .drop_duplicates("pid", keep="first").drop(columns="_n").sort_values("pid").reset_index(drop=True))
    print(f"\nTotal patients: {len(feat_df)} (from {n_before} rows)")


    if args.fmcib_csv:
        emb = pd.read_csv(args.fmcib_csv)
        if "pid" not in emb.columns:
            sys.exit("ERROR: --fmcib_csv must contain pid and FMCIB pred_* columns")
        emb_cols = [c for c in emb.columns if re.fullmatch(r"pred_\d+", str(c))]
        if not emb_cols:
            sys.exit("ERROR: no pred_<index> embedding columns found in --fmcib_csv")
        emb = emb[["pid", *emb_cols]].copy()
        emb["pid"] = emb["pid"].astype(str)
        emb = emb.drop_duplicates("pid", keep="first")
        emb = emb.rename(columns={c: "fmcib_" + c for c in emb_cols})
        for c in ["fmcib_" + c for c in emb_cols]:
            emb[c] = pd.to_numeric(emb[c], errors="coerce")
        feat_df = feat_df.drop(columns=[c for c in feat_df.columns if str(c).startswith("fmcib_")], errors="ignore")
        feat_df = feat_df.merge(emb, on="pid", how="left", validate="one_to_one")
        matched = int(feat_df[["fmcib_" + c for c in emb_cols]].notna().any(axis=1).sum())
        print(f"CT FMCIB embeddings: {matched}/{len(feat_df)} patients matched; {len(emb_cols)} image features")
    elif args.profile == "image":
        sys.exit("ERROR: --profile image requires --fmcib_csv; provide CT-only FMCIB embeddings.")


    if args.box_features:
        feat_df, n_seg = ts.merge_box_features(feat_df, args.box_features)
        print(f"Box-segmentation sizes: {n_seg}/{len(feat_df)} patients with seg_*_size_mm > 0")
    else:
        print("NOTE: --box_features not given -> model uses raw box sizes only (they are ~1.5x the lesion size).")
    if args.solid_features:
        feat_df, n_sol = ts.merge_box_features(feat_df, args.solid_features)
        print(f"Solid-component sizes: {n_sol}/{len(feat_df)} patients with solid_open_mm > 0")
    if args.yolo_summary:
        feat_df, n_y = ts.merge_yolo_summary(feat_df, args.yolo_summary)
        print(f"YOLOv11 summary: {n_y}/{len(feat_df)} patients matched")


    parts = []
    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv)
        if not {"pid", "T_stage"} <= set(lab.columns):
            sys.exit("ERROR: --labels_csv must have columns pid,T_stage")
        parts.append(lab[["pid", "T_stage"]])
    parts += [m[["pid", "T_stage"]] for m in manis if "T_stage" in m.columns]
    if "T_stage" in feat_df.columns:
        parts.append(feat_df[["pid", "T_stage"]])
    gt = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pid", "T_stage"])
    gt["pid"] = gt["pid"].astype(str)
    gt["T_true"] = gt["T_stage"].map(ts._norm4)
    gt = gt[gt["T_true"] != ""].drop_duplicates("pid", keep="first")[["pid", "T_stage", "T_true"]]
    train_df = feat_df.drop(columns=["T_stage"], errors="ignore").merge(gt[["pid", "T_stage"]], on="pid", how="left")
    labelled = train_df["T_stage"].notna()
    print(f"Patients with a usable T_stage label: {int(labelled.sum())}/{len(train_df)}")
    lab_df = feat_df.merge(gt[["pid", "T_true"]], on="pid", how="inner")
    print("Label distribution:", lab_df["T_true"].value_counts().sort_index().to_dict())
    sanity_report(feat_df, lab_df)

    work = train_df[labelled].copy().reset_index(drop=True)
    work["_T"] = work["T_stage"].map(ts._norm4)
    work = work[work["_T"].isin(ts._T_LABELS)].reset_index(drop=True)
    ml_possible = len(work) >= 20 and work["_T"].nunique() >= 2
    final_pkl = ckpt_dir / "t_stage_model.pkl"
    bundle, oof = None, {}
    results_cv = {}

    if not ml_possible:
        print("\nWARNING: <20 usable labelled patients -> rule-based staging only (no metrics).")
        ts._ML_MODEL = None
        ts._load_ml_model = lambda: None
    else:

        print("\n" + "=" * 78 + "\n  REPEATED CROSS-VALIDATION (model selection happens inside each training fold)\n" + "=" * 78)
        y_idx = work["_T"].map(ts._T_TO_IDX).to_numpy()
        k = min(args.folds, int(np.bincount(y_idx).min()))
        if k < 2:
            print("  Skipped: a class has <2 patients.")
        else:
            cvs = RepeatedStratifiedKFold(n_splits=k, n_repeats=args.repeats, random_state=args.seed)
            modes = list(ts.OVERRIDE_MODES)
            mode_preds = {m: {r: {} for r in range(args.repeats)} for m in modes}
            probs = {r: {} for r in range(args.repeats)}
            tsets = {r: {} for r in range(args.repeats)}
            for i, (tr, te) in enumerate(cvs.split(work, y_idx)):
                r = i // k
                tr_df = work.iloc[tr].drop(columns=["_T"])
                te_df = work.iloc[te].drop(columns=["_T", "T_stage"])
                b = quiet(ts.train_t_stage_xgboost, tr_df, model_path=ckpt_dir / "_fold.pkl", seed=args.seed + i)
                if b is None:
                    sys.exit("ERROR: model training returned None inside cross-validation.")
                ts._ML_MODEL = b
                res = ts.batch_stage_patients(te_df)
                feat_by_pid = {str(d["pid"]): d for d in te_df.to_dict("records")}
                for _, row in res.iterrows():
                    for m in modes:
                        st = ts.apply_override(feat_by_pid[str(row["pid"])], row["predicted_T_ml"], m)[0]
                        mode_preds[m][r][row["pid"]] = ts._T_TO_IDX[st]
                    probs[r][row["pid"]] = [row[f"p_{t}"] for t in ts._T_LABELS]
                    tsets[r][row["pid"]] = str(row.get("T_set", "") or "")
                if i % k == k - 1:
                    print(f"  repeat {r + 1}/{args.repeats} done")
            (ckpt_dir / "_fold.pkl").unlink(missing_ok=True)

            truth = dict(zip(work["pid"].astype(str), y_idx))
            truth_lbl = {p: ts._T_LABELS[v] for p, v in truth.items()}
            noev = ts._no_tumor_evidence(work).to_numpy()
            pid_list = work["pid"].astype(str).tolist()
            size = pd.to_numeric(work.get("tumor_size_2d_inplane_mm"), errors="coerce").fillna(0.0).to_numpy()
            diag = pd.to_numeric(work.get("tumor_size_box_diag_mm"), errors="coerce").fillna(0.0).to_numpy() \
                if "tumor_size_box_diag_mm" in work.columns else None
            rule = ts.rule_t_from_size(size).astype(int) - 1
            maj = np.zeros(len(work), dtype=int)
            yt_all = np.array([truth[p] for p in pid_list])


            def _mode_score(m):
                sc = []
                for r in range(args.repeats):
                    mr = metric_row(yt_all, np.array([mode_preds[m][r][p] for p in pid_list]))
                    sc.append(0.5 * mr["qwk"] + 0.5 * mr["macro_f1"])
                return float(np.mean(sc))
            mode_scores = {m: _mode_score(m) for m in modes}
            if args.override == "auto":
                chosen_mode = modes[0]
                for m in modes[1:]:
                    if mode_scores[m] > mode_scores[chosen_mode] + 0.003:
                        chosen_mode = m
            else:
                chosen_mode = args.override
            os.environ["TNM_OVERRIDE"] = chosen_mode
            print("\n  IASLC override policy (0.5*QWK + 0.5*macro-F1, mean over repeats): "
                  + ", ".join(f"{m}={v:.3f}" for m, v in mode_scores.items()) + f"  ->  using '{chosen_mode}'")
            preds = mode_preds[chosen_mode]
            for subset_name, mask in [("ALL labelled patients", np.ones(len(work), bool)),
                                      ("patients WITH an extracted tumor", ~noev)]:
                if mask.sum() < 10:
                    continue
                table = {}
                rows = []
                for r in range(args.repeats):
                    yp = np.array([preds[r][p] for p in pid_list])
                    ph = np.array([probs[r][p][2] + probs[r][p][3] for p in pid_list])
                    rows.append(metric_row(yt_all[mask], yp[mask], ph[mask]))
                table["MODEL (cross-validated)"] = summarize(rows)
                for m in modes:
                    rows_m = [metric_row(yt_all[mask], np.array([mode_preds[m][r][p] for p in pid_list])[mask],
                                         np.array([probs[r][p][2] + probs[r][p][3] for p in pid_list])[mask])
                              for r in range(args.repeats)]
                    table[f"  variant: override={m}"] = summarize(rows_m)
                table["baseline: always T1"] = summarize([metric_row(yt_all[mask], maj[mask])])

                rows_th = []
                cv_th = RepeatedStratifiedKFold(n_splits=k, n_repeats=args.repeats, random_state=args.seed)
                th_pred = [np.zeros(len(work), int) for _ in range(args.repeats)]
                szf = np.where(np.isfinite(size), size, 0.0)
                for i_f, (tr_i, te_i) in enumerate(cv_th.split(szf, yt_all)):
                    th_pred[i_f // k][te_i] = _pred_th(szf[te_i], learn_thresholds(szf[tr_i], yt_all[tr_i]))
                for tp in th_pred:
                    rows_th.append(metric_row(yt_all[mask], tp[mask], np.clip(szf[mask], 0, 250)))
                table["baseline: learned 3 size cut-offs"] = summarize(rows_th)
                table["baseline: IASLC size rule"] = summarize([metric_row(yt_all[mask], rule[mask], np.clip(size[mask], 0, 200))])
                print_table(subset_name + f" (n={int(mask.sum())})", table)
                results_cv[subset_name] = table

            metric_rows = []
            for subset_name, table in results_cv.items():
                for model_name, metrics in table.items():
                    row = {"subset": subset_name, "model": model_name}
                    for metric, (mean, sd) in metrics.items():
                        row[metric + "_mean"] = mean
                        row[metric + "_sd"] = sd
                    metric_rows.append(row)
            pd.DataFrame(metric_rows).to_csv(out_dir / "cross_validated_model_comparison.csv", index=False)

            for p in pid_list:
                v = [preds[r][p] for r in range(args.repeats)]
                oof[p] = ts._T_LABELS[int(pd.Series(v).mode().iloc[0])]
            cov = np.mean([[truth_lbl[p] in tsets[r][p].split("|") for p in pid_list] for r in range(args.repeats)]) if all(
                tsets[0].get(p) for p in pid_list) else float("nan")
            avg_set = np.mean([[len(tsets[r][p].split("|")) for p in pid_list] for r in range(args.repeats)]) if not np.isnan(cov) else float("nan")
            if not np.isnan(cov):
                print(f"\n  Conformal prediction sets (target coverage {100 * (1 - 0.10):.0f}%): observed coverage {cov:.1%}, mean set size {avg_set:.2f} stages")
            conf = np.mean([[probs[r][p][preds[r][p]] for p in pid_list] for r in range(args.repeats)])
            acc = np.mean([np.mean([preds[r][p] == truth[p] for p in pid_list]) for r in range(args.repeats)])
            print(f"\n  Mean predicted confidence {conf:.2f} vs accuracy {acc:.2f}  (a big gap = overconfident model)")


        print("\n" + "=" * 78 + "\n  FINAL MODEL (all labelled patients)\n" + "=" * 78)
        bundle = ts.train_t_stage_xgboost(train_df[labelled].copy(), model_path=final_pkl, seed=args.seed)
        if bundle and bundle.get("label_issues"):
            pd.DataFrame(bundle["label_issues"]).to_csv(out_dir / "label_issues.csv", index=False)
            print(f"  {len(bundle['label_issues'])} suspect labels written to {out_dir / 'label_issues.csv'} "
                  f"(look at these CTs with inspect_boxes.py)")
        if bundle is None:
            sys.exit("ERROR: final model training failed.")
        ts._ML_MODEL = bundle
        os.environ["TNM_T_MODEL_PATH"] = str(final_pkl)


    results = ts.batch_stage_patients(feat_df.drop(columns=["T_stage", "source", "_shard_no"], errors="ignore"))
    results["pid"] = results["pid"].astype(str)
    results = results.rename(columns={"predicted_T": "predicted_T_final"})
    out = (results.merge(feat_df[["pid", "source"]], on="pid", how="left")
                  .merge(gt.rename(columns={"T_true": "true_T"})[["pid", "true_T"]], on="pid", how="left"))
    out["predicted_T_crossval"] = out["pid"].map(oof) if oof else np.nan
    n_noev = int(out["no_tumor_evidence"].sum())
    if n_noev:
        print(f"\n  WARNING: {n_noev} patient(s) had NO tumor evidence and were set to T1 (confidence 0.20). "
              f"Fix the extraction for these instead of trusting the T1.")
    cols = ["pid", "source", "true_T", "predicted_T_final", "predicted_T_crossval", "T_confidence", "T_borderline",
            "predicted_T_ml", "p_T1", "p_T2", "p_T3", "p_T4", "T_set", "tumor_size_mm", "overall_stage", "T_deciding", "T_anatomy", "multifocal",
            "lung_invaded", "medi_invaded", "heart_invaded", "vessels_invaded", "no_tumor_evidence"]
    out_csv = out_dir / "final_t_staging_predictions.csv"
    out[[c for c in cols if c in out.columns]].to_csv(out_csv, index=False)
    print(f"\n  Predicted T distribution (all {len(out)}): {out['predicted_T_final'].value_counts().sort_index().to_dict()}")
    print(f"  Saved: {out_csv}")


    if ml_possible:
        lab = out[out["true_T"].notna()]
        cv = lab[lab["predicted_T_crossval"].notna()]
        if len(cv):
            report_metrics(ev, cv["true_T"].tolist(), cv["predicted_T_crossval"].tolist(),
                           "T-Stage - CROSS-VALIDATED, modal over repeats (report this)", out_dir / "cm_T_crossval.png")
        report_metrics(ev, lab["true_T"].tolist(), lab["predicted_T_final"].tolist(),
                       "T-Stage - TRAINING DATA (optimistic, do not report)", out_dir / "cm_T_train.png")
        print(f"\n  Final model: {bundle['model_type']} {bundle['best_config']} | inner-CV score {bundle.get('cv_macro_f1')}")
        print(f"  Model file: {final_pkl}")
    print(f"  Outputs in: {out_dir}")
    print("\n  NOTE: N and M are not estimated (defaults N0/M0): 'overall_stage' is a T-only convenience value.")


if __name__ == "__main__":
    main()
