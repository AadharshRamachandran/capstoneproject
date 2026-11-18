import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ALL = {"box", "seg", "solid", "anat", "emb"}
VARIANTS = [("full", set()),
            ("add: box only", ALL - {"box"}),
            ("add: box + TumorNet", ALL - {"box", "seg"}),
            ("add: + HU solid", ALL - {"box", "seg", "solid"}),
            ("add: + anatomy (= minus embedding)", {"emb"}),
            ("minus box", {"box"}), ("minus TumorNet", {"seg"}), ("minus solid", {"solid"}), ("minus anatomy", {"anat"})]


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--fmcib_csv", required=True)
    ap.add_argument("--box_features", default="")
    ap.add_argument("--solid_features", default="")
    ap.add_argument("--out_root", default="ablation_v5log2")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="comma list of variant names (or slugs) to run")
    ap.add_argument("--summarize_only", action="store_true", help="do not run anything; only build the summary table from the finished folders")
    ap.add_argument("--code_dir", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    only = {slug(x) for x in a.only.split(",") if x}
    t0 = time.time()
    rows = []
    for name, drop in VARIANTS:
        if only and slug(name) not in only:
            continue
        out = root / slug(name)
        csv = out / "cross_validated_model_comparison.csv"
        if not csv.exists() and a.summarize_only:
            print(f"{name}: not finished (no {csv.name})")
            continue
        if not csv.exists():
            cmd = [sys.executable, str(Path(a.code_dir) / "final_t_staging.py"), "--features_dir", a.features_dir, "--labels_csv", a.labels_csv,
                   "--fmcib_csv", a.fmcib_csv, "--out_dir", str(out), "--profile", "image", "--override", "none", "--repeats", str(a.repeats),
                   "--code_dir", a.code_dir]
            if a.box_features:
                cmd += ["--box_features", a.box_features]
            if a.solid_features:
                cmd += ["--solid_features", a.solid_features]
            env = dict(os.environ, TNM_ABLATE=",".join(sorted(drop)), PYTHONUNBUFFERED="1")
            print(f"\n=== {name}  (blank: {sorted(drop) or 'nothing'}) ===", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "log.txt", "w", encoding="utf-8") as lg:
                rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not csv.exists():
                print(f"  FAILED (see {out / 'log.txt'})")
                continue
        d = pd.read_csv(csv)
        d = d[(d["model"] == "MODEL (cross-validated)") & (d["subset"].str.startswith("ALL"))]
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append({"variant": name, "blanked": ",".join(sorted(drop)) or "-", **{m: r.get(m + "_mean") for m in ("acc", "macro_f1", "qwk", "within1", "auc_T34")},
                     "acc_sd": r.get("acc_sd")})
        pf = out / "final_t_staging_predictions.csv"
        if pf.exists():
            try:
                from sklearn.metrics import accuracy_score, f1_score
                pr = pd.read_csv(pf)
                pr = pr[pr["true_T"].isin(["T1", "T2", "T3", "T4"])]
                yt, yp = pr["true_T"].str[1].astype(int), pr["predicted_T_crossval"].astype(str).str[1].astype(int)
                rows[-1].update(acc_vote=accuracy_score(yt, yp), macro_f1_vote=f1_score(yt, yp, average="macro"), weighted_f1_vote=f1_score(yt, yp, average="weighted"))
            except Exception as e:
                print("  (vote metrics skipped:", e, ")")
        x = rows[-1]
        print(f"  acc={x['acc']:.3f}±{x['acc_sd']:.3f} f1={x['macro_f1']:.3f} qwk={x['qwk']:.3f} within1={x['within1']:.3f} auc34={x['auc_T34']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if rows:
        res = pd.DataFrame(rows)
        if (res.variant == "full").any():
            base = float(res.loc[res.variant == "full", "acc"].iloc[0])
            res["d_acc_vs_full"] = res["acc"] - base
        res.to_csv(root / "ablation_v5log2_summary.csv", index=False)
        pd.set_option("display.width", 200)
        print("\n" + res.round(3).to_string(index=False) + f"\n\nsaved {root / 'ablation_v5log2_summary.csv'}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
