import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run(cmd, label):
    print(f"\n--- {label}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(HERE))
    if r.returncode != 0:
        sys.exit(f"{label} failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", default=str(HERE / "sample_scan"))
    ap.add_argument("--model", default=str(HERE / "model" / "t_stage_model.pkl"))
    ap.add_argument("--ckpt_dir", required=True, help="TumorNet checkpoint folder")
    ap.add_argument("--fmcib_python", default=sys.executable, help="python of the FMCIB (3.10) environment")
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--cv_predictions", default="")
    a = ap.parse_args()

    scan = Path(a.scan_dir)
    pidf = scan / "pid.txt"
    pid = pidf.read_text().strip() if pidf.exists() else next((p.name for p in (scan / "Lung-PET-CT-Dx").iterdir() if p.is_dir()), None)
    if not pid or not Path(a.model).exists():
        sys.exit("sample_scan or model missing - run prepare_sample_scan.py first")
    W = scan / "_work"
    cache, fts, fxml = W / "cache", W / "features_ts", W / "features_xml"
    box, solid, fm = W / "box_seg_features.csv", W / "box_solid_features.csv", W / "fmcib.csv"
    W.mkdir(exist_ok=True)
    py = sys.executable

    if not (fxml / "patient_features_shard000.zip").exists():
        if not (fts / "patient_features_shard000.zip").exists():
            run([py, "local_shard_features.py", "--shards_dir", scan, "--shard_num", 0, "--ckpt_dir", a.ckpt_dir, "--out_dir", fts,
                 "--merge_ts_masks", "--align_ts_axes", "--cache_dir", cache], "TotalSegmentator organ masks")
        run([py, "add_boxes_to_cache.py", "--shard_dir", scan, "--cache_dir", cache, "--shard_num", 0], "radiologist boxes -> cache")
        run([py, "local_shard_features.py", "--shards_dir", scan, "--shard_num", 0, "--from_cache", "--cache_dir", cache, "--out_dir", fxml,
             "--tumor_source", "xml"], "patient features")
    if not box.exists():
        run([py, "box_seg_sizes.py", "--shard_dirs", scan, "--ckpt_dir", a.ckpt_dir, "--out_csv", box], "TumorNet inside the box")
    if not solid.exists():
        run([py, "box_solid_sizes.py", "--shard_dirs", scan, "--out_csv", solid], "HU solid component")
    if not fm.exists():
        run([a.fmcib_python, "extract_fmcib_features.py", "--shard_dirs", scan, "--out_csv", fm, "--patch_dir", W / "fmcib_patches",
             "--batch_size", "1", "--workers", "0"], "FMCIB embedding")


    os.environ.update(TNM_T_MODEL_PATH=str(a.model), TNM_OVERRIDE="none", TNM_FEATURE_PROFILE="image")
    import tnm_staging as ts
    from final_t_staging import load_zip
    _, f, _ = load_zip(fxml / "patient_features_shard000.zip")
    f["pid"] = f["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    f = f[f["pid"] == pid.replace("Lung_Dx_", "Lung_Dx-")].reset_index(drop=True)
    if f.empty:
        sys.exit("the patient is missing from the feature file")
    emb = pd.read_csv(fm)
    emb["pid"] = emb["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    cols = [c for c in emb.columns if re.fullmatch(r"pred_\d+", str(c))]
    emb = emb[["pid", *cols]].rename(columns={c: "fmcib_" + c for c in cols}).drop_duplicates("pid")
    f = f.drop(columns=[c for c in f.columns if str(c).startswith("fmcib_")], errors="ignore").merge(emb, on="pid", how="left")
    f, _ = ts.merge_box_features(f, box, min_coverage=0.0)
    f, _ = ts.merge_box_features(f, solid, min_coverage=0.0)

    r = ts.classify_t_stage(f.iloc[0].to_dict())
    out = {"pid": pid, "T_stage": r.t_stage, "probabilities": {k: round(v, 3) for k, v in r.uncertainty_probs.items()},
           "conformal_set_90pct": r.t_set, "confidence": round(r.confidence, 3), "tumour_size_mm": round(r.tumor_size_mm, 1),
           "how": r.deciding_factor, "anatomy": r.anatomy_evidence, "no_tumour_evidence": bool(r.no_tumor_evidence)}
    if a.labels_csv and Path(a.labels_csv).exists():
        lab = pd.read_csv(a.labels_csv)
        lab["pid"] = lab["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
        m = lab[lab["pid"] == out["pid"]]
        if len(m):
            out["recorded_T_stage"] = str(m.iloc[0]["T_stage"])
    if a.cv_predictions and Path(a.cv_predictions).exists():
        cv = pd.read_csv(a.cv_predictions)
        m = cv[cv["pid"].astype(str) == out["pid"]]
        if len(m):
            out["heldout_crossvalidated_T"] = str(m.iloc[0].get("predicted_T_crossval", ""))
    (scan / "result.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 60)
    print(f"Patient {out['pid']}  ->  predicted T-stage: {out['T_stage']}")
    for k, v in out.items():
        if k not in ("pid", "T_stage"):
            print(f"  {k}: {v}")
    print("=" * 60 + f"\nsaved {scan / 'result.json'}")


if __name__ == "__main__":
    main()