import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(r"C:\Users\HP\Downloads")
BASE_DIR    = PROJECT_DIR / "lung_data" / "laptop_dataset"
WORK_DIR    = PROJECT_DIR / "working"
DEFAULT_MSD = PROJECT_DIR / "working" / "msd_processed"

PREPROC_DIR = WORK_DIR / "preprocessed"
CKPT_DIR    = WORK_DIR / "checkpoints"
RESULTS_DIR = WORK_DIR / "results"
FEAT_DIR    = WORK_DIR / "features"
EXPORT_DIR  = WORK_DIR / "dataset_export"

for _d in [PREPROC_DIR, CKPT_DIR, RESULTS_DIR, FEAT_DIR, EXPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = 0
PIN_MEMORY  = True
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_runtime_info():
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("  WARNING: No GPU detected — training will be very slow.")
    print(f"Workers: {NUM_WORKERS}  (Windows - using 0)")

def load_manifest(data_dir: Path) -> pd.DataFrame:
    p = EXPORT_DIR / "patient_manifest.csv"
    if p.exists():
        df = pd.read_csv(p)
        print(f"Loaded manifest: {len(df)} patients from {p}")
        return df

    print(f"Manifest not found, scanning {data_dir} ...")
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    stage_map: dict = {}
    for csv_name in ["slice_manifest.csv", "series_manifest.csv"]:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            continue
        tmp    = pd.read_csv(csv_path)
        id_col = next((c for c in ["pid", "patient_id", "PatientID"]
                       if c in tmp.columns), None)
        if id_col is None:
            continue
        for _, row in tmp.iterrows():
            pid = str(row[id_col])
            if pid not in stage_map:
                stage_map[pid] = {
                    "T_stage":     str(row.get("T_stage", row.get("t_stage", "Unknown"))),
                    "N_stage":     str(row.get("N_stage", row.get("n_stage", "Unknown"))),
                    "M_stage":     str(row.get("M_stage", row.get("m_stage", "Unknown"))),
                    "xml_working": str(row.get("xml_working", "")),
                }
        print(f"  Stage labels loaded from {csv_name}: {len(stage_map)} entries")
        break

    ct_slices_dir = data_dir / "ct_slices"
    scan_root     = ct_slices_dir if ct_slices_dir.exists() else data_dir

    records = []
    for pid_dir in sorted(scan_root.iterdir()):
        if not pid_dir.is_dir() or pid_dir.name in ("annotations", "ct_slices"):
            continue
        pid       = pid_dir.name
        extra     = stage_map.get(pid, {})
        ann_dir   = data_dir / "annotations" / pid
        xml_paths = sorted(ann_dir.rglob("*.xml")) if ann_dir.exists() else []
        records.append({
            "pid":              pid,
            "dicom_root_input": str(pid_dir),
            "T_stage":          extra.get("T_stage", "Unknown"),
            "N_stage":          extra.get("N_stage", "Unknown"),
            "M_stage":          extra.get("M_stage", "Unknown"),
            "xml_working":      extra.get("xml_working",
                                          "|".join(str(x) for x in xml_paths)),
        })

    if not records:
        print(f"ERROR: No patient folders found under {scan_root}")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(p, index=False)
    print(f"Built manifest: {len(df)} patients")
    print(f"  Sample IDs   : {df['pid'].tolist()[:5]}")
    print(f"  T_stage known: {(df['T_stage'] != 'Unknown').sum()}/{len(df)}")
    return df

def step_preprocess(manifest: pd.DataFrame, fast_totalseg: bool = False) -> pd.DataFrame:
    print("\n" + "█" * 55)
    print("  STEP 1: Preprocessing + TotalSegmentator")
    print("█" * 55)

    if not fast_totalseg:
        try:
            vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if torch.cuda.is_available() else 0)
            if vram < 8.0:
                fast_totalseg = True
                print(f"  Auto-enabled fast TotalSegmentator (VRAM={vram:.1f} GB < 8 GB)")
        except Exception:
            pass

    rec_path   = PREPROC_DIR / "preprocess_records.csv"
    valid_pids = set(manifest["pid"].astype(str))

    if rec_path.exists():
        existing  = pd.read_csv(rec_path)
        done_pids = set(
            existing[
                (existing["status"] == "ok") &
                existing["pid"].astype(str).isin(valid_pids)
            ]["pid"].astype(str)
        )
        remaining = manifest[~manifest["pid"].astype(str).isin(done_pids)]
        if remaining.empty:
            print(f"  All {len(done_pids)} patients already preprocessed.")
            return existing
        print(f"  Already done: {len(done_pids)}, remaining: {len(remaining)}")
    else:
        remaining = manifest
        existing  = pd.DataFrame()

    try:
        import totalsegmentator
        print("  TotalSegmentator: already installed")
    except ImportError:
        import subprocess
        print("  Installing TotalSegmentator...")
        subprocess.run(["pip", "install", "TotalSegmentator", "--quiet",
                        "--break-system-packages"], check=False)

    from data_pipeline import preprocess_patient_full
    new_records = []
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        print(f"  [{i}/{len(remaining)}] Processing: {row['pid']}")
        new_records.append(
            preprocess_patient_full(row, str(PREPROC_DIR), fast_totalseg=fast_totalseg)
        )

    all_recs = (
        pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
        if not existing.empty else pd.DataFrame(new_records)
    )
    all_recs = (all_recs[all_recs["pid"].astype(str).isin(valid_pids)]
                .drop_duplicates(subset=["pid"], keep="last")
                .reset_index(drop=True))
    all_recs.to_csv(rec_path, index=False)

    ok  = (all_recs["status"] == "ok").sum()
    err = (all_recs["status"] != "ok").sum()
    print(f"\n  Preprocessed: {len(all_recs)}  Success: {ok}  Errors: {err}")
    if err > 0:
        for _, r in all_recs[all_recs["status"] != "ok"].iterrows():
            print(f"    {r['pid']}: {r['status']}")
    return all_recs

def step_train_tumornet(msd_dir: Path, skip_3d: bool = False):
    print("\n" + "█" * 55)
    print("  STEP 3: Train TumorNet (on MSD Task06_Lung)")
    print("█" * 55)

    from data_pipeline import CachedTumorSegDataset2D, TumorSegDataset3D
    from models import get_segmentation_model, AttentionUNet3D
    from training import SegmentationTrainer, FocalDiceLoss, DiceLoss3D, seg_metrics
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

    msd_manifest = msd_dir / "msd_slice_manifest.csv"
    if not msd_manifest.exists():
        print(f"  MSD manifest not found: {msd_manifest}")
        print(f"  Run: python msd_loader.py --msd_dir <Task06_Lung> --out_dir {msd_dir}")
        return {}, {}

    import random as _rng
    records = pd.read_csv(msd_manifest).to_dict("records")
    n       = len(records)
    _rng.seed(42)
    _rng.shuffle(records)
    n_test  = max(1, int(n * 0.10))
    n_val   = max(1, int(n * 0.20))
    test_r  = records[:n_test]
    val_r   = records[n_test: n_test + n_val]
    train_r = records[n_test + n_val:]
    print(f"  MSD slices: {n}  train: {len(train_r)}  val: {len(val_r)}  test: {len(test_r)}")

    ckpt_2d   = CKPT_DIR / "TumorNet_2D"
    best_ckpt = ckpt_2d / "TumorNet_2D_fold0_best.pth"
    metrics_2d: dict = {}
    hist_2d:    dict = {}

    if best_ckpt.exists():
        print(f"\n  TumorNet 2D checkpoint found — skipping training.")
        print(f"  Checkpoint: {best_ckpt}")
        mp = RESULTS_DIR / "tumornet_2d_metrics.csv"
        if mp.exists():
            metrics_2d = pd.read_csv(mp).iloc[0].to_dict()
            print(f"  Loaded metrics: DSC={metrics_2d.get('DSC', 0)*100:.2f}%  "
                  f"IoU={metrics_2d.get('IoU', 0)*100:.2f}%")
    else:
        print("\n  Training TumorNet 2D  (ResNet152 + FPN) ...")
        print("  LR: enc=3e-5, dec=3e-4 | WD=1e-4 | BS=16 | Epochs=100 | Patience=20")
        print("  Loss: FocalDice | Opt: AdamW (layer-wise) | Sched: CosineAnnealingWarmRestarts T0=30")

        train_ds = CachedTumorSegDataset2D(train_r, "train")
        val_ds   = CachedTumorSegDataset2D(val_r,   "val")
        test_ds  = CachedTumorSegDataset2D(test_r,  "test")
        kw       = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,  **kw)
        val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, **kw)
        test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False, **kw)

        model_2d = get_segmentation_model("tumornet_2d", pretrained=True,
                                           mc_dropout=False).to(DEVICE)
        enc_p = [p for n, p in model_2d.named_parameters() if "encoder" in n]
        dec_p = [p for n, p in model_2d.named_parameters() if "encoder" not in n]
        opt_2d = AdamW([{"params": enc_p, "lr": 3e-5},
                        {"params": dec_p, "lr": 3e-4}], weight_decay=1e-4)
        sched_2d = CosineAnnealingWarmRestarts(opt_2d, T_0=30, T_mult=2, eta_min=1e-7)

        ckpt_2d.mkdir(parents=True, exist_ok=True)
        trainer_2d = SegmentationTrainer(
            model=model_2d,
            loss_fn=FocalDiceLoss(gamma=2.0, alpha=0.75, w_focal=0.4, w_dice=0.6),
            optimizer=opt_2d, scheduler=sched_2d, device=DEVICE,
            output_dir=str(ckpt_2d), model_name="TumorNet_2D",
            max_epochs=50, patience=20, use_amp=(DEVICE.type == "cuda"),
        )
        hist_2d = trainer_2d.fit(train_dl, val_dl, fold=0)
        print("\n  Tuning decision threshold on val set ...")
        best_thresh, best_dsc = 0.5, 0.0
        model_2d.eval()
        with torch.no_grad():
            vl_all, vm_all = [], []
            for batch in val_dl:
                imgs, msks = batch[0].to(DEVICE), batch[1].to(DEVICE)
                out   = model_2d(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                vl_all.append(logits.cpu())
                vm_all.append(msks.cpu())
            vl_cat = torch.cat(vl_all)
            vm_cat = torch.cat(vm_all)
        for thresh in [t / 100 for t in range(30, 56, 5)]:
            d = seg_metrics(vl_cat, vm_cat, thresh=thresh)["DSC"]
            print(f"    thresh={thresh:.2f}  val DSC={d*100:.2f}%")
            if d > best_dsc:
                best_dsc, best_thresh = d, thresh
        print(f"  Best threshold: {best_thresh:.2f}  (val DSC={best_dsc*100:.2f}%)")
        torch.save({"threshold": best_thresh}, ckpt_2d / "TumorNet_2D_threshold.pth")

        metrics_2d = trainer_2d.evaluate(test_dl)
        pd.DataFrame([{**metrics_2d, "threshold": best_thresh}]).to_csv(
            RESULTS_DIR / "tumornet_2d_metrics.csv", index=False)

        print("\n" + "─" * 55)
        print("  TumorNet 2D — Test Set Results")
        print("─" * 55)
        for k, v in metrics_2d.items():
            if isinstance(v, float):
                if k == "threshold":
                    print(f"    {k:<16}: {v:.2f}")
                else:
                    print(f"    {k:<16}: {v*100:.2f}%")
        print("─" * 55)

    metrics_3d: dict = {}
    hist_3d:    dict = {}
    if not skip_3d:
        ckpt_3d      = CKPT_DIR / "TumorNet_3D"
        vol_manifest = msd_dir / "msd_volume_manifest.csv"
        if (ckpt_3d / "TumorNet_3D_fold0_best.pth").exists():
            print("\n  TumorNet 3D checkpoint found — skipping training.")
        elif vol_manifest.exists():
            print("\n  Training TumorNet 3D  (AttentionUNet3D) ...")
            vrecs  = pd.read_csv(vol_manifest).to_dict("records")
            nv     = len(vrecs)
            nvt    = max(1, int(nv * 0.10))
            nvv    = max(1, int(nv * 0.20))
            kw     = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            tr3_dl = DataLoader(TumorSegDataset3D(vrecs[nvt + nvv:], "train"),
                                batch_size=2, shuffle=True, **kw)
            va3_dl = DataLoader(TumorSegDataset3D(vrecs[nvt: nvt + nvv], "val"),
                                batch_size=2, shuffle=False)
            te3_dl = DataLoader(TumorSegDataset3D(vrecs[:nvt], "test"),
                                batch_size=2, shuffle=False)
            model_3d = AttentionUNet3D(1, 1).to(DEVICE)
            opt_3d   = AdamW(model_3d.parameters(), lr=1e-4, weight_decay=1e-4)
            ckpt_3d.mkdir(parents=True, exist_ok=True)
            trainer_3d = SegmentationTrainer(
                model=model_3d, loss_fn=DiceLoss3D(), optimizer=opt_3d,
                scheduler=ReduceLROnPlateau(opt_3d, patience=5, factor=0.5),
                device=DEVICE, output_dir=str(ckpt_3d), model_name="TumorNet_3D",
                max_epochs=50, patience=15, use_amp=(DEVICE.type == "cuda"),
            )
            hist_3d    = trainer_3d.fit(tr3_dl, va3_dl, fold=0)
            metrics_3d = trainer_3d.evaluate(te3_dl)
            pd.DataFrame([metrics_3d]).to_csv(RESULTS_DIR / "tumornet_3d_metrics.csv",
                                              index=False)
            print("\n  TumorNet 3D — Test Set Results")
            for k, v in metrics_3d.items():
                if isinstance(v, float):
                    print(f"    {k:<16}: {v*100:.2f}%")
        else:
            print("  3D manifest not found — skipping 3D training.")
    else:
        print("  Skipping 3D training (--skip_3d).")

    return {"2D": metrics_2d, "3D": metrics_3d}, {"2D": hist_2d, "3D": hist_3d}

def step_run_tumornet_inference(manifest: pd.DataFrame, force_reinfer: bool = False):
    """
    Run TumorNet 2D on every patient's preprocessed volume.
    
    BUG FIX
    not full chest CT slices. Before inference, crop each slice to the lung region
    (from TotalSegmentator masks) to match the training distribution.
    """
    ckpt_path = CKPT_DIR / "TumorNet_2D" / "TumorNet_2D_fold0_best.pth"
    if not ckpt_path.exists():
        print("  TumorNet checkpoint not found — skipping inference.")
        return

    from models import get_segmentation_model
    import cv2 as _cv2

    thresh_path = CKPT_DIR / "TumorNet_2D" / "TumorNet_2D_threshold.pth"
    threshold   = 0.5
    if thresh_path.exists():
        threshold = float(
            torch.load(str(thresh_path), map_location="cpu",
                       weights_only=True).get("threshold", 0.5)
        )
    effective_threshold = min(threshold, 0.25)
    print(f"  TumorNet inference threshold: {effective_threshold:.2f} (raw={threshold:.2f}, mode=recall-priority)")

    model = get_segmentation_model("tumornet_2d", pretrained=False,
                                    mc_dropout=False).to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    spacing_map: dict = {}
    rec_path = PREPROC_DIR / "preprocess_records.csv"
    if rec_path.exists():
        for _, r in pd.read_csv(rec_path).iterrows():
            spacing_map[str(r["pid"])] = float(r.get("slice_thickness", 5.0) or 5.0)

    tumor_dir = PREPROC_DIR / "tumor_masks"
    tumor_dir.mkdir(parents=True, exist_ok=True)
    vol_dir   = PREPROC_DIR / "volumes"
    anat_dir  = PREPROC_DIR / "anatomy_masks"

    IMG_SIZE          = 256
    TARGET_SPACING_MM = 0.75
    BATCH_SIZE        = 8
    TUMOR_PADDING     = 20
    done = skipped = 0

    for _, row in manifest.iterrows():
        pid       = str(row["pid"])
        vol_path  = vol_dir   / f"{pid}.npy"
        mask_path = tumor_dir / f"{pid}_tumor.npy"

        if (not force_reinfer) and mask_path.exists():
            if np.load(str(mask_path)).sum() > 0:
                skipped += 1
                continue

        if not vol_path.exists():
            continue

        vol     = np.load(str(vol_path))
        D, H, W = vol.shape
        vol_f   = (
            (vol.astype(np.float32) / 255.0) if vol.dtype == np.uint8
            else ((vol.astype(np.float32) - vol.min()) /
                  max(float(vol.max() - vol.min()), 1e-8))
        )
        lung_mask = None
        lung_l_path = anat_dir / f"{pid}_lung_left.npy"
        lung_r_path = anat_dir / f"{pid}_lung_right.npy"
        if lung_l_path.exists() and lung_r_path.exists():
            try:
                lung_l = np.load(str(lung_l_path)) > 0
                lung_r = np.load(str(lung_r_path)) > 0
                lung_mask = lung_l | lung_r
            except Exception:
                lung_mask = None

        st     = spacing_map.get(pid, 5.0)
        stride = max(1, int(round(TARGET_SPACING_MM / max(st, 0.5))))
        sampled = list(range(0, D, stride))
        def prep_input_slice(slc_2d: np.ndarray, d: int) -> np.ndarray:
            """Crop to lung ROI if available, else use full slice. Resize to IMG_SIZE."""
            if lung_mask is not None and d < lung_mask.shape[0]:
                mask_d = lung_mask[d]
                if mask_d.sum() > 100:
                    rows = np.where(np.any(mask_d, axis=1))[0]
                    cols = np.where(np.any(mask_d, axis=0))[0]
                    if len(rows) > 0 and len(cols) > 0:
                        pad = TUMOR_PADDING
                        r0 = max(0, rows[0] - pad)
                        r1 = min(slc_2d.shape[0], rows[-1] + pad + 1)
                        c0 = max(0, cols[0] - pad)
                        c1 = min(slc_2d.shape[1], cols[-1] + pad + 1)
                        slc_cropped = slc_2d[r0:r1, c0:c1]
                        return _cv2.resize(slc_cropped, (IMG_SIZE, IMG_SIZE))
            return _cv2.resize(slc_2d, (IMG_SIZE, IMG_SIZE))
        sampled_preds = np.zeros(len(sampled), dtype=np.uint8)
        with torch.no_grad():
            for b0 in range(0, len(sampled), BATCH_SIZE):
                bidx   = sampled[b0: b0 + BATCH_SIZE]
                batch  = [prep_input_slice(vol_f[d], d) for d in bidx]
                bt     = torch.from_numpy(np.stack(batch)[:, np.newaxis]).float().to(DEVICE)
                logits = model(bt)
                logits = logits[0] if isinstance(logits, tuple) else logits
                probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                for i, prob in enumerate(probs):
                    sampled_preds[b0 + i] = int((prob > effective_threshold).any())
        pred_full = np.zeros((D, H, W), dtype=np.uint8)
        with torch.no_grad():
            if stride == 1:
                for b0 in range(0, D, BATCH_SIZE):
                    bidx   = list(range(b0, min(b0 + BATCH_SIZE, D)))
                    batch  = [prep_input_slice(vol_f[d], d) for d in bidx]
                    bt     = torch.from_numpy(np.stack(batch)[:, np.newaxis]).float().to(DEVICE)
                    logits = model(bt)
                    logits = logits[0] if isinstance(logits, tuple) else logits
                    probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                    for i, (d, prob) in enumerate(zip(bidx, probs)):
                        pb = (prob > effective_threshold).astype(np.uint8)
                        if pb.shape != (H, W):
                            pb = _cv2.resize(pb, (W, H), interpolation=_cv2.INTER_NEAREST)
                        pred_full[d] = pb
            else:
                for si, d in enumerate(sampled):
                    if sampled_preds[si] == 0:
                        continue
                    slc    = prep_input_slice(vol_f[d], d)
                    t      = torch.from_numpy(slc[np.newaxis, np.newaxis]).float().to(DEVICE)
                    logits = model(t)
                    logits = logits[0] if isinstance(logits, tuple) else logits
                    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
                    pb     = (prob > effective_threshold).astype(np.uint8)
                    if pb.shape != (H, W):
                        pb = _cv2.resize(pb, (W, H), interpolation=_cv2.INTER_NEAREST)
                    pred_full[d] = pb
        from scipy.ndimage import binary_fill_holes
        from skimage.morphology import remove_small_objects
        pred_full_cleaned = pred_full.astype(bool)
        for d in range(D):
            if pred_full[d].sum() > 0:
                pred_full_cleaned[d] = binary_fill_holes(pred_full[d] > 0.5)
        pred_full_cleaned = remove_small_objects(pred_full_cleaned, min_size=5).astype(np.uint8)
        pred_full = pred_full_cleaned

        np.save(str(mask_path), pred_full)
        n_tumor = int((pred_full.sum(axis=(1, 2)) > 0).sum())
        print(f"  Inference: {pid}  tumor slices={n_tumor}/{D}"
              f"  stride={stride} (spacing={st:.1f} mm)")
        done += 1

    if force_reinfer:
        print(f"\n  TumorNet inference complete: {done} processed, cache skip disabled")
    else:
        print(f"\n  TumorNet inference complete: {done} processed, {skipped} skipped")

def step_extract_features(manifest: pd.DataFrame, force_reinfer: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  STEP 4: Feature Extraction (Pathway A)")
    print("=" * 55)

    print("  Running TumorNet inference on patient volumes ...")
    step_run_tumornet_inference(manifest, force_reinfer=force_reinfer)

    from feature_extraction import extract_all_patients

    feat_path  = FEAT_DIR / "patient_features.csv"
    valid_pids = set(manifest["pid"].astype(str))

    if force_reinfer and feat_path.exists():
        print("  --force_reinfer set: removing cached patient_features.csv")
        feat_path.unlink(missing_ok=True)

    if feat_path.exists():
        try:
            existing = pd.read_csv(feat_path)
        except pd.errors.EmptyDataError:
            print("  Cached feature CSV is empty — deleting and re-extracting.")
            feat_path.unlink(missing_ok=True)
            existing = None
        except Exception as e:
            print(f"  Cached feature CSV could not be read ({e}) — deleting and re-extracting.")
            feat_path.unlink(missing_ok=True)
            existing = None

        if existing is not None:
            if "pid" not in existing.columns:
                print("  Cached feature CSV is missing 'pid' — deleting and re-extracting.")
                feat_path.unlink(missing_ok=True)
            else:
                existing = existing[existing["pid"].astype(str).isin(valid_pids)]
                if len(existing) == len(manifest):
                    ip_col = "tumor_size_2d_inplane_mm"
                    if ip_col in existing.columns and existing[ip_col].sum() == 0:
                        print("  Cache has all-zero in-plane sizes — re-extracting.")
                        feat_path.unlink(missing_ok=True)
                    else:
                        print(f"  Cached features found — loading: {feat_path}")
                        print(f"  {len(existing)} patients, {len(existing.columns)} features")
                        return existing

    preproc_path = PREPROC_DIR / "preprocess_records.csv"
    if preproc_path.exists():
        preproc_df = pd.read_csv(preproc_path)
        cols = [c for c in ["pid", "pixel_spacing_x", "pixel_spacing_y", "slice_thickness"]
                if c in preproc_df.columns]
        manifest = manifest.merge(preproc_df[cols], on="pid", how="left")

    feat_df = extract_all_patients(manifest, str(PREPROC_DIR))
    feat_df.to_csv(feat_path, index=False)
    print(f"  Saved: {feat_path}  ({len(feat_df)} patients, {len(feat_df.columns)} features)")
    return feat_df

def step_tnm_staging(feat_df: pd.DataFrame, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    print("\n" + "█" * 55)
    print("  STEP 5: T Staging — Pathway A")
    print("█" * 55)

    from tnm_staging import batch_stage_patients, train_t_stage_xgboost
    from evaluation import (normalize_t, STAGE_LABELS,
                             compute_classification_metrics, print_metrics_table)

    out_path = RESULTS_DIR / "staging_pathwayA_2d.csv"
    train_df = feat_df.copy()
    if manifest is not None and "T_stage" in manifest.columns:
        gt = manifest[["pid", "T_stage"]].copy()
        if "T_stage" in train_df.columns:
            train_df = train_df.drop(columns=["T_stage"])
        train_df = train_df.merge(gt, on="pid", how="left")
    xgb_bundle = None
    try:
        xgb_bundle = train_t_stage_xgboost(train_df)
    except Exception as e:
        print(f"  XGB training skipped due to error: {e}")
    if xgb_bundle is not None:
        os.environ["TNM_T_MODEL_PATH"] = str(CKPT_DIR / "t_stage_xgb.pkl")
    results = batch_stage_patients(feat_df, use_3d=False)
    results.to_csv(out_path, index=False)
    print("\n  Staging Summary (Pathway A):")
    print(f"  {'PID':<12} {'T':<8} {'Stage':<14} {'T-Size':>9} {'T-Deciding'}")
    print("  " + "─" * 95)
    for _, r in results.iterrows():
        sz  = f"{r.get('tumor_size_mm', 0):.1f} mm" if pd.notna(r.get("tumor_size_mm")) else "N/A"
        dec = str(r.get("T_deciding", ""))[:40]
        print(f"  {str(r['pid']):<12} "
              f"{str(r.get('predicted_T', '')):<8} "
              f"{str(r.get('overall_stage', '')):<14} "
              f"{sz:>9}  {dec}")

    print(f"\n  T-Stage distribution:")
    if "predicted_T" in results.columns:
        for stg, cnt in results["predicted_T"].value_counts().sort_index().items():
            print(f"    {stg}: {cnt} patients")

    print(f"\n  Saved: {out_path}")
    _T_NORM4 = {
        "T1a": "T1", "T1b": "T1", "T1c": "T1", "T1": "T1",
        "T2a": "T2", "T2b": "T2", "T2": "T2",
        "T3":  "T3", "T4":  "T4", "TX": "T4",
    }
    if "T_stage" in feat_df.columns:
        merged  = results.merge(feat_df[["pid", "T_stage"]], on="pid", how="inner")
        y_true  = [_T_NORM4.get(normalize_t(x), "") for x in merged["T_stage"].tolist()]
        y_pred  = [_T_NORM4.get(normalize_t(x), "") for x in merged["predicted_T"].tolist()]
        pairs   = [(t, p) for t, p in zip(y_true, y_pred)
                   if t in STAGE_LABELS and p in STAGE_LABELS]
        if pairs:
            yt, yp       = zip(*pairs)
            valid_labels = [l for l in STAGE_LABELS if l in set(yt) | set(yp)]
            m = compute_classification_metrics(list(yt), list(yp), labels=valid_labels)
            print_metrics_table(m, f"Pathway A — T-Stage Metrics (n={len(pairs)})")
            print("\n  Classification Report:")
            print(m["report"])
        else:
            print("\n  No valid T-stage ground-truth pairs "
                  "(all labels may be 'Unknown' in this dataset).")

    return results

def step_evaluate(staging_results, manifest: pd.DataFrame,
                  seg_metrics: dict, seg_histories: dict):
    print("\n" + "█" * 55)
    print("  STEP 6: Final Evaluation — Pathway A")
    print("█" * 55)

    import os as _os
    _os.environ["TNM_RESULTS_DIR"] = str(RESULTS_DIR)
    from evaluation import run_full_evaluation

    gt = manifest[["pid", "T_stage"]].copy()

    run_full_evaluation(
        staging_results          = staging_results,
        image_classifier_results = {},
        ground_truth             = gt,
        seg_metrics_2d           = seg_metrics.get("2D", {}),
        seg_metrics_3d           = seg_metrics.get("3D", {}),
        hist_2d                  = seg_histories.get("2D", {}),
        hist_3d                  = seg_histories.get("3D", {}),
    )

def main():
    parser = argparse.ArgumentParser(description="Lung Cancer TNM Staging Pipeline")
    parser.add_argument(
        "--mode", default="train_seg_only",
        choices=["full", "preprocess_only", "train_seg_only", "eval_only"],
    )
    parser.add_argument("--skip_3d",       action="store_true")
    parser.add_argument("--force_reinfer", action="store_true",
                        help="Re-run TumorNet inference for all patients, ignoring cached masks")
    parser.add_argument("--fast_totalseg", action="store_true")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR))
    parser.add_argument("--msd_dir",  type=str, default=str(DEFAULT_MSD))
    args = parser.parse_args()

    log_runtime_info()
    data_dir = Path(args.data_dir)
    msd_dir  = Path(args.msd_dir)

    print(f"MSD dir: {msd_dir}")
    print(f"Data   : {data_dir}")
    print("=" * 55)
    print("  LUNG CANCER TNM STAGING PIPELINE")
    print("  Pathway A: TotalSegmentator + TumorNet + weighted XGBoost")
    print("=" * 55)

    t0       = time.time()
    manifest = load_manifest(data_dir)

    if args.mode in ("full", "preprocess_only"):
        step_preprocess(manifest, fast_totalseg=args.fast_totalseg)
    if args.mode == "preprocess_only":
        print(f"\nDone in {(time.time()-t0)/60:.1f} min")
        return

    seg_metrics = seg_histories = {}
    if args.mode in ("full", "train_seg_only"):
        seg_metrics, seg_histories = step_train_tumornet(
            msd_dir=msd_dir, skip_3d=args.skip_3d
        )
    if args.mode == "train_seg_only":
        print(f"\nDone in {(time.time()-t0)/60:.1f} min")
        print(f"Checkpoints: {CKPT_DIR}")
        return

    feat_df         = step_extract_features(manifest, force_reinfer=args.force_reinfer)
    staging_results = step_tnm_staging(feat_df, manifest=manifest)

    if args.mode == "eval_only":
        p_seg = RESULTS_DIR / "staging_pathwayA_2d.csv"
        if p_seg.exists():
            staging_results = pd.read_csv(p_seg)

    step_evaluate(staging_results, manifest, seg_metrics, seg_histories)

    print(f"\n{'█'*55}")
    print(f"  COMPLETE in {(time.time()-t0)/60:.1f} min")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("█" * 55)

if __name__ == "__main__":
    main()
