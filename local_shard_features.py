import argparse
import gc
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

HERE = Path(__file__).resolve().parent

DEFAULT_SHARDS = r"C:\Users\HP\Downloads\archive (2)"
DEFAULT_CKPT   = r"C:\Users\HP\Downloads\archive (1)"
DEFAULT_OUT    = r"C:\Users\HP\Downloads\working\shard_features"

CKPT_NAME   = "TumorNet_2D_fold0_best.pth"
THRESH_NAME = "TumorNet_2D_threshold.pth"
HEAVY_SUBDIRS = ["volumes", "images", "tumor_masks", "anatomy_masks", "nifti", "totalseg_out"]
_LOBE_MAP = {"lobe_LUL": "lung_upper_lobe_left", "lobe_LLL": "lung_lower_lobe_left",
             "lobe_RUL": "lung_upper_lobe_right", "lobe_RML": "lung_middle_lobe_right",
             "lobe_RLL": "lung_lower_lobe_right"}

_FEATURE_MASKS = ["lung", "mediastinum", "trachea", "esophagus", "heart", "aorta", "vertebrae"] + list(_LOBE_MAP)


CFG = SimpleNamespace()
TS_LOG = []
SCRATCH = PREPROC_DIR = None
DEVICE = None
tumor_model = None
effective_threshold = 0.25
_mask_report_done = False


def parse_args():
    p = argparse.ArgumentParser(description="Local sharded feature extraction (TotalSegmentator + TumorNet)")
    p.add_argument("--shards_dir", default=DEFAULT_SHARDS,
                   help="folder (or .zip) holding the downloaded shard(s); sub-folders are searched for patient_manifest.csv")
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT, help="folder (or .zip) holding TumorNet_2D_fold0_best.pth")
    p.add_argument("--out_dir", default=DEFAULT_OUT, help="where patient_features_shardXXX.zip files are written")
    p.add_argument("--code_dir", default=str(HERE), help="folder with data_pipeline.py, models.py, feature_extraction.py")
    p.add_argument("--scratch_dir", default="", help="temp working folder (default <out_dir>/_scratch). Keep the path SHORT on Windows")
    p.add_argument("--labels_csv", default="", help="optional CSV with columns pid,T_stage to merge into the manifests")
    p.add_argument("--shard_num", type=int, default=None,
                   help="ONE-SHARD-PER-RUN mode: the shard in --shards_dir is saved as patient_features_shardNNN.zip with this number "
                        "(use 0..7 for your 8 shards; every run must use a different number)")
    p.add_argument("--shard_idx", type=int, nargs="*", default=None, help="only run these shard numbers (when one folder holds several shards)")
    p.add_argument("--limit", type=int, default=0, help="smoke test: only the first N patients of each shard (output is tagged _testN)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--ts_fast", action="store_true", help="always use TotalSegmentator's 3 mm --fast model (much quicker on CPU, less accurate)")
    p.add_argument("--ts_timeout_min", type=int, default=25,
                   help="kill a TotalSegmentator run after this many minutes and try the next fallback mode (0 = never)")
    p.add_argument("--ts_threads", type=int, default=0,
                   help="CPU threads for TotalSegmentator/torch (0 = leave defaults). On a CPU-only PC set this to your core count")
    p.add_argument("--align_ts_axes", action="store_true", help="transpose TS masks (must be identical for ALL shards)")
    p.add_argument("--merge_ts_masks", action="store_true",
                   help="build lung / lung_left / lung_right / vertebrae / lobe_* masks from TotalSegmentator v2 names (identical for ALL shards)")
    p.add_argument("--split_if_slices_over", type=int, default=350, help="CTs with more slices start TS in low-RAM force_split mode")
    p.add_argument("--no_fast_fallback", action="store_true", help="never fall back to the 3 mm fast model after an out-of-memory failure")
    p.add_argument("--max_fails_at_start", type=int, default=3, help="abort if the first N patients ALL fail")
    p.add_argument("--skip_failed", action="store_true", help="write the shard zip even if some patients raised exceptions")
    p.add_argument("--overwrite", action="store_true", help="redo shards whose output zip already exists")
    p.add_argument("--keep_scratch", action="store_true", help="do not delete the scratch folder at the end")
    p.add_argument("--inspect", action="store_true", help="only list the shards / checkpoint that were found, then exit")
    p.add_argument("--cache_dir", default="", help="save every patient's tumor + anatomy masks here (compressed .npz) so feature "
                                                   "extraction can be re-run later WITHOUT TotalSegmentator (--from_cache)")
    p.add_argument("--tumor_source", choices=["xml", "yolo", "paper"], default="xml",
                   help="--from_cache only: which cached tumour mask to use. xml = radiologist boxes, yolo = YOLOv11 boxes (yolo_boxes.py build), "
                        "paper = TumorNet masks on YOLO crops (paper_mode.py masks)")
    p.add_argument("--from_cache", action="store_true", help="skip DICOM/TotalSegmentator/TumorNet: rebuild features from --cache_dir")
    p.add_argument("--seg_threshold", type=float, default=0.0,
                   help="TumorNet probability threshold for patients WITHOUT an XML box (0 = old behaviour min(saved, 0.25))")
    p.add_argument("--only_pids", default="", help="comma-separated pids, or a .txt file with one pid per line: process only these")
    p.add_argument("--with_radiomics", action="store_true", help="also compute first-order radiomics (needs the CT volume; off by default)")
    a = p.parse_args()
    os.environ["TNM_TUMOR_SOURCE"] = a.tumor_source
    return a


def resolve_dir(p, extract_root, what):
    p = Path(p)
    if p.is_dir():
        return p
    z = None
    if p.is_file() and p.suffix.lower() == ".zip":
        z = p
    elif p.with_name(p.name + ".zip").is_file():
        z = p.with_name(p.name + ".zip")
    if z is None:
        raise FileNotFoundError(f"{what} not found: {p}")
    dest = Path(extract_root) / z.stem
    marker = dest / ".extracted_ok"
    if not marker.exists():
        print(f"extracting {z.name} -> {dest} (one-off) ...")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        marker.touch()
    return dest


def find_file_deep(root, filename, extract_root):
    root = Path(root)
    hits = sorted(root.rglob(filename))
    if hits:
        return hits[0]
    for z in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    if name.replace("\\", "/").split("/")[-1] == filename:
                        dest = Path(extract_root) / "ckpt" / z.stem
                        dest.mkdir(parents=True, exist_ok=True)
                        zf.extract(name, dest)
                        return dest / name
        except zipfile.BadZipFile:
            continue
    return None


def _zip_has_top_level(z, filename):
    try:
        with zipfile.ZipFile(z) as zf:
            return any(n.replace("\\", "/").strip("/") == filename for n in zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def build_manifest_fallback(base_dir):
    base_dir = Path(base_dir)
    images_root = base_dir / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx"
    if not images_root.exists():
        images_root = base_dir / "Lung-PET-CT-Dx"
    ann_root = base_dir / "Annotation" / "Annotation"
    if not ann_root.exists():
        ann_root = base_dir / "Annotation"
    if not images_root.exists():
        return None
    recs = []
    for d in sorted(images_root.iterdir()):
        if not d.is_dir():
            continue
        short = d.name.replace("Lung_Dx-", "").replace("Lung_Dx_", "")
        recs.append({"pid": d.name, "short_pid": short, "T_stage": "Unknown", "N_stage": "Unknown", "M_stage": "Unknown"})
    return pd.DataFrame(recs) if recs else None


def discover_shards(shards_dir, extract_root, max_depth=6):
    root = resolve_dir(shards_dir, extract_root, "Shards folder")
    dirs = []

    def walk(d, depth):
        if (d / "patient_manifest.csv").is_file():
            dirs.append(d)
            return
        for z in sorted(d.glob("*.zip")):
            if _zip_has_top_level(z, "patient_manifest.csv"):
                dirs.append(resolve_dir(z, extract_root, "Shard archive"))
        if depth >= max_depth:
            return
        for e in sorted(d.iterdir()):
            if e.is_dir() and not e.name.startswith((".", "__")) and e.name not in ("_work", "_extracted", "_scratch"):
                walk(e, depth + 1)

    walk(root, 0)
    shards, used = [], set()
    for i, d in enumerate(dirs):
        m = re.search(r"shard[-_ ]?0*(\d+)", d.name, re.I) or re.search(r"shard[-_ ]?0*(\d+)", d.parent.name, re.I)
        idx = int(m.group(1)) if m else i
        while idx in used:
            idx += 1
        used.add(idx)
        shards.append({"idx": idx, "dir": d, "manifest": d / "patient_manifest.csv"})
    if not shards:
        mf = build_manifest_fallback(root)
        if mf is not None:
            dest = Path(extract_root) / "fallback_manifest"
            dest.mkdir(parents=True, exist_ok=True)
            mf.to_csv(dest / "patient_manifest.csv", index=False)
            print(f"NOTE: no patient_manifest.csv found - built one from the folder layout ({len(mf)} patients).")
            shards.append({"idx": 0, "dir": root, "manifest": dest / "patient_manifest.csv"})
    if not shards:
        listing = sorted(p.name for p in root.iterdir())[:30]
        raise FileNotFoundError(f"No shard (folder with patient_manifest.csv) found under {root}. Top-level entries: {listing}")
    return sorted(shards, key=lambda s: s["idx"])


def disk_free_gb(path=None):
    return shutil.disk_usage(str(path or SCRATCH)).free / 1e9


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9 if psutil else 0.0


def ram_total_gb():
    return psutil.virtual_memory().total / 1e9 if psutil else 0.0


TS_WORKER_SRC = '''
import sys, json, inspect, multiprocessing


def main():
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    import torch
    print("[ts_worker] torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| requested device:", a.get("device"), flush=True)
    from totalsegmentator.python_api import totalsegmentator
    sup = set(inspect.signature(totalsegmentator).parameters)
    cand = dict(fast=a["fast"], force_split=a["force_split"], nr_thr_resamp=a["threads"], nr_thr_saving=a["threads"],
                quiet=False, verbose=False, statistics=False, preview=False)
    if a.get("device"):
        cand["device"] = a["device"]
    print("[ts_worker] starting segmentation (first run may download model weights)", flush=True)
    totalsegmentator(a["inp"], a["out"], **{k: v for k, v in cand.items() if k in sup})
    print("[ts_worker] finished", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
'''


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    except Exception:
        pass


def _last_log_line(log_path):
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 600))
            tail = fh.read().decode("utf-8", "replace")
        parts = [x.strip() for x in re.split(r"[\r\n]+", tail) if x.strip()]
        return parts[-1][:110] if parts else "(no output yet)"
    except OSError:
        return "(no output yet)"


def _ts_attempt(nii_path, out_dir, fast, split):
    for f in Path(out_dir).glob("*.nii.gz"):
        f.unlink()
    args_file = Path(out_dir) / "_ts_args.json"
    log_path = Path(out_dir) / "_ts_worker.log"
    args_file.write_text(json.dumps({
        "inp": str(nii_path), "out": str(out_dir), "fast": bool(fast), "force_split": bool(split),
        "threads": max(1, CFG.ts_threads), "device": "gpu" if DEVICE.type == "cuda" else "cpu"}), encoding="utf-8")
    env = dict(os.environ, nnUNet_n_proc_DA="0", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if CFG.ts_threads > 0:
        env["OMP_NUM_THREADS"] = str(CFG.ts_threads)
    else:
        env.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    timed_out = False
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(CFG.ts_worker), str(args_file)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                try:
                    proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    el = time.time() - t0
                    m, s_ = divmod(int(el), 60)
                    print(f"      ... TotalSegmentator running ({m}m{s_:02d}s) | {_last_log_line(log_path)}", flush=True)
                    if CFG.ts_timeout_min and el > CFG.ts_timeout_min * 60:
                        print(f"      !! exceeded {CFG.ts_timeout_min} min - killing it (see --ts_timeout_min)", flush=True)
                        _kill_tree(proc)
                        proc.wait()
                        timed_out = True
                        break
        except KeyboardInterrupt:
            _kill_tree(proc)
            raise
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    args_file.unlink(missing_ok=True)
    if "[ts_worker] torch" in text:
        print("      " + next(l for l in text.splitlines() if "[ts_worker] torch" in l), flush=True)
    rc = -999 if timed_out else proc.returncode
    ok = rc == 0 and any(Path(out_dir).glob("*.nii.gz"))
    return ok, rc, text[-600:], time.time() - t0


def run_totalsegmentator_subprocess(nii_path, output_dir, fast=False):
    import nibabel as nib
    nii_path, output_dir = Path(nii_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast = bool(fast or CFG.ts_fast)
    pid = nii_path.name.replace("_ct.nii.gz", "")
    n_sl = int(nib.load(str(nii_path)).shape[2])
    start_split = n_sl > CFG.split_if_slices_over
    modes = [("force_split" if start_split else "default", fast, start_split)]
    if not start_split:
        modes.append(("force_split", fast, True))
    if CFG.allow_fast_fallback and not fast:
        modes.append(("fast+force_split", True, True))
    for k, (name, f, sp) in enumerate(modes):
        ok, rc, err, secs = _ts_attempt(nii_path, output_dir, f, sp)
        if ok:
            TS_LOG.append({"pid": pid, "n_slices": n_sl, "mode": name, "attempts": k + 1, "seconds": round(secs, 1)})
            return {p.stem.replace(".nii", ""): p for p in output_dir.glob("*.nii.gz")}
        low = err.lower()
        oom = rc in (-9, 137) or "memory" in low or "alloc" in low
        why = "probably out of RAM" if oom else f"exit code {rc}"
        print(f"    TotalSegmentator mode '{name}' failed ({why}). Last stderr: {err[-300:]!r}")
    TS_LOG.append({"pid": pid, "n_slices": n_sl, "mode": "FAILED", "attempts": len(modes), "seconds": 0.0})
    return {}


class ShardData:
    def __init__(self, shard_dir, manifest, scratch):
        self.shard_dir, self.scratch = Path(shard_dir), Path(scratch)
        self.pids = set(manifest["pid"].astype(str))
        self.shorts = set(manifest["short_pid"].astype(str))
        self.zip, self.members, self.img_root, self.xml = None, {}, None, {}
        self._open_images(str(manifest["pid"].iloc[0]))
        self._index_annotations()

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None

    def _open_images(self, first_pid):
        for cand in [self.shard_dir / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx", self.shard_dir / "Lung-PET-CT-Dx", self.shard_dir]:
            if (cand / first_pid).is_dir():
                self.img_root = cand
                print(f"images: read directly from {cand} (no copying)")
                return
        zp = self.shard_dir / "Lung-PET-CT-Dx.zip"
        if not zp.exists():
            raise FileNotFoundError(f"No image folder or Lung-PET-CT-Dx.zip in {self.shard_dir}: "
                                    f"{sorted(p.name for p in self.shard_dir.iterdir())}")
        self.zip = zipfile.ZipFile(zp)
        for info in self.zip.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            for k, part in enumerate(parts):
                if part in self.pids:
                    self.members.setdefault(part, []).append((info, k))
                    break
        print(f"images: zip archive {zp.name} indexed ({sum(len(v) for v in self.members.values())} files, "
              f"{len(self.members)} patients); each patient is extracted on demand and deleted afterwards")

    def _index_annotations(self):
        ann_dir = None
        for cand in [self.shard_dir / "Annotation" / "Annotation", self.shard_dir / "Annotation"]:
            if cand.is_dir() and next(cand.rglob("*.xml"), None) is not None:
                ann_dir = cand
                break
        if ann_dir is None and (self.shard_dir / "Annotation.zip").exists():
            ann_dir = self.scratch / "Annotation"
            shutil.rmtree(ann_dir, ignore_errors=True)
            with zipfile.ZipFile(self.shard_dir / "Annotation.zip") as z:
                z.extractall(ann_dir)
        if ann_dir is None:
            print("WARNING: no Annotation folder/zip found - tumor boxes will be missing")
            return
        for p in sorted(ann_dir.rglob("*.xml")):
            for part in p.relative_to(ann_dir).parts[:-1]:
                if part in self.shorts:
                    self.xml.setdefault(part, []).append(str(p))
                    break
        print(f"annotations: {sum(len(v) for v in self.xml.values())} XML files for {len(self.xml)} patients")

    def prepare(self, pid):
        if self.zip is None:
            return self.img_root / pid
        dest_root = self.scratch / "dicom" / "Lung-PET-CT-Dx"
        shutil.rmtree(dest_root / pid, ignore_errors=True)
        for info, k in self.members.get(pid, []):
            parts = info.filename.split("/")[k:]
            if ".." in parts:
                continue
            out = dest_root.joinpath(*parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            with self.zip.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return dest_root / pid

    def release(self, pid):
        if self.zip is not None:
            shutil.rmtree(self.scratch / "dicom" / "Lung-PET-CT-Dx" / pid, ignore_errors=True)


def _clear_heavy():
    for sub in HEAVY_SUBDIRS:
        d = PREPROC_DIR / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def align_totalseg_masks(pid):
    for p in (PREPROC_DIR / "anatomy_masks").glob(f"{pid}_*.npy"):
        m = np.load(str(p))
        if m.ndim == 3:
            np.save(str(p), np.ascontiguousarray(m.transpose(0, 2, 1)))


def merge_totalseg_masks(pid):
    md = PREPROC_DIR / "anatomy_masks"

    def _load(name):
        p = md / f"{pid}_{name}.npy"
        return (np.load(str(p)) > 0) if p.exists() else None

    def _union(ms):
        ms = [m for m in ms if m is not None]
        if not ms:
            return None
        out = ms[0].copy()
        for m in ms[1:]:
            out |= m
        return out

    lobes = {k: _load(v) for k, v in _LOBE_MAP.items()}
    for k, m in lobes.items():
        if m is not None:
            np.save(str(md / f"{pid}_{k}.npy"), m.astype(np.uint8))
    left = _union([lobes["lobe_LUL"], lobes["lobe_LLL"]])
    right = _union([lobes["lobe_RUL"], lobes["lobe_RML"], lobes["lobe_RLL"]])
    both = _union([left, right])
    for name, m in [("lung_left", left), ("lung_right", right), ("lung", both)]:
        if m is not None:
            np.save(str(md / f"{pid}_{name}.npy"), m.astype(np.uint8))
    verts = None
    for p in sorted(md.glob(f"{pid}_vertebrae_*.npy")):
        m = np.load(str(p)) > 0
        verts = m if verts is None else (verts | m)
    if verts is not None:
        np.save(str(md / f"{pid}_vertebrae.npy"), verts.astype(np.uint8))


def report_mask_coverage(pid):
    global _mask_report_done
    if _mask_report_done:
        return
    _mask_report_done = True
    md = PREPROC_DIR / "anatomy_masks"
    have = sorted(p.name[len(pid) + 1:-4] for p in md.glob(f"{pid}_*.npy"))
    missing = [m for m in _FEATURE_MASKS if m not in have]
    print(f"    anatomy masks saved ({len(have)}): {have[:12]}{' ...' if len(have) > 12 else ''}")
    if missing:
        print(f"    NOTE: feature extractor expects these masks but they are absent -> their distances become inf / flags 0: {missing}")
        if not CFG.merge_ts_masks:
            print("          (if lung/vertebrae/lobe_* are in that list, re-run with --merge_ts_masks - keep it identical for ALL shards)")


def _largest_component_3d(mask3d):
    from scipy.ndimage import label
    lab, n = label(mask3d)
    if n <= 1:
        return mask3d
    sizes = np.bincount(lab.ravel())[1:]
    return (lab == (int(sizes.argmax()) + 1)).astype(mask3d.dtype)


def run_tumor_inference(shard_manifest, spacing_map):
    import cv2
    import torch
    IMG_SIZE, BATCH_SIZE, TUMOR_PADDING = 256, 8, 20
    tumor_dir = PREPROC_DIR / "tumor_masks"
    vol_dir = PREPROC_DIR / "volumes"
    anat_dir = PREPROC_DIR / "anatomy_masks"
    tumor_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for _, row in shard_manifest.iterrows():
        pid = str(row["pid"])
        vol_path, mask_path = vol_dir / f"{pid}.npy", tumor_dir / f"{pid}_tumor.npy"
        if mask_path.exists() and np.load(str(mask_path)).sum() > 0:
            skipped += 1
            continue
        if not vol_path.exists():
            continue
        vol = np.load(str(vol_path))
        D, H, W = vol.shape
        vol_f = (vol.astype(np.float32) / 255.0) if vol.dtype == np.uint8 else \
                ((vol.astype(np.float32) - vol.min()) / max(float(vol.max() - vol.min()), 1e-8))
        lung_mask = None
        ll, lr = anat_dir / f"{pid}_lung_left.npy", anat_dir / f"{pid}_lung_right.npy"
        if ll.exists() and lr.exists():
            try:
                lung_mask = (np.load(str(ll)) > 0) | (np.load(str(lr)) > 0)
                if lung_mask.shape != vol.shape:
                    lung_mask = lung_mask.transpose(0, 2, 1) if lung_mask.transpose(0, 2, 1).shape == vol.shape else None
            except Exception:
                lung_mask = None

        def prep_input_slice(slc_2d, d):
            box = (0, H, 0, W)
            if lung_mask is not None and d < lung_mask.shape[0]:
                mask_d = lung_mask[d]
                if mask_d.sum() > 100:
                    rows = np.where(np.any(mask_d, axis=1))[0]
                    cols = np.where(np.any(mask_d, axis=0))[0]
                    if len(rows) > 0 and len(cols) > 0:
                        box = (max(0, rows[0] - TUMOR_PADDING), min(H, rows[-1] + TUMOR_PADDING + 1),
                               max(0, cols[0] - TUMOR_PADDING), min(W, cols[-1] + TUMOR_PADDING + 1))
            r0, r1, c0, c1 = box
            return cv2.resize(slc_2d[r0:r1, c0:c1], (IMG_SIZE, IMG_SIZE)), box

        pred_full = np.zeros((D, H, W), dtype=np.uint8)
        with torch.no_grad():
            for b0 in range(0, D, BATCH_SIZE):
                bidx = list(range(b0, min(D, b0 + BATCH_SIZE)))
                items = [prep_input_slice(vol_f[d], d) for d in bidx]
                bt = torch.from_numpy(np.stack([it[0] for it in items])[:, np.newaxis]).float().to(DEVICE)
                logits = tumor_model(bt)
                logits = logits[0] if isinstance(logits, tuple) else logits
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                for d, prob, (_, (r0, r1, c0, c1)) in zip(bidx, probs, items):
                    pb = (prob > effective_threshold).astype(np.uint8)
                    pb = cv2.resize(pb, (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST)
                    pred_full[d, r0:r1, c0:c1] = pb
        from scipy.ndimage import binary_fill_holes
        for d in range(D):
            if pred_full[d].any():
                pred_full[d] = binary_fill_holes(pred_full[d] > 0)
        pred_full = _largest_component_3d(pred_full)
        np.save(str(mask_path), pred_full.astype(np.uint8))
        done += 1
    print(f"    TumorNet inference: {done} processed, {skipped} skipped (had XML box mask)")


def save_patient_cache(pid, rec):
    if not CFG.cache_dir:
        return
    cache = Path(CFG.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    payload = {"meta": np.array([float(rec.get("pixel_spacing_x", 1.0) or 1.0), float(rec.get("pixel_spacing_y", 1.0) or 1.0),
                                 float(rec.get("slice_thickness", 1.0) or 1.0), float(rec.get("n_slices", 0) or 0)])}
    tp = PREPROC_DIR / "tumor_masks" / f"{pid}_tumor.npy"
    if tp.exists():
        payload["tumor"] = (np.load(str(tp)) > 0).astype(np.uint8)
        payload["tumor_src"] = np.array("box" if int(rec.get("n_xml_matched", 0) or 0) > 0 else "model")
    for p in sorted((PREPROC_DIR / "anatomy_masks").glob(f"{pid}_*.npy")):
        payload["anat__" + p.stem[len(pid) + 1:]] = (np.load(str(p)) > 0).astype(np.uint8)
    old = cache / f"{pid}.npz"
    if old.exists():
        try:
            z = np.load(str(old))
            for k in ("tumor_box", "tumornet", "tumornet_p", "tumor_yolo", "tumor_paper"):
                if k in z.files and k not in payload:
                    payload[k] = z[k]
        except Exception:
            pass
    np.savez_compressed(str(cache / f"{pid}.npz"), **payload)


def restore_patient_cache(pid):
    f = Path(CFG.cache_dir) / f"{pid}.npz"
    if not f.exists():
        return None
    z = np.load(str(f))
    (PREPROC_DIR / "tumor_masks").mkdir(parents=True, exist_ok=True)
    (PREPROC_DIR / "anatomy_masks").mkdir(parents=True, exist_ok=True)
    tum = None
    mode = os.environ.get("TNM_TUMOR_SOURCE", "xml")
    if mode in ("yolo", "paper"):
        key = "tumor_yolo" if mode == "yolo" else "tumor_paper"
        from sidecar import load_side
        tum = load_side(CFG.cache_dir, pid, key)
        if tum is None and key in z.files:
            tum = z[key]
        if tum is not None:
            pass
        else:
            ref = next((z[k] for k in ("tumor_box", "tumor") if k in z.files), None)
            tum = np.zeros(ref.shape, np.uint8) if ref is not None else None
        src = "box" if mode == "yolo" else "model"
    else:
        if "tumor_box" in z.files and z["tumor_box"].any():
            tum = z["tumor_box"]
        elif "tumor" in z.files:
            tum = z["tumor"]
        src = "box" if (tum is not None and "tumor_box" in z.files and z["tumor_box"].any()) else \
              (str(z["tumor_src"]) if "tumor_src" in z.files else "model")
    if tum is not None:
        np.save(str(PREPROC_DIR / "tumor_masks" / f"{pid}_tumor.npy"), tum)
        (PREPROC_DIR / "tumor_masks" / f"{pid}_tumor_src.txt").write_text(src)
    if "tumornet" in z.files and z["tumornet"].any():
        (PREPROC_DIR / "tumornet_masks").mkdir(parents=True, exist_ok=True)
        np.save(str(PREPROC_DIR / "tumornet_masks" / f"{pid}_tumornet.npy"), z["tumornet"])
    for k in z.files:
        if k.startswith("anat__"):
            np.save(str(PREPROC_DIR / "anatomy_masks" / f"{pid}_{k[6:]}.npy"), z[k])
    nd = PREPROC_DIR / "anatomynet_masks"
    for old in nd.glob(f"{pid}_*.npy") if nd.exists() else []:
        old.unlink()
    side = Path(CFG.cache_dir) / os.environ.get("TNM_ANATOMYNETS_DIR", "_anatomynets") / f"{pid}.npz"
    if mode == "paper" and side.exists() and os.environ.get("TNM_PAPER_MASKS", "auto").lower() != "ts":
        nd.mkdir(parents=True, exist_ok=True)
        sz = np.load(str(side))
        for k in ("lungnet", "medinet"):
            if k in sz.files:
                np.save(str(nd / f"{pid}_{k}.npy"), sz[k])
    m = z["meta"]
    return {"pid": pid, "status": "ok", "pixel_spacing_x": float(m[0]), "pixel_spacing_y": float(m[1]),
            "slice_thickness": float(m[2]), "n_slices": int(m[3])}


def process_shard(shard, out_dir, mods):
    global _mask_report_done
    preprocess_patient_full, extract_all_patients = mods
    tag = f"shard{shard['idx']:03d}" + (f"_test{CFG.limit}" if CFG.limit else "")
    partial_csv = out_dir / f"features_{tag}.partial.csv"
    out_zip = out_dir / f"patient_features_{tag}.zip"
    if out_zip.exists() and not CFG.overwrite:
        print(f"\n[{tag}] {out_zip.name} already exists - skipping (use --overwrite to redo). "
              f"If this is a DIFFERENT shard, give it its own number with --shard_num.")
        CFG.overall_done += CFG.shard_sizes[shard["idx"]]
        return None

    manifest = pd.read_csv(shard["manifest"])
    manifest["pid"] = manifest["pid"].astype(str)
    if "short_pid" not in manifest.columns:
        manifest["short_pid"] = (manifest["pid"].str.replace("Lung_Dx-", "", regex=False)
                                 .str.replace("Lung_Dx_", "", regex=False))
    manifest["short_pid"] = manifest["short_pid"].astype(str)
    if CFG.labels_csv:
        lab = pd.read_csv(CFG.labels_csv)
        lab["pid"] = lab["pid"].astype(str)
        manifest = (manifest.drop(columns=["T_stage"], errors="ignore")
                    .merge(lab[["pid", "T_stage"]].drop_duplicates("pid"), on="pid", how="left"))
        manifest["T_stage"] = manifest["T_stage"].fillna("Unknown")
        print(f"labels merged: {(manifest['T_stage'] != 'Unknown').sum()}/{len(manifest)} patients have a T_stage")
    elif "T_stage" in manifest.columns and (manifest["T_stage"].astype(str) == "Unknown").all():
        print("NOTE: every T_stage in this shard's manifest is 'Unknown' (use --labels_csv, or pass labels to final_t_staging.py).")
    if CFG.limit:
        manifest = manifest.head(CFG.limit).reset_index(drop=True)
    if CFG.only_pids:
        keep = set(CFG.only_pids)
        manifest = manifest[manifest["pid"].isin(keep)].reset_index(drop=True)
        print(f"--only_pids: {len(manifest)} patient(s) of this shard selected")
        if manifest.empty:
            return None

    if CFG.from_cache:
        class _NoData:
            xml = {}
            def prepare(self, pid): return ""
            def release(self, pid): pass
            def close(self): pass
        data = _NoData()
        manifest["xml_working"] = ""
    else:
        data = ShardData(shard["dir"], manifest, SCRATCH)
        manifest["xml_working"] = manifest["short_pid"].map(lambda sp: "|".join(data.xml.get(sp, [])))
    n = len(manifest)
    print(f"\n[{tag}] {n} patients | RAM total {ram_total_gb():.0f} GB | scratch free {disk_free_gb():.1f} GB | device {DEVICE}")

    done_rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done_pids = {str(r["pid"]) for r in done_rows}
    if done_pids:
        print(f"resuming: {len(done_pids)} patients already done")
    offset = CFG.overall_done
    bad_in_a_row = n_ok = 0
    failed = []
    _mask_report_done = False
    t_start = time.time()

    try:
        for i in range(n):
            pid = str(manifest["pid"].iloc[i])
            if pid in done_pids:
                continue
            t0 = time.time()
            print(f"\n>>> starting patient {len(done_rows) + 1}/{n} of {tag}: {pid} "
                  f"(overall {offset + len(done_rows)}/{CFG.overall_total} done so far)", flush=True)
            _clear_heavy()
            try:
                one = manifest.iloc[[i]].copy()
                if CFG.from_cache:
                    rec = restore_patient_cache(pid)
                    if rec is None:
                        raise FileNotFoundError(f"no cache file for {pid} in {CFG.cache_dir}")
                    n_ok += 1
                    bad_in_a_row = 0
                else:
                    one["dicom_root_input"] = str(data.prepare(pid))
                    rec = preprocess_patient_full(one.iloc[0], str(PREPROC_DIR), fast_totalseg=False)
                    if rec["status"] == "ok":
                        n_ok += 1
                        bad_in_a_row = 0
                        if CFG.align_ts_axes:
                            align_totalseg_masks(pid)
                        if CFG.merge_ts_masks:
                            merge_totalseg_masks(pid)
                        report_mask_coverage(pid)
                    else:
                        print(f"    preprocess status: {rec['status']}")
                        bad_in_a_row += 1
                        if n_ok == 0 and bad_in_a_row >= CFG.max_fails_at_start:
                            raise RuntimeError(f"The first {bad_in_a_row} patients all failed (last status: {rec['status']}). "
                                               f"Stopping instead of producing empty features - see the messages above.")
                    spacing_map = {pid: float(rec.get("slice_thickness", 5.0) or 5.0)}
                    run_tumor_inference(one, spacing_map)
                    (PREPROC_DIR / "tumor_masks").mkdir(parents=True, exist_ok=True)
                    (PREPROC_DIR / "tumor_masks" / f"{pid}_tumor_src.txt").write_text(
                        "box" if int(rec.get("n_xml_matched", 0) or 0) > 0 else "model")
                    save_patient_cache(pid, rec)
                rdf = pd.DataFrame([rec]).reindex(columns=["pid", "pixel_spacing_x", "pixel_spacing_y", "slice_thickness"])
                feats = extract_all_patients(one.merge(rdf, on="pid", how="left"), str(PREPROC_DIR))
                done_rows.extend(feats.to_dict("records"))
                pd.DataFrame(done_rows).to_csv(partial_csv, index=False)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if isinstance(e, RuntimeError) and "first" in str(e) and "failed" in str(e):
                    raise
                traceback.print_exc()
                failed.append(pid)
                bad_in_a_row += 1
                print(f"    ERROR on {pid}: {e!r} (will be retried on the next run)")
                if n_ok == 0 and bad_in_a_row >= CFG.max_fails_at_start:
                    raise RuntimeError(f"The first {bad_in_a_row} patients all failed - stopping. Last error: {e!r}")
                continue
            finally:
                data.release(pid)
            mode = next((r["mode"] for r in reversed(TS_LOG) if r["pid"] == pid), "-")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            k = len(done_rows)
            eta = (time.time() - t_start) / max(1, k - len(done_pids)) * (n - k) / 60
            print(f"  DONE {offset + k}/{CFG.overall_total} patients processed  ({tag}: {k}/{n})", flush=True)
            print(f"  [{k}/{n}] {pid} | {rec.get('n_slices', '?')} slices | TS mode: {mode} | {time.time() - t0:.0f}s | "
                  f"RAM {rss_gb():.1f} GB | free disk {disk_free_gb():.0f} GB | ETA ~{eta:.0f} min")
    finally:
        data.close()
        _clear_heavy()
    CFG.overall_done = offset + n

    feat_df = pd.DataFrame(done_rows)
    if failed and not CFG.skip_failed:
        print(f"\n[{tag}] {len(failed)} patient(s) raised errors: {failed[:10]}{' ...' if len(failed) > 10 else ''}")
        print(f"[{tag}] NOT writing {out_zip.name}; progress is kept in {partial_csv.name}. Re-run to retry only those "
              f"patients (or add --skip_failed to write the zip without them).")
        return feat_df
    man_out = manifest.drop(columns=[c for c in ["dicom_root_input", "ann_dir_input", "xml_working"] if c in manifest.columns])
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("patient_features.csv", feat_df.to_csv(index=False))
        z.writestr("patient_manifest.csv", man_out.to_csv(index=False))
        z.writestr("totalseg_log.csv", pd.DataFrame(TS_LOG).to_csv(index=False))
    partial_csv.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print(f"{tag} DONE: {len(feat_df)} patients, {feat_df.shape[1]} feature columns")
    size = pd.to_numeric(feat_df.get("tumor_size_2d_inplane_mm", pd.Series(dtype=float)), errors="coerce").fillna(0)
    print(f"patients with no tumor size (check these): {int((size <= 0).sum())}")
    if "tumor_in_lung_frac" in feat_df.columns:
        lf = pd.to_numeric(feat_df["tumor_in_lung_frac"], errors="coerce")
        print(f"tumor inside lung mask: median {lf.median():.2f}; patients <50%: {int((lf < 0.5).sum())} "
              f"(high = tumor/anatomy misaligned; see align_choice column)")
    if TS_LOG:
        log = pd.DataFrame(TS_LOG)
        print("TotalSegmentator modes used:", log["mode"].value_counts().to_dict())
        print(f"TotalSegmentator time per patient: median {log['seconds'].median():.0f}s, max {log['seconds'].max():.0f}s")
    print(f"written: {out_zip}  ({out_zip.stat().st_size / 1e6:.1f} MB)")
    return feat_df


def main():
    global SCRATCH, PREPROC_DIR, DEVICE, tumor_model, effective_threshold
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_root = out_dir / "_extracted"
    extract_root.mkdir(exist_ok=True)


    shards = discover_shards(args.shards_dir, extract_root)
    if args.shard_num is not None:
        if len(shards) != 1:
            raise SystemExit(f"--shard_num needs --shards_dir to contain exactly ONE shard, but {len(shards)} were found "
                             f"({[str(s['dir']) for s in shards]}). Point --shards_dir at a single shard's folder.")
        shards[0]["idx"] = args.shard_num
    if args.shard_idx:
        shards = [s for s in shards if s["idx"] in set(args.shard_idx)]
    ckpt_path = thresh_path = None
    if not args.from_cache:
        ckpt_root = resolve_dir(args.ckpt_dir, extract_root, "Checkpoint folder")
        ckpt_path = find_file_deep(ckpt_root, CKPT_NAME, extract_root)
        thresh_path = find_file_deep(ckpt_root, THRESH_NAME, extract_root)
        if ckpt_path is None:
            raise FileNotFoundError(f"{CKPT_NAME} not found under {ckpt_root}")

    print("shards found :")
    for s in shards:
        npat = len(pd.read_csv(s["manifest"], usecols=["pid"]))
        print(f"  shard{s['idx']:03d}: {npat} patients  <- {s['dir']}")
    print("checkpoint   :", ckpt_path)
    print("threshold    :", thresh_path if thresh_path else "(not found, will use 0.5)")
    CFG.shard_sizes = {}
    for s in shards:
        npat = len(pd.read_csv(s["manifest"], usecols=["pid"]))
        CFG.shard_sizes[s["idx"]] = min(npat, args.limit) if args.limit else npat
    CFG.overall_total, CFG.overall_done = sum(CFG.shard_sizes.values()), 0
    print(f"TOTAL patients to process: {CFG.overall_total}")
    idxs = sorted(s["idx"] for s in shards)
    if idxs and idxs != list(range(idxs[0], idxs[-1] + 1)):
        print(f"WARNING: shard numbers {idxs} are not contiguous - some shards may not be in {args.shards_dir}")
    if args.inspect:
        return


    import torch
    DEVICE = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and DEVICE.type != "cuda":
        print("WARNING: --device cuda requested but CUDA is not available - using CPU")
    if args.ts_threads > 0:
        torch.set_num_threads(args.ts_threads)

    scratch_base = Path(args.scratch_dir) if args.scratch_dir else out_dir / "_scratch"
    SCRATCH = scratch_base / "lung_scratch"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    PREPROC_DIR = SCRATCH / "preprocessed"
    PREPROC_DIR.mkdir(parents=True, exist_ok=True)

    code_dir = Path(args.code_dir)
    CFG.ts_worker = code_dir / "ts_worker.py"
    CFG.ts_fast, CFG.ts_threads = args.ts_fast, args.ts_threads
    CFG.ts_timeout_min = args.ts_timeout_min
    CFG.align_ts_axes, CFG.merge_ts_masks = args.align_ts_axes, args.merge_ts_masks
    CFG.split_if_slices_over = args.split_if_slices_over
    CFG.allow_fast_fallback = not args.no_fast_fallback
    CFG.max_fails_at_start = args.max_fails_at_start
    CFG.skip_failed, CFG.overwrite, CFG.limit = args.skip_failed, args.overwrite, args.limit
    CFG.labels_csv = args.labels_csv
    CFG.cache_dir, CFG.from_cache = args.cache_dir, args.from_cache
    _op = args.only_pids.strip()
    if _op and Path(_op).is_file():
        _op = ",".join(l.strip() for l in Path(_op).read_text(encoding="utf-8").splitlines() if l.strip())
    CFG.only_pids = [x.strip() for x in _op.split(",") if x.strip()]
    CFG.with_radiomics = args.with_radiomics
    if CFG.from_cache and not CFG.cache_dir:
        raise SystemExit("--from_cache needs --cache_dir")

    for need in ["data_pipeline.py", "models.py", "feature_extraction.py"]:
        if not (code_dir / need).exists():
            raise FileNotFoundError(f"{need} not found in {code_dir} (use --code_dir)")
    CFG.ts_worker.write_text(TS_WORKER_SRC, encoding="utf-8")
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir))

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | device {DEVICE} | RAM {ram_total_gb():.1f} GB | "
          f"scratch {SCRATCH} ({disk_free_gb():.1f} GB free)")
    if DEVICE.type == "cpu":
        print("NOTE: running on CPU. TotalSegmentator can take 10-40 min per patient this way; consider --ts_fast "
              "and --ts_threads <cores>.")
    if not args.from_cache:
        try:
            from totalsegmentator.python_api import totalsegmentator
            import nnunetv2
        except Exception:
            traceback.print_exc()
            raise RuntimeError("TotalSegmentator / nnunetv2 is not importable - pip install TotalSegmentator first.")

    import data_pipeline, feature_extraction
    importlib.reload(data_pipeline)
    importlib.reload(feature_extraction)
    if not hasattr(data_pipeline, "_KEEP_STRUCT_PREFIXES"):
        print("WARNING: this data_pipeline.py is the OLD version (saves every TotalSegmentator mask, ~100+ files per patient, "
              "slow and disk-heavy). Replace it with the data_pipeline.py delivered alongside this script.")
    data_pipeline.run_totalsegmentator = run_totalsegmentator_subprocess
    _extract = feature_extraction.extract_all_patients
    if args.with_radiomics:
        _extract = lambda m, d: feature_extraction.extract_all_patients(m, d, with_radiomics=True)
    mods = (data_pipeline.preprocess_patient_full, _extract)


    if not args.from_cache:
        from models import get_segmentation_model
        threshold = 0.5
        if thresh_path:
            threshold = float(torch.load(str(thresh_path), map_location="cpu", weights_only=True).get("threshold", 0.5))
        effective_threshold = args.seg_threshold if args.seg_threshold > 0 else min(threshold, 0.25)
        print(f"Inference threshold: {effective_threshold:.2f} (raw={threshold:.2f})")
        tumor_model = get_segmentation_model("tumornet_2d", pretrained=False, mc_dropout=False).to(DEVICE)
        tumor_model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True))
        tumor_model.eval()
        print("TotalSegmentator runs in a separate process per patient (RAM fully released after each one).")
    else:
        print("--from_cache: features are rebuilt from cached masks (no DICOM, TotalSegmentator or TumorNet).")


    for s in shards:
        process_shard(s, out_dir, mods)

    if not args.keep_scratch:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\nAll shards finished. Next:  python final_t_staging.py --features_dir \"" + str(out_dir) + "\" [--labels_csv labels.csv]")


if __name__ == "__main__":
    main()