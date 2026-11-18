import os
import io
import re
import warnings
import contextlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from dicom_utils import select_series
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage

warnings.filterwarnings("ignore")

LUNG_WW = 1400
LUNG_WC = -700
IMG_SIZE_2D = 256
IMG_SIZE_3D = (64, 128, 128)
CLAHE_CLIP = 1.0
CLAHE_GRID = (16, 16)
NUM_WORKERS = 0
PIN_MEMORY = True
_TOTALSEG_SUPPRESS_NOTICE_SHOWN = False
_TOTALSEG_TB_SHOWN = False
_KEEP_STRUCT_PREFIXES = ("lung", "vertebrae", "heart", "aorta", "trachea", "esophagus", "pulmonary")

_GROUP_STRUCTS = {"rib_": "rib_all", "sternum": "sternum", "costal_cartilages": "sternum",
                  "superior_vena_cava": "superior_vena_cava", "inferior_vena_cava": "inferior_vena_cava",
                  "brachiocephalic": "brachiocephalic_all", "subclavian": "brachiocephalic_all"}

def _report_totalseg_error(prefix, e):
    global _TOTALSEG_TB_SHOWN
    print(f"  {prefix}: {e!r}")
    if not _TOTALSEG_TB_SHOWN:
        import traceback
        traceback.print_exc()
        _TOTALSEG_TB_SHOWN = True

def load_dicom_series(series_dir: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    import pydicom
    series_dir = Path(series_dir)
    dcm_files = sorted(series_dir.glob("*.dcm"), key=lambda f: f.stem)
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")
    slices = [pydicom.dcmread(str(f), force=True) for f in dcm_files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        pass
    ref = slices[0]
    ps = getattr(ref, "PixelSpacing", [1.0, 1.0])
    meta = {
        "pixel_spacing_x": float(ps[1]),
        "pixel_spacing_y": float(ps[0]),
        "slice_thickness": float(getattr(ref, "SliceThickness", 1.0)),
        "n_slices": len(slices),
        "rows": int(getattr(ref, "Rows", 512)),
        "cols": int(getattr(ref, "Columns", 512)),
    }
    target_h: Optional[int] = None
    target_w: Optional[int] = None
    volume: List[np.ndarray] = []
    sop_uids: List[str] = []

    def _to_2d_frames(arr: np.ndarray) -> List[np.ndarray]:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            if arr.shape[1] > 16 and arr.shape[2] > 16 and arr.shape[0] > 1:
                return [arr[i] for i in range(arr.shape[0])]
            if arr.shape[-1] in (3, 4):
                return [arr[..., 0]]
            return [np.squeeze(arr)]
        return [np.squeeze(arr)]

    for s in slices:
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        frames = _to_2d_frames(s.pixel_array)
        for frame in frames:
            frame = np.asarray(frame, dtype=np.float32)
            while frame.ndim > 2:
                frame = frame[0]
            if frame.ndim != 2:
                continue
            if target_h is None or target_w is None:
                target_h, target_w = frame.shape
            if frame.shape != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            hu = frame * slope + intercept
            volume.append(hu)
            sop_uids.append(str(getattr(s, "SOPInstanceUID", "")))

    if not volume:
        raise ValueError(f"No valid 2D DICOM slices found in {series_dir}")

    meta["sop_uids"] = sop_uids
    return np.stack(volume, 0), meta

def apply_lung_window(vol: np.ndarray) -> np.ndarray:
    lo = LUNG_WC-LUNG_WW/2
    hi = LUNG_WC+LUNG_WW/2
    return ((np.clip(vol, lo, hi)-lo)/(hi-lo)*255).astype(np.uint8)

def apply_clahe_volume(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol)
    if vol.ndim == 2:
        vol = vol[None, ...]

    out_slices: List[np.ndarray] = []
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

    for i in range(vol.shape[0]):
        slc = np.asarray(vol[i])
        slc = np.squeeze(slc)
        while slc.ndim > 2:
            slc = slc[0]
        if slc.ndim != 2:
            continue

        if slc.dtype not in (np.uint8, np.uint16):
            slc = np.clip(slc, 0, 255).astype(np.uint8)

        out_slices.append(clahe.apply(slc))

    if not out_slices:
        raise ValueError("No valid 2D slices available for CLAHE preprocessing")

    return np.stack(out_slices, 0)

def preprocess_volume(vol_hu: np.ndarray) -> np.ndarray:
    return apply_clahe_volume(apply_lung_window(vol_hu))

def dicom_series_to_nifti(series_dir: Union[str, Path], out_path: Union[str, Path]) -> bool:
    import pydicom
    series_dir = Path(series_dir)
    out_path   = Path(out_path)

    def _safe_float(v, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _estimate_z_spacing(dcm_files: List[Path]) -> float:
        z_vals: List[float] = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is not None and len(ipp) >= 3:
                    z_vals.append(_safe_float(ipp[2], 0.0))
            except Exception:
                continue
        if len(z_vals) >= 2:
            z_sorted = sorted(z_vals)
            diffs = [abs(z_sorted[i+1] - z_sorted[i]) for i in range(len(z_sorted)-1)]
            diffs = [d for d in diffs if d > 1e-6]
            if diffs:
                return float(np.median(diffs))
        try:
            ref = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
            z = _safe_float(getattr(ref, "SpacingBetweenSlices", 0.0), 0.0)
            if z > 1e-6:
                return z
            z = _safe_float(getattr(ref, "SliceThickness", 0.0), 0.0)
            if z > 1e-6:
                return z
        except Exception:
            pass
        return 1.0
    try:
        vol_hu, meta = load_dicom_series(series_dir)
        vol_hu = np.asarray(vol_hu, dtype=np.float32)
    except Exception as e:
        print(f"  DICOM load failed for {series_dir.name}: {e}")
        return False
    if vol_hu.ndim == 2:
        vol_hu = vol_hu[np.newaxis, ...]
    while vol_hu.ndim > 3:
        vol_hu = vol_hu[..., 0]
    if vol_hu.ndim != 3:
        print(f"  Cannot reduce volume to 3D (shape={vol_hu.shape})")
        return False
    dcm_files = sorted(series_dir.glob("*.dcm"), key=lambda f: f.stem)
    try:
        ref = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
        ps  = getattr(ref, "PixelSpacing", [1.0, 1.0])
        sx  = max(_safe_float(ps[1], 1.0), 1e-6)
        sy  = max(_safe_float(ps[0], 1.0), 1e-6)
    except Exception:
        sx = sy = 1.0
    sz = _estimate_z_spacing(dcm_files)
    try:
        import nibabel as _nib
        vol_nib = vol_hu.transpose(1, 2, 0)
        affine  = np.diag([sx, sy, sz, 1.0])
        nii     = _nib.Nifti1Image(vol_nib, affine)
        nii.header.set_data_shape(vol_nib.shape)
        nii.header.set_zooms((sx, sy, sz))
        _nib.save(nii, str(out_path))
        _check = _nib.load(str(out_path))
        if len(_check.shape) != 3:
            raise RuntimeError(f"Written NIfTI has shape {_check.shape} — not 3D")
        print(f"  NIfTI written: shape={vol_nib.shape}  spacing=({sx:.3f},{sy:.3f},{sz:.3f})mm")
        return True
    except Exception as e:
        print(f"  nibabel NIfTI write failed: {e}")
        out_path.unlink(missing_ok=True)
        return False

def run_totalsegmentator(nii_path: Union[str, Path], output_dir: Union[str, Path], fast: bool=False) -> Dict[str, Path]:
    try:
        global _TOTALSEG_SUPPRESS_NOTICE_SHOWN
        import inspect
        from totalsegmentator.python_api import totalsegmentator
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sig = inspect.signature(totalsegmentator)
        supported = set(sig.parameters.keys())

        kwargs = {}
        if "output" in supported:
            kwargs["output"] = str(output_dir)
        elif "output_path" in supported:
            kwargs["output_path"] = str(output_dir)
        kwargs_fast = fast
        try:
            import torch as _torch
            vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9 if _torch.cuda.is_available() else 0
            if vram_gb < 8.0:
                kwargs_fast = True
        except Exception:
            pass
        if "fast" in supported:
            kwargs["fast"] = kwargs_fast
        if "nr_thr_resamp" in supported:
            kwargs["nr_thr_resamp"] = 1
        if "nr_thr_saving" in supported:
            kwargs["nr_thr_saving"] = 1
        if "num_threads" in supported:
            kwargs["num_threads"] = 1

        if "quiet" in supported:
            kwargs["quiet"] = True
        if "verbose" in supported:
            kwargs["verbose"] = False
        if "statistics" in supported:
            kwargs["statistics"] = False
        if "preview" in supported:
            kwargs["preview"] = False
        os.environ.setdefault("nnUNet_n_proc_DA", "0")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        show_backend_logs = os.getenv("TOTALSEG_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}

        if show_backend_logs:
            if kwargs:
                totalsegmentator(str(nii_path), **kwargs)
            else:
                totalsegmentator(str(nii_path), str(output_dir))
        else:
            if not _TOTALSEG_SUPPRESS_NOTICE_SHOWN:
                print("  TotalSegmentator backend logs suppressed (set TOTALSEG_VERBOSE=1 to enable).")
                _TOTALSEG_SUPPRESS_NOTICE_SHOWN = True
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if kwargs:
                    totalsegmentator(str(nii_path), **kwargs)
                else:
                    totalsegmentator(str(nii_path), str(output_dir))

        return {f.stem.replace(".nii", ""): f for f in output_dir.glob("*.nii.gz")}
    except ImportError as e:
        _report_totalseg_error("TotalSegmentator import failed", e)
        return {}
    except Exception as e:
        _report_totalseg_error("TotalSegmentator failed", e)
        return {}

def load_nifti_as_numpy(nii_path: Union[str, Path]) -> np.ndarray:
    try:
        import SimpleITK as sitk
        return (sitk.GetArrayFromImage(sitk.ReadImage(str(nii_path)))>0).astype(np.uint8)
    except Exception:
        return np.zeros((1, 512, 512), dtype=np.uint8)

def parse_annotation_xml(xml_path: str) -> List[Dict]:
    try:
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append({
                "xmin": int(float(bb.findtext("xmin","0"))),
                "ymin": int(float(bb.findtext("ymin","0"))),
                "xmax": int(float(bb.findtext("xmax","512"))),
                "ymax": int(float(bb.findtext("ymax","512"))),
            })
        return boxes
    except Exception:
        return []

def annotation_to_mask(boxes: List[Dict], h: int=512, w: int=512) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        mask[b["ymin"]:b["ymax"], b["xmin"]:b["xmax"]] = 1
    return mask

def _build_seg_transforms(mode: str, img_size: int) -> A.Compose:
    augs = [A.Resize(img_size, img_size)]
    if mode == "train":
        augs += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.08, scale_limit=0.12, rotate_limit=20,
                border_mode=cv2.BORDER_REFLECT_101, p=0.6
            ),
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,
                border_mode=cv2.BORDER_REFLECT_101, p=0.4
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=img_size // 16, max_width=img_size // 16,
                fill_value=0, p=0.2
            ),
        ]
    augs += [A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0), ToTensorV2()]
    return A.Compose(augs)

class CachedTumorSegDataset2D(Dataset):
    def __init__(self, samples: List[Dict], mode: str="train", img_size: int=IMG_SIZE_2D):
        self.transform = _build_seg_transforms(mode, img_size)
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        missing_img = 0
        missing_mask = 0
        print(f"    [CachedDataset] Caching {len(samples)} slices (mode={mode})...", end=" ", flush=True)
        for s in samples:
            img_path = s.get("image_path","")
            img = None
            if img_path and Path(img_path).exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((img_size, img_size), dtype=np.uint8)
                missing_img += 1
            msk_path = s.get("mask_path","")
            msk = None
            if msk_path and Path(msk_path).exists():
                msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                msk = np.zeros_like(img)
                missing_mask += 1
            self.images.append(img)
            self.masks.append(msk)
        print(f"done. (missing images:{missing_img}, missing masks:{missing_mask})")
        if missing_img>len(samples)*0.5:
            print(f"    WARNING: >50% image paths invalid. Check manifest CSV paths.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        aug = self.transform(image=self.images[idx].copy().astype(np.uint8), mask=(self.masks[idx].copy()>127).astype(np.uint8))
        return aug["image"], aug["mask"].long()

class TumorSegDataset2D(Dataset):
    def __init__(self, samples: List[Dict], mode: str="train", img_size: int=IMG_SIZE_2D):
        self.samples = samples
        self.img_size = img_size
        self.transform = _build_seg_transforms(mode, img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img_path = s.get("image_path","")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) if img_path and Path(img_path).exists() else None
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        msk_path = s.get("mask_path","")
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE) if msk_path and Path(msk_path).exists() else None
        if msk is None:
            msk = np.zeros_like(img)
        aug = self.transform(image=img.astype(np.uint8), mask=(msk>127).astype(np.uint8))
        return aug["image"], aug["mask"].long()

class TumorSegDataset3D(Dataset):
    def __init__(self, records: List[Dict], mode: str="train", vol_size: Tuple=IMG_SIZE_3D):
        self.records = records
        self.mode = mode
        self.vol_size = vol_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.records[idx]
        vol_path = r.get("volume_path","")
        vol = np.load(vol_path).astype(np.float32) if vol_path and Path(vol_path).exists() else np.zeros(self.vol_size, dtype=np.float32)
        msk_path = r.get("mask_path","")
        msk = np.load(msk_path).astype(np.uint8) if msk_path and Path(msk_path).exists() else np.zeros(vol.shape, dtype=np.uint8)
        zoom = [t/s for t,s in zip(self.vol_size, vol.shape)]
        vol = ndimage.zoom(vol, zoom, order=1)
        msk = (ndimage.zoom(msk, zoom, order=1)>0.5).astype(np.uint8)
        vol = (vol-vol.min())/(vol.max()-vol.min()+1e-8)
        if self.mode=="train":
            for ax in range(3):
                if np.random.rand()>0.5:
                    vol = np.flip(vol, ax).copy()
                    msk = np.flip(msk, ax).copy()
        return torch.from_numpy(vol).unsqueeze(0).float(), torch.from_numpy(msk).long()

class StagingImageDataset(Dataset):
    LABEL_MAPS = {
        "T": {"T1":0,"T2":1,"T3":2,"T4":3},
        "N": {"N0":0,"N1":1,"N2":2,"N3":3},
        "M": {"M0":0,"M1":1,"M1a":1,"M1b":1,"M1c":1},
    }

    def __init__(self, samples: List[Dict], mode: str="train", img_size: int=224, stage_type: str="T"):
        self.samples = samples
        self.label_map = self.LABEL_MAPS.get(stage_type, self.LABEL_MAPS["T"])
        self.stage_type = stage_type
        augs = [A.Resize(img_size, img_size)]
        if mode=="train":
            augs += [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.CLAHE(p=0.3),
            ]
        augs += [A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], max_pixel_value=255.0), ToTensorV2()]
        self.transform = A.Compose(augs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = cv2.imread(s.get("image_path",""), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        aug = self.transform(image=img3)
        label = self.label_map.get(s.get(f"{self.stage_type}_stage","T1"), 0)
        return aug["image"], torch.tensor(label, dtype=torch.long), s

def make_seg_loaders(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict],
                     batch_size: int=32, img_size: int=IMG_SIZE_2D, use_cache: bool=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    DS = CachedTumorSegDataset2D if use_cache else TumorSegDataset2D
    kw = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return (
        DataLoader(DS(train_samples, "train", img_size), batch_size=batch_size, shuffle=True, **kw),
        DataLoader(DS(val_samples, "val", img_size), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(DS(test_samples, "test", img_size), batch_size=batch_size, shuffle=False, **kw),
    )

def preprocess_patient_full(row: pd.Series, output_base: str, fast_totalseg: bool=False) -> Dict:
    pid = str(row.get("pid","unknown"))
    dicom_root = str(row.get("dicom_root_input",""))
    output_base = Path(output_base)
    result = {"pid": pid, "status": "ok"}
    try:
        dicom_path = Path(dicom_root)
        if not dicom_path.exists():
            result["status"] = f"error: dicom_root not found: {dicom_path}"
            return result

        ann_paths = [p for p in str(row.get("xml_working", "")).split("|") if p]
        xml_uids = {Path(p).stem for p in ann_paths}
        ct_series, _sinfo, _how = select_series(dicom_path, xml_uids)
        if ct_series is None:
            result["status"] = "error: no_ct_series"
            return result
        print(f"  series chosen for {pid}: {_how} | modality={_sinfo['modality']} | {_sinfo['n']} slices | {_sinfo['desc'][:40]}")
        result["series_choice"] = _how
        result["series_modality"] = _sinfo["modality"]

        vol_hu, meta = load_dicom_series(ct_series)
        vol_pre = preprocess_volume(vol_hu)

        vol_dir = output_base/"volumes"
        vol_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(vol_dir/f"{pid}.npy"), vol_pre)
        result.update({
            "volume_path": str(vol_dir/f"{pid}.npy"),
            "pixel_spacing_x": meta["pixel_spacing_x"],
            "pixel_spacing_y": meta["pixel_spacing_y"],
            "slice_thickness": meta["slice_thickness"],
            "n_slices": meta["n_slices"],
        })

        img_dir = output_base/"images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(vol_pre.shape[0]):
            cv2.imwrite(str(img_dir/f"{pid}_s{i:04d}.png"), vol_pre[i])

        nii_dir = output_base/"nifti"/pid
        seg_dir = output_base/"totalseg_out"/pid
        nii_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        nii_path = nii_dir/f"{pid}_ct.nii.gz"
        if nii_path.exists():
            _needs_regen = False
            try:
                import nibabel as _nib
                _nib_img = _nib.load(str(nii_path))
                _ndim = len(_nib_img.shape)
                if _ndim != 3:
                    print(f"  NIfTI for {pid} has {_ndim}D shape {_nib_img.shape} (nibabel) — regenerating as 3D.")
                    _needs_regen = True
            except Exception as _nib_err:
                print(f"  NIfTI for {pid} unreadable ({_nib_err}) — regenerating.")
                _needs_regen = True
            if _needs_regen:
                nii_path.unlink(missing_ok=True)
                import shutil as _shutil
                if seg_dir.exists():
                    _shutil.rmtree(str(seg_dir))
                seg_dir.mkdir(parents=True, exist_ok=True)

        if not nii_path.exists():
            ok = dicom_series_to_nifti(ct_series, nii_path)
            if not ok:
                result["status"] = "error: nifti_conversion_failed"
                return result

        if not list(seg_dir.glob("*.nii.gz")):
            run_totalsegmentator(nii_path, seg_dir, fast=fast_totalseg)
        if not list(seg_dir.glob("*.nii.gz")):
            result["status"] = "error: totalseg_failed"
            return result

        mask_dir = output_base/"anatomy_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        grouped = {}
        for nii_f in seg_dir.glob("*.nii.gz"):
            struct_name = nii_f.stem.replace(".nii","")
            gname = next((g for pre, g in _GROUP_STRUCTS.items() if struct_name.startswith(pre)), None)
            if gname is not None:
                m = load_nifti_as_numpy(nii_f) > 0
                grouped[gname] = m if gname not in grouped else (grouped[gname] | m)
                continue
            if not struct_name.startswith(_KEEP_STRUCT_PREFIXES):
                continue
            np.save(str(mask_dir/f"{pid}_{struct_name}.npy"), load_nifti_as_numpy(nii_f))
        for gname, m in grouped.items():
            np.save(str(mask_dir/f"{pid}_{gname}.npy"), m.astype(np.uint8))
        result["anatomy_masks_dir"] = str(mask_dir)

        tumor_masks = np.zeros(vol_pre.shape, dtype=np.uint8)
        uid_to_idx = {u: i for i, u in enumerate(meta.get("sop_uids", []))}
        n_xml = n_match = 0
        for xml_path in ann_paths:
            if not Path(xml_path).exists():
                continue
            boxes = parse_annotation_xml(xml_path)
            if not boxes:
                continue
            n_xml += 1
            uid = Path(xml_path).stem
            s_idx = uid_to_idx.get(uid)
            if s_idx is None:
                nums = re.findall(r"\d+", uid)
                if nums and len(nums[-1]) <= 4:
                    s_idx = int(nums[-1]) - 1
            if s_idx is not None and 0 <= s_idx < vol_pre.shape[0]:
                tumor_masks[s_idx] = np.maximum(tumor_masks[s_idx], annotation_to_mask(boxes, vol_pre.shape[1], vol_pre.shape[2]))
                n_match += 1
        result["n_xml"], result["n_xml_matched"] = n_xml, n_match
        print(f"  XML boxes matched to slices: {n_match}/{n_xml}")
        if n_xml and n_match < n_xml:
            print(f"  WARNING: {n_xml - n_match} annotated slice(s) of {pid} could not be matched")

        tumor_dir = output_base/"tumor_masks"
        tumor_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(tumor_dir/f"{pid}_tumor.npy"), tumor_masks)
        result["tumor_mask_path"] = str(tumor_dir/f"{pid}_tumor.npy")

    except Exception as e:
        result["status"] = f"error: {e}"
    return result
