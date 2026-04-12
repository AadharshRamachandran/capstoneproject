import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

LOBE_NAMES  = ["lobe_LUL", "lobe_LLL", "lobe_RUL", "lobe_RML", "lobe_RLL"]
RIGHT_LOBES = {"lobe_RUL", "lobe_RML", "lobe_RLL"}
LEFT_LOBES  = {"lobe_LUL", "lobe_LLL"}
_THRESH: Dict[str, float] = {
    "lung":        3.0,
    "mediastinum": 3.0,
    "diaphragm":   3.0,
    "trachea":     1.0,
    "esophagus":   1.0,
    "heart":       1.0,
    "aorta":       1.0,
    "vertebrae":   1.0,
}
_DEFAULT_THRESH = 1.0
_STRUCTS = ["lung", "mediastinum", "trachea", "esophagus", "heart", "aorta", "vertebrae"]

def extract_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Largest external contour of a 2-D binary mask, or None."""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def extract_all_contours(mask: np.ndarray) -> List[np.ndarray]:
    """All external contours larger than 10 pixels."""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in cnts if cv2.contourArea(c) > 10]

def _pts(cnt: np.ndarray) -> np.ndarray:
    p = cnt.reshape(-1, 2).astype(np.float64)
    return p[:: max(1, len(p) // 500)]

def min_dist_contours(c1: np.ndarray, c2: np.ndarray) -> float:
    p1, p2 = _pts(c1), _pts(c2)
    if len(p1) == 0 or len(p2) == 0:
        return float("inf")
    return float(cdist(p1, p2).min())

def max_tumor_dim_2d(cnt: np.ndarray, pixel_spacing: float) -> float:
    """Maximum in-plane Feret diameter (mm)."""
    p = _pts(cnt)
    return float(cdist(p, p).max()) * pixel_spacing if len(p) >= 2 else 0.0

def estimate_diaphragm(lung_mask: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Approximate the diaphragm as the bottom `frac` of the lung mask."""
    rows = np.where(np.any(lung_mask > 0, axis=1))[0]
    if len(rows) == 0:
        return np.zeros_like(lung_mask)
    inf_row = rows.max()
    height  = int((rows.max() - rows.min()) * frac)
    d_mask  = np.zeros_like(lung_mask)
    d_mask[max(0, inf_row - height): inf_row + 1, :] = \
        lung_mask[max(0, inf_row - height): inf_row + 1, :]
    return d_mask

def max_tumor_dim_3d(tumor_3d: np.ndarray, voxel_spacing: Tuple) -> float:
    """
    3-D Feret diameter (mm).

    Uses convex-hull vertices for large clouds; random sample as fallback.
    IMPORTANT: Only reads voxels where tumor_3d > 0 — shape of the volume
    does NOT affect the result.
    """
    vox = np.argwhere(tumor_3d > 0).astype(np.float64)
    if len(vox) < 2:
        return 0.0
    vox_mm = vox * np.array(voxel_spacing)
    if len(vox_mm) <= 2000:
        return float(cdist(vox_mm, vox_mm).max())
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vox_mm)
        hp   = vox_mm[hull.vertices]
        return float(cdist(hp, hp).max())
    except Exception:
        idx = np.random.choice(len(vox_mm), 500, replace=False)
        return float(cdist(vox_mm[idx], vox_mm[idx]).max())

def min_dist_3d(a: np.ndarray, b: np.ndarray, vs: Tuple) -> float:
    """Minimum 3-D distance (mm) between foreground voxels of two masks."""
    pa = np.argwhere(a > 0).astype(np.float64)
    pb = np.argwhere(b > 0).astype(np.float64)
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")
    pa = pa * np.array(vs)
    pb = pb * np.array(vs)
    if len(pa) > 500:
        pa = pa[np.random.choice(len(pa), 500, replace=False)]
    if len(pb) > 500:
        pb = pb[np.random.choice(len(pb), 500, replace=False)]
    return float(cdist(pa, pb).min())

def tumor_volume_3d(tumor_3d: np.ndarray, vs: Tuple) -> float:
    return float(tumor_3d.sum() * vs[0] * vs[1] * vs[2] / 1000.0)

def detect_multifocal(tumor_mask_2d: np.ndarray,
                       lobe_masks_2d: Dict[str, np.ndarray]) -> Dict:
    """
    Flag multifocal disease in one slice.

    A case is multifocal when nodule contours span more than one lobe of the
    same lung (IASLC 8th Ed. T4 criterion: separate nodule in different lobe).
    """
    nodules      = extract_all_contours(tumor_mask_2d)
    nodule_lobes: List[str] = []
    for cnt in nodules:
        nod = np.zeros_like(tumor_mask_2d)
        cv2.drawContours(nod, [cnt], -1, 1, thickness=-1)
        best_lobe, best_overlap = "unknown", 0
        for lname in LOBE_NAMES:
            lm      = lobe_masks_2d.get(lname, np.zeros_like(tumor_mask_2d))
            overlap = int((nod * (lm > 0).astype(np.uint8)).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_lobe    = lname
        nodule_lobes.append(best_lobe)
    rh = {l for l in nodule_lobes if l in RIGHT_LOBES}
    lh = {l for l in nodule_lobes if l in LEFT_LOBES}
    return {
        "n_nodules":        len(nodules),
        "nodule_lobes":     nodule_lobes,
        "is_multifocal":    len(rh) > 1 or len(lh) > 1,
        "multifocal_right": len(rh) > 1,
        "multifocal_left":  len(lh) > 1,
    }

def handcrafted_radiomics(img: np.ndarray, mask: np.ndarray) -> Dict:
    """First-order + shape radiomics from the tumor ROI bounding box."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return {}
    roi_img  = img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(np.float32)
    roi_mask = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(bool)
    vals     = roi_img[roi_mask]
    if len(vals) == 0:
        return {}
    f  = {}
    m  = vals.mean()
    s  = vals.std()
    f["rad_mean"]      = float(m)
    f["rad_std"]       = float(s)
    f["rad_skewness"]  = float(((vals - m) ** 3).mean() / (s ** 3 + 1e-8))
    f["rad_kurtosis"]  = float(((vals - m) ** 4).mean() / (s ** 4 + 1e-8)) - 3.0
    h, _               = np.histogram(vals, bins=64, density=True)
    hp                 = h[h > 0]
    f["rad_entropy"]   = float(-np.sum(hp * np.log2(hp + 1e-12)))
    f["rad_energy"]    = float(np.sum(vals ** 2))
    cnt = extract_contour(mask.astype(np.uint8))
    if cnt is not None:
        area   = cv2.contourArea(cnt)
        perim  = cv2.arcLength(cnt, True) + 1e-8
        _, _, w, h_bb = cv2.boundingRect(cnt)
        hull_a = cv2.contourArea(cv2.convexHull(cnt)) + 1e-8
        f["rad_compactness"]  = float(4.0 * np.pi * area / perim ** 2)
        f["rad_solidity"]     = float(area / hull_a)
        f["rad_aspect_ratio"] = float(w / (h_bb + 1e-8))
        f["rad_extent"]       = float(area / (w * h_bb + 1e-8))
    return f

def propagate_uncertainty(size_mm: float, size_std: float) -> Dict:
    """
    Gaussian uncertainty propagation over IASLC 8th Ed. T-stage size thresholds.

    Returns per-stage probabilities, a confidence score, and a borderline flag
    (True when the top-2 probabilities differ by < 20 %).
    """
    from scipy import stats as scipy_stats
    bounds = {"T1": (0, 39), "T2": (39, 56), "T3": (56, 80), "T4": (80, 1000)}
    dist   = scipy_stats.norm(loc=size_mm, scale=max(size_std, 0.1))
    probs  = {s: float(dist.cdf(hi) - dist.cdf(lo)) for s, (lo, hi) in bounds.items()}
    sp     = sorted(probs.values(), reverse=True)
    return {
        "uncertainty_stage_probs":   probs,
        "uncertainty_confidence":    float(sp[0]),
        "uncertainty_is_borderline": bool(len(sp) >= 2 and sp[0] - sp[1] < 0.20),
        "uncertainty_size_std_mm":   float(size_std),
    }

def extract_patient_features(ct_volume:       np.ndarray,
                               tumor_masks:    np.ndarray,
                               anatomy_masks:  Dict[str, np.ndarray],
                               tumor_mask_std: Optional[np.ndarray],
                               pixel_spacing:  float = 1.0,
                               slice_thickness: float = 1.0) -> Dict:
    """
    Compute all features for one patient.

    Key outputs
    -----------
    tumor_size_2d_inplane_mm  PRIMARY size field. Maximum in-plane Feret diameter
                              (mm) across tumor-positive slices. No depth blending.
                              tnm_staging.py uses this field first.
    tumor_size_3d_mm          Secondary 3-D Feret diameter for paper reporting.
    tumor_size_2d_mm          Legacy blended value (kept for CSV compatibility).
    {struct}_invaded_2d/3d    Binary invasion flags at anatomy-specific thresholds.
    uncertainty_*             Gaussian size uncertainty fields.
    mean/std_rad_*            Aggregated per-slice radiomics.
    is_multifocal             IASLC T4 multifocal criterion flag.
    """
    feats: Dict = {}
    vs = (slice_thickness, pixel_spacing, pixel_spacing)
    feats["tumor_size_3d_mm"]    = max_tumor_dim_3d(tumor_masks, vs)
    feats["tumor_volume_3d_cm3"] = tumor_volume_3d(tumor_masks, vs)

    for struct in _STRUCTS:
        amask = anatomy_masks.get(struct, np.zeros_like(tumor_masks))
        dist  = min_dist_3d(tumor_masks, amask, vs)
        thr   = _THRESH.get(struct, _DEFAULT_THRESH)
        feats[f"dist_{struct}_3d_mm"] = dist
        feats[f"{struct}_invaded_3d"] = int(dist < thr)
    D = ct_volume.shape[0]

    tumor_sizes_2d:  List[float] = []
    struct_dists:    Dict[str, List[float]] = defaultdict(list)
    for s in _STRUCTS:
        struct_dists[s]
    all_radiomics:   List[Dict]  = []
    size_stds:       List[float] = []
    multifocal_flags: List[Dict] = []

    lobe_vol: Dict[str, np.ndarray] = {
        ln: anatomy_masks.get(ln, np.zeros(ct_volume.shape, dtype=np.uint8))
        for ln in LOBE_NAMES
    }

    for d in range(D):
        t_mask = tumor_masks[d]
        if t_mask.max() == 0:
            continue
        t_cnt = extract_contour(t_mask)
        if t_cnt is None:
            continue

        sz = max_tumor_dim_2d(t_cnt, pixel_spacing)
        tumor_sizes_2d.append(sz)

        if tumor_mask_std is not None and t_mask.sum() > 0:
            var = float(tumor_mask_std[d][t_mask > 0].mean())
            size_stds.append(var * sz)
        for struct in _STRUCTS:
            am = anatomy_masks.get(struct, np.zeros_like(t_mask))
            if am.ndim == 3:
                am = am[d]
            while am.ndim > 2:
                am = am[0]
            a_cnt = extract_contour(am)
            if a_cnt is not None:
                struct_dists[struct].append(
                    min_dist_contours(t_cnt, a_cnt) * pixel_spacing
                )
        lung_2d = anatomy_masks.get("lung", np.zeros_like(t_mask))
        if lung_2d.ndim == 3:
            lung_2d = lung_2d[d]
        while lung_2d.ndim > 2:
            lung_2d = lung_2d[0]
        d_cnt = extract_contour(estimate_diaphragm(lung_2d))
        if d_cnt is not None:
            struct_dists["diaphragm"].append(
                min_dist_contours(t_cnt, d_cnt) * pixel_spacing
            )
        lobe_sl: Dict[str, np.ndarray] = {}
        for ln in LOBE_NAMES:
            lm = lobe_vol[ln]
            lobe_sl[ln] = lm[d] if lm.ndim == 3 else lm
            while lobe_sl[ln].ndim > 2:
                lobe_sl[ln] = lobe_sl[ln][0]
        multifocal_flags.append(detect_multifocal(t_mask, lobe_sl))

        rad = handcrafted_radiomics(ct_volume[d], t_mask)
        if rad:
            all_radiomics.append(rad)

    size_2d_inplane = float(max(tumor_sizes_2d)) if tumor_sizes_2d else 0.0
    n_tumor_slices  = len(tumor_sizes_2d)
    depth_est       = n_tumor_slices * slice_thickness
    feats["tumor_size_2d_inplane_mm"] = size_2d_inplane
    feats["tumor_depth_mm"]           = depth_est
    feats["tumor_size_2d_mm"]         = max(size_2d_inplane, depth_est)
    feats["n_tumor_slices"]           = n_tumor_slices

    for struct, dists in struct_dists.items():
        d_min = float(min(dists)) if dists else float("inf")
        thr   = _THRESH.get(struct, _DEFAULT_THRESH)
        feats[f"dist_{struct}_2d_mm"] = d_min
        feats[f"{struct}_invaded_2d"] = int(d_min < thr)

    feats["vessels_invaded_2d"] = int(
        bool(feats.get("aorta_invaded_2d", 0)) or
        bool(feats.get("pulmonary_artery_invaded_2d", 0))
    )
    feats["vessels_invaded_3d"] = int(
        bool(feats.get("aorta_invaded_3d", 0)) or
        bool(feats.get("pulmonary_artery_invaded_3d", 0))
    )

    ext = ["trachea", "esophagus", "heart", "aorta", "vertebrae"]
    feats["any_extended_organ_invaded_2d"] = int(
        any(feats.get(f"{o}_invaded_2d", 0) for o in ext)
    )
    feats["any_extended_organ_invaded_3d"] = int(
        any(feats.get(f"{o}_invaded_3d", 0) for o in ext)
    )

    feats["is_multifocal"]        = int(any(mf["is_multifocal"] for mf in multifocal_flags))
    feats["max_nodules_per_slice"] = max(
        (mf["n_nodules"] for mf in multifocal_flags), default=0
    )

    if all_radiomics:
        rad_df = pd.DataFrame(all_radiomics)
        for col in rad_df.columns:
            feats[f"mean_{col}"] = float(rad_df[col].mean())
            feats[f"std_{col}"]  = float(rad_df[col].std(ddof=0))

    if size_stds:
        feats.update(propagate_uncertainty(size_2d_inplane, float(np.mean(size_stds))))
    elif size_2d_inplane > 0:
        feats.update(propagate_uncertainty(size_2d_inplane, 5.0))

    return feats

def extract_all_patients(patient_manifest: pd.DataFrame,
                          preproc_dir: str) -> pd.DataFrame:
    """Extract features for all patients in the manifest."""
    preproc_dir = Path(preproc_dir)
    rows: List[Dict] = []

    for _, row in patient_manifest.iterrows():
        pid       = row["pid"]
        print(f"  Extracting: {pid}", end=" ")

        vol_path   = preproc_dir / "volumes"     / f"{pid}.npy"
        tumor_path = preproc_dir / "tumor_masks" / f"{pid}_tumor.npy"
        masks_dir  = preproc_dir / "anatomy_masks"

        if not vol_path.exists() or not tumor_path.exists():
            print("→ missing files, skipping")
            rows.append({"pid": pid, "T_stage": row.get("T_stage", "Unknown")})
            continue

        vol         = np.load(str(vol_path))
        tumor_masks = np.load(str(tumor_path))

        anatomy_masks: Dict[str, np.ndarray] = {}
        for sname in _STRUCTS + LOBE_NAMES:
            fpath = masks_dir / f"{pid}_{sname}.npy"
            if fpath.exists():
                anatomy_masks[sname] = np.load(str(fpath))

        ps = float(row.get("pixel_spacing_x", 1.0) or 1.0)
        st = float(row.get("slice_thickness",  1.0) or 1.0)

        try:
            feats = extract_patient_features(
                ct_volume       = vol,
                tumor_masks     = tumor_masks,
                anatomy_masks   = anatomy_masks,
                tumor_mask_std  = None,
                pixel_spacing   = ps,
                slice_thickness = st,
            )
        except Exception as e:
            print(f"→ error: {e}")
            feats = {}

        feats["pid"]     = pid
        feats["T_stage"] = row.get("T_stage", "Unknown")
        rows.append(feats)

        ip = feats.get("tumor_size_2d_inplane_mm", 0.0)
        d3 = feats.get("tumor_size_3d_mm", 0.0)
        print(f"→ done (inplane={ip:.1f} mm  3d={d3:.1f} mm)")

    df = pd.DataFrame(rows)
    print(f"\nExtracted {len(df)} patients, {len(df.columns)} features")
    return df