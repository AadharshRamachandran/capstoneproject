import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

LOBE_NAMES = ["lobe_LUL", "lobe_LLL", "lobe_RUL", "lobe_RML", "lobe_RLL"]
RIGHT_LOBES = {"lobe_RUL", "lobe_RML", "lobe_RLL"}
LEFT_LOBES = {"lobe_LUL", "lobe_LLL"}
_LOBE_MAP = {"lobe_LUL": "lung_upper_lobe_left", "lobe_LLL": "lung_lower_lobe_left",
             "lobe_RUL": "lung_upper_lobe_right", "lobe_RML": "lung_middle_lobe_right",
             "lobe_RLL": "lung_lower_lobe_right"}
_STRUCTS = ["lung", "mediastinum", "trachea", "esophagus", "heart", "aorta", "vertebrae", "pulmonary_artery"]
_DIST_CAP_MM = 50.0
_WINDOW_PAD_MM = 50.0


def _feret_mm(cnt: np.ndarray, psx: float, psy: float) -> float:
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float64)
    pts = hull * np.array([psx, psy])
    if len(pts) < 2:
        return float(max(psx, psy))
    return float(cdist(pts, pts).max() + 0.5 * (psx + psy))


def _contours(mask2d: np.ndarray) -> List[np.ndarray]:
    m = np.ascontiguousarray((mask2d > 0).astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in cnts if len(c) >= 1]


def boxes_to_ellipses(tumor: np.ndarray, force=None) -> Tuple[np.ndarray, bool, Dict]:
    out = np.zeros_like(tumor, dtype=np.uint8)
    n_rect = n_other = 0
    max_w = max_h = 0
    for d in np.where(tumor.reshape(tumor.shape[0], -1).any(axis=1))[0]:
        m = np.ascontiguousarray((tumor[d] > 0).astype(np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if w < 2 or h < 2:
                out[d][lab == k] = 1
                n_other += 1
                continue
            tmp = np.zeros(m.shape, np.uint8)
            cv2.ellipse(tmp, (int(x + (w - 1) // 2), int(y + (h - 1) // 2)),
                        (max(1, (int(w) - 1) // 2), max(1, (int(h) - 1) // 2)), 0, 0, 360, 1, -1)
            if area >= 0.98 * w * h:
                n_rect += 1
                out[d][tmp > 0] = 1
            else:
                n_other += 1
                out[d][(tmp > 0) & (lab == k)] = 1
            max_w, max_h = max(max_w, int(w)), max(max_h, int(h))
    is_box = (n_rect + n_other > 0) if force is True else (False if force is False else (n_rect > 0 and n_rect >= 0.8 * (n_rect + n_other)))
    if not is_box:
        return (tumor > 0).astype(np.uint8), False, {}
    return out, True, {"max_w_px": max_w, "max_h_px": max_h}


def _crop_slices(tumor: np.ndarray, vs: Tuple[float, float, float], pad_mm: float = _WINDOW_PAD_MM):
    idx = np.argwhere(tumor)
    lo, hi = idx.min(0), idx.max(0) + 1
    pad = np.ceil(pad_mm / np.array(vs)).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(tumor.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _axis_variants(allow_flips: bool = False):
    for tr in (False, True):
        for fz in ((False, True) if allow_flips else (False,)):
            for fy in ((False, True) if allow_flips else (False,)):
                for fx in ((False, True) if allow_flips else (False,)):
                    yield (tr, fz, fy, fx)


def _apply_axes(m: np.ndarray, key: Tuple[bool, bool, bool, bool]) -> np.ndarray:
    tr, fz, fy, fx = key
    if m.ndim != 3:
        return m
    if tr:
        m = m.transpose(0, 2, 1)
    if fz:
        m = m[::-1]
    if fy:
        m = m[:, ::-1]
    if fx:
        m = m[:, :, ::-1]
    return m


def _key_name(key) -> str:
    tr, fz, fy, fx = key
    parts = (["transpose"] if tr else []) + (["flipZ"] if fz else []) + (["flipY"] if fy else []) + (["flipX"] if fx else [])
    return "+".join(parts) if parts else "identity"


def auto_align(anat: Dict[str, np.ndarray], tumor_bool: np.ndarray, mode: str = "auto"):
    ident = (False, False, False, False)
    lung = anat.get("lung")
    if mode == "identity" or lung is None or not tumor_bool.any():
        return ident, np.nan, np.nan
    idx = np.nonzero(tumor_bool)
    scores = {}
    for key in _axis_variants(allow_flips=(mode == "flips")):
        m = _apply_axes(lung, key)
        if m.shape == tumor_bool.shape:
            scores[key] = float((m[idx] > 0).mean())
    if ident not in scores:
        return ident, np.nan, np.nan
    s_id = scores[ident]
    best = max(scores, key=scores.get)
    if scores[best] - s_id >= 0.25 and scores[best] >= 0.6:
        return best, scores[best], s_id
    return ident, s_id, s_id


def load_anatomy(masks_dir: Path, pid: str) -> Dict[str, np.ndarray]:
    masks_dir = Path(masks_dir)
    raw: Dict[str, np.ndarray] = {}
    for p in sorted(masks_dir.glob(f"{pid}_*.npy")):
        name = p.stem[len(pid) + 1:]
        try:
            raw[name] = np.load(str(p), mmap_mode="r")
        except Exception:
            continue

    def _union(names):
        arrs = [np.asarray(raw[n]) > 0 for n in names if n in raw and raw[n].ndim == 3]
        if not arrs:
            return None
        out = arrs[0].copy()
        for a in arrs[1:]:
            if a.shape == out.shape:
                out |= a
        return out.astype(np.uint8)

    anat = dict(raw)
    for k, v in _LOBE_MAP.items():
        if k not in anat and v in raw:
            anat[k] = raw[v]
    if "lung" not in anat:
        lobes = [v for v in _LOBE_MAP.values() if v in raw] or [k for k in _LOBE_MAP if k in raw]
        u = _union(lobes)
        if u is not None:
            anat["lung"] = u
    if "vertebrae" not in anat:
        u = _union([n for n in raw if n.startswith("vertebrae")])
        if u is not None:
            anat["vertebrae"] = u
    return anat


def _mediastinum_proxy(lung_view: np.ndarray, z0: int, z1: int) -> np.ndarray:
    zs = z1 - z0
    H, W = lung_view.shape[1:]
    out = np.zeros((zs, H, W), dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for i in range(zs):
        l2 = np.ascontiguousarray((np.asarray(lung_view[z0 + i]) > 0).astype(np.uint8))
        if l2.sum() < 500:
            continue
        pts = cv2.findNonZero(l2)
        if pts is None:
            continue
        hull = cv2.convexHull(pts)
        hm = np.zeros(l2.shape, np.uint8)
        cv2.fillConvexPoly(hm, hull, 1)
        med = (hm > 0) & (l2 == 0)
        out[i] = cv2.morphologyEx(med.astype(np.uint8), cv2.MORPH_OPEN, k)
    return out


def handcrafted_radiomics(img: np.ndarray, mask: np.ndarray) -> Dict:
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return {}
    roi_img = img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(np.float32)
    roi_mask = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(bool)
    vals = roi_img[roi_mask]
    if len(vals) == 0:
        return {}
    m, s = vals.mean(), vals.std()
    h, _ = np.histogram(vals, bins=64, density=True)
    hp = h[h > 0]
    return {"rad_mean": float(m), "rad_std": float(s),
            "rad_skewness": float(((vals - m) ** 3).mean() / (s ** 3 + 1e-8)),
            "rad_kurtosis": float(((vals - m) ** 4).mean() / (s ** 4 + 1e-8)) - 3.0,
            "rad_entropy": float(-np.sum(hp * np.log2(hp + 1e-12)))}


def extract_patient_features(ct_volume: Optional[np.ndarray],
                             tumor_masks: np.ndarray,
                             anatomy_masks: Dict[str, np.ndarray],
                             tumor_mask_std=None,
                             pixel_spacing: float = 1.0,
                             slice_thickness: float = 1.0,
                             pixel_spacing_y: Optional[float] = None,
                             with_radiomics: bool = False,
                             align_mode: str = "auto",
                             tumor_is_box=None,
                             tumornet_mask=None,
                             anatomynet=None) -> Dict:
    psx = float(pixel_spacing or 1.0)
    psy = float(pixel_spacing_y or psx)
    st = float(slice_thickness or 1.0)
    vs = (st, psy, psx)
    feats: Dict = {}

    raw = tumor_masks > 0
    if not raw.any():
        feats.update({"tumor_size_2d_inplane_mm": 0.0, "tumor_size_2d_mm": 0.0, "tumor_size_3d_mm": 0.0,
                      "tumor_volume_3d_cm3": 0.0, "n_tumor_slices": 0, "tumor_depth_mm": 0.0})
        return feats


    tum, is_box, box = boxes_to_ellipses(tumor_masks, tumor_is_box)
    tum_b = tum > 0
    feats["tumor_mask_is_box"] = int(is_box)
    z_idx = np.where(tum_b.reshape(tum_b.shape[0], -1).any(axis=1))[0]
    n_sl = int(len(z_idx))
    z_extent = float((z_idx.max() - z_idx.min() + 1) * st)


    sizes, raw_diag = [], []
    for d in z_idx:
        for c in _contours(tum_b[d]):
            sizes.append(_feret_mm(c, psx, psy))
    inplane = float(max(sizes)) if sizes else 0.0
    if is_box:
        feats["tumor_size_box_side_mm"] = float(max(box["max_w_px"] * psx, box["max_h_px"] * psy))
        feats["tumor_size_box_diag_mm"] = float(np.hypot(box["max_w_px"] * psx, box["max_h_px"] * psy))
    feats["tumor_size_2d_inplane_mm"] = inplane
    feats["tumor_size_2d_mm"] = inplane
    feats["tumor_depth_mm"] = z_extent
    feats["n_tumor_slices"] = n_sl


    lab3, n3 = ndi.label(tum_b)
    if n3 > 1:
        sz3 = np.bincount(lab3.ravel())[1:]
        main = lab3 == (int(sz3.argmax()) + 1)
    else:
        main = tum_b
    vox = np.argwhere(main).astype(np.float64) * np.array(vs)
    if len(vox) > 2:
        if len(vox) > 3000:
            try:
                from scipy.spatial import ConvexHull
                vox = vox[ConvexHull(vox).vertices]
            except Exception:
                vox = vox[np.linspace(0, len(vox) - 1, 1500).astype(int)]
        feats["tumor_size_3d_mm"] = float(cdist(vox, vox).max())
    else:
        feats["tumor_size_3d_mm"] = inplane
    feats["tumor_volume_3d_cm3"] = float(tum_b.sum() * st * psy * psx / 1000.0)
    feats["tumor_size_max_mm"] = float(max(inplane, feats["tumor_size_3d_mm"]))


    key, score, score_id = auto_align(anatomy_masks, tum_b, align_mode)
    feats["align_choice"] = _key_name(key)
    feats["align_score"] = score
    feats["align_identity_score"] = score_id
    anat = {k: _apply_axes(v, key) for k, v in anatomy_masks.items() if v is not None and v.ndim == 3}
    anat = {k: v for k, v in anat.items() if v.shape == tum_b.shape}


    win = _crop_slices(tum_b, vs)
    t_c = tum_b[win]
    z0, z1 = win[0].start, win[0].stop

    def crop(name):
        m = anat.get(name)
        return None if m is None else (np.asarray(m[win]) > 0)

    present = {k: bool(np.asarray(v).any()) for k, v in anat.items() if k in _STRUCTS or k in LOBE_NAMES}
    organ_union = np.zeros_like(t_c, dtype=bool)
    for s in ["heart", "aorta", "trachea", "esophagus", "pulmonary_artery"]:
        m = crop(s)
        if m is not None:
            organ_union |= m
    med = None
    if "lung" in anat and present.get("lung"):
        med = _mediastinum_proxy(anat["lung"], z0, z1)[(slice(None), win[1], win[2])] > 0
        med = med | organ_union

    lung_c = crop("lung")
    feats["tumor_in_lung_frac"] = float(lung_c[t_c].mean()) if lung_c is not None and lung_c.any() else np.nan
    if lung_c is not None and lung_c.any():
        d_in = ndi.distance_transform_edt(lung_c, sampling=vs)
        feats["lung_depth_mm"] = float(d_in[t_c].min())
    else:
        feats["lung_depth_mm"] = np.nan

    for s in _STRUCTS:
        if s == "lung":
            sm = lung_c
        elif s == "mediastinum":
            sm = med
        else:
            sm = crop(s)
        have_any = (s == "mediastinum" and med is not None) or present.get(s, False)
        if sm is None or not have_any:
            feats[f"dist_{s}_mm"] = np.nan
            feats[f"contact5_{s}"] = np.nan
            feats[f"contact10_{s}"] = np.nan
            continue
        if not sm.any():
            feats[f"dist_{s}_mm"] = _DIST_CAP_MM
            feats[f"contact5_{s}"] = 0.0
            feats[f"contact10_{s}"] = 0.0
            continue
        dist = ndi.distance_transform_edt(~sm, sampling=vs)
        td = dist[t_c]
        feats[f"dist_{s}_mm"] = float(min(td.min(), _DIST_CAP_MM))
        feats[f"contact5_{s}"] = float((td <= 5.0).mean())
        feats[f"contact10_{s}"] = float((td <= 10.0).mean())


    d_t = ndi.distance_transform_edt(~t_c, sampling=vs)
    lab, n = ndi.label(d_t <= 5.0)
    comps = []
    lobe_c = {ln: crop(ln) for ln in LOBE_NAMES}
    for k in range(1, n + 1):
        comp = (lab == k) & t_c
        if comp.sum() < 30:
            continue
        best, best_o = "unknown", 0
        for ln, lm in lobe_c.items():
            if lm is None:
                continue
            o = int((comp & lm).sum())
            if o > best_o:
                best, best_o = ln, o
        comps.append(best)
    feats["n_tumor_components"] = len(comps)
    r = [c for c in comps if c in RIGHT_LOBES]
    l = [c for c in comps if c in LEFT_LOBES]
    same_lobe = (len(r) > len(set(r))) or (len(l) > len(set(l)))
    diff_ipsi = (len(set(r)) > 1) or (len(set(l)) > 1)
    contra = bool(r) and bool(l)
    feats["multi_same_lobe"] = int(same_lobe)
    feats["multi_diff_lobe_ipsi"] = int(diff_ipsi)
    feats["multi_contralateral"] = int(contra)
    feats["is_multifocal"] = int(diff_ipsi)


    try:
        from anatomy_v2 import extract_v2
        tn_full = None
        if tumornet_mask is not None and np.asarray(tumornet_mask).shape == tum_b.shape:
            tn_full = _apply_axes(np.asarray(tumornet_mask), key) > 0
        feats.update(extract_v2(tum_b, anat, vs, win, med, tn_full, size_max_mm=feats.get("tumor_size_max_mm", 0.0)))
    except Exception as e:
        feats["a2_error"] = repr(e)[:120]

    try:
        from paper_rules import paper_features
        if anatomynet is not None and np.asarray(anatomynet[0]).shape == tum_b.shape and np.asarray(anatomynet[1]).shape == tum_b.shape:
            lung_n = np.asarray(anatomynet[0]) > 0
            medi_n = (np.asarray(anatomynet[1]) > 0)[win]
            feats.update(paper_features(tum_b, {"lung": lung_n, "trachea": anat.get("trachea")}, vs, win, medi_n))
            feats["paper_masks_src"] = 1
        else:
            feats.update(paper_features(tum_b, anat, vs, win, med))
            feats["paper_masks_src"] = 0
    except Exception as e:
        feats["paper_error"] = repr(e)[:120]


    if with_radiomics and ct_volume is not None:
        rows = [handcrafted_radiomics(ct_volume[d], tum_b[d]) for d in z_idx]
        rows = [x for x in rows if x]
        if rows:
            df = pd.DataFrame(rows)
            for c in df.columns:
                feats[f"mean_{c}"] = float(df[c].mean())
    return feats


def extract_all_patients(patient_manifest: pd.DataFrame, preproc_dir: str, with_radiomics: bool = False) -> pd.DataFrame:
    preproc_dir = Path(preproc_dir)
    rows: List[Dict] = []
    for _, row in patient_manifest.iterrows():
        pid = str(row["pid"])
        print(f"  Extracting: {pid}", end=" ")
        vol_path = preproc_dir / "volumes" / f"{pid}.npy"
        tumor_path = preproc_dir / "tumor_masks" / f"{pid}_tumor.npy"
        masks_dir = preproc_dir / "anatomy_masks"
        if not tumor_path.exists():
            print("-> missing tumor mask, skipping")
            rows.append({"pid": pid, "T_stage": row.get("T_stage", "Unknown")})
            continue
        tumor = np.load(str(tumor_path))
        vol = np.load(str(vol_path), mmap_mode="r") if (with_radiomics and vol_path.exists()) else None
        anat = load_anatomy(masks_dir, pid)
        tn_path = preproc_dir / "tumornet_masks" / f"{pid}_tumornet.npy"
        tn_mask = np.load(str(tn_path)) if tn_path.exists() else None
        nets = None
        lp, mp = preproc_dir / "anatomynet_masks" / f"{pid}_lungnet.npy", preproc_dir / "anatomynet_masks" / f"{pid}_medinet.npy"
        if lp.exists() and mp.exists():
            nets = (np.load(str(lp)), np.load(str(mp)))
        src_file = preproc_dir / "tumor_masks" / f"{pid}_tumor_src.txt"
        is_box = True if (src_file.exists() and src_file.read_text().strip() == "box") else None
        psx = float(row.get("pixel_spacing_x", 1.0) or 1.0)
        psy = float(row.get("pixel_spacing_y", psx) or psx)
        st = float(row.get("slice_thickness", 1.0) or 1.0)
        try:
            feats = extract_patient_features(vol, tumor, anat, None, pixel_spacing=psx, slice_thickness=st,
                                             pixel_spacing_y=psy, with_radiomics=with_radiomics, tumor_is_box=is_box,
                                             tumornet_mask=tn_mask, anatomynet=nets)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"-> error: {e}")
            feats = {}
        feats["pid"] = pid
        feats["T_stage"] = row.get("T_stage", "Unknown")
        rows.append(feats)
        print(f"-> size={feats.get('tumor_size_2d_inplane_mm', 0):.1f} mm "
              f"(box-diag {feats.get('tumor_size_box_diag_mm', float('nan')):.1f}) "
              f"in-lung={feats.get('tumor_in_lung_frac', float('nan')):.2f} align={feats.get('align_choice', '-')}")
    df = pd.DataFrame(rows)
    print(f"\nExtracted {len(df)} patients, {len(df.columns)} columns")
    return df
