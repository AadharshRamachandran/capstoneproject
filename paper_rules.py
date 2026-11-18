from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi

CAP = 100.0


def _z_info(anat):
    lung = anat.get("lung")
    if lung is None:
        return None
    zl = np.where(np.asarray(lung > 0).reshape(lung.shape[0], -1).any(1))[0]
    if len(zl) < 5:
        return None
    z0, z1 = int(zl.min()), int(zl.max())
    apex_at_max = True
    tr = anat.get("trachea")
    if tr is not None:
        zt = np.where(np.asarray(tr > 0).reshape(tr.shape[0], -1).any(1))[0]
        if len(zt) >= 3:
            apex_at_max = bool(zt.mean() > (z0 + z1) / 2.0)
    return z0, z1, apex_at_max


def paper_features(t_full: np.ndarray, anat: Dict[str, np.ndarray], vs, win, med: Optional[np.ndarray]) -> Dict[str, float]:
    f = {"paper_d_medi_mm": np.nan, "paper_d_lung_mm": np.nan, "paper_outside_lung_frac": np.nan, "paper_dia_gap_mm": np.nan}
    if not t_full.any():
        return f
    t = t_full[win]
    zs = np.where(t.reshape(t.shape[0], -1).any(1))[0]
    sp2 = (vs[1], vs[2])
    lung = anat.get("lung")
    lung_c = (np.asarray(lung[win]) > 0) if lung is not None and np.asarray(lung).any() else None

    def d2min(S, to_inside=False):
        best = np.inf
        for z in zs:
            if S[z].any():
                d = ndi.distance_transform_edt(S[z] if to_inside else ~S[z], sampling=sp2)
                best = min(best, float(d[t[z]].min()))
            elif to_inside:
                best = 0.0
        return min(best, CAP) if np.isfinite(best) else np.nan

    if med is not None and med.any():
        f["paper_d_medi_mm"] = d2min(med)
    if lung_c is not None and lung_c.any():
        f["paper_d_lung_mm"] = d2min(lung_c, to_inside=True)
        f["paper_outside_lung_frac"] = float((t & ~lung_c).sum() / max(t.sum(), 1))
    zi = _z_info(anat)
    if zi is not None:
        z0, z1, apex_at_max = zi
        h = max(z1 - z0, 1)
        band = 0.10 * h
        tz = np.where(t_full.reshape(t_full.shape[0], -1).any(1))[0]
        dia_z = z0 if apex_at_max else z1
        edge = dia_z + band if apex_at_max else dia_z - band
        tz_near = tz.min() if apex_at_max else tz.max()
        gap = (tz_near - edge) if apex_at_max else (edge - tz_near)
        f["paper_dia_gap_mm"] = float(max(0.0, gap) * vs[0])
    return f


def apply_rules(size_mm, d_medi, d_dia, d_lung, outside_frac, touch_mm=0.0, use_medi=True, use_dia=True,
                t1_needs_surround=True, outside_tol=0.02):
    size = np.asarray(size_mm, float)
    inv = np.zeros(size.shape, bool)
    if use_medi:
        inv |= np.nan_to_num(np.asarray(d_medi, float), nan=np.inf) <= touch_mm
    if use_dia:
        inv |= np.nan_to_num(np.asarray(d_dia, float), nan=np.inf) <= touch_mm
    surrounded = (np.nan_to_num(np.asarray(d_lung, float), nan=0.0) > touch_mm) & (np.nan_to_num(np.asarray(outside_frac, float), nan=1.0) <= outside_tol)
    t = np.where((size > 70) | inv, 3, np.where(size > 50, 2, np.where(size > 30, 1, np.where(surrounded | (not t1_needs_surround), 0, 1))))
    t = np.where(np.isfinite(size), t, -1)
    return t.astype(int)
