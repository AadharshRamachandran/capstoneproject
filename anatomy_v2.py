from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

VITAL = ["heart", "aorta", "pulmonary_artery", "trachea", "esophagus", "vertebrae"]
CORE = VITAL + ["mediastinum"]
OPTIONAL = {
    "ribs": ("rib_", "rib"), "sternum": ("sternum", "costal_cartilages"), "svc": ("superior_vena_cava",),
    "brachiocephalic": ("brachiocephalic", "subclavian"), "atrium_left": ("heart_atrium_left",),
    "pulm_vein": ("pulmonary_vein",), "ivc": ("inferior_vena_cava",),
}
LOBES = ["lobe_LUL", "lobe_LLL", "lobe_RUL", "lobe_RML", "lobe_RLL"]
CAP = 50.0
INV_D, INV_SURF, GATE_MM = 1.5, 0.05, 50.0


def _union(anat: Dict[str, np.ndarray], prefixes, win) -> Optional[np.ndarray]:
    out = None
    for k, v in anat.items():
        if any(k == p or k.startswith(p) for p in prefixes) and getattr(v, "ndim", 0) == 3:
            a = np.asarray(v[win]) > 0
            out = a if out is None else (out | a)
    return out


def _dist2d_min(S: np.ndarray, t: np.ndarray, sp2: Tuple[float, float]) -> float:
    best = np.inf
    for z in np.where(t.reshape(t.shape[0], -1).any(1))[0]:
        if not S[z].any():
            continue
        d = ndi.distance_transform_edt(~S[z], sampling=sp2)
        best = min(best, float(d[t[z]].min()))
    return best if np.isfinite(best) else CAP


def struct_feats(S: np.ndarray, t: np.ndarray, vs, surf: np.ndarray, tag: str, pre: str = "a2_") -> Dict[str, float]:
    nan = {f"{pre}d3_{tag}": np.nan, f"{pre}d2_{tag}": np.nan, f"{pre}ov_{tag}": np.nan,
           f"{pre}surf2_{tag}": np.nan, f"{pre}inv_{tag}": np.nan}
    if S is None:
        return nan
    if not S.any():
        return {f"{pre}d3_{tag}": CAP, f"{pre}d2_{tag}": CAP, f"{pre}ov_{tag}": 0.0, f"{pre}surf2_{tag}": 0.0, f"{pre}inv_{tag}": 0.0}
    d3 = ndi.distance_transform_edt(~S, sampling=vs)
    td = d3[t]
    dmin3 = float(min(td.min(), CAP))
    dmin2 = float(min(_dist2d_min(S, t, (vs[1], vs[2])), CAP))
    surf2 = float((d3[surf] <= 2.0).mean()) if surf.any() else 0.0
    return {f"{pre}d3_{tag}": dmin3, f"{pre}d2_{tag}": dmin2, f"{pre}ov_{tag}": float((S & t).sum() / max(t.sum(), 1)),
            f"{pre}surf2_{tag}": surf2,
            f"{pre}inv_{tag}": float(dmin3 <= INV_D and dmin2 <= INV_D and surf2 >= INV_SURF)}


def _surface(t: np.ndarray) -> np.ndarray:
    return t & ~ndi.binary_erosion(t, iterations=1, border_value=0)


def _feret_xy(m2: np.ndarray, psx: float, psy: float) -> float:
    import cv2
    cs, _ = cv2.findContours(m2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = 0.0
    for c in cs:
        p = c.reshape(-1, 2).astype(np.float64) * np.array([psx, psy])
        if len(p) > 1:
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            best = max(best, float(d.max()))
    return best


def lung_geometry(anat, vs, t_zc: float) -> Dict[str, float]:
    out = {k: np.nan for k in ("a2_dist_diaphragm_mm", "a2_dist_apex_mm", "a2_dist_carina_mm", "a2_below_carina_mm", "a2_z_rel")}
    lung = anat.get("lung")
    if lung is None:
        return out
    zl = np.where(np.asarray(lung > 0).reshape(lung.shape[0], -1).any(1))[0]
    if len(zl) < 5:
        return out
    z0, z1 = int(zl.min()), int(zl.max())
    h = max(z1 - z0, 1)
    apex_at_max = None
    tr = anat.get("trachea")
    ztr = None
    if tr is not None:
        ztr = np.where(np.asarray(tr > 0).reshape(tr.shape[0], -1).any(1))[0]
        if len(ztr) >= 3:
            apex_at_max = bool(ztr.mean() > (z0 + z1) / 2.0)
    if apex_at_max is None:
        apex_at_max = True
    st = vs[0]
    apex_z, dia_z = (z1, z0) if apex_at_max else (z0, z1)
    out["a2_dist_apex_mm"] = float(abs(t_zc - apex_z) * st)
    out["a2_dist_diaphragm_mm"] = float(abs(t_zc - dia_z) * st)
    out["a2_z_rel"] = float(abs(t_zc - apex_z) / h)
    if ztr is not None and len(ztr) >= 3:
        zc = int(ztr.min() if apex_at_max else ztr.max())
        pts = np.argwhere(np.asarray(tr[zc]) > 0)
        if len(pts):
            cy, cx = pts.mean(0)
            out["a2_carina"] = (zc, cy, cx)
            out["a2_below_carina_mm"] = float((zc - t_zc if apex_at_max else t_zc - zc) * st)
    return out


def extract_v2(t_full: np.ndarray, anat: Dict[str, np.ndarray], vs, win, med: Optional[np.ndarray],
               tn_full: Optional[np.ndarray] = None, size_max_mm: float = 0.0, box_area_mm2: float = np.nan) -> Dict[str, float]:
    f: Dict[str, float] = {}
    t = t_full[win]
    surf = _surface(t)
    zc = float(np.argwhere(t_full.reshape(t_full.shape[0], -1).any(1)).mean()) if t_full.any() else 0.0

    masks = {}
    for s in VITAL:
        a = anat.get(s)
        masks[s] = None if a is None or not np.asarray(a).any() else (np.asarray(a[win]) > 0)
    masks["mediastinum"] = med
    for o, pre in OPTIONAL.items():
        u = _union(anat, pre, win)
        if u is not None and u.any():
            masks[o] = u
    lung = anat.get("lung")
    lung_w = (np.asarray(lung[win]) > 0) if lung is not None and np.asarray(lung).any() else None

    for tag in list(masks):
        f.update(struct_feats(masks[tag], t, vs, surf, tag))
    for o in OPTIONAL:
        if o not in masks:
            f.update(struct_feats(None, t, vs, surf, o))

    vit = [tag for tag in VITAL if masks.get(tag) is not None]
    inv = [f[f"a2_inv_{k}"] for k in vit if np.isfinite(f.get(f"a2_inv_{k}", np.nan))]
    f["a2_n_inv"] = float(np.nansum(inv)) if inv else np.nan
    f["a2_inv_any"] = float(np.nanmax(inv)) if inv else np.nan
    f["a2_inv_gated"] = float(f["a2_inv_any"] * (size_max_mm >= GATE_MM)) if inv else np.nan
    d3 = [f[f"a2_d3_{k}"] for k in vit]
    d2 = [f[f"a2_d2_{k}"] for k in vit]
    sf = [f[f"a2_surf2_{k}"] for k in vit]
    f["a2_min_vital_d3"] = float(np.min(d3)) if d3 else np.nan
    f["a2_min_vital_d2"] = float(np.min(d2)) if d2 else np.nan
    f["a2_surf2_vital"] = float(np.max(sf)) if sf else np.nan

    g = lung_geometry(anat, vs, zc)
    car = g.pop("a2_carina", None)
    f.update(g)
    if car is not None and t.any():
        zc_i, cy, cx = car
        idx = np.argwhere(t_full)
        if len(idx) > 4000:
            idx = idx[np.linspace(0, len(idx) - 1, 4000).astype(int)]
        d = np.sqrt(((idx[:, 0] - zc_i) * vs[0]) ** 2 + ((idx[:, 1] - cy) * vs[1]) ** 2 + ((idx[:, 2] - cx) * vs[2]) ** 2)
        f["a2_dist_carina_mm"] = float(d.min())
    else:
        f["a2_dist_carina_mm"] = np.nan


    n_l, any_l = 0, False
    for ln in LOBES:
        a = anat.get(ln)
        if a is None:
            continue
        any_l = True
        if (np.asarray(a[win])[t] > 0).mean() >= 0.05:
            n_l += 1
    f["a2_n_lobes"] = float(n_l) if any_l else np.nan
    f["a2_fissure_cross"] = float(n_l >= 2) if any_l else np.nan


    if lung_w is not None and lung_w.any() and surf.any():
        d_out = ndi.distance_transform_edt(lung_w, sampling=vs)
        near_edge = (d_out[surf] <= 2.0)
        if med is not None and med.any():
            d_med = ndi.distance_transform_edt(~med, sampling=vs)[surf]
            toward_med = d_med <= 10.0
        else:
            toward_med = np.zeros_like(near_edge)
        f["a2_chestwall_contact"] = float((near_edge & ~toward_med).mean())
        f["a2_med_pleura_contact"] = float((near_edge & toward_med).mean())
    else:
        f["a2_chestwall_contact"] = f["a2_med_pleura_contact"] = np.nan


    if tn_full is not None and tn_full.any():
        tn = tn_full[win] > 0
        tsurf = _surface(tn)
        f["a2_tn_size_mm"], f["a2_tn_vol_cm3"] = tn_size_vol(tn_full > 0, vs)
        f["a2_tn_fill"] = float(f["a2_tn_vol_cm3"] * 1000.0 / max(t_full.sum() * vs[0] * vs[1] * vs[2], 1e-6))
        f["a2_tn_in_lung"] = float(lung_w[tn].mean()) if lung_w is not None and lung_w.any() and tn.any() else np.nan
        for tag in ("heart", "aorta", "pulmonary_artery", "trachea", "esophagus", "vertebrae", "mediastinum"):
            r = struct_feats(masks.get(tag), tn, vs, tsurf, tag, pre="a2_tn_") if tn.any() else {}
            f.update({k: v for k, v in r.items() if "d3_" in k or "surf2_" in k or "inv_" in k})
        if lung_w is not None and lung_w.any() and tn.any():
            f["a2_tn_lung_depth"] = float(ndi.distance_transform_edt(lung_w, sampling=vs)[tn].min())
    return f


def tn_size_vol(tn: np.ndarray, vs) -> Tuple[float, float]:
    best = 0.0
    for z in np.where(tn.reshape(tn.shape[0], -1).any(1))[0]:
        best = max(best, _feret_xy(tn[z], vs[2], vs[1]))
    return best, float(tn.sum() * vs[0] * vs[1] * vs[2] / 1000.0)
