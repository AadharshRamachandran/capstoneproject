from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="Skipping features without any observed values")
_T_LABELS = ["T1", "T2", "T3", "T4"]
_T_TO_IDX = {t: i for i, t in enumerate(_T_LABELS)}
_ML_MODEL = None
try:
    import gpu_utils as _gpu
except Exception:
    class _gpu:
        xgb_kwargs = staticmethod(lambda: {})
        catboost_kwargs = staticmethod(lambda: {})
        torch_device = staticmethod(lambda: "cpu")
_FEATURE_SET_VERSION = 15
FEATURE_PROFILE = os.environ.get("TNM_FEATURE_PROFILE", "image")
CONFORMAL_ALPHA = 0.10
OVERRIDE_MODES = ("none", "size", "full")


SIZE_T1, SIZE_T2, SIZE_T3 = 30.0, 50.0, 70.0
_VITAL = ["heart", "aorta", "pulmonary_artery", "trachea", "esophagus", "vertebrae"]


BOX_SIZE_FEATURES = ["f_size", "f_size_max", "f_size3d", "f_logvol", "f_side", "f_depth", "f_nsl", "f_rule_T", "f_rule_T_max"]


SEG_FEATURES = ["f_seg_roi", "f_seg_full", "f_seg_mean", "f_seg_rule_T", "f_seg_vol", "f_seg_fill", "f_seg_shrink"]


SOLID_FEATURES = ["f_solid_open", "f_solid_raw", "f_solid_any", "f_solid_rule_T", "f_solid_frac", "f_ggo_frac",
                  "f_solid_hu", "f_solid_vol", "f_solid_to_box", "f_solid_rect", "f_solid_eq", "f_no_solid"]
SIZE_FEATURES = BOX_SIZE_FEATURES + SEG_FEATURES + SOLID_FEATURES
SHAPE_FEATURES = ["f_sphericity", "f_elongation", "f_flatness"]
CONTACT_FEATURES = [
    "f_lung_depth", "f_dist_medi", "f_dist_heart", "f_dist_vessel", "f_dist_airway_eso", "f_dist_vert",
    "f_contact_medi", "f_contact_heart",
    "f_pleural_contact2",
    "f_contact2_medi",
    "f_contact2_heart",
]
INTERACTION_FEATURES = [
    "f_size_depth_ratio",
    "f_size_x_medi",
    "f_size_x_vessel",
    "f_pleu_flag",
]
MULTI_FEATURES = ["f_multi_ipsi_diff", "f_multi_same_lobe", "f_in_lung"]


_VITAL_V2 = ["heart", "aorta", "pulmonary_artery", "trachea", "esophagus", "vertebrae"]
ANAT2_FEATURES = (["f2_inv_n", "f2_inv_any", "f2_inv_gated", "f2_min_vital_d3", "f2_min_vital_d2", "f2_surf_vital", "f2_ov_vital"]
                  + [f"f2_surf_{s}" for s in _VITAL_V2 + ["mediastinum"]]
                  + ["f2_d2_heart", "f2_d2_great", "f2_d2_airway", "f2_d2_vert"]
                  + ["f2_dia", "f2_apex", "f2_carina", "f2_below_carina", "f2_z_rel", "f2_n_lobes", "f2_fissure",
                     "f2_chestwall", "f2_med_pleura"]
                  + ["f2_d_ribs", "f2_surf_ribs", "f2_d_sternum", "f2_d_svc", "f2_d_brachio", "f2_d_atrium_l", "f2_d_pv"]
                  + ["f2_tn_size", "f2_tn_fill", "f2_tn_min_vital_d3", "f2_tn_surf_vital", "f2_tn_depth", "f2_tn_inv_any"])

YOLO_FEATURES = ["f_y_solid", "f_y_box", "f_y_dz", "f_y_conf", "f_y_conf_max", "f_y_iou", "f_y_n_lesions", "f_y_size_mean", "f_y_size_gap", "f_y_solid_frac"]
FULL_FEATURES = SIZE_FEATURES + SHAPE_FEATURES + CONTACT_FEATURES + INTERACTION_FEATURES + MULTI_FEATURES + ANAT2_FEATURES + YOLO_FEATURES
MODEL_FEATURES = FULL_FEATURES

_MONOTONE_UP = {"f_y_solid", "f_y_size_mean", "f2_inv_n", "f2_inv_any", "f2_inv_gated", "f2_tn_size",
                "f_size", "f_size_max", "f_size3d", "f_logvol", "f_rule_T", "f_rule_T_max",
                "f_seg_roi", "f_seg_full", "f_seg_mean", "f_seg_rule_T", "f_seg_vol",
                "f_solid_open", "f_solid_raw", "f_solid_any", "f_solid_rule_T", "f_solid_vol", "f_solid_rect", "f_solid_eq"}


_T_NORM4 = {t: t for t in _T_LABELS}


def feature_names(profile=None, columns=None):
    selected = profile or FEATURE_PROFILE
    if selected == "image":

        available = [] if columns is None else list(columns)
        image_cols = sorted(c for c in available if str(c).startswith("fmcib_"))
        return FULL_FEATURES + image_cols
    return list(SIZE_FEATURES if selected == "size" else FULL_FEATURES)


def _norm4(label) -> str:
    s = str(label).strip().upper()
    if not s:
        return ""
    if s[0] in "1234":
        s = "T" + s
    if s in ("TX", "T0", "UNKNOWN", "NAN"):
        return ""
    for t in _T_LABELS:
        if s.startswith(t):
            return t
    return ""


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(default, index=df.index, dtype=float)


def rule_t_from_size(size_mm):
    s = np.nan_to_num(np.asarray(size_mm, dtype=float), nan=0.0)
    return np.where(s <= SIZE_T1, 1, np.where(s <= SIZE_T2, 2, np.where(s <= SIZE_T3, 3, 4))).astype(float)


def merge_box_features(df: pd.DataFrame, csv_path, min_coverage: float = 0.9) -> tuple:
    box = pd.read_csv(csv_path)
    if "pid" not in box.columns:
        raise ValueError("box features CSV needs a 'pid' column")
    box["pid"] = box["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    keep = ["pid"] + [c for c in box.columns if str(c).startswith(("seg_", "solid_"))] + [c for c in ("box_side_mm",) if c in box.columns]
    box = box[keep].drop_duplicates("pid", keep="first")
    for c in keep[1:]:
        box[c] = pd.to_numeric(box[c], errors="coerce")
    out = df.drop(columns=[c for c in keep[1:] if c in df.columns], errors="ignore").copy()
    out["pid"] = out["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    out = out.merge(box, on="pid", how="left")
    chk = [c for c in keep[1:] if (c.startswith("seg_") and c.endswith("_size_mm")) or c == "solid_open_mm"]
    have = pd.Series(False, index=out.index)
    for c in chk:
        have |= out[c].notna() if c == "solid_open_mm" else out[c].gt(0)
    n = int(have.sum()) if chk else 0
    if len(out) and n / len(out) < min_coverage:
        print(f"  WARNING: {Path(str(csv_path)).name} has values for only {n}/{len(out)} patients (< {min_coverage:.0%}) -> NOT used. "
              f"Finish the extraction for all shards first (a partial column acts as a shortcut, not a measurement).")
        return df, 0
    return out, n


def merge_yolo_summary(df: pd.DataFrame, csv_path, min_coverage: float = 0.9) -> tuple:
    y = pd.read_csv(csv_path)
    if "pid" not in y.columns:
        raise ValueError("yolo summary CSV needs a 'pid' column")
    y["pid"] = y["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    keep = ["pid"] + [c for c in y.columns if str(c).startswith("yolo_")]
    y = y[keep].drop_duplicates("pid", keep="first")
    for c in keep[1:]:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    out = df.drop(columns=[c for c in keep[1:] if c in df.columns], errors="ignore").copy()
    out["pid"] = out["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    out = out.merge(y, on="pid", how="left")
    n = int(out["yolo_n_boxes"].notna().sum()) if "yolo_n_boxes" in out else 0
    if len(out) and n / len(out) < min_coverage:
        print(f"  WARNING: {Path(str(csv_path)).name} covers only {n}/{len(out)} patients (< {min_coverage:.0%}) -> NOT used.")
        return df, 0
    return out, n


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)


    size = _col(df, "tumor_size_2d_inplane_mm").clip(0, 180)
    size = size.where(size > 0)
    s3   = _col(df, "tumor_size_3d_mm").clip(0, 250)
    vol  = _col(df, "tumor_volume_3d_cm3").clip(lower=0)
    side = _col(df, "tumor_size_box_side_mm").clip(0, 200)
    dep  = _col(df, "tumor_depth_mm").clip(0, 300)

    smax_col = _col(df, "tumor_size_max_mm").clip(0, 250)
    smax_alt = pd.concat([size, s3.where(s3 > 0), dep.where(dep > 0)], axis=1).max(axis=1)
    smax = smax_col.where(smax_col > 0, smax_alt)
    smax = pd.concat([smax, size], axis=1).max(axis=1)
    out["f_size"]       = size
    out["f_size_max"]   = smax.where(smax > 0)
    out["f_size3d"]     = s3.where(s3 > 0)
    out["f_logvol"]     = np.log1p(vol.where(vol > 0))
    out["f_rule_T"]     = rule_t_from_size(size.fillna(0.0))
    out["f_rule_T_max"] = rule_t_from_size(smax.fillna(0.0))
    out["f_side"]       = side.where(side > 0)
    out["f_depth"]      = dep
    out["f_nsl"]        = _col(df, "n_tumor_slices").clip(0, 300)


    def _pos(name, cap):
        v = _col(df, name).clip(0, cap)
        return v.where(v > 0)
    seg_roi, seg_full = _pos("seg_roi_size_mm", 250), _pos("seg_full_size_mm", 250)
    seg_mean = pd.concat([seg_roi, seg_full], axis=1).mean(axis=1)
    out["f_seg_roi"]    = seg_roi
    out["f_seg_full"]   = seg_full
    out["f_seg_mean"]   = seg_mean
    out["f_seg_rule_T"] = pd.Series(rule_t_from_size(seg_mean.fillna(0.0)), index=df.index).where(seg_mean.notna())
    seg_vol = pd.concat([_pos("seg_roi_vol_cm3", 1e5), _pos("seg_full_vol_cm3", 1e5)], axis=1).mean(axis=1)
    out["f_seg_vol"]    = np.log1p(seg_vol)
    out["f_seg_fill"]   = pd.concat([_col(df, "seg_roi_fill_at_max"), _col(df, "seg_full_fill_at_max")], axis=1).mean(axis=1).clip(0, 1.5)
    out["f_seg_shrink"] = (seg_mean / _pos("box_side_mm", 250).where(lambda v: v > 0, side.where(side > 0))).clip(0, 2)


    def _nn(name, cap):
        return _col(df, name).clip(0, cap)
    s_open, s_raw, s_any = _nn("solid_open_mm", 250), _nn("solid_raw_mm", 250), _nn("solid_any_mm", 250)
    out["f_solid_open"]   = s_open
    out["f_solid_raw"]    = s_raw
    out["f_solid_any"]    = s_any
    out["f_solid_rule_T"] = pd.Series(rule_t_from_size(s_open.fillna(0.0)), index=df.index).where(s_open.notna())
    out["f_solid_frac"]   = _col(df, "solid_frac").clip(0, 1)
    out["f_ggo_frac"]     = _col(df, "solid_ggo_frac").clip(0, 1)
    out["f_solid_hu"]     = _col(df, "solid_hu_mean").clip(-400, 400)
    out["f_solid_vol"]    = np.log1p(_nn("solid_vol_cm3", 1e5))
    out["f_solid_rect"]   = _nn("solid_rect_mm", 250)
    out["f_solid_eq"]     = _nn("solid_eq_mm", 250)
    out["f_no_solid"]     = (s_open == 0).astype(float)
    out["f_solid_to_box"] = (s_open / _pos("box_side_mm", 250).where(lambda v: v > 0, side.where(side > 0))).clip(0, 1.5)


    y_s = _nn("yolo_solid_open_mm", 250)
    out["f_y_solid"]      = y_s
    out["f_y_box"]        = _nn("yolo_box_side_mm", 250)
    out["f_y_dz"]         = _nn("yolo_solid_dz_mm", 400)
    out["f_y_conf"]       = _col(df, "yolo_conf_mean").clip(0, 1)
    out["f_y_conf_max"]   = _col(df, "yolo_conf_max").clip(0, 1)
    out["f_y_iou"]        = _col(df, "yolo_iou3d").clip(0, 1)
    out["f_y_n_lesions"]  = _col(df, "yolo_n_lesions").clip(0, 10)
    out["f_y_size_mean"]  = pd.concat([s_open, y_s], axis=1).mean(axis=1)
    out["f_y_size_gap"]   = (y_s - s_open).abs().clip(0, 250)
    out["f_y_solid_frac"] = _col(df, "yolo_solid_frac").clip(0, 1)


    sph_col = _col(df, "sphericity_3d")
    elo_col = _col(df, "elongation_3d")
    flt_col = _col(df, "flatness_3d")
    vol_sphere = (4.0 / 3.0) * np.pi * ((s3.fillna(0.0) / 2.0) ** 3) / 1000.0
    sph_proxy  = (vol.where(vol > 0) / vol_sphere.where(vol_sphere > 0)).clip(0, 1)
    out["f_sphericity"] = sph_col.where(sph_col.notna(), other=sph_proxy)
    elo_proxy = (side.where(side > 0) / s3.where(s3 > 0)).clip(0, 1)
    out["f_elongation"] = elo_col.where(elo_col.notna(), other=elo_proxy)
    flt_proxy = (dep.where(dep > 0) / s3.where(s3 > 0)).clip(0, 1)
    out["f_flatness"]   = flt_col.where(flt_col.notna(), other=flt_proxy)


    ld  = _col(df, "lung_depth_mm").clip(0, 50)
    dm  = _col(df, "dist_mediastinum_mm").clip(0, 50)
    dh  = _col(df, "dist_heart_mm").clip(0, 50)
    dv  = pd.concat([_col(df, "dist_aorta_mm"), _col(df, "dist_pulmonary_artery_mm")], axis=1).min(axis=1).clip(0, 50)
    da  = pd.concat([_col(df, "dist_trachea_mm"), _col(df, "dist_esophagus_mm")], axis=1).min(axis=1).clip(0, 50)
    dve = _col(df, "dist_vertebrae_mm").clip(0, 50)
    cm5 = _col(df, "contact5_mediastinum")
    ch5 = _col(df, "contact5_heart")
    out["f_lung_depth"]      = ld
    out["f_dist_medi"]       = dm
    out["f_dist_heart"]      = dh
    out["f_dist_vessel"]     = dv
    out["f_dist_airway_eso"] = da
    out["f_dist_vert"]       = dve
    out["f_contact_medi"]    = cm5
    out["f_contact_heart"]   = ch5
    out["f_pleural_contact2"]= _col(df, "contact2_lung")
    out["f_contact2_medi"]   = _col(df, "contact2_mediastinum")
    out["f_contact2_heart"]  = _col(df, "contact2_heart")


    sz_safe  = size.fillna(0.0)
    dep_safe = dep.fillna(1.0).replace(0.0, 1.0)
    out["f_size_depth_ratio"] = (sz_safe / dep_safe).clip(0, 10)
    out["f_size_x_medi"]   = (sz_safe * cm5.fillna(0.0)).clip(0, 180)
    out["f_size_x_vessel"] = (sz_safe * (1.0 - dv.fillna(50.0) / 50.0).clip(0, 1)).clip(0, 180)
    out["f_pleu_flag"] = ((ld.fillna(99.0) <= 3.0) & (sz_safe > SIZE_T1)).astype(float)


    out["f_multi_ipsi_diff"] = _col(df, "multi_diff_lobe_ipsi", 0.0).fillna(0.0)
    out["f_multi_same_lobe"] = _col(df, "multi_same_lobe", 0.0).fillna(0.0)
    out["f_in_lung"]         = _col(df, "tumor_in_lung_frac")


    def _a(name):
        return pd.to_numeric(_col(df, name), errors="coerce").replace([np.inf, -np.inf], np.nan)
    def _mx(names):
        return pd.concat([_a(n) for n in names], axis=1).max(axis=1, skipna=True) if names else pd.Series(np.nan, index=df.index)
    def _mn(names):
        return pd.concat([_a(n) for n in names], axis=1).min(axis=1, skipna=True) if names else pd.Series(np.nan, index=df.index)
    out["f2_inv_n"], out["f2_inv_any"], out["f2_inv_gated"] = _a("a2_n_inv"), _a("a2_inv_any"), _a("a2_inv_gated")
    out["f2_min_vital_d3"], out["f2_min_vital_d2"] = _a("a2_min_vital_d3").clip(0, 50), _a("a2_min_vital_d2").clip(0, 50)
    out["f2_surf_vital"] = _a("a2_surf2_vital")
    out["f2_ov_vital"] = _mx([f"a2_ov_{s}" for s in _VITAL_V2])
    for s in _VITAL_V2 + ["mediastinum"]:
        out[f"f2_surf_{s}"] = _a(f"a2_surf2_{s}")
    out["f2_d2_heart"] = _a("a2_d2_heart").clip(0, 50)
    out["f2_d2_great"] = _mn(["a2_d2_aorta", "a2_d2_pulmonary_artery", "a2_d2_svc", "a2_d2_brachiocephalic"]).clip(0, 50)
    out["f2_d2_airway"] = _mn(["a2_d2_trachea", "a2_d2_esophagus"]).clip(0, 50)
    out["f2_d2_vert"] = _a("a2_d2_vertebrae").clip(0, 50)
    out["f2_dia"], out["f2_apex"] = _a("a2_dist_diaphragm_mm").clip(0, 400), _a("a2_dist_apex_mm").clip(0, 400)
    out["f2_carina"], out["f2_below_carina"] = _a("a2_dist_carina_mm").clip(0, 300), _a("a2_below_carina_mm").clip(-300, 300)
    out["f2_z_rel"], out["f2_n_lobes"], out["f2_fissure"] = _a("a2_z_rel"), _a("a2_n_lobes"), _a("a2_fissure_cross")
    out["f2_chestwall"], out["f2_med_pleura"] = _a("a2_chestwall_contact"), _a("a2_med_pleura_contact")
    out["f2_d_ribs"], out["f2_surf_ribs"] = _a("a2_d3_ribs").clip(0, 50), _a("a2_surf2_ribs")
    out["f2_d_sternum"], out["f2_d_svc"] = _a("a2_d3_sternum").clip(0, 50), _a("a2_d3_svc").clip(0, 50)
    out["f2_d_brachio"] = _a("a2_d3_brachiocephalic").clip(0, 50)
    out["f2_d_atrium_l"], out["f2_d_pv"] = _a("a2_d3_atrium_left").clip(0, 50), _a("a2_d3_pulm_vein").clip(0, 50)
    out["f2_tn_size"], out["f2_tn_fill"] = _a("a2_tn_size_mm").clip(0, 250), _a("a2_tn_fill").clip(0, 2)
    out["f2_tn_min_vital_d3"] = _mn([f"a2_tn_d3_{s}" for s in _VITAL_V2]).clip(0, 50)
    out["f2_tn_surf_vital"] = _mx([f"a2_tn_surf2_{s}" for s in _VITAL_V2])
    out["f2_tn_depth"] = _a("a2_tn_lung_depth").clip(0, 100)
    out["f2_tn_inv_any"] = _mx([f"a2_tn_inv_{s}" for s in _VITAL_V2])

    out = out[FULL_FEATURES]
    _abl = {x for x in os.environ.get("TNM_ABLATE", "").split(",") if x}
    if _abl:
        _tn = [c for c in ANAT2_FEATURES if c.startswith("f2_tn_")]
        _blocks = {"box": BOX_SIZE_FEATURES, "seg": SEG_FEATURES + _tn, "solid": SOLID_FEATURES,
                   "anat": SHAPE_FEATURES + CONTACT_FEATURES + INTERACTION_FEATURES + MULTI_FEATURES + [c for c in ANAT2_FEATURES if c not in _tn]}
        for _b in _abl & set(_blocks):
            out[[c for c in _blocks[_b] if c in out.columns]] = np.nan
    image_cols = sorted(c for c in df.columns if str(c).startswith("fmcib_"))
    if image_cols:
        image = df[image_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "emb" in _abl:
            image[:] = np.nan
        out = pd.concat([out, image], axis=1)
    return out


def _no_tumor_evidence(df: pd.DataFrame) -> pd.Series:
    size = _col(df, "tumor_size_2d_inplane_mm", 0.0).fillna(0.0)
    vol = _col(df, "tumor_volume_3d_cm3", 0.0).fillna(0.0)
    nsl = _col(df, "n_tumor_slices", 0.0).fillna(0.0)
    return (size <= 0) & (vol <= 0) & (nsl <= 0)


def _tabpfn_enabled():
    if os.environ.get("TNM_USE_TABPFN", "auto") == "0":
        return False
    try:
        import tabpfn
        return True
    except Exception:
        if not getattr(_tabpfn_enabled, "_warned", False):
            print("  (optional) TabPFN not installed: pip install tabpfn   -> adds a small-data foundation-model candidate")
            _tabpfn_enabled._warned = True
        return False


def _candidates():
    if FEATURE_PROFILE == "image":
        c = [
            ("ordinal_lr", {"C": 0.1, "size_only": True}),
            ("ordinal_lr", {"C": 0.1, "size_only": True, "soft_tau": 0.6}),
            ("ordinal_lr", {"C": 0.1}),
            ("ordinal_lr", {"C": 0.1, "n_components": 16}),
            ("ordinal_lr", {"C": 0.1, "n_components": 16, "proj": "pls"}),
            ("ordinal_lr", {"C": 0.1, "n_components": 16, "proj": "pls", "soft_tau": 0.6}),
            ("reg_ridge", {"alpha": 30.0, "n_components": 16, "proj": "pls"}),
            ("image_fusion", {"C": 0.03, "n_components": 24}),
            ("ordinal_xgb", {"n_estimators": 150, "max_depth": 2}),
        ]
        try:
            import catboost
            c.append(("image_catboost", {"iterations": 300, "depth": 4, "n_components": 16, "bag": 3}))
            c.append(("reg_catboost", {"iterations": 300, "depth": 3, "n_components": 16, "proj": "pls"}))
        except ImportError:
            print("  CatBoost not installed; add it with: pip install catboost")
        if _tabpfn_enabled():
            c.append(("tabpfn_fusion", {"n_estimators": 4, "n_components": 12, "proj": "pls"}))
        return c

    c = [("ordinal_lr", {"C": v, "size_only": True}) for v in (0.1, 0.5)]
    c += [("ordinal_lr", {"C": v}) for v in (0.1, 0.3)]
    c += [("lr", {"C": v}) for v in (0.1, 0.3)]
    c += [("ordinal_xgb", {"n_estimators": 150, "max_depth": 2}),
          ("xgb", {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8,
                   "colsample_bytree": 0.8, "reg_lambda": 5.0, "min_child_weight": 5, "gamma": 0.5})]
    try:
        import catboost
        c.append(("catboost", {"iterations": 300, "depth": 3, "learning_rate": 0.04, "l2_leaf_reg": 10.0}))
    except ImportError:
        pass
    try:
        import lightgbm
        c.append(("lgbm", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.04, "num_leaves": 7,
                           "min_child_samples": 5, "reg_lambda": 8.0, "subsample": 0.8, "colsample_bytree": 0.7}))
    except ImportError:
        pass
    if _tabpfn_enabled():
        c.append(("tabpfn", {"n_estimators": 8}))
    return c


def _fit(kind: str, params: dict, X: np.ndarray, y: np.ndarray, seed: int = 42):
    classes = np.unique(y)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sc  = StandardScaler()
        Xp  = sc.fit_transform(imp.fit_transform(X))
        cnt = np.bincount(y, minlength=len(_T_LABELS)).astype(float)
        cnt[cnt == 0] = 1.0
        weights = (cnt.sum() / (len(_T_LABELS) * cnt))[y]
        m = CatBoostClassifier(
            iterations=params.get("iterations", 300),
            depth=params.get("depth", 3),
            learning_rate=params.get("learning_rate", 0.04),
            loss_function="MultiClass",
            l2_leaf_reg=params.get("l2_leaf_reg", 10.0),
            verbose=False, random_seed=seed, thread_count=1, **_gpu.catboost_kwargs(), allow_writing_files=False,
        )
        m.fit(Xp, y, sample_weight=weights)
        class _CatWrap:
            def __init__(self, imp_, sc_, m_, cls_):
                self.imp_, self.sc_, self.m_ = imp_, sc_, m_
                self.classes_ = cls_
            def predict_proba(self, X_):
                return self.m_.predict_proba(self.sc_.transform(self.imp_.transform(X_)))
        return _CatWrap(imp, sc, m, np.asarray(m.classes_, dtype=int))
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        cnt = np.bincount(y, minlength=len(_T_LABELS)).astype(float)
        cnt[cnt == 0] = 1.0
        weights = (cnt.sum() / (len(_T_LABELS) * cnt))[y]
        m = Pipeline([
            ("imp",  SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc",   StandardScaler()),
            ("lgbm", LGBMClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.04),
                num_leaves=params.get("num_leaves", 7),
                min_child_samples=params.get("min_child_samples", 5),
                reg_lambda=params.get("reg_lambda", 8.0),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.7),
                class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1,
            )),
        ])
        m.fit(X, y, lgbm__sample_weight=weights)
        return m
    if kind in ("ordinal_lr", "ordinal_xgb", "reg_ridge", "reg_catboost", "tabpfn_fusion"):
        n_tab = len(FULL_FEATURES) if FEATURE_PROFILE == "image" else X.shape[1]
        common = dict(n_tabular=n_tab, n_use=params.get("n_use") or (len(SIZE_FEATURES) if params.get("size_only") else None),
                      n_components=params.get("n_components", 0), seed=seed, proj=params.get("proj", "pca"))
        if kind in ("ordinal_lr", "ordinal_xgb"):
            return OrdinalClassifier(base="lr" if kind == "ordinal_lr" else "xgb", C=params.get("C", 0.1),
                                     n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth", 2),
                                     soft_tau=params.get("soft_tau"), **common).fit(X, y)
        if kind.startswith("reg_"):
            return RegOrdinalClassifier(base="ridge" if kind == "reg_ridge" else "catboost", alpha=params.get("alpha", 30.0),
                                        iterations=params.get("iterations", 300), depth=params.get("depth", 3), **common).fit(X, y)
        return TabPFNFusion(tabpfn_estimators=params.get("n_estimators", 4), **common).fit(X, y)
    if kind == "image_catboost" and params.get("bag", 1) > 1:
        return BagModel([ImageCatBoostClassifier(n_tabular=len(FULL_FEATURES), n_components=params["n_components"],
                                                 iterations=params["iterations"], depth=params["depth"],
                                                 seed=seed + 17 * i).fit(X, y) for i in range(int(params["bag"]))])
    if kind == "lr":
        m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("lr", LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000, random_state=seed))])
        m.fit(X, y)
        return m
    if kind == "image_fusion":
        return ImageFusionClassifier(
            n_tabular=len(FULL_FEATURES), n_components=params["n_components"],
            C=params["C"], seed=seed,
        ).fit(X, y)
    if kind == "image_catboost":
        return ImageCatBoostClassifier(
            n_tabular=len(FULL_FEATURES), n_components=params["n_components"],
            iterations=params["iterations"], depth=params["depth"], seed=seed,
        ).fit(X, y)
    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(n_estimators=params.get("n_estimators", 8), random_state=seed)
        m.fit(X, y)
        return m
    from xgboost import XGBClassifier
    cnt = np.bincount(y, minlength=len(_T_LABELS)).astype(float)
    cnt[cnt == 0] = 1.0
    w = (cnt.sum() / (len(_T_LABELS) * cnt))[y]
    m = XGBClassifier(objective="multi:softprob", num_class=len(_T_LABELS), eval_metric="mlogloss",
                      random_state=seed, n_jobs=1, tree_method="hist", **_gpu.xgb_kwargs(), **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y, sample_weight=w)
    return m


class ImageFusionClassifier:
    def __init__(self, n_tabular, n_components=24, C=0.1, seed=42):
        self.n_tabular = int(n_tabular)
        self.n_components = int(n_components)
        self.C = float(C)
        self.seed = int(seed)

    def fit(self, X, y):
        self.tab_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.tab_scaler = StandardScaler()
        tab = self.tab_scaler.fit_transform(self.tab_imputer.fit_transform(X[:, :self.n_tabular]))
        self.image_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        image = self.image_imputer.fit_transform(X[:, self.n_tabular:])
        self.image_scaler = StandardScaler()
        image = self.image_scaler.fit_transform(image)
        ncomp = max(1, min(self.n_components, X.shape[0] - 1, image.shape[1]))
        self.pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=self.seed)
        image = self.pca.fit_transform(image)
        self.classifier = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed)
        self.classifier.fit(np.column_stack([tab, image]), y)
        self.classes_ = self.classifier.classes_
        return self

    def predict_proba(self, X):
        tab = self.tab_scaler.transform(self.tab_imputer.transform(X[:, :self.n_tabular]))
        image = self.image_scaler.transform(self.image_imputer.transform(X[:, self.n_tabular:]))
        image = self.pca.transform(image)
        return self.classifier.predict_proba(np.column_stack([tab, image]))


class ImageCatBoostClassifier:
    def __init__(self, n_tabular, n_components=16, iterations=300, depth=4, seed=42):
        self.n_tabular = int(n_tabular)
        self.n_components = int(n_components)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.seed = int(seed)

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.tab_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        tab = self.tab_imputer.fit_transform(X[:, :self.n_tabular])
        self.tab_scaler = StandardScaler()
        tab = self.tab_scaler.fit_transform(tab)
        self.image_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        image = self.image_imputer.fit_transform(X[:, self.n_tabular:])
        self.image_scaler = StandardScaler()
        image = self.image_scaler.fit_transform(image)
        ncomp = max(1, min(self.n_components, X.shape[0] - 1, image.shape[1]))
        self.pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=self.seed)
        image = self.pca.fit_transform(image)
        x = np.column_stack([tab, image])
        counts = np.bincount(y, minlength=len(_T_LABELS)).astype(float)
        weights = len(y) / (len(_T_LABELS) * np.maximum(counts, 1.0))
        self.classifier = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=0.03,
            loss_function="MultiClass", verbose=False, random_seed=self.seed,
            thread_count=1, **_gpu.catboost_kwargs(), allow_writing_files=False,
        )
        self.classifier.fit(x, y, sample_weight=weights[y])
        self.classes_ = np.asarray(self.classifier.classes_, dtype=int)
        return self

    def predict_proba(self, X):
        tab = self.tab_scaler.transform(self.tab_imputer.transform(X[:, :self.n_tabular]))
        image = self.image_scaler.transform(self.image_imputer.transform(X[:, self.n_tabular:]))
        image = self.pca.transform(image)
        return self.classifier.predict_proba(np.column_stack([tab, image]))


def _proba4(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    cls = getattr(model, "classes_", None)
    if cls is None:
        cls = model.named_steps["lr"].classes_
    out = np.zeros((len(X), len(_T_LABELS)))
    for j, c in enumerate(cls):
        out[:, int(c)] = p[:, j]
    return out


class OrdinalClassifier:
    def __init__(self, base="lr", C=0.1, n_tabular=None, n_use=None, n_components=0, seed=42,
                 n_estimators=150, max_depth=2, proj="pca", soft_tau=None):
        self.base, self.C, self.seed = base, float(C), int(seed)
        self.proj = proj
        self.soft_tau = soft_tau
        self.n_tabular, self.n_use, self.n_components = n_tabular, n_use, int(n_components)
        self.n_estimators, self.max_depth = int(n_estimators), int(max_depth)
        self.classes_ = np.arange(len(_T_LABELS))

    def _design(self, X, fit):
        nt, nu = self._nt, self._nu
        if fit:
            self.imp = SimpleImputer(strategy="median", keep_empty_features=True)
            self.sc = StandardScaler()
            tab = self.sc.fit_transform(self.imp.fit_transform(X[:, :nu]))
        else:
            tab = self.sc.transform(self.imp.transform(X[:, :nu]))
        parts = [tab]
        if self.use_img:
            if fit:
                self.iimp = SimpleImputer(strategy="median", keep_empty_features=True)
                self.isc = StandardScaler()
                img = self.isc.fit_transform(self.iimp.fit_transform(X[:, nt:]))
                nc = max(1, min(self.n_components, X.shape[0] - 1, img.shape[1]))
                self.pca = None
                if self.proj == "pls" and getattr(self, "_y_fit", None) is not None:
                    try:
                        from sklearn.cross_decomposition import PLSRegression
                        self.pca = PLSRegression(n_components=nc, scale=False).fit(img, self._y_fit.astype(float))
                        parts.append(self.pca.transform(img))
                    except Exception:
                        self.pca = None
                if self.pca is None:
                    self.pca = PCA(n_components=nc, svd_solver="randomized", random_state=self.seed)
                    parts.append(self.pca.fit_transform(img))
            else:
                parts.append(self.pca.transform(self.isc.transform(self.iimp.transform(X[:, nt:]))))
        return np.column_stack(parts)

    def fit(self, X, y):
        X = np.asarray(X, float)
        self._nt = X.shape[1] if self.n_tabular is None else min(int(self.n_tabular), X.shape[1])
        self._nu = self._nt if self.n_use is None else min(int(self.n_use), self._nt)
        self.use_img = self.n_components > 0 and X.shape[1] > self._nt
        self._y_fit = np.asarray(y)
        Z = self._design(X, fit=True)
        mono = None
        if self.base == "xgb":
            flags = [1 if n in _MONOTONE_UP else 0 for n in FULL_FEATURES[:self._nu]]
            flags += [0] * (Z.shape[1] - len(flags))
            mono = "(" + ",".join(str(v) for v in flags) + ")"
        self.models_ = []
        for k in range(len(_T_LABELS) - 1):
            t = (y > k).astype(int)
            if t.min() == t.max():
                self.models_.append(float(t[0]))
                continue
            cnt = np.bincount(t, minlength=2).astype(float)
            w = (len(t) / (2.0 * cnt))[t]
            if self.soft_tau:

                q = self._soft_gt(np.asarray(y), k)
                Z2, t2 = np.vstack([Z, Z]), np.r_[np.ones(len(t)), np.zeros(len(t))].astype(int)
                w2 = np.r_[q * w, (1.0 - q) * w]
                ok = w2 > 1e-4
                if self.base == "lr":
                    m = LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)
                    m.fit(Z2[ok], t2[ok], sample_weight=w2[ok])
                else:
                    from xgboost import XGBClassifier
                    m = XGBClassifier(objective="binary:logistic", n_estimators=self.n_estimators, max_depth=self.max_depth,
                                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                                      min_child_weight=3, monotone_constraints=mono, tree_method="hist", **_gpu.xgb_kwargs(),
                                      n_jobs=1, random_state=self.seed, eval_metric="logloss")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(Z2[ok], t2[ok], sample_weight=w2[ok])
            elif self.base == "lr":
                m = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed)
                m.fit(Z, t)
            else:
                from xgboost import XGBClassifier
                m = XGBClassifier(objective="binary:logistic", n_estimators=self.n_estimators, max_depth=self.max_depth,
                                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                                  min_child_weight=3, monotone_constraints=mono, tree_method="hist", **_gpu.xgb_kwargs(),
                                  n_jobs=1, random_state=self.seed, eval_metric="logloss")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(Z, t, sample_weight=w)
            self.models_.append(m)
        return self

    def _soft_gt(self, y, k):
        j = np.arange(len(_T_LABELS))[None, :]
        sdist = np.exp(-np.abs(j - y[:, None]) / float(self.soft_tau))
        sdist /= sdist.sum(axis=1, keepdims=True)
        return sdist[:, k + 1:].sum(axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Z = self._design(X, fit=False)
        pg = np.column_stack([m.predict_proba(Z)[:, 1] if hasattr(m, "predict_proba") else np.full(len(Z), m)
                              for m in self.models_])
        pg = np.minimum.accumulate(pg, axis=1)
        p = np.column_stack([1 - pg[:, 0], pg[:, 0] - pg[:, 1], pg[:, 1] - pg[:, 2], pg[:, 2]])
        p = np.clip(p, 1e-6, None)
        return p / p.sum(axis=1, keepdims=True)


class RegOrdinalClassifier(OrdinalClassifier):
    def __init__(self, base="ridge", alpha=30.0, iterations=300, depth=3, **kw):
        super().__init__(base="lr", **kw)
        self.reg_base, self.alpha, self.iterations, self.depth = base, float(alpha), int(iterations), int(depth)

    def fit(self, X, y):
        X = np.asarray(X, float)
        self._nt = X.shape[1] if self.n_tabular is None else min(int(self.n_tabular), X.shape[1])
        self._nu = self._nt if self.n_use is None else min(int(self.n_use), self._nt)
        self.use_img = self.n_components > 0 and X.shape[1] > self._nt
        self._y_fit = np.asarray(y)
        Z = self._design(X, fit=True)
        cnt = np.maximum(np.bincount(y, minlength=len(_T_LABELS)).astype(float), 1.0)
        w = (len(y) / (len(_T_LABELS) * cnt))[y]
        yf = y.astype(float)
        if self.reg_base == "ridge":
            from sklearn.linear_model import Ridge
            self.reg = Ridge(alpha=self.alpha).fit(Z, yf, sample_weight=w)
        else:
            from catboost import CatBoostRegressor
            self.reg = CatBoostRegressor(iterations=self.iterations, depth=self.depth, learning_rate=0.04, loss_function="RMSE",
                                         verbose=False, random_seed=self.seed, thread_count=1, **_gpu.catboost_kwargs(), allow_writing_files=False)
            self.reg.fit(Z, yf, sample_weight=w)
        res = np.clip(self.reg.predict(Z), 0, 3) - yf
        self.sigma_ = float(max(0.5, 1.25 * np.sqrt(np.average(res ** 2, weights=w))))
        return self

    def predict_proba(self, X):
        Z = self._design(np.asarray(X, float), fit=False)
        mu = np.clip(self.reg.predict(Z), 0, 3)[:, None]
        p = np.exp(-((mu - np.arange(len(_T_LABELS))[None, :]) ** 2) / (2 * self.sigma_ ** 2))
        p = np.clip(p, 1e-6, None)
        return p / p.sum(axis=1, keepdims=True)


class TabPFNFusion(OrdinalClassifier):
    def __init__(self, tabpfn_estimators=4, **kw):
        super().__init__(base="lr", **kw)
        self.tabpfn_estimators = int(tabpfn_estimators)

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        X = np.asarray(X, float)
        self._nt = X.shape[1] if self.n_tabular is None else min(int(self.n_tabular), X.shape[1])
        self._nu = self._nt if self.n_use is None else min(int(self.n_use), self._nt)
        self.use_img = self.n_components > 0 and X.shape[1] > self._nt
        self._y_fit = np.asarray(y)
        Z = self._design(X, fit=True)
        self.clf = TabPFNClassifier(n_estimators=self.tabpfn_estimators, random_state=self.seed, device=_gpu.torch_device())
        self.clf.fit(Z, y)
        return self

    def predict_proba(self, X):
        raw = self.clf.predict_proba(self._design(np.asarray(X, float), fit=False))
        p = np.full((raw.shape[0], len(_T_LABELS)), 1e-6)
        for j, c in enumerate(self.clf.classes_):
            p[:, int(c)] = raw[:, j]
        return p / p.sum(axis=1, keepdims=True)


class BagModel:
    def __init__(self, models):
        self.models = models
        self.classes_ = np.arange(len(_T_LABELS))

    def predict_proba(self, X):
        return np.mean([_proba4(m, X) for m in self.models], axis=0)


def _find_label_issues(y, p, max_frac=0.06):
    y = np.asarray(y)
    pred = p.argmax(1)
    sc = p[np.arange(len(y)), y]
    pp = p[np.arange(len(y)), pred]
    flag = (np.abs(pred - y) >= 2) & (sc <= 0.12) & (pp >= 0.40)
    idx = np.where(flag)[0]
    idx = idx[np.argsort(-(pp[idx] - sc[idx]))]
    return idx[: int(max_frac * len(y))]


def _conf_matrix(y, yp, k=4):
    return np.bincount(np.asarray(y, int) * k + np.asarray(yp, int), minlength=k * k).reshape(k, k).astype(float)


def _qwk(y, yp, k=4):
    O = _conf_matrix(y, yp, k)
    n = O.sum()
    if n == 0:
        return 0.0
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / float((k - 1) ** 2)
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else 0.0


def _macro_f1(y, yp, k=4):
    O = _conf_matrix(y, yp, k)
    tp = np.diag(O)
    prec = np.divide(tp, O.sum(0), out=np.zeros(k), where=O.sum(0) > 0)
    rec = np.divide(tp, O.sum(1), out=np.zeros(k), where=O.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    present = O.sum(1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def _score_pred(y, yp):
    return 0.5 * _qwk(y, yp) + 0.5 * _macro_f1(y, yp)


def _score(y, p):
    return _score_pred(y, np.asarray(p).argmax(1))


def _best_score(y, p, seed=42):
    s_arg = _score(y, p)
    try:
        cuts = _fit_cutpoints(p, y, seed, n_boot=0)
        return max(s_arg, _score_pred(y, _decode_cut(p, cuts)) - 0.01)
    except Exception:
        return s_arg


def _fit_temperature(p, y):
    logp = np.log(np.clip(p, 1e-9, 1.0))
    best_t, best = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.4), np.log(4.0), 41)):
        z = logp / t
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(q[np.arange(len(y)), y] + 1e-12))
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def _apply_temperature(p, t):
    if t is None or abs(t - 1.0) < 1e-9:
        return p
    z = np.log(np.clip(p, 1e-9, 1.0)) / t
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def _expected_stage(p):
    return np.asarray(p) @ np.arange(p.shape[1])


def _decode_cut(p, cuts):
    e = _expected_stage(p)
    return (e > cuts[0]).astype(int) + (e > cuts[1]).astype(int) + (e > cuts[2]).astype(int)


def _fit_cutpoints(p, y, seed=42, n_boot=15):
    base = np.array([0.5, 1.5, 2.5])

    def one(pp, yy):
        e = _expected_stage(pp)
        c = base.copy()

        def sc(cc):
            return _score_pred(yy, (e > cc[0]).astype(int) + (e > cc[1]) + (e > cc[2]))
        best = sc(c)
        for _ in range(2):
            for k in range(3):
                lo = (c[k - 1] + 0.05) if k > 0 else 0.05
                hi = (c[k + 1] - 0.05) if k < 2 else 2.95
                for v in np.linspace(max(lo, base[k] - 0.6), min(hi, base[k] + 0.6), 25):
                    c2 = c.copy()
                    c2[k] = v
                    s = sc(c2)
                    if s > best + 1e-9:
                        best, c = s, c2
        return c
    rng = np.random.default_rng(seed)
    cuts = [one(p, y)]
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        cuts.append(one(p[idx], y[idx]))
    return np.median(np.vstack(cuts), axis=0)


def _decode_w(p, w):
    return (np.asarray(p) * np.asarray(w)[None, :]).argmax(1)


def _fit_class_weights(p, y, seed=42, n_boot=15):
    grid = np.linspace(-1.0, 1.6, 27)

    def one(pp, yy):
        lw = np.zeros(pp.shape[1])

        def sc(v):
            return _score_pred(yy, (pp * np.exp(v)).argmax(1))
        best = sc(lw)
        for _ in range(2):
            for k in range(1, pp.shape[1]):
                for g in grid:
                    v = lw.copy()
                    v[k] = g
                    sv = sc(v)
                    if sv > best + 1e-9:
                        best, lw = sv, v
        return lw
    rng = np.random.default_rng(seed)
    ws = [one(p, y)]
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        ws.append(one(p[idx], y[idx]))
    return np.exp(np.median(np.vstack(ws), axis=0))


def _decode(p, bundle):
    mode = os.environ.get("TNM_DECODE", "auto").strip().lower()
    cw = bundle.get("class_weights")
    if cw is not None and (mode == "weights" or (mode == "auto" and bundle.get("decode") == "weights")):
        return _decode_w(p, cw)
    cuts = bundle.get("cutpoints")
    if cuts is not None and (mode == "cutpoints" or (mode == "auto" and bundle.get("decode") == "cutpoints")):
        return _decode_cut(p, cuts)
    return np.asarray(p).argmax(1)


def train_t_stage_xgboost(feature_df: pd.DataFrame, model_path=None, min_rows: int = 20, seed: int = 42):
    global _ML_MODEL
    if "T_stage" not in feature_df.columns:
        print("  Train skipped: no T_stage column.")
        return None
    if FEATURE_PROFILE == "image" and not any(str(c).startswith("fmcib_") for c in feature_df.columns):
        raise ValueError("Image-only profile requires FMCIB CT image embeddings (fmcib_* columns). Run the Bash pipeline.")
    work = feature_df.copy()
    work["_T"] = work["T_stage"].map(_norm4)
    work = work[work["_T"].isin(_T_LABELS)]
    work = work[~_no_tumor_evidence(work)].reset_index(drop=True)
    if len(work) < min_rows or work["_T"].nunique() < 2:
        print(f"  Train skipped: {len(work)} usable rows / {work['_T'].nunique()} classes.")
        return None
    names = feature_names(columns=work.columns)
    X = _make_features(work)[names].to_numpy(dtype=float)
    y = work["_T"].map(_T_TO_IDX).to_numpy(dtype=int)
    counts = np.bincount(y, minlength=len(_T_LABELS))
    min_class = int(counts[counts > 0].min())


    results = []
    if min_class >= 2:
        n_splits = max(2, min(5, min_class))
        n_rep = max(1, int(os.environ.get("TNM_INNER_REPEATS", "1")))
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_rep, random_state=seed)
        splits = list(cv.split(X, y))
        for kind, params in _candidates():
            oof_sum, oof_cnt = np.zeros((len(y), len(_T_LABELS))), np.zeros(len(y))
            ok = True
            for tr, te in splits:
                try:
                    m = _fit(kind, params, X[tr], y[tr], seed)
                    p4 = _proba4(m, X[te])
                except Exception as exc:
                    warnings.warn(f"  _fit({kind}) failed in inner CV: {exc}")
                    ok = False
                    break
                oof_sum[te] += p4
                oof_cnt[te] += 1
            if not ok or oof_cnt.min() == 0:
                continue
            oof = oof_sum / oof_cnt[:, None]
            results.append({"kind": kind, "params": params, "score": _best_score(y, oof, seed), "oof": oof})

    if not results:
        chosen = [("lr", {"C": 0.1})]
        ens_oof, best_score, single_score = None, float("nan"), float("nan")
    else:
        best_i, best_score = 0, -1.0
        for i, r in enumerate(results):
            if r["score"] > best_score + 0.005:
                best_i, best_score = i, r["score"]
        ranked = sorted(range(len(results)), key=lambda i: -results[i]["score"])
        top = ranked[:3]
        ens_oof = np.mean([results[i]["oof"] for i in top], axis=0)
        ens_score = _best_score(y, ens_oof, seed)
        single_score = results[best_i]["score"]
        if len(top) >= 2 and ens_score >= single_score - 0.005:
            chosen = [(results[i]["kind"], results[i]["params"]) for i in top]
            best_score = ens_score
        else:
            chosen = [(results[best_i]["kind"], results[best_i]["params"])]
            ens_oof = results[best_i]["oof"]
            best_score = single_score


    keep = np.ones(len(y), bool)
    issues = []
    if ens_oof is not None:
        bad = _find_label_issues(y, ens_oof, float(os.environ.get("TNM_CLEAN_MAX", "0.06")))
        if os.environ.get("TNM_CLEAN", "0") == "1":
            keep[bad] = False
        for i in bad:
            issues.append({"pid": str(work.loc[i, "pid"]) if "pid" in work.columns else str(i), "T_label": _T_LABELS[y[i]],
                           "T_oof_pred": _T_LABELS[int(ens_oof[i].argmax())], "p_label": float(ens_oof[i, y[i]]),
                           "p_pred": float(ens_oof[i].max()),
                           "tumor_size_mm": float(work.loc[i, "tumor_size_2d_inplane_mm"]) if "tumor_size_2d_inplane_mm" in work.columns else np.nan})
        if len(bad):
            print(f"  Suspect labels: {len(bad)} look inconsistent with their CT features (>=2 stages off, confident OOF) "
                  f"{'-> EXCLUDED from fitting' if os.environ.get('TNM_CLEAN', '0') == '1' else '(reported only)'}: "
                  f"{[d['pid'] for d in issues[:8]]}")


    models = [_fit(k, p, X[keep], y[keep], seed) for k, p in chosen]


    temperature, cutpoints, decode, qhat, class_w = 1.0, None, "argmax", None, None
    if ens_oof is not None:
        temperature = _fit_temperature(ens_oof, y)
        cal = _apply_temperature(ens_oof, temperature)
        s_arg = _score_pred(y, cal.argmax(1))
        try:
            cutpoints = _fit_cutpoints(cal, y, seed)
            s_cut = _score_pred(y, _decode_cut(cal, cutpoints))
            decode = "cutpoints" if s_cut >= s_arg + 0.01 else "argmax"
        except Exception as exc:
            warnings.warn(f"  cut-point fit failed: {exc}")
        try:
            if os.environ.get("TNM_WEIGHTS", "0") != "1":
                raise RuntimeError("class-weight decoding off (set TNM_WEIGHTS=1 to try it)")
            class_w = _fit_class_weights(cal, y, seed)
            s_w = _score_pred(y, _decode_w(cal, class_w))
            s_cur = s_cut if decode == "cutpoints" else s_arg
            if s_w >= max(s_cur, s_arg) + 0.015:
                decode = "weights"
        except Exception as exc:
            pass
        sc_arr = 1.0 - cal[np.arange(len(y)), y]
        k = min(len(sc_arr), int(np.ceil((len(sc_arr) + 1) * (1 - CONFORMAL_ALPHA))))
        qhat = float(np.sort(sc_arr)[k - 1])

    desc = "+".join(f"{k}" for k, _ in chosen)
    bundle = {
        "models":              models,
        "model":               models[0],
        "model_type":          ("ensemble(" + desc + ")") if len(models) > 1 else chosen[0][0],
        "feature_names":       names,
        "profile":             FEATURE_PROFILE,
        "conformal_qhat":      qhat,
        "conformal_alpha":     CONFORMAL_ALPHA,
        "feature_set_version": _FEATURE_SET_VERSION,
        "label_decoder":       {i: t for i, t in enumerate(_T_LABELS)},
        "cv_macro_f1":         best_score if best_score == best_score else float("nan"),
        "n_train":             int(len(work)),
        "best_config":         {"members": [{"kind": k, **p} for k, p in chosen]},
        "temperature":         temperature,
        "cutpoints":           cutpoints,
        "decode":              decode,
        "class_weights":       class_w,
        "label_issues":        issues,
    }
    model_path = Path(model_path) if model_path else Path(__file__).resolve().parent / "checkpoints" / "t_stage_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)
    _ML_MODEL = bundle
    cv_str = f"{best_score:.3f}" if best_score == best_score else "N/A"
    print(f"  Model trained: {bundle['model_type']} | n={len(work)} | features={len(names)} ({FEATURE_PROFILE}) "
          f"| inner-CV score (0.5 QWK + 0.5 macro-F1)={cv_str} | T={temperature:.2f} | decode={decode}")
    print(f"  Members: {bundle['best_config']['members']}")
    print(f"  Saved: {model_path}")
    return bundle


train_t_stage_model = train_t_stage_xgboost


def _load_ml_model():
    global _ML_MODEL
    if _ML_MODEL is not None:
        return _ML_MODEL
    p = os.environ.get("TNM_T_MODEL_PATH", "").strip()
    if p and Path(p).exists():
        try:
            with open(p, "rb") as fh:
                b = pickle.load(fh)
            if b.get("feature_set_version") == _FEATURE_SET_VERSION:
                _ML_MODEL = b
        except Exception:
            pass
    return _ML_MODEL


@dataclass
class TStagingResult:
    t_stage: str
    deciding_factor: str
    tumor_size_mm: float = 0.0
    lung_invaded: bool = False
    medi_invaded: bool = False
    trachea_invaded: bool = False
    esophagus_invaded: bool = False
    vertebral_invaded: bool = False
    heart_invaded: bool = False
    vessels_invaded: bool = False
    is_multifocal: bool = False
    confidence: float = 1.0
    is_borderline: bool = False
    uncertainty_probs: Dict = field(default_factory=dict)
    anatomy_evidence: str = ""
    no_tumor_evidence: bool = False
    t_set: str = ""
    ml_t_stage: str = ""


@dataclass
class TNMResult:
    t_result: TStagingResult
    tnm_string: str = ""
    overall_stage: str = ""
    report: str = ""

    def __post_init__(self):
        self.tnm_string = self.t_result.t_stage
        self.overall_stage = tnm_to_overall_stage(self.t_result.t_stage)


def _f(d: Dict, k: str, default=0.0) -> float:
    try:
        v = float(d.get(k, default))
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _evidence(feats: Dict) -> Dict:
    size = min(max(_f(feats, "tumor_size_2d_inplane_mm"), 0.0), 180.0)
    d = lambda s: _f(feats, f"dist_{s}_mm", 99.0)
    overlap = {s: d(s) <= 0.5 for s in _VITAL}
    flags = {
        "lung": _f(feats, "lung_depth_mm", 99.0) <= 3.0,
        "medi": _f(feats, "contact5_mediastinum", 0.0) >= 0.25,
        "heart": overlap["heart"], "vessels": overlap["aorta"] or overlap["pulmonary_artery"],
        "trachea": overlap["trachea"], "esophagus": overlap["esophagus"], "vert": overlap["vertebrae"],
        "multi": bool(_f(feats, "multi_diff_lobe_ipsi", 0.0)),
        "same_lobe": bool(_f(feats, "multi_same_lobe", 0.0)),
    }
    names = {"lung": "near lung boundary", "medi": "mediastinal contact", "heart": "heart", "vessels": "great vessels",
             "trachea": "trachea", "esophagus": "esophagus", "vert": "vertebrae", "multi": "separate nodule other lobe",
             "same_lobe": "separate nodule same lobe"}
    ev = [names[k] for k, v in flags.items() if v]
    no_ev = size <= 0 and _f(feats, "tumor_volume_3d_cm3") <= 0 and _f(feats, "n_tumor_slices") <= 0
    return {"size": size, **flags, "no_evidence": no_ev, "text": ("Anatomy: " + ", ".join(ev)) if ev else "No invasion detected"}


def _rule_stage(e: Dict) -> tuple[str, str]:
    s = e["size"]
    t4 = [n for n, k in [("heart", "heart"), ("great vessels", "vessels"), ("trachea", "trachea"),
                         ("esophagus", "esophagus"), ("vertebrae", "vert")] if e[k]]
    if s > SIZE_T3:
        return "T4", f"Size {s / 10:.1f} cm > 7 cm"
    if e["multi"]:
        return "T4", "Separate nodule in a different ipsilateral lobe"
    if t4:
        return "T4", "Tumor overlaps " + ", ".join(t4)
    if s > SIZE_T2 or e["same_lobe"]:
        return "T3", f"Size {s / 10:.1f} cm (5-7 cm)" if s > SIZE_T2 else "Separate nodule in same lobe"
    if s > SIZE_T1:
        return "T2", f"Size {s / 10:.1f} cm (3-5 cm)"
    return "T1", f"Size {s / 10:.1f} cm (<= 3 cm)"


def _override_mode():
    m = os.environ.get("TNM_OVERRIDE", "none").strip().lower()
    return m if m in OVERRIDE_MODES else "none"


def _iaslc_hard_override(e: Dict, ml_stage: str, mode: str | None = None) -> tuple[str, str, bool]:
    mode = (mode or _override_mode()).lower()
    if mode == "none":
        return ml_stage, "", False
    s = e["size"]
    if s > SIZE_T3:
        return "T4", f"IASLC override: size {s:.0f} mm > 70 mm -> T4", ml_stage != "T4"
    if mode == "size":
        return ml_stage, "", False
    for key, text in [("multi", "separate nodule in different ipsilateral lobe"), ("heart", "tumour overlaps heart"),
                      ("vessels", "tumour overlaps great vessels (aorta / PA)"), ("trachea", "tumour overlaps trachea"),
                      ("esophagus", "tumour overlaps esophagus"), ("vert", "tumour overlaps vertebrae")]:
        if e[key]:
            return "T4", f"IASLC override: {text} -> T4", ml_stage != "T4"
    if e["same_lobe"] and ml_stage in ("T1", "T2"):
        return "T3", "IASLC override: separate nodule same lobe -> T3 (floor)", True
    return ml_stage, "", False


def apply_override(features: Dict, ml_stage: str, mode: str | None = None) -> tuple[str, str, bool]:
    return _iaslc_hard_override(_evidence(features), ml_stage, mode)


def _ml_predict(feat_df: pd.DataFrame, bundle):
    X = _make_features(feat_df)[bundle["feature_names"]].to_numpy(dtype=float)
    models = bundle.get("models") or [bundle["model"]]
    p = np.mean([_proba4(m, X) for m in models], axis=0)
    p = _apply_temperature(p, bundle.get("temperature"))
    return p, _decode(p, bundle)


def _stage_from_probs(features: Dict, p0: np.ndarray, k: int, bundle) -> "TStagingResult":
    e = _evidence(features)
    base = dict(tumor_size_mm=e["size"], lung_invaded=e["lung"], medi_invaded=e["medi"],
                trachea_invaded=e["trachea"], esophagus_invaded=e["esophagus"],
                vertebral_invaded=e["vert"], heart_invaded=e["heart"],
                vessels_invaded=e["vessels"], is_multifocal=e["multi"],
                anatomy_evidence=e["text"])
    ml_stage = _T_LABELS[int(k)]
    srt = np.sort(p0)[::-1]
    tset = ""
    if bundle.get("conformal_qhat") is not None:
        members = [i for i in range(len(_T_LABELS)) if 1.0 - p0[i] <= bundle["conformal_qhat"]] or [int(k)]
        tset = "|".join(_T_LABELS[i] for i in members)
    final_stage, reason, was = _iaslc_hard_override(e, ml_stage)
    p0 = np.array(p0, float)
    if was:
        k_final = _T_TO_IDX[final_stage]
        deciding = f"ML ({bundle['model_type']}) -> {ml_stage}, then {reason}; size={e['size'] / 10:.1f} cm"
        p0 = np.clip(p0, 0.01, None)
        p0[k_final] = max(p0[k_final], 0.70)
        p0 /= p0.sum()
        conf, border = float(p0[k_final]), False
    else:
        deciding = f"ML ({bundle['model_type']}), size={e['size'] / 10:.1f} cm"
        conf, border = float(p0[int(k)]), bool(srt[0] - srt[1] < 0.15)
    return TStagingResult(t_stage=final_stage, deciding_factor=deciding, confidence=conf, is_borderline=border,
                          uncertainty_probs={t: float(p0[i]) for i, t in enumerate(_T_LABELS)},
                          t_set=tset, ml_t_stage=ml_stage, **base)


def _no_evidence_result(features: Dict) -> "TStagingResult":
    e = _evidence(features)
    return TStagingResult(
        t_stage="T1", deciding_factor="No tumor evidence extracted (flagged; likely extraction failure)",
        confidence=0.20, is_borderline=True, no_tumor_evidence=True, ml_t_stage="T1",
        uncertainty_probs={"T1": 0.25, "T2": 0.25, "T3": 0.25, "T4": 0.25},
        tumor_size_mm=e["size"], lung_invaded=e["lung"], medi_invaded=e["medi"], trachea_invaded=e["trachea"],
        esophagus_invaded=e["esophagus"], vertebral_invaded=e["vert"], heart_invaded=e["heart"],
        vessels_invaded=e["vessels"], is_multifocal=e["multi"], anatomy_evidence=e["text"])


def _rule_result(features: Dict) -> "TStagingResult":
    e = _evidence(features)
    t, why = _rule_stage(e)
    return TStagingResult(t_stage=t, deciding_factor="Rule: " + why, confidence=0.5, is_borderline=False, ml_t_stage=t,
                          uncertainty_probs={x: float(x == t) for x in _T_LABELS},
                          tumor_size_mm=e["size"], lung_invaded=e["lung"], medi_invaded=e["medi"],
                          trachea_invaded=e["trachea"], esophagus_invaded=e["esophagus"],
                          vertebral_invaded=e["vert"], heart_invaded=e["heart"], vessels_invaded=e["vessels"],
                          is_multifocal=e["multi"], anatomy_evidence=e["text"])


def classify_t_stage(features: Dict) -> TStagingResult:
    if _evidence(features)["no_evidence"]:
        return _no_evidence_result(features)
    bundle = _load_ml_model()
    if bundle is not None:
        try:
            p, k = _ml_predict(pd.DataFrame([features]), bundle)
            return _stage_from_probs(features, p[0], int(k[0]), bundle)
        except Exception as ex:
            print(f"  Model inference error: {ex} - falling back to rules")
    return _rule_result(features)


_OVERALL = {"T1": "Stage IA", "T2": "Stage IB", "T3": "Stage IIB", "T4": "Stage IIIA"}


def tnm_to_overall_stage(t: str, n: str = "N0", m: str = "M0") -> str:
    if m.startswith("M1"):
        return "Stage IV"
    return _OVERALL.get(t[:2], "Stage Unknown") if n == "N0" else "Stage (N unknown)"


def generate_staging_report(result: TNMResult, patient_id: str = "") -> str:
    t = result.t_result
    lines = ["=" * 60, "  LUNG CANCER T-STAGE REPORT  (IASLC 8th Ed.)", "=" * 60,
             f"  Patient ID   : {patient_id}", f"  T Stage      : {result.tnm_string}",
             f"  Tumor size   : {t.tumor_size_mm:.1f} mm", f"  Factor       : {t.deciding_factor}",
             f"  Anatomy      : {t.anatomy_evidence}", f"  Confidence   : {t.confidence * 100:.1f}%",
             "  N / M        : not estimated (CT-only placeholder N0/M0)", "=" * 60]
    return "\n".join(lines)


def run_full_tnm_staging(features: Dict, patient_id: str = "", use_3d: bool = False) -> TNMResult:
    r = TNMResult(t_result=classify_t_stage(features))
    r.report = generate_staging_report(r, patient_id)
    return r


def _record(pid: str, r: TNMResult) -> Dict:
    t = r.t_result
    rec = {"pid": pid, "predicted_T": t.t_stage, "predicted_T_ml": t.ml_t_stage or t.t_stage,
           "predicted_N": "N0", "predicted_M": "M0",
           "predicted_TNM": r.tnm_string, "overall_stage": r.overall_stage, "T_deciding": t.deciding_factor,
           "T_anatomy": t.anatomy_evidence, "T_confidence": t.confidence, "T_borderline": t.is_borderline,
           "tumor_size_mm": t.tumor_size_mm, "multifocal": t.is_multifocal, "lung_invaded": t.lung_invaded,
           "medi_invaded": t.medi_invaded, "heart_invaded": t.heart_invaded, "vessels_invaded": t.vessels_invaded,
           "no_tumor_evidence": t.no_tumor_evidence, "T_set": t.t_set,
           "T_set_size": (len(t.t_set.split("|")) if t.t_set else np.nan)}
    for x in _T_LABELS:
        rec[f"p_{x}"] = t.uncertainty_probs.get(x, np.nan)
    return rec


def batch_stage_patients(feature_df: pd.DataFrame, use_3d: bool = False) -> pd.DataFrame:
    feature_df = feature_df.reset_index(drop=True)
    bundle = _load_ml_model()
    P = K = None
    if bundle is not None and len(feature_df):
        try:
            P, K = _ml_predict(feature_df, bundle)
        except Exception as ex:
            print(f"  Model inference error: {ex} - falling back to rules")
    rows = []
    for i, feats in enumerate(feature_df.to_dict("records")):
        pid = str(feats.get("pid", feats.get("patient_id", "")))
        if _evidence(feats)["no_evidence"]:
            res = _no_evidence_result(feats)
        elif P is not None:
            res = _stage_from_probs(feats, P[i], int(K[i]), bundle)
        else:
            res = _rule_result(feats)
        tnm = TNMResult(t_result=res)
        tnm.report = ""
        rows.append(_record(pid, tnm))
    return pd.DataFrame(rows)
