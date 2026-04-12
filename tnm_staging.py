from __future__ import annotations
from dataclasses import dataclass,field
from typing import Dict
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings

_T_LABELS=["T1","T2","T3","T4"]
_T_TO_IDX={t:i for i,t in enumerate(_T_LABELS)}
_IDX_TO_T={i:t for i,t in enumerate(_T_LABELS)}
_ML_MODEL=None
_LEAKAGE_COLS={
    "pid","patient_id","predicted_t","predicted_label","overall_stage",
    "t_deciding","t_anatomy","t_stage","n_stage","m_stage","tnm_stage","stage",
    "predicted_n","predicted_m","predicted_tnm"
}

def _norm4(label:str)->str:
    s=str(label).strip().upper()
    if not s:return ""
    if s[0].isdigit() and s[0] in("1","2","3","4"):s="T"+s
    if s in("TX","T0","UNKNOWN","NAN"):return ""
    if s.startswith("T1"):return "T1"
    if s.startswith("T2"):return "T2"
    if s.startswith("T3"):return "T3"
    if s.startswith("T4"):return "T4"
    return ""

def _is_numeric(sr:pd.Series)->bool:
    if sr.dtype.kind in "biufc":return True
    try:
        pd.to_numeric(sr.dropna().head(50),errors="raise")
        return True
    except Exception:return False

def _build_X(df:pd.DataFrame,cols:list[str])->np.ndarray:
    mats=[]
    for c in cols:
        if c in df.columns:
            sr=pd.to_numeric(df[c],errors="coerce")
        else:
            sr=pd.Series(np.zeros(len(df),dtype=float),index=df.index)
        vals=sr.fillna(0.0).to_numpy(dtype=float)
        vals=np.nan_to_num(vals,nan=0.0,posinf=0.0,neginf=0.0)
        mats.append(vals)
    if not mats:return np.zeros((len(df),0),dtype=float)
    return np.vstack(mats).T

def _make_novel_features(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    sz=pd.to_numeric(out.get("tumor_size_2d_inplane_mm",pd.Series(0.0,index=out.index)),errors="coerce").fillna(0.0)
    vol=pd.to_numeric(out.get("tumor_volume_3d_cm3",pd.Series(0.0,index=out.index)),errors="coerce").fillna(0.0)
    nsl=pd.to_numeric(out.get("n_tumor_slices",pd.Series(0.0,index=out.index)),errors="coerce").fillna(0.0)
    st=pd.to_numeric(out.get("slice_thickness",pd.Series(1.0,index=out.index)),errors="coerce").fillna(1.0)
    d3=pd.to_numeric(out.get("tumor_size_3d_mm",pd.Series(0.0,index=out.index)),errors="coerce").fillna(0.0)
    sphere_d=2.0*((3.0*vol.clip(lower=0.05))/(4.0*np.pi))**(1.0/3.0)*10.0
    sphere_d=sphere_d.where(vol>0.05,0.0)
    depth=(nsl*st).clip(upper=70.0)
    out["feat_size_max"]=np.maximum(np.maximum(sz,sphere_d.clip(upper=180.0)),depth)
    out["feat_size_feret"]=sz
    out["feat_sphere_d"]=sphere_d.clip(upper=180.0)
    out["feat_depth_mm"]=depth
    out["feat_size_3d"]=d3.clip(upper=180.0)
    out["feat_vol_cm3"]=vol
    dist_lung=pd.to_numeric(out.get("dist_lung_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    dist_medi=pd.to_numeric(out.get("dist_mediastinum_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    dist_heart=pd.to_numeric(out.get("dist_heart_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    dist_aorta=pd.to_numeric(out.get("dist_aorta_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    dist_vert=pd.to_numeric(out.get("dist_vertebrae_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    dist_trach=pd.to_numeric(out.get("dist_trachea_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    dist_esoph=pd.to_numeric(out.get("dist_esophagus_2d_mm",pd.Series(1e6,index=out.index)),errors="coerce").fillna(1e6)
    out["feat_inv_lung"]=(dist_lung<3.0).astype(int)
    out["feat_inv_medi"]=(dist_medi<3.0).astype(int)
    out["feat_inv_heart"]=(dist_heart<1.0).astype(int)
    out["feat_inv_aorta"]=(dist_aorta<1.0).astype(int)
    out["feat_inv_vert"]=(dist_vert<1.0).astype(int)
    out["feat_inv_trach"]=(dist_trach<1.0).astype(int)
    out["feat_inv_esoph"]=(dist_esoph<1.0).astype(int)
    out["feat_dist_lung"]=dist_lung.clip(upper=200.0)
    out["feat_dist_medi"]=dist_medi.clip(upper=200.0)
    out["feat_dist_heart"]=dist_heart.clip(upper=200.0)
    out["feat_dist_aorta"]=dist_aorta.clip(upper=200.0)
    out["feat_any_t4_inv"]=(
        (dist_medi<3.0)|(dist_heart<1.0)|(dist_aorta<1.0)|
        (dist_vert<1.0)|(dist_trach<1.0)|(dist_esoph<1.0)
    ).astype(int)
    multifocal=pd.to_numeric(out.get("is_multifocal",pd.Series(0,index=out.index)),errors="coerce").fillna(0)
    out["feat_multifocal"]=multifocal
    out["feat_size_x_medi_inv"]=(out["feat_size_max"]*out["feat_inv_medi"])
    out["feat_size_x_heart_inv"]=(out["feat_size_max"]*out["feat_inv_heart"])
    out["feat_size_bin"]=pd.cut(
        out["feat_size_max"],
        bins=[-1,10,20,35,50,70,180,1e9],
        labels=[0,1,2,3,4,5,6]
    ).astype(float).fillna(0.0)
    nsl_safe=nsl.replace(0,1)
    out["feat_mean_sz_per_sl"]=sz/nsl_safe
    out["feat_depth_ratio"]=depth/out["feat_size_max"].replace(0,1)
    return out

def _get_feature_cols(df:pd.DataFrame)->list[str]:
    cols=[]
    for c in df.columns:
        cl=c.lower()
        if cl in _LEAKAGE_COLS or cl.startswith("gt_"):continue
        if c in("T_stage","T_stage_norm"):continue
        if c.startswith("feat_") or _is_numeric(df[c]):
            cols.append(c)
    return cols

def train_t_stage_xgboost(feature_df:pd.DataFrame,model_path:str|Path|None=None,min_rows:int=20)->dict|None:
    global _ML_MODEL
    if "T_stage" not in feature_df.columns:
        print("  XGB train skipped: no T_stage column.")
        return None
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        print(f"  XGB train skipped: {e}")
        return None
    work=feature_df.copy()
    work["T_stage_norm"]=work["T_stage"].map(_norm4)
    work=work[work["T_stage_norm"].isin(_T_LABELS)].reset_index(drop=True)
    if len(work)<min_rows:
        print(f"  XGB train skipped: only {len(work)} rows.")
        return None
    work=_make_novel_features(work)
    feature_names=_get_feature_cols(work)
    X=_build_X(work,feature_names)
    y=np.array([_T_TO_IDX[t] for t in work["T_stage_norm"].tolist()],dtype=int)
    counts=np.bincount(y,minlength=len(_T_LABELS)).astype(float)
    counts[counts==0]=1.0
    cls_w=counts.sum()/(len(_T_LABELS)*counts)
    sw=np.array([cls_w[v] for v in y],dtype=float)
    grid=[
        {"n_estimators":400,"max_depth":4,"learning_rate":0.05,"subsample":0.85,"colsample_bytree":0.85,"reg_lambda":1.5,"min_child_weight":2,"gamma":0.1},
        {"n_estimators":600,"max_depth":5,"learning_rate":0.03,"subsample":0.9,"colsample_bytree":0.8,"reg_lambda":1.0,"min_child_weight":1,"gamma":0.0},
        {"n_estimators":300,"max_depth":3,"learning_rate":0.08,"subsample":0.9,"colsample_bytree":0.9,"reg_lambda":2.0,"min_child_weight":3,"gamma":0.2},
        {"n_estimators":800,"max_depth":4,"learning_rate":0.02,"subsample":0.8,"colsample_bytree":0.85,"reg_lambda":1.2,"min_child_weight":2,"gamma":0.05},
    ]
    unique,cnt=np.unique(y,return_counts=True)
    min_class=int(cnt.min()) if len(cnt)>0 else 0
    n_splits=max(2,min(5,min_class)) if min_class>=2 else 0
    best_cfg=grid[0]
    best_cv=-1.0
    if n_splits>=2:
        skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
        for cfg in grid:
            scores=[]
            for tr,te in skf.split(X,y):
                xgb=XGBClassifier(
                    objective="multi:softprob",num_class=len(_T_LABELS),
                    eval_metric="mlogloss",random_state=42,n_jobs=0,
                    use_label_encoder=False,**cfg
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xgb.fit(X[tr],y[tr],sample_weight=sw[tr])
                yp=xgb.predict(X[te])
                yt=[_IDX_TO_T[int(v)] for v in y[te]]
                yp_t=[_IDX_TO_T[int(v)] for v in yp]
                scores.append(f1_score(yt,yp_t,average="macro",zero_division=0))
            s=float(np.mean(scores))
            if s>best_cv:
                best_cv=s
                best_cfg=cfg
    best_model=XGBClassifier(
        objective="multi:softprob",num_class=len(_T_LABELS),
        eval_metric="mlogloss",random_state=42,n_jobs=0,
        use_label_encoder=False,**best_cfg
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X,y,sample_weight=sw)
    bundle={
        "model":best_model,"model_type":"xgboost",
        "feature_names":feature_names,"label_decoder":_IDX_TO_T,
        "cv_macro_f1":(best_cv if best_cv>=0 else float("nan")),
        "n_train":int(len(work)),"best_config":best_cfg,
    }
    if model_path is None:
        model_path=Path(__file__).resolve().parent/"checkpoints"/"t_stage_xgb.pkl"
    model_path=Path(model_path)
    model_path.parent.mkdir(parents=True,exist_ok=True)
    with open(model_path,"wb") as f:
        pickle.dump(bundle,f)
    _ML_MODEL=bundle
    cv_str=f"{best_cv*100:.2f}%" if best_cv>=0 else "N/A"
    print(f"  XGB trained: n={len(work)} | features={len(feature_names)} | CV macro-F1={cv_str}")
    print(f"  XGB best config: {best_cfg}")
    print(f"  XGB saved: {model_path}")
    return bundle

def _load_ml_model()->dict|None:
    global _ML_MODEL
    if _ML_MODEL is not None:return _ML_MODEL
    env_path=os.environ.get("TNM_T_MODEL_PATH","").strip()
    candidates=[]
    if env_path:candidates.append(Path(env_path))
    candidates+=[
        Path.home()/"Downloads"/"working"/"checkpoints"/"t_stage_xgb.pkl",
        Path(r"c:\Users\HP\Downloads\working\checkpoints\t_stage_xgb.pkl"),
    ]
    for parent in Path(__file__).parents:
        candidates.append(parent/"checkpoints"/"t_stage_xgb.pkl")
    for mp in candidates:
        if not mp.exists():continue
        try:
            with open(mp,"rb") as f:
                _ML_MODEL=pickle.load(f)
                return _ML_MODEL
        except Exception:pass
    return None

@dataclass
class TStagingResult:
    t_stage:str
    deciding_factor:str
    tumor_size_mm:float=0.0
    lung_invaded:bool=False
    medi_invaded:bool=False
    trachea_invaded:bool=False
    esophagus_invaded:bool=False
    vertebral_invaded:bool=False
    heart_invaded:bool=False
    vessels_invaded:bool=False
    is_multifocal:bool=False
    confidence:float=1.0
    is_borderline:bool=False
    uncertainty_probs:Dict=field(default_factory=dict)
    anatomy_evidence:str=""

@dataclass
class TNMResult:
    t_result:TStagingResult
    tnm_string:str=""
    overall_stage:str=""
    report:str=""
    def __post_init__(self):
        self.tnm_string=f"{self.t_result.t_stage}"
        self.overall_stage=tnm_to_overall_stage(self.t_result.t_stage)

def _f(features:Dict,key:str,default:float=0.0)->float:
    v=features.get(key,default)
    try:
        fv=float(v)
        return default if(np.isnan(fv) or np.isinf(fv)) else fv
    except(TypeError,ValueError):return default

def _b(features:Dict,key:str)->bool:
    return bool(features.get(key,0))

def _collect_evidence(features: Dict) -> Dict:
    size_mm = _f(features, "tumor_size_2d_inplane_mm")
    if size_mm <= 0:
        size_mm = _f(features, "tumor_size_2d_mm")
    if size_mm <= 0:
        size_mm = _f(features, "tumor_size_3d_mm")
    size_mm = min(max(size_mm, 0.0), 180.0)

    n_sl = _f(features, "n_tumor_slices")
    vol_cm3 = _f(features, "tumor_volume_3d_cm3")

    inv_lung = _b(features, "lung_invaded_2d") or _b(features, "lung_invaded_3d")
    inv_medi = _b(features, "mediastinum_invaded_2d") or _b(features, "mediastinum_invaded_3d")
    inv_heart = _b(features, "heart_invaded_2d") or _b(features, "heart_invaded_3d")
    inv_vessels = (
        _b(features, "vessels_invaded_2d") or _b(features, "vessels_invaded_3d") or
        _b(features, "aorta_invaded_2d") or _b(features, "aorta_invaded_3d")
    )
    inv_trach = _b(features, "trachea_invaded_2d") or _b(features, "trachea_invaded_3d")
    inv_esoph = _b(features, "esophagus_invaded_2d") or _b(features, "esophagus_invaded_3d")
    inv_vert = (
        _b(features, "vertebrae_invaded_2d") or _b(features, "vertebrae_invaded_3d") or
        _b(features, "vertebral_invaded_2d")
    )
    multifocal = _b(features, "is_multifocal")

    any_t4_invasion = any([inv_medi, inv_heart, inv_vessels, inv_trach, inv_esoph, inv_vert, multifocal])
    no_tumor_evidence = (size_mm <= 0.0 and n_sl <= 0.0 and vol_cm3 <= 0.0 and not inv_lung and not any_t4_invasion)

    ev = []
    if inv_heart: ev.append("heart")
    if inv_vessels: ev.append("great vessels")
    if inv_medi: ev.append("mediastinum")
    if inv_trach: ev.append("trachea")
    if inv_esoph: ev.append("esophagus")
    if inv_vert: ev.append("vertebrae")
    if inv_lung: ev.append("lung wall")
    if multifocal: ev.append("multifocal")

    return {
        "size_mm": size_mm,
        "n_slices": n_sl,
        "vol_cm3": vol_cm3,
        "inv_lung": inv_lung,
        "inv_medi": inv_medi,
        "inv_heart": inv_heart,
        "inv_vessels": inv_vessels,
        "inv_trach": inv_trach,
        "inv_esoph": inv_esoph,
        "inv_vert": inv_vert,
        "multifocal": multifocal,
        "any_t4_invasion": any_t4_invasion,
        "no_tumor_evidence": no_tumor_evidence,
        "anatomy_evidence": "Anatomy: "+",".join(ev)+" invaded" if ev else "No invasion detected",
    }

def classify_t_stage(features:Dict)->TStagingResult:
    evidence = _collect_evidence(features)
    size_mm = evidence["size_mm"]
    if evidence["no_tumor_evidence"]:
        return TStagingResult(
            t_stage="T1",
            deciding_factor="No tumor evidence extracted; conservative fallback instead of ML upstaging",
            tumor_size_mm=size_mm,
            lung_invaded=evidence["inv_lung"],
            medi_invaded=evidence["inv_medi"],
            trachea_invaded=evidence["inv_trach"],
            esophagus_invaded=evidence["inv_esoph"],
            vertebral_invaded=evidence["inv_vert"],
            heart_invaded=evidence["inv_heart"],
            vessels_invaded=evidence["inv_vessels"],
            is_multifocal=evidence["multifocal"],
            confidence=0.20,
            is_borderline=True,
            uncertainty_probs={"T1": 0.55, "T2": 0.25, "T3": 0.12, "T4": 0.08},
            anatomy_evidence=evidence["anatomy_evidence"],
        )

    bundle=_load_ml_model()
    if bundle is not None:
        try:
            model=bundle["model"]
            cols=bundle["feature_names"]
            decoder=bundle["label_decoder"]
            row_df=pd.DataFrame([features])
            row_df=_make_novel_features(row_df)
            X=_build_X(row_df,cols)
            pred_idx=int(model.predict(X)[0])
            prob_arr=model.predict_proba(X)[0]
            t_pred=decoder[pred_idx]
            u_probs={decoder[i]:float(prob_arr[i]) for i in range(len(decoder))}
            conf=float(prob_arr[pred_idx])
            if t_pred == "T4" and (not evidence["any_t4_invasion"]) and size_mm <= 0.0 and conf < 0.60:
                t_pred = "T1"
                u_probs["T1"] = max(u_probs.get("T1", 0.0), conf)
                conf = max(conf, u_probs.get("T1", 0.0))
            return TStagingResult(
                t_stage=t_pred,
                deciding_factor=f"ML prediction (size={size_mm/10:.1f}cm, model-based)",
                tumor_size_mm=size_mm,
                lung_invaded=evidence["inv_lung"],medi_invaded=evidence["inv_medi"],
                trachea_invaded=evidence["inv_trach"],esophagus_invaded=evidence["inv_esoph"],
                vertebral_invaded=evidence["inv_vert"],heart_invaded=evidence["inv_heart"],
                vessels_invaded=evidence["inv_vessels"],is_multifocal=evidence["multifocal"],
                confidence=conf,is_borderline=(conf<0.6),
                uncertainty_probs=u_probs,anatomy_evidence=evidence["anatomy_evidence"]
            )
        except Exception as e:
            print(f"  XGB inference error: {e}, falling back to rules")
    size_mm=evidence["size_mm"]
    if size_mm<=0:
        n_sl=_f(features,"n_tumor_slices")
        st=_f(features,"slice_thickness")
        if n_sl>0 and st>0:size_mm=min(n_sl*st,70.0)
    size_mm=min(max(size_mm,0.0),180.0)
    size_cm=size_mm/10.0
    inv_heart=evidence["inv_heart"]
    inv_medi=evidence["inv_medi"]
    inv_vessels=evidence["inv_vessels"]
    inv_trach=evidence["inv_trach"]
    inv_esoph=evidence["inv_esoph"]
    inv_vert=evidence["inv_vert"]
    inv_lung=evidence["inv_lung"]
    multifocal=evidence["multifocal"]
    anatomy_ev=evidence["anatomy_evidence"]
    t4_reasons=[]
    if size_cm>7.0:t4_reasons.append(f"Tumor {size_cm:.1f}cm > 7cm")
    if size_mm>=50 and inv_medi:t4_reasons.append("Mediastinal invasion")
    if size_mm>=50 and inv_heart:t4_reasons.append("Heart invasion")
    if size_mm>=50 and inv_vessels:t4_reasons.append("Vessel invasion")
    if size_mm>=15 and inv_trach:t4_reasons.append("Tracheal invasion")
    if size_mm>=15 and inv_esoph:t4_reasons.append("Esophageal invasion")
    if size_mm>=15 and inv_vert:t4_reasons.append("Vertebral invasion")
    if multifocal:t4_reasons.append("Multifocal")
    if t4_reasons:
        return TStagingResult(
            t_stage="T4",deciding_factor="; ".join(t4_reasons),tumor_size_mm=size_mm,
            lung_invaded=inv_lung,medi_invaded=inv_medi,trachea_invaded=inv_trach,
            esophagus_invaded=inv_esoph,vertebral_invaded=inv_vert,heart_invaded=inv_heart,
            vessels_invaded=inv_vessels,is_multifocal=multifocal,anatomy_evidence=anatomy_ev
        )
    if size_mm>55:
        return TStagingResult(
            t_stage="T3",deciding_factor=f"Tumor {size_cm:.1f}cm (T3: 5.5-7cm calibrated)",
            tumor_size_mm=size_mm,lung_invaded=inv_lung,medi_invaded=inv_medi,
            trachea_invaded=inv_trach,esophagus_invaded=inv_esoph,vertebral_invaded=inv_vert,
            heart_invaded=inv_heart,vessels_invaded=inv_vessels,is_multifocal=multifocal,
            anatomy_evidence=anatomy_ev
        )
    if size_mm>35:
        sub="T2b" if size_mm>45 else "T2a"
        return TStagingResult(
            t_stage=sub,deciding_factor=f"Tumor {size_cm:.1f}cm ({sub} calibrated)",
            tumor_size_mm=size_mm,lung_invaded=inv_lung,medi_invaded=inv_medi,
            trachea_invaded=inv_trach,esophagus_invaded=inv_esoph,vertebral_invaded=inv_vert,
            heart_invaded=inv_heart,vessels_invaded=inv_vessels,is_multifocal=multifocal,
            anatomy_evidence=anatomy_ev
        )
    if inv_lung and size_mm>0:
        return TStagingResult(
            t_stage="T2",deciding_factor=f"Tumor {size_cm:.1f}cm + lung wall contact",
            tumor_size_mm=size_mm,lung_invaded=True,medi_invaded=inv_medi,
            trachea_invaded=inv_trach,esophagus_invaded=inv_esoph,vertebral_invaded=inv_vert,
            heart_invaded=inv_heart,vessels_invaded=inv_vessels,is_multifocal=multifocal,
            anatomy_evidence=anatomy_ev
        )
    if size_mm<=10:sub,reason="T1a",f"Tumor {size_cm:.1f}cm <= 1cm (T1a)"
    elif size_mm<=20:sub,reason="T1b",f"Tumor {size_cm:.1f}cm (T1b: 1-2cm)"
    else:sub,reason="T1c",f"Tumor {size_cm:.1f}cm (T1c: 2-3.5cm calibrated)"
    return TStagingResult(
        t_stage=sub,deciding_factor=reason,tumor_size_mm=size_mm,
        lung_invaded=inv_lung,medi_invaded=inv_medi,trachea_invaded=inv_trach,
        esophagus_invaded=inv_esoph,vertebral_invaded=inv_vert,heart_invaded=inv_heart,
        vessels_invaded=inv_vessels,is_multifocal=multifocal,anatomy_evidence=anatomy_ev
    )

def tnm_to_overall_stage(t:str,n:str="N0",m:str="M0")->str:
    if m in("M1a","M1b","M1c"):return "Stage IV"
    t_base=t[:2] if len(t)>=2 else t
    table={
        ("T1a","N0"):"Stage IA1",("T1b","N0"):"Stage IA2",("T1c","N0"):"Stage IA3",
        ("T1","N0"):"Stage IA",("T2a","N0"):"Stage IB",("T2","N0"):"Stage IB",
        ("T2b","N0"):"Stage IIA",("T3","N0"):"Stage IIB",("T4","N0"):"Stage IIIA",
        ("T1a","N1"):"Stage IIB",("T1b","N1"):"Stage IIB",("T1c","N1"):"Stage IIB",
        ("T1","N1"):"Stage IIB",("T2a","N1"):"Stage IIB",("T2b","N1"):"Stage IIB",
        ("T2","N1"):"Stage IIB",("T3","N1"):"Stage IIIA",("T4","N1"):"Stage IIIA",
        ("T1a","N2"):"Stage IIIA",("T1b","N2"):"Stage IIIA",("T1c","N2"):"Stage IIIA",
        ("T1","N2"):"Stage IIIA",("T2a","N2"):"Stage IIIA",("T2b","N2"):"Stage IIIA",
        ("T2","N2"):"Stage IIIA",("T3","N2"):"Stage IIIB",("T4","N2"):"Stage IIIB",
        ("T1a","N3"):"Stage IIIB",("T1b","N3"):"Stage IIIB",("T1c","N3"):"Stage IIIB",
        ("T1","N3"):"Stage IIIB",("T2a","N3"):"Stage IIIB",("T2b","N3"):"Stage IIIB",
        ("T2","N3"):"Stage IIIB",("T3","N3"):"Stage IIIC",("T4","N3"):"Stage IIIC",
    }
    return table.get((t,n)) or table.get((t_base,n),"Stage Unknown")

def generate_staging_report(result:TNMResult,patient_id:str="")->str:
    t=result.t_result
    u=t.uncertainty_probs
    size_str=f"{t.tumor_size_mm:.1f}mm ({t.tumor_size_mm/10:.2f}cm)"
    lines=[
        "="*60,
        f"  LUNG CANCER T-STAGE REPORT  (IASLC 8th Ed.)",
        "="*60,
        f"  Patient ID   : {patient_id}",
        f"  T Stage      : {result.tnm_string}",
        f"  Overall Stage: {result.overall_stage}",
        f"  Tumor size   : {size_str}",
        f"  Factor       : {t.deciding_factor}",
        f"  Anatomy      : {t.anatomy_evidence}",
        f"  Confidence   : {t.confidence*100:.1f}%",
        f"  Borderline   : {'YES' if t.is_borderline else 'No'}",
    ]
    if u:
        lines.append("  Stage probs (XGBoost):")
        for stage in ["T1","T2","T3","T4"]:
            p=u.get(stage,0.0)
            bar="█"*int(p*20)+"░"*(20-int(p*20))
            lines.append(f"    {stage}: [{bar}] {p*100:.1f}%")
    lines+=[
        f"  N-STAGE: N0 (CT-only placeholder)",
        f"  M-STAGE: M0 (CT-only placeholder)",
        "="*60,
    ]
    return "\n".join(lines)

def run_full_tnm_staging(features:Dict,patient_id:str="",use_3d:bool=False)->TNMResult:
    t_result=classify_t_stage(features)
    result=TNMResult(t_result=t_result)
    result.report=generate_staging_report(result,patient_id)
    return result

def batch_stage_patients(feature_df:pd.DataFrame,use_3d:bool=False)->pd.DataFrame:
    rows=[]
    for _,row in feature_df.iterrows():
        feats=row.to_dict()
        pid=str(feats.get("pid",feats.get("patient_id","")))
        r=run_full_tnm_staging(feats,pid)
        t=r.t_result
        rows.append({
            "pid":pid,
            "predicted_T":t.t_stage,
            "predicted_N":"N0",
            "predicted_M":"M0",
            "predicted_TNM":r.tnm_string,
            "overall_stage":r.overall_stage,
            "T_deciding":t.deciding_factor,
            "T_anatomy":t.anatomy_evidence,
            "T_confidence":t.confidence,
            "T_borderline":t.is_borderline,
            "tumor_size_mm":t.tumor_size_mm,
            "multifocal":t.is_multifocal,
            "lung_invaded":t.lung_invaded,
            "medi_invaded":t.medi_invaded,
            "heart_invaded":t.heart_invaded,
            "vessels_invaded":t.vessels_invaded,
        })
    return pd.DataFrame(rows)
