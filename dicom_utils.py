from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def to_2d(px):
    px = np.asarray(px)
    while px.ndim > 2:
        px = px.mean(axis=-1) if px.shape[-1] in (3, 4) and px.ndim == 3 else px[0]
    return px


def _hdr(f):
    import pydicom
    return pydicom.dcmread(str(f), stop_before_pixels=True, force=True)


def list_series_dirs(patient_dir) -> List[Path]:
    return sorted({p.parent for p in Path(patient_dir).rglob("*.dcm")})


def series_info(series_dir) -> Dict:
    files = sorted(Path(series_dir).glob("*.dcm"), key=lambda f: f.stem)
    hs = [_hdr(f) for f in files]
    try:
        zs = [float(h.ImagePositionPatient[2]) for h in hs]
        order = sorted(range(len(hs)), key=lambda i: zs[i])
    except Exception:
        zs, order = None, list(range(len(hs)))
    ref = hs[order[0]] if hs else None
    return {
        "dir": Path(series_dir), "n": len(hs),
        "files": [files[i] for i in order],
        "uids": [str(getattr(hs[i], "SOPInstanceUID", "")) for i in order],
        "z": [zs[i] for i in order] if zs else None,
        "modality": str(getattr(ref, "Modality", "")) if ref is not None else "",
        "series_uid": str(getattr(ref, "SeriesInstanceUID", "")) if ref is not None else "",
        "desc": str(getattr(ref, "SeriesDescription", "")) if ref is not None else "",
        "rows": int(getattr(ref, "Rows", 512)) if ref is not None else 512,
        "cols": int(getattr(ref, "Columns", 512)) if ref is not None else 512,
    }


def select_series(patient_dir, xml_uids: Optional[Set[str]] = None) -> Tuple[Optional[Path], Optional[Dict], str]:
    dirs = list_series_dirs(patient_dir)
    if not dirs:
        return None, None, "no_dicom"
    infos = [series_info(d) for d in dirs]
    if xml_uids:
        scored = [(len(xml_uids & set(i["uids"])), i) for i in infos]
        best_n, best = max(scored, key=lambda t: t[0])
        if best_n > 0:
            return best["dir"], best, f"uid_match({best_n}/{len(xml_uids)})"
    cts = [i for i in infos if i["modality"].upper() == "CT"]
    if cts:
        best = max(cts, key=lambda i: i["n"])
        return best["dir"], best, "ct_most_slices"
    return infos[0]["dir"], infos[0], "first_series"


def read_hu_slice(path):
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    px = ds.pixel_array
    if px.ndim != 2:
        return None
    return px.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))


def pixel_spacing(path):
    import pydicom
    h = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    ps = getattr(h, "PixelSpacing", [1.0, 1.0])
    return float(ps[1]), float(ps[0])


def old_selection(patient_dir) -> Optional[Path]:
    p = Path(patient_dir)
    for study in p.iterdir():
        if not study.is_dir():
            continue
        for series in study.iterdir():
            if series.is_dir() and list(series.glob("*.dcm")):
                return series
    for sub in p.rglob("*.dcm"):
        return sub.parent
    return None
