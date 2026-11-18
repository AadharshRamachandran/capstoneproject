import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

Box = Tuple[int, int, int, int]


def parse_xml_boxes(xml_path) -> List[Box]:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return []
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is not None:
            out.append((int(float(bb.findtext("xmin", "0"))), int(float(bb.findtext("ymin", "0"))),
                        int(float(bb.findtext("xmax", "512"))), int(float(bb.findtext("ymax", "512")))))
    return out


def short_pid(pid: str) -> str:
    return str(pid).replace("Lung_Dx-", "").replace("Lung_Dx_", "")


def norm_pid(pid: str) -> str:
    return str(pid).replace("Lung_Dx_", "Lung_Dx-")


class BoxSource:
    def __init__(self, kind: str, ann_root=None, yolo_csv=None, min_conf: float = 0.0):
        if kind not in ("xml", "yolo"):
            raise ValueError("kind must be 'xml' or 'yolo'")
        self.kind = kind
        self.ann_root = Path(ann_root) if ann_root else None
        self._y = None
        if kind == "yolo":
            if not yolo_csv or not Path(yolo_csv).exists():
                raise FileNotFoundError(f"yolo boxes csv not found: {yolo_csv} (run yolo_boxes.py build first)")
            d = pd.read_csv(yolo_csv)
            d["pid"] = d["pid"].astype(str).map(norm_pid)
            self._y = d[d["conf"] >= min_conf] if "conf" in d else d
        elif self.ann_root is None:
            raise ValueError("xml source needs ann_root")

    def boxes(self, pid: str) -> Dict[str, List[Box]]:
        pid = norm_pid(pid)
        out: Dict[str, List[Box]] = {}
        if self.kind == "xml":
            d = self.ann_root / short_pid(pid)
            if d.is_dir():
                for x in sorted(d.glob("*.xml")):
                    b = parse_xml_boxes(x)
                    if b:
                        out[x.stem] = b
            return out
        d = self._y[self._y["pid"] == pid]
        for r in d.itertuples():
            out.setdefault(str(r.uid), []).append((int(r.x0), int(r.y0), int(r.x1), int(r.y1)))
        return out

    def confidences(self, pid: str) -> List[float]:
        if self.kind != "yolo":
            return []
        return self._y.loc[self._y["pid"] == norm_pid(pid), "conf"].astype(float).tolist()
