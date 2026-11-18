import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dicom_utils import old_selection, select_series, series_info


def parse_boxes(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    out = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        out.append((int(float(bb.findtext("xmin", "0"))), int(float(bb.findtext("ymin", "0"))),
                    int(float(bb.findtext("xmax", "512"))), int(float(bb.findtext("ymax", "512"))), (obj.findtext("name") or "")))
    return out


def find_roots(shard_dir):
    shard_dir = Path(shard_dir)
    img = None
    for c in [shard_dir / "Lung-PET-CT-Dx" / "Lung-PET-CT-Dx", shard_dir / "Lung-PET-CT-Dx", shard_dir]:
        if c.is_dir() and any(p.is_dir() and p.name.startswith("Lung_Dx") for p in c.iterdir()):
            img = c
            break
    ann = None
    for c in [shard_dir / "Annotation" / "Annotation", shard_dir / "Annotation"]:
        if c.is_dir():
            ann = c
            break
    if img is None or ann is None:
        sys.exit(f"ERROR: could not find Lung-PET-CT-Dx/ and Annotation/ under {shard_dir}")
    return img, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--shard_num", type=int, default=0)
    ap.add_argument("--report_dir", default="")
    ap.add_argument("--force", action="store_true", help="redo patients whose cache file already has tumor_box (default: skip them - rewriting the compressed npz is the slow part)")
    args = ap.parse_args()

    img_root, ann_root = find_roots(args.shard_dir)
    cache = Path(args.cache_dir)
    rep_dir = Path(args.report_dir) if args.report_dir else cache / "_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rows, mismatch = [], []
    pdirs = sorted(p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Dx"))
    print(f"{len(pdirs)} patients in shard {args.shard_num}")
    for pdir in pdirs:
        pid = pdir.name
        short = pid.replace("Lung_Dx-", "").replace("Lung_Dx_", "")
        cf = cache / f"{pid}.npz"
        rec = {"pid": pid, "shard": args.shard_num, "status": "ok"}
        if not cf.exists():
            rec["status"] = "no_cache_file"
            rows.append(rec)
            continue
        xmls = sorted((ann_root / short).glob("*.xml")) if (ann_root / short).is_dir() else []
        uids = {x.stem for x in xmls}
        rec["n_xml"] = len(xmls)
        if not args.force:
            try:
                if "tumor_box" in np.load(str(cf)).files:
                    rec["status"] = "already_done"
                    rows.append(rec)
                    continue
            except Exception:
                pass
        try:
            sdir, info, how = select_series(pdir, uids)
            rec.update({"series_choice": how, "series_modality": info["modality"], "n_slices_series": info["n"]})
            z = np.load(str(cf))
            data = {k: z[k] for k in z.files}
            cache_D = int(data["anat__lung"].shape[0]) if "anat__lung" in data else (int(data["tumor"].shape[0]) if "tumor" in data else -1)
            rec["cache_n_slices"] = cache_D


            old_dir = old_selection(pdir)
            same = True
            if old_dir is not None and Path(old_dir) != Path(sdir):
                oi = series_info(old_dir)
                same = (oi["n"] == info["n"]) and (oi["z"] is not None and info["z"] is not None and
                                                    np.allclose(oi["z"], info["z"], atol=0.5))
                rec["old_series_modality"] = oi["modality"]
            rec["old_series_same_geometry"] = bool(same)
            if cache_D != info["n"]:
                same = False
            if not same:
                rec["status"] = "anatomy_from_other_series"
                mismatch.append(pid)

            uid_to_idx = {u: i for i, u in enumerate(info["uids"])}
            box = np.zeros((info["n"], info["rows"], info["cols"]), dtype=np.uint8)
            n_match = 0
            names = set()
            for x in xmls:
                idx = uid_to_idx.get(x.stem)
                if idx is None:
                    continue
                for (x0, y0, x1, y1, nm) in parse_boxes(x):
                    box[idx, max(0, y0):y1, max(0, x0):x1] = 1
                    names.add(nm)
                n_match += 1
            rec["n_xml_matched"] = n_match
            rec["box_slices"] = int(box.reshape(box.shape[0], -1).any(1).sum())
            rec["box_class"] = ",".join(sorted(names))
            if n_match == 0:
                rec["status"] = "no_box_matched" if rec["status"] == "ok" else rec["status"]
            else:
                data["tumor_box"] = box
                np.savez_compressed(str(cf), **data)
        except Exception as e:
            rec["status"] = f"error: {e!r}"[:200]
        rows.append(rec)
        print(f"  {pid}: {rec['status']} | matched {rec.get('n_xml_matched', 0)}/{rec.get('n_xml', 0)} | "
              f"{rec.get('series_choice', '-')} | box slices {rec.get('box_slices', 0)}")

    df = pd.DataFrame(rows)
    df.to_csv(rep_dir / f"boxes_report_shard{args.shard_num:03d}.csv", index=False)
    (rep_dir / f"mismatch_pids_shard{args.shard_num:03d}.txt").write_text("\n".join(mismatch), encoding="utf-8")
    print("\nstatus counts:", df["status"].value_counts().to_dict())
    print(f"patients whose cached anatomy came from another series (need a full re-run): {len(mismatch)}")
    print(f"report: {rep_dir}")


if __name__ == "__main__":
    main()
