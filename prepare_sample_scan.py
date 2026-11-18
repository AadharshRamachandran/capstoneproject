import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from box_seg_sizes import find_roots
from box_source import BoxSource, short_pid
from dicom_utils import select_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True, help="e.g. Lung_Dx-A0001")
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--model", default="", help="trained t_stage_model.pkl (final_t_staging.py writes it to <out_dir>\\checkpoints)")
    ap.add_argument("--dest", default=str(HERE / "sample_scan"))
    a = ap.parse_args()
    pid = a.pid.replace("Lung_Dx_", "Lung_Dx-")
    found = None
    for sd in a.shard_dirs:
        try:
            img, ann = find_roots(sd)
        except SystemExit:
            continue
        if (img / pid).is_dir():
            found = (img / pid, ann)
            break
    if not found:
        sys.exit(f"{pid} not found in the given shard folders")
    pdir, ann = found
    boxes = BoxSource("xml", ann_root=ann).boxes(pid)
    sdir, info, how = select_series(pdir, set(boxes))
    if sdir is None:
        sys.exit("no DICOM series found")
    dest = Path(a.dest)
    out_img = dest / "Lung-PET-CT-Dx" / pid / Path(sdir).name
    out_ann = dest / "Annotation" / short_pid(pid)
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in Path(sdir).iterdir():
        if f.is_file():
            shutil.copy2(f, out_img / f.name)
            n += 1
    xml = sorted((ann / short_pid(pid)).glob("*.xml"))
    for x in xml:
        shutil.copy2(x, out_ann / x.name)
    print(f"copied {n} DICOM slices ({how}) -> {out_img}\ncopied {len(xml)} XML files -> {out_ann}")
    if a.model:
        (HERE / "model").mkdir(exist_ok=True)
        shutil.copy2(a.model, HERE / "model" / "t_stage_model.pkl")
        print("copied model -> model\\t_stage_model.pkl")
    (dest / "pid.txt").write_text(pid)
    print("done. Now run:  python predict_one_scan.py")


if __name__ == "__main__":
    main()
