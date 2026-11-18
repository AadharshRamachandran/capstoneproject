import argparse, os, re
from pathlib import Path
import pandas as pd


def find_col(df, hints, given=None, skip=()):
    if given:
        return given
    for h in hints:
        for c in df.columns:
            if c not in skip and h in c.lower():
                return c
    raise SystemExit(f"Could not find a column like {hints} in {list(df.columns)}; pass it explicitly.")


def t_value(x):
    m = re.search(r"[1-4]", str(x))
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--xml_dir", default=None)
    ap.add_argument("--slice_csv", default=None)
    ap.add_argument("--slice_id_col", default=None)
    ap.add_argument("--pool_work", default=None, help="yolo work folder: count annotated slices from <pool_work>\\pool\\labels (non-empty label file)")
    a = ap.parse_args()

    lab = pd.read_csv(a.labels_csv)
    idc = find_col(lab, ["patient", "case", "subject", "id"], a.id_col)
    tc = find_col(lab, ["t_stage", "tstage", "t-stage", "t_label", "label", "stage"], a.label_col, skip=(idc,))
    lab = lab[[idc, tc]].copy()
    lab.columns = ["pid", "raw"]
    lab["T"] = lab["raw"].map(t_value)
    lab = lab.dropna(subset=["T"]).drop_duplicates("pid")
    lab["T"] = lab["T"].astype(int)
    lab["pid"] = lab["pid"].astype(str).str.strip()

    if a.pool_work:
        rows = {}
        for f in (Path(a.pool_work) / "pool" / "labels").glob("*.txt"):
            if f.stat().st_size > 0:
                pid = f.stem.rsplit("_", 1)[0]
                rows[pid] = rows.get(pid, 0) + 1
        sl = pd.DataFrame({"pid": list(rows), "n_slices": list(rows.values())})
    elif a.xml_dir:
        rows = []
        for root, _, files in os.walk(a.xml_dir):
            n = sum(f.lower().endswith(".xml") for f in files)
            if n:
                pid = next((p for p in re.split(r"[\\/]", root)[::-1] if p in set(lab.pid)), os.path.basename(root))
                rows.append((pid, n))
        sl = pd.DataFrame(rows, columns=["pid", "n_slices"]).groupby("pid", as_index=False).sum()
    elif a.slice_csv:
        s = pd.read_csv(a.slice_csv)
        sc = find_col(s, ["patient", "case", "subject", "id"], a.slice_id_col)
        sl = s.groupby(sc).size().reset_index(name="n_slices")
        sl.columns = ["pid", "n_slices"]
        sl["pid"] = sl["pid"].astype(str).str.strip()
    else:
        raise SystemExit("give --pool_work, --xml_dir or --slice_csv")

    d = sl.merge(lab, on="pid", how="inner")
    print(f"patients with slices and a T label: {len(d)} (labelled patients: {len(lab)})")
    per = d.groupby("T")["n_slices"].agg(["count", "sum"])
    per.columns = ["patients", "slices"]
    per["slice_share_%"] = (100 * per["slices"] / per["slices"].sum()).round(1)
    per["patient_share_%"] = (100 * per["patients"] / per["patients"].sum()).round(1)
    print(per)
    print()
    print(f"Always-T1, per slice  : {per.loc[1, 'slice_share_%']:.1f} %   "
          f"({per.loc[1, 'slices']} of {per['slices'].sum()} slices)")
    print(f"Always-T1, per patient: {per.loc[1, 'patient_share_%']:.1f} %   "
          f"({per.loc[1, 'patients']} of {per['patients'].sum()} patients)")

    print("(with a majority vote over slices, 'always T1' gives the per-patient value)")


if __name__ == "__main__":
    main()
