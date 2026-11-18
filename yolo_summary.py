import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solid_csv", required=True)
    ap.add_argument("--agreement_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()
    s = pd.read_csv(a.solid_csv)
    s["pid"] = s["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    keep = {"box_side_mm": "yolo_box_side_mm", "solid_open_mm": "yolo_solid_open_mm", "solid_rect_mm": "yolo_solid_rect_mm",
            "solid_dz_mm": "yolo_solid_dz_mm", "solid_frac": "yolo_solid_frac"}
    s = s[["pid"] + [c for c in keep if c in s.columns]].rename(columns=keep).drop_duplicates("pid")
    g = pd.read_csv(a.agreement_csv)
    g["pid"] = g["pid"].astype(str).str.replace("Lung_Dx_", "Lung_Dx-", regex=False)
    out = g.merge(s, on="pid", how="outer")
    out.to_csv(a.out_csv, index=False)
    print(f"{len(out)} patients -> {a.out_csv}")


if __name__ == "__main__":
    main()
