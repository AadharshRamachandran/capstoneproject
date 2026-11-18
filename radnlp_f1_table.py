import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def head_scores(t, p):
    labs = sorted(set(t) | set(p))
    return {"acc": accuracy_score(t, p),
            "micro": f1_score(t, p, average="micro", labels=labs, zero_division=0),
            "macro": f1_score(t, p, average="macro", labels=labs, zero_division=0),
            "weighted": f1_score(t, p, average="weighted", labels=labs, zero_division=0)}


def model_scores(d):
    out = {}
    per = {h: head_scores(d[f"{h}_true"].astype(str), d[f"{h}_pred"].astype(str)) for h in "TNM"}
    for m in ("acc", "micro", "macro", "weighted"):
        out[m] = float(np.mean([per[h][m] for h in "TNM"]))
    for h in "TNM":
        out[f"acc_{h}"] = per[h]["acc"]
    out["joint"] = float(np.mean([(d[f"{h}_true"].astype(str) == d[f"{h}_pred"].astype(str)) for h in "TNM"], axis=0).min() if False else
                         np.all([d[f"{h}_true"].astype(str).to_numpy() == d[f"{h}_pred"].astype(str).to_numpy() for h in "TNM"], axis=0).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out_tex", default="radnlp_f1_table.tex")
    ap.add_argument("--out_csv", default="radnlp_f1_table.csv")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f).assign(_file=f) for f in a.pred_csv], ignore_index=True)
    if "model" not in df:
        df["model"] = df["_file"]
    if "split" in df:
        df = df[df["split"].astype(str).str.lower() == a.split.lower()]
    rng = np.random.default_rng(0)
    rows = []
    for name, d in df.groupby("model", sort=False):
        d = d.reset_index(drop=True)
        s = model_scores(d)
        boots = [model_scores(d.iloc[rng.integers(0, len(d), len(d))])["macro"] for _ in range(a.boot)]
        rows.append({"model": name, "n_notes": len(d), **s, "macro_ci_lo": np.percentile(boots, 2.5), "macro_ci_hi": np.percentile(boots, 97.5)})
    res = pd.DataFrame(rows)
    res.to_csv(a.out_csv, index=False)
    best = {c: res[c].max() for c in ("acc", "micro", "macro", "weighted")}
    lines = [r"\begin{table}[h!]", r"\centering",
             rf"\caption{{Report-staging results on the {a.split} notes (n={int(res.n_notes.iloc[0])}). Accuracy, micro, macro and weighted F1 are averaged over the T, N and M heads.}}",
             r"\label{tab:nlp-main}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Model} & \textbf{T acc} & \textbf{N acc} & \textbf{Micro F1} & \textbf{Macro F1} & \textbf{Weighted F1} \\", r"\midrule"]
    for _, r in res.iterrows():
        f = lambda c, v: (rf"\textbf{{{v:.3f}}}" if abs(r[c] - best.get(c, -1)) < 1e-12 else f"{v:.3f}")
        lines.append(f"{r.model} & {r.acc_T:.3f} & {r.acc_N:.3f} & {f('micro', r.micro)} & {f('macro', r.macro)} & {f('weighted', r.weighted)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    open(a.out_tex, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"\nwrote {a.out_tex} and {a.out_csv}")


if __name__ == "__main__":
    main()
