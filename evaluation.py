import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)

warnings.filterwarnings("ignore")

STAGE_LABELS = ["T1", "T2", "T3", "T4"]
T_SUB_LABELS = ["T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]

_default_results = Path(__file__).resolve().parent.parent / "working" / "results"
RESULTS_DIR = Path(os.environ.get("TNM_RESULTS_DIR", str(_default_results)))

_T_NORM_MAP = {
    "T1":  "T1", "T1A": "T1", "T1B": "T1", "T1C": "T1",
    "T1a": "T1", "T1b": "T1", "T1c": "T1",
    "1A":  "T1", "1B":  "T1", "1C":  "T1", "1":   "T1",
    "T2":  "T2", "T2A": "T2", "T2B": "T2", "T2a": "T2", "T2b": "T2",
    "2":   "T2", "2A":  "T2", "2B":  "T2",
    "T3":  "T3", "3":   "T3",
    "T4":  "T4", "4":   "T4",
}
_T_NORM4 = {
    "T1a": "T1", "T1b": "T1", "T1c": "T1", "T1": "T1",
    "T2a": "T2", "T2b": "T2", "T2": "T2",
    "T3": "T3", "T4": "T4", "TX": "T4",
}

def normalize_t(label) -> str:
    if label is None:
        return "Unknown"
    key = str(label).strip().upper().replace(" ", "")
    return _T_NORM_MAP.get(key, str(label).strip())

def compute_classification_metrics(y_true: List, y_pred: List,
                                    labels: Optional[List] = None) -> Dict:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    per_f1   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_rec  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "macro_f1":        f1_score(y_true, y_pred, average="macro",
                                    zero_division=0, labels=labels),
        "macro_precision": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0, labels=labels),
        "macro_recall":    recall_score(y_true, y_pred, average="macro",
                                        zero_division=0, labels=labels),
        "per_class_f1":    per_f1.tolist(),
        "per_class_prec":  per_prec.tolist(),
        "per_class_rec":   per_rec.tolist(),
        "labels":          labels,
        "report":          classification_report(y_true, y_pred,
                                                  labels=labels, zero_division=0),
    }

def print_metrics_table(metrics: Dict, title: str):
    labels   = metrics.get("labels", [])
    per_f1   = metrics.get("per_class_f1",  [])
    per_prec = metrics.get("per_class_prec", [])
    per_rec  = metrics.get("per_class_rec",  [])

    print(f"\n  {title}")
    print("  " + "-" * 60)
    print(f"  Overall Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1         : {metrics['macro_f1']*100:.2f}%")
    print(f"  Macro Precision  : {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall     : {metrics['macro_recall']*100:.2f}%")
    print()
    print(f"  {'Stage':<8} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("  " + "-" * 42)
    for i, lbl in enumerate(labels):
        p = per_prec[i] * 100 if i < len(per_prec) else 0.0
        r = per_rec[i]  * 100 if i < len(per_rec)  else 0.0
        f = per_f1[i]   * 100 if i < len(per_f1)   else 0.0
        print(f"  {lbl:<8} {p:>9.2f}% {r:>9.2f}% {f:>9.2f}%")
    print("  " + "-" * 60)

def plot_confusion_matrix(y_true: List, y_pred: List, labels: List,
                           title: str, out_path: str):
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.7, f"n={cm[i, j]}",
                    ha="center", va="center", fontsize=8, color="gray")
    ax.set_xlabel("Predicted Stage")
    ax.set_ylabel("True Stage")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def plot_training_history(history: Dict, title: str, out_path: str):
    if not history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "train_loss" in history and history["train_loss"]:
        axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
        if "val_loss" in history and history["val_loss"]:
            axes[0].plot(history["val_loss"], label="Val Loss", color="orange")
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    mk  = "val_dsc" if "val_dsc" in history else "val_f1"
    mla = "DSC"     if mk == "val_dsc"       else "F1"
    if mk in history and history[mk]:
        axes[1].plot(history[mk], label=f"Val {mla}", color="green")
        best_ep  = int(np.argmax(history[mk]))
        best_val = history[mk][best_ep]
        axes[1].axvline(best_ep, color="red", linestyle="--", alpha=0.5,
                        label=f"Best ep {best_ep + 1}")
        axes[1].set_title(f"{mla} over Epochs  (best={best_val*100:.2f}%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(mla)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def evaluate_segmentation(seg_metrics_2d: Dict, seg_metrics_3d: Dict,
                           hist_2d: Dict, hist_3d: Dict, out_dir: Path):
    if seg_metrics_2d:
        print("\n  ─── TumorNet 2D — Segmentation Metrics ───────────────")
        print(f"  {'Metric':<16} {'Value':>10}")
        print("  " + "─" * 30)
        for k, v in seg_metrics_2d.items():
            if isinstance(v, float):
                print(f"  {k:<16} {v*100:>9.2f}%")
        print("  " + "─" * 30)

    if seg_metrics_3d:
        print("\n  ─── TumorNet 3D — Segmentation Metrics ───────────────")
        for k, v in seg_metrics_3d.items():
            if isinstance(v, float):
                print(f"  {k:<16} {v*100:>9.2f}%")

    if hist_2d:
        plot_training_history(hist_2d, "TumorNet 2D Training History",
                              str(out_dir / "hist_tumornet2d.png"))
    if hist_3d:
        plot_training_history(hist_3d, "TumorNet 3D Training History",
                              str(out_dir / "hist_tumornet3d.png"))

def evaluate_pathway_a(staging_results: pd.DataFrame,
                        ground_truth: pd.DataFrame,
                        out_dir: Path) -> Dict:
    if staging_results is None or staging_results.empty:
        print("  No Pathway A results to evaluate.")
        return {}

    gt_cols = [c for c in ["pid", "T_stage"] if c in ground_truth.columns]
    merged  = staging_results.merge(ground_truth[gt_cols], on="pid", how="inner")

    if merged.empty or "predicted_T" not in merged.columns:
        print("  Cannot merge Pathway A results with ground truth.")
        return {}
    if "T_stage" not in merged.columns:
        print("  T_stage column missing from ground truth — skipping.")
        return {}

    y_true_raw = [normalize_t(x) for x in merged["T_stage"].tolist()]
    y_pred_raw = [normalize_t(x) for x in merged["predicted_T"].tolist()]
    y_true = [_T_NORM4.get(t, t) for t in y_true_raw]
    y_pred = [_T_NORM4.get(p, p) for p in y_pred_raw]

    pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t in STAGE_LABELS and p in STAGE_LABELS
    ]
    if not pairs:
        print(
            "\n  Pathway A T-stage: no valid label pairs after normalisation "
            "(all ground-truth labels may be 'Unknown')."
        )
        return {}

    yt, yp       = zip(*pairs)
    valid_labels = [l for l in STAGE_LABELS if l in set(yt) | set(yp)]
    m            = compute_classification_metrics(list(yt), list(yp), labels=valid_labels)

    print_metrics_table(m, f"Pathway A — T-Stage (Segmentation-Based, n={len(yt)})")
    print("\n  Full Classification Report:")
    print(m["report"])
    plot_confusion_matrix(
        list(yt), list(yp), valid_labels,
        "Pathway A — T-Stage Confusion Matrix",
        str(out_dir / "cm_pathwayA_T.png"),
    )

    print("\n  Pathway A — T-stage evaluation complete.")
    if "T_borderline" in merged.columns:
        bl = merged["T_borderline"].fillna(False).astype(bool)
        n_bl = int(bl.sum())
        if n_bl > 0:
            print(f"\n  Borderline T-stage cases: {n_bl}/{len(merged)} patients")
            for subset_name, subset_idx in [("Borderline", bl), ("Non-borderline", ~bl)]:
                sub = merged[subset_idx]
                if sub.empty:
                    continue
                yt_s = [_T_NORM4.get(normalize_t(x), "") for x in sub["T_stage"].tolist()]
                yp_s = [_T_NORM4.get(normalize_t(x), "") for x in sub["predicted_T"].tolist()]
                pairs_s = [(t, p) for t, p in zip(yt_s, yp_s)
                           if t in STAGE_LABELS and p in STAGE_LABELS]
                if pairs_s:
                    acc = sum(t == p for t, p in pairs_s) / len(pairs_s)
                    print(f"    {subset_name} accuracy: {acc*100:.1f}% (n={len(pairs_s)})")
            print("  (Borderline cases have elevated staging uncertainty — "
                  "clinical review recommended)")

    return {"T": m}

def evaluate_pathway_b(image_results: Dict,
                        ground_truth: pd.DataFrame,
                        out_dir: Path) -> Dict:
    if not image_results:
        print("  No Pathway B results to evaluate.")
        return {}

    print(f"\n  ─── Pathway B — Image Classifiers ─────────────────────")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Macro F1':>10} "
          f"{'Precision':>10} {'Recall':>10}")
    print("  " + "─" * 62)

    all_metrics: Dict = {}
    for name, m in sorted(image_results.items(),
                           key=lambda x: -x[1].get("macro_f1", 0)):
        if not isinstance(m, dict) or "accuracy" not in m:
            continue
        print(f"  {name:<22} {m['accuracy']*100:>9.2f}% "
              f"{m['macro_f1']*100:>9.2f}% "
              f"{m.get('macro_precision', 0)*100:>9.2f}% "
              f"{m.get('macro_recall', 0)*100:>9.2f}%")
        all_metrics[name] = m
    print("  " + "─" * 62)
    return all_metrics

def compare_pathways(pathway_a_t_metrics: Dict,
                     pathway_b_metrics: Dict,
                     out_dir: Path):
    rows = []
    if pathway_a_t_metrics and "accuracy" in pathway_a_t_metrics:
        rows.append({
            "approach":        "Pathway A (Seg-Based)",
            "accuracy":        pathway_a_t_metrics.get("accuracy", 0),
            "macro_f1":        pathway_a_t_metrics.get("macro_f1", 0),
            "macro_precision": pathway_a_t_metrics.get("macro_precision", 0),
            "macro_recall":    pathway_a_t_metrics.get("macro_recall", 0),
        })
    for name, m in pathway_b_metrics.items():
        if isinstance(m, dict) and "accuracy" in m:
            rows.append({
                "approach":        name,
                "accuracy":        m.get("accuracy", 0),
                "macro_f1":        m.get("macro_f1", 0),
                "macro_precision": m.get("macro_precision", 0),
                "macro_recall":    m.get("macro_recall", 0),
            })
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(out_dir / "pathway_comparison.csv", index=False)

    print(f"\n  ─── Pathway A vs Pathway B Comparison ─────────────────")
    print(f"  {'Approach':<26} {'Accuracy':>10} {'Macro F1':>10} "
          f"{'Precision':>10} {'Recall':>10}")
    print("  " + "─" * 70)
    for _, r in df.iterrows():
        print(f"  {r['approach']:<26} {r['accuracy']*100:>9.2f}% "
              f"{r['macro_f1']*100:>9.2f}% "
              f"{r['macro_precision']*100:>9.2f}% "
              f"{r['macro_recall']*100:>9.2f}%")
    print("  " + "─" * 70)

    best = df.iloc[0]
    print(f"\n  Best approach: {best['approach']} "
          f"(Macro F1 = {best['macro_f1']*100:.2f}%)")
    print(f"  Comparison saved: {out_dir / 'pathway_comparison.csv'}")

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.8), 5))
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - w * 1.5, df["accuracy"]        * 100, w, label="Accuracy",   color="steelblue")
    ax.bar(x - w * 0.5, df["macro_f1"]         * 100, w, label="Macro F1",  color="seagreen")
    ax.bar(x + w * 0.5, df["macro_precision"]  * 100, w, label="Precision", color="darkorange")
    ax.bar(x + w * 1.5, df["macro_recall"]     * 100, w, label="Recall",    color="mediumpurple")
    ax.set_xticks(x)
    ax.set_xticklabels(df["approach"], rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Pathway A (Anatomy-Aware) vs Pathway B (Image Classifiers) — T-Stage")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(str(out_dir / "pathway_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {out_dir / 'pathway_comparison.png'}")

def run_full_evaluation(staging_results,
                         image_classifier_results: Dict,
                         ground_truth: pd.DataFrame,
                         seg_metrics_2d: Dict,
                         seg_metrics_3d: Dict,
                         hist_2d: Dict,
                         hist_3d: Dict):
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 55)
    print("  EVALUATION RESULTS")
    print("█" * 55)

    evaluate_segmentation(seg_metrics_2d, seg_metrics_3d, hist_2d, hist_3d, out_dir)

    pa_metrics = evaluate_pathway_a(staging_results, ground_truth, out_dir)
    pb_metrics = evaluate_pathway_b(image_classifier_results, ground_truth, out_dir)

    compare_pathways(pa_metrics.get("T", {}), pb_metrics, out_dir)

    saved = ["pathway_comparison.csv", "pathway_comparison.png", "cm_pathwayA_T.png"]
    if hist_2d:
        saved.append("hist_tumornet2d.png")
    if hist_3d:
        saved.append("hist_tumornet3d.png")
    print(f"\n  All results saved to: {out_dir}")
    print(f"  Files: {', '.join(saved)}")
