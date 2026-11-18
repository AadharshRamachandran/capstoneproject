# Interpretable Lung Cancer T-Staging from CT (6th semester work)

Code, trained models and the results reported in the paper
*"An Interpretable Segmentation-Based Pipeline for Lung Cancer T-Staging from CT Scans"*
(Lung-PET-CT-Dx, 351 labelled patients, patient-level nested cross-validation).

Pipeline: radiologist (XML) boxes locate the tumour -> TotalSegmentator organ masks,
TumorNet 2D (ResNet-152 + FPN) sizes inside the box, HU solid-component size and an FMCIB
embedding -> ordinal ensemble (logistic regression, XGBoost, CatBoost) + temperature scaling.

How to run every step: see [README_PAPER.md](README_PAPER.md) and `run_paper.ps1`.

## Models (`models/`, stored with Git LFS)
| File | What it is |
|---|---|
| `TumorNet_2D_fold0_best.pth` | TumorNet 2D (ResNet-152 + FPN) trained on MSD Task06_Lung, DSC 89.3% on the test split |
| `TumorNet_2D_threshold.pth` | Saved inference threshold for TumorNet |
| `t_stage_model.pkl` | Final T-stage ensemble (CatBoost + ordinal LR + ordinal XGBoost), trained on all labelled patients |

FMCIB weights (`model_weights.torch`) are downloaded automatically on first use and are not stored here.
Run `git lfs pull` after cloning to fetch the model files.

## Results (`results/`)
| Paper table / section | Files |
|---|---|
| Main result and per-stage table (Acc 62.3%, macro F1 56.6%, QWK 0.69, within-one 94.8%, AUC 0.90) | `results/main/` (`cross_validated_model_comparison.csv`, held-out `final_t_staging_predictions.csv`, `cm_T_crossval.png`) |
| Ablation of the input blocks | `results/ablation/<variant>/` (metrics, predictions, log per variant) and `ablation_v5log2_summary.csv` |
| Comparison of single model / feature configurations | `results/model_comparison/` |
| One-scan demo output | `results/sample_prediction/result.json` |

All reported numbers are cross-validated with whole patients held out; no training-set scores are reported.
The CT scans themselves (Lung-PET-CT-Dx, TCIA) are not included.
