# Code for the report: CT T-staging pipeline and the other experiments in the paper

Open PowerShell in this folder, edit the paths at the top of `run_paper.ps1`, then:

    pip install -r requirements.txt
    .\run_paper.ps1 -List                      # shows every step and its output file
    .\run_paper.ps1                            # MAIN pipeline (steps 1-5)
    .\run_paper.ps1 -Steps ablation            # then any other group: explore | yolo | medsam2 | yolofeat | baselines

Every step skips itself when its output file already exists (add `-Redo` to force). Stop with Ctrl+C and run the same command to continue.
Nothing here was run on your data by me; the scripts were tested on small synthetic data and on the outputs you sent.

## What each part of the paper runs

| Paper section / table | Step (group) | Script | Output |
|---|---|---|---|
| Series choice, radiologist boxes, TotalSegmentator organ masks | 1 (main) | `local_shard_features.py`, `add_boxes_to_cache.py` | `shard_features_xml\*.zip`, cache |
| TumorNet inside the box | 2 (main) | `box_seg_sizes.py` | `box_seg_features.csv` |
| HU solid-component size | 3 (main) | `box_solid_sizes.py` | `box_solid_features.csv` |
| FMCIB embedding | 4 (main) | `extract_fmcib_features.py` (Python 3.10 venv) | `fmcib_features_ct.csv` |
| **Proposed ensemble** (Table: main, per-stage) | 5 (main) | `final_t_staging.py --profile image --override none` | `results_mode_xml\` |
| Ablation of input blocks | 6 (ablation) | `ablation_v5log2.py` | `ablation_v5log2\ablation_v5log2_summary.csv` |
| Single-model comparison (Table: explore) | 7 (explore) | `experiment_ct_models.py`, `experiment_v15.py` | `experiments_v5\`, `experiment_v15.csv` |
| YOLOv11 second reader | 8-9 (yolo) | `yolo_boxes.py`, `yolo_map_eval.py`, `yolo_summary.py` | `yolo\`, `results_mode_both\` |
| MedSAM2 lesion size | 10 (medsam2) | `box_medsam2_sizes.py` | `medsam2_test50.csv` |
| Our implementation of Wehbe et al. (Table: yolofeat) | 11 (yolofeat) | `yolo_tnm_classifier.py` (uses `yolo_feature_staging.py`) | `yolo_replication\` |
| Always-T1 (slices / patients) | 12 (baselines) | `always_t1_slice_baseline.py` | printed |
| ResNet-152 / DenseNet-121 / Swin-Tiny | 13 (baselines) | `dl_slice_baselines.py` | `dl_baselines.csv` |
| One-scan demo | - | `prepare_sample_scan.py`, `predict_one_scan.py` | `sample_scan\result.json` |
| Part A: RadNLP table | - | `radnlp_f1_table.py` (from saved predictions) | `.tex` table |

Rough times (your PC): step 1 hours (TotalSegmentator, only if the cache is missing); steps 2-4 about 1-3 h; step 5 about 1 h; ablation 9 variants about 12-20 h in total (use `ablation_v5log2.py --only "variant1,variant2"`); YOLO folds several hours; step 11 about 30 min + 5 passes of feature extraction on the first run.

## Things to know
* **Checkpoints needed:** TumorNet (`$Ckpt`, used by steps 1-2), YOLOv11 weights are produced by step 8 (`yolo11n.pt` is the starting point, included), FMCIB weights download on first use, MedSAM2 needs its own repo and checkpoint (see `box_medsam2_sizes.py`).
* **Step 13 (ResNet / DenseNet / Swin):** this is a re-implementation (AdamW 3e-5 / 3e-4, class-weighted focal loss, majority vote) scored with whole patients held out. The numbers in the paper's baseline table came from the earlier full-dataset runs, so they will not match exactly; do not mix the two without saying which is which.
* **Part A (RadNLP fine-tuning of BioBERT, BioClinicalBERT, Clinical Longformer, GatorTron-base) is not in this folder.** It was run as a Kaggle notebook; `radnlp_f1_table.py` builds the table from its saved test predictions. Send the notebook if you want it added.
* **TumorNet training on the Medical Segmentation Decathlon** is also not included (it was run in a notebook); only the trained checkpoint is used here.
* `feature_extraction.py` imports `paper_rules.py` as a helper for distance features. No fixed-rule or override result is used or reported.

## Predict the T-stage of ONE scan (demo)
The scan, its XML boxes and the trained model are kept inside this folder, so the shard disks are needed only once.

    # once: copy one patient + the trained model into this folder
    python prepare_sample_scan.py --pid Lung_Dx-A0001 --shard_dirs D:\hi\shard_000 D:\hi\shard_001 ... C:\Users\HP\Downloads\working\_local_shards\shard_007 ^
           --model C:\Users\HP\Downloads\working\results_mode_xml\checkpoints\t_stage_model.pkl

    # any time: T-stage of that scan (first run ~ TotalSegmentator + TumorNet + FMCIB; later runs take seconds)
    python predict_one_scan.py --ckpt_dir "C:\Users\HP\Downloads\archive (1)" --fmcib_python C:\Users\HP\Downloads\working\fmcib_venv\Scripts\python.exe ^
           --labels_csv C:\Users\HP\Downloads\C__lung_labels.csv --cv_predictions C:\Users\HP\Downloads\working\results_mode_xml\final_t_staging_predictions.csv

It prints the predicted T-stage, the four probabilities, the 90 % conformal set, the tumour size and the anatomy evidence, and writes `sample_scan\result.json`.
The saved model was trained on all labelled patients, so for a patient from the dataset the printed stage is a training-set prediction; the honest one is
`heldout_crossvalidated_T` (from `--cv_predictions`). Pick a patient that has a recorded label if you want to compare.
