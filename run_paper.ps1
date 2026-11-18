param([string[]]$Steps = @("main"), [switch]$Redo, [switch]$List, [string]$Device = "0")
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Pen     = "D:\hi"
$Work    = "C:\Users\HP\Downloads\working"
$Ckpt    = "C:\Users\HP\Downloads\archive (1)"
$Cache   = "C:\cache"
$Labels  = "C:\Users\HP\Downloads\C__lung_labels.csv"
$FmcibPy = "$Work\fmcib_venv\Scripts\python.exe"
$Ms2Ckpt = "C:\MedSAM2\checkpoints\MedSAM2_latest.pt"
$Ms2Cfg  = "configs/sam2.1_hiera_t512.yaml"
Set-Location $PSScriptRoot
$Shards = @(); foreach ($n in 0..7) { $t = "{0:D3}" -f $n; $d = if ($n -eq 7) { "$Work\_local_shards\shard_$t" } else { "$Pen\shard_$t" }; if (Test-Path $d) { $Shards += $d } else { Write-Warning "shard $n not found: $d" } }
$Xml   = "$Work\shard_features_xml"
$Fm    = "$Work\fmcib_features_ct.csv"
$Box   = "$Work\box_seg_features.csv"
$Solid = "$Work\box_solid_features.csv"
$Yolo  = "$Work\yolo"

function Step($name, $group, $out, [scriptblock]$cmd) {
    if ($List) { Write-Host ("{0,-34} group={1,-9} output={2}" -f $name, $group, $out); return }
    if (-not ($Steps -contains $group)) { return }
    if ($out -and (Test-Path $out) -and -not $Redo) { Write-Host "[skip] $name  (exists: $out)"; return }
    $t = Get-Date; Write-Host "`n=== $name"
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("    done in {0:N1} min" -f ((Get-Date) - $t).TotalMinutes)
}

Step "1 TotalSegmentator + boxes, per shard" "main" "$Xml\patient_features_shard000.zip" {
    foreach ($n in 0..7) { $tag = "{0:D3}" -f $n; $sd = $Shards | Where-Object { $_ -like "*shard_$tag" } | Select-Object -First 1; if (-not $sd) { continue }
        python local_shard_features.py --shards_dir $sd --shard_num $n --ckpt_dir $Ckpt --out_dir "$Work\shard_features_ts" --merge_ts_masks --align_ts_axes --cache_dir $Cache
        if ($LASTEXITCODE -ne 0) { throw "shard $n (TotalSegmentator) failed" }
        python add_boxes_to_cache.py --shard_dir $sd --cache_dir $Cache --shard_num $n
        if ($LASTEXITCODE -ne 0) { throw "shard $n (boxes) failed" }
        python local_shard_features.py --shards_dir $sd --shard_num $n --from_cache --cache_dir $Cache --out_dir $Xml --tumor_source xml
        if ($LASTEXITCODE -ne 0) { throw "shard $n (features) failed" } } }
Step "2 TumorNet in box" "main" $Box { python box_seg_sizes.py --shard_dirs @Shards --ckpt_dir $Ckpt --out_csv $Box }
Step "3 solid component (HU)" "main" $Solid { python box_solid_sizes.py --shard_dirs @Shards --out_csv $Solid }
Step "4 FMCIB embedding" "main" $Fm { & $FmcibPy extract_fmcib_features.py --shard_dirs @Shards --out_csv $Fm --patch_dir "$Work\fmcib_patches" --batch_size 4 --workers 3 }
Step "5 final T-staging (main result)" "main" "$Work\results_mode_xml\final_t_staging_predictions.csv" {
    python final_t_staging.py --features_dir $Xml --labels_csv $Labels --out_dir "$Work\results_mode_xml" --fmcib_csv $Fm --box_features $Box --solid_features $Solid --profile image --override none }
Step "6 block ablation (9 variants, ~1-2 h each)" "ablation" "$Work\ablation_v5log2\ablation_v5log2_summary.csv" {
    python ablation_v5log2.py --features_dir $Xml --labels_csv $Labels --fmcib_csv $Fm --box_features $Box --solid_features $Solid --out_root "$Work\ablation_v5log2" }
Step "7 single-model comparison" "explore" "$Work\experiments_v5\ct_model_experiments.csv" {
    python experiment_ct_models.py --features_dir $Xml --labels_csv $Labels --fmcib_csv $Fm --box_features $Box --solid_features $Solid --out_csv "$Work\experiments_v5\ct_model_experiments.csv" }
Step "7b extra size-feature experiments" "explore" "$Work\experiment_v15.csv" {
    python experiment_v15.py --features_dir $Xml --labels_csv $Labels --fmcib_csv $Fm --box_features $Box --solid_features $Solid --out_csv "$Work\experiment_v15.csv" }
Step "8 YOLOv11 folds + train + predict (long)" "yolo" "$Yolo\raw\fold4.csv" {
    if (-not (Test-Path "$Yolo\folds.csv")) { python yolo_boxes.py folds --shard_dirs @Shards --labels_csv $Labels --out_csv "$Yolo\folds.csv" }
    python yolo_boxes.py all --shard_dirs @Shards --folds_csv "$Yolo\folds.csv" --work $Yolo --cache_dir $Cache --epochs 40 --patience 10 --device $Device }
Step "9 YOLO boxes + agreement + mAP" "yolo" "$Work\yolo_map.csv" {
    python yolo_boxes.py build --folds_csv "$Yolo\folds.csv" --work $Yolo --cache_dir $Cache --out_csv "$Yolo\yolo_boxes.csv"
    python yolo_map_eval.py --shard_dirs @Shards --cache_dir $Cache --raw_dir "$Yolo\raw" --out_csv "$Work\yolo_map.csv" }
Step "9b XML + YOLO final staging" "yolo" "$Work\results_mode_both\final_t_staging_predictions.csv" {
    python box_solid_sizes.py --shard_dirs @Shards --out_csv "$Work\yolo_solid_features.csv" --source yolo --yolo_csv "$Yolo\yolo_boxes.csv"
    python yolo_summary.py --solid_csv "$Work\yolo_solid_features.csv" --agreement_csv "$Yolo\yolo_agreement.csv" --out_csv "$Work\yolo_summary.csv"
    python final_t_staging.py --features_dir $Xml --labels_csv $Labels --out_dir "$Work\results_mode_both" --fmcib_csv $Fm --box_features $Box --solid_features $Solid --yolo_summary "$Work\yolo_summary.csv" --profile image --override none }
Step "10 MedSAM2 lesion size (needs the MedSAM2 repo)" "medsam2" "$Work\medsam2_test50.csv" {
    python box_medsam2_sizes.py --shard_dirs @Shards --checkpoint $Ms2Ckpt --cfg $Ms2Cfg --out_csv "$Work\medsam2_test50.csv" --limit 50 --device cuda }
Step "11 YOLO backbone features -> PCA -> TNMClassifier" "yolofeat" "$Work\yolo_replication\yolo_tnm_classifier_results.csv" {
    python yolo_tnm_classifier.py --work $Yolo --folds_csv "$Yolo\folds.csv" --labels_csv $Labels --out_dir "$Work\yolo_replication" --random_split --device $Device }
Step "12 always-T1 (slices and patients)" "baselines" "" {
    python always_t1_slice_baseline.py --labels_csv $Labels --pool_work $Yolo }
Step "13 ResNet-152 / DenseNet-121 / Swin-Tiny slice-level" "baselines" "$Work\dl_baselines.csv" {
    python dl_slice_baselines.py --work $Yolo --folds_csv "$Yolo\folds.csv" --out_csv "$Work\dl_baselines.csv" --device $Device }
if (-not $List) { Write-Host "`nfinished: $($Steps -join ', ')" }
