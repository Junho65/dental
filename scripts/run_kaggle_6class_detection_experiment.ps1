param(
    [string]$Dataset = "lokisilvres/dental-disease-panoramic-detection-dataset",
    [string]$RawRoot = "data\raw\kaggle\dental_disease_panoramic_detection",
    [string]$KaggleOutRoot = "data\detection_kaggle_6class",
    [string]$ExistingRoot = "data\detection_hierarchical_zenodo_5class",
    [string]$MergedOutRoot = "data\detection_hierarchical_zenodo_kaggle_6class",
    [string]$Weights = "runs\detect\artifacts\detection\yolov8s_5class_img640_e100_pretrained\run01\weights\best.pt",
    [string]$RunProject = "artifacts/detection/yolov8s_6class_kaggle_continue",
    [string]$RunName = "run01",
    [int]$Epochs = 80,
    [int]$Patience = 15,
    [int]$ImgSize = 640,
    [int]$Batch = 8,
    [int]$Workers = 0,
    [string]$Device = "0",
    [double]$FlipLR = 0.0,
    [double]$Mosaic = 1.0,
    [int]$CloseMosaic = 10,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path .\train_detection.py)) {
    throw "Run this script from the dental project root, for example: cd C:\Pywork\dental\dental"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$notebookOut = "reports\kaggle_notebook_analysis\$timestamp"
$auditOut = "reports\kaggle_audit\$timestamp"
$datasetAuditOut = "reports\kaggle_dataset_audit\$timestamp"
$trainOutLog = "reports\process_logs\$timestamp.kaggle_6class_detection.out.log"
$trainErrLog = "reports\process_logs\$timestamp.kaggle_6class_detection.err.log"

New-Item -ItemType Directory -Path "reports\process_logs" -Force | Out-Null

Write-Host "Step 1/7: Analyze Kaggle notebooks"
python .\scripts\analyze_kaggle_notebooks.py --dataset $Dataset --out $notebookOut
if ($LASTEXITCODE -ne 0) { throw "analyze_kaggle_notebooks.py failed with exit code $LASTEXITCODE" }

Write-Host "Step 2/7: Download Kaggle dataset"
python .\scripts\download_kaggle_dental_panoramic.py --dataset $Dataset --out $RawRoot
if ($LASTEXITCODE -ne 0) { throw "download_kaggle_dental_panoramic.py failed with exit code $LASTEXITCODE" }

Write-Host "Step 3/7: Audit duplicates against existing dataset"
python .\scripts\audit_roboflow_duplicates.py `
    --roboflow-root $RawRoot `
    --existing-root $ExistingRoot `
    --out $auditOut
if ($LASTEXITCODE -ne 0) { throw "audit_roboflow_duplicates.py failed with exit code $LASTEXITCODE" }

Write-Host "Step 4/7: Remap Kaggle labels to 6-class detection"
python .\scripts\prepare_roboflow_6class_yolo.py `
    --raw $RawRoot `
    --out $KaggleOutRoot `
    --keep-csv (Join-Path $auditOut "roboflow_keep.csv") `
    --stem-prefix "kg_" `
    --target-names "caries_family,periapical_lesion,impacted_tooth,bone_loss,cyst,retained_root"
if ($LASTEXITCODE -ne 0) { throw "prepare_roboflow_6class_yolo.py failed with exit code $LASTEXITCODE" }

Write-Host "Step 5/7: Merge existing + Kaggle with class-aware split"
python .\scripts\merge_yolo_6class_stratified.py `
    --base-data (Join-Path $ExistingRoot "hierarchical_zenodo_5class.yaml") `
    --extra-data (Join-Path $KaggleOutRoot "roboflow_6class.yaml") `
    --out $MergedOutRoot
if ($LASTEXITCODE -ne 0) { throw "merge_yolo_6class_stratified.py failed with exit code $LASTEXITCODE" }

Write-Host "Step 6/7: Audit final merged detection dataset"
$mergedYaml = Join-Path $MergedOutRoot "hierarchical_zenodo_kaggle_6class.yaml"
python .\scripts\audit_yolo_detection_dataset.py `
    --data $mergedYaml `
    --out $datasetAuditOut `
    --require-all-classes-all-splits
if ($LASTEXITCODE -ne 0) { throw "audit_yolo_detection_dataset.py failed with exit code $LASTEXITCODE" }

if ($SkipTraining) {
    Write-Host "SkipTraining was set. Dataset is ready."
    Write-Host "Notebook analysis: $notebookOut"
    Write-Host "Duplicate audit: $auditOut"
    Write-Host "Dataset audit: $datasetAuditOut"
    Write-Host "Dataset YAML: $mergedYaml"
    exit 0
}

Write-Host "Step 7/7: Start YOLO detection training"
$argsList = @(
    ".\train_detection.py",
    "--model", $Weights,
    "--data", $mergedYaml,
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--imgsz", "$ImgSize",
    "--batch", "$Batch",
    "--workers", "$Workers",
    "--device", $Device,
    "--fliplr", "$FlipLR",
    "--mosaic", "$Mosaic",
    "--close-mosaic", "$CloseMosaic",
    "--project", $RunProject,
    "--name", $RunName
)

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList $argsList `
    -WorkingDirectory (Resolve-Path .) `
    -RedirectStandardOutput $trainOutLog `
    -RedirectStandardError $trainErrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started Kaggle 6-class detection training."
Write-Host "PID: $($process.Id)"
Write-Host "Notebook analysis: $notebookOut"
Write-Host "Duplicate audit: $auditOut"
Write-Host "Dataset audit: $datasetAuditOut"
Write-Host "Dataset YAML: $mergedYaml"
Write-Host "Out log: $trainOutLog"
Write-Host "Err log: $trainErrLog"
