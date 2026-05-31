param(
    [string]$RawRoot = "data\raw\zenodo\panoramic_radiography_yolo_dataset_14_classes",
    [string]$OutRoot = "data\detection_zenodo_5class",
    [string]$ExistingRoot = "data\detection_hierarchical",
    [string]$Weights = "runs\detect\artifacts\detection\yolov8s_hierarchical_progressive\02_continue40\weights\best.pt",
    [string]$RunProject = "artifacts/detection/yolov8s_hierarchical_progressive",
    [string]$RunName = "04_zenodo_5class",
    [int]$Epochs = 50,
    [int]$Patience = 10,
    [int]$ImgSize = 416,
    [int]$Batch = 8,
    [int]$Workers = 0,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path .\train_detection.py)) {
    throw "Run this script from the dental project root, for example: cd C:\Pywork\dental\dental"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$auditOut = "reports\zenodo_audit\$timestamp"
$trainOutLog = "reports\process_logs\$timestamp.zenodo_5class.out.log"
$trainErrLog = "reports\process_logs\$timestamp.zenodo_5class.err.log"

New-Item -ItemType Directory -Path "reports\process_logs" -Force | Out-Null

python .\scripts\download_zenodo_dental_conditions.py --out $RawRoot
if ($LASTEXITCODE -ne 0) { throw "download_zenodo_dental_conditions.py failed with exit code $LASTEXITCODE" }

python .\scripts\audit_roboflow_duplicates.py `
    --roboflow-root $RawRoot `
    --existing-root $ExistingRoot `
    --out $auditOut
if ($LASTEXITCODE -ne 0) { throw "audit_roboflow_duplicates.py failed with exit code $LASTEXITCODE" }

python .\scripts\prepare_roboflow_6class_yolo.py `
    --raw $RawRoot `
    --out $OutRoot `
    --keep-csv (Join-Path $auditOut "roboflow_keep.csv") `
    --target-names "caries_family,periapical_lesion,impacted_tooth,bone_loss,retained_root"
if ($LASTEXITCODE -ne 0) { throw "prepare_roboflow_6class_yolo.py failed with exit code $LASTEXITCODE" }

$argsList = @(
    ".\train_detection.py",
    "--model", $Weights,
    "--data", (Join-Path $OutRoot "roboflow_6class.yaml"),
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--imgsz", "$ImgSize",
    "--batch", "$Batch",
    "--workers", "$Workers",
    "--device", $Device,
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

Write-Host "Started Zenodo open-data training."
Write-Host "PID: $($process.Id)"
Write-Host "Audit: $auditOut"
Write-Host "Dataset: $OutRoot"
Write-Host "Out log: $trainOutLog"
Write-Host "Err log: $trainErrLog"
