param(
    [string]$ApiKeyEnv = "ROBOFLOW_API_KEY",
    [string]$RawRoot = "data\raw\roboflow\dental-x-ray-panoramic",
    [string]$OutRoot = "data\detection_roboflow_6class",
    [string]$ExistingRoot = "data\detection_hierarchical",
    [string]$Weights = "runs\detect\artifacts\detection\yolov8s_hierarchical_progressive\02_continue40\weights\best.pt",
    [string]$RunProject = "artifacts/detection/yolov8s_hierarchical_progressive",
    [string]$RunName = "04_roboflow_6class",
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

if (-not [Environment]::GetEnvironmentVariable($ApiKeyEnv)) {
    throw "$ApiKeyEnv is not set. Example: `$env:$ApiKeyEnv = '<your-roboflow-api-key>'"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$auditOut = "reports\roboflow_audit\$timestamp"
$trainOutLog = "reports\process_logs\$timestamp.roboflow_6class.out.log"
$trainErrLog = "reports\process_logs\$timestamp.roboflow_6class.err.log"

New-Item -ItemType Directory -Path "reports\process_logs" -Force | Out-Null

python .\scripts\download_roboflow_dataset.py `
    --workspace opg-unbmz `
    --project dental-x-ray-panoramic `
    --version 1 `
    --format yolov8 `
    --out $RawRoot `
    --api-key-env $ApiKeyEnv

python .\scripts\audit_roboflow_duplicates.py `
    --roboflow-root $RawRoot `
    --existing-root $ExistingRoot `
    --out $auditOut

python .\scripts\prepare_roboflow_6class_yolo.py `
    --raw $RawRoot `
    --out $OutRoot `
    --keep-csv (Join-Path $auditOut "roboflow_keep.csv")

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

Write-Host "Started Roboflow 6-class training."
Write-Host "PID: $($process.Id)"
Write-Host "Audit: $auditOut"
Write-Host "Dataset: $OutRoot"
Write-Host "Out log: $trainOutLog"
Write-Host "Err log: $trainErrLog"
