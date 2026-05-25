param(
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "reports\process_logs"
$ReportDir = "reports\comparison_$RunStamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
Start-Transcript -Path (Join-Path $LogDir "$RunStamp.comparison.transcript.log") -Append | Out-Null

$DetectRunRoot = "runs\detect\artifacts\detection"
$SegmentRunRoot = "runs\segment\artifacts\segment"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$PyArgs
    )

    $OutLog = Join-Path $LogDir "$RunStamp.$Name.out.log"
    $ErrLog = Join-Path $LogDir "$RunStamp.$Name.err.log"
    $CommandLine = "python $($PyArgs -join ' ')"
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $Name"
    $CommandLine | Tee-Object -FilePath $OutLog
    if ($DryRun) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DRY   $Name"
        return
    }
    & python @PyArgs 1>> $OutLog 2>> $ErrLog
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE. See $OutLog and $ErrLog"
    }
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $Name"
}

$NRun = "cmp_yolov8n_hierarchical_$RunStamp"
$SRun = "cmp_yolov8s_hierarchical_$RunStamp"
$NSRun = "cmp_nsitnov8024_freeze20_rectseg_img320_b1_e30_$RunStamp"

Run-Step "train_yolov8n" @(
    "train_detection.py",
    "--model", "yolov8n.pt",
    "--data", "data\detection_hierarchical\hierarchical_detection.yaml",
    "--imgsz", "416",
    "--batch", "16",
    "--epochs", "50",
    "--name", $NRun
)

Run-Step "eval_yolov8n" @(
    "scripts\eval_yolo_box.py",
    "--weights", "$DetectRunRoot\$NRun\weights\best.pt",
    "--data", "data\detection_hierarchical\hierarchical_detection.yaml",
    "--imgsz", "416",
    "--task", "detect",
    "--out", "$ReportDir\yolov8n_box_metrics.json"
)

Run-Step "train_yolov8s" @(
    "train_detection.py",
    "--model", "yolov8s.pt",
    "--data", "data\detection_hierarchical\hierarchical_detection.yaml",
    "--imgsz", "416",
    "--batch", "8",
    "--epochs", "50",
    "--name", $SRun
)

Run-Step "eval_yolov8s" @(
    "scripts\eval_yolo_box.py",
    "--weights", "$DetectRunRoot\$SRun\weights\best.pt",
    "--data", "data\detection_hierarchical\hierarchical_detection.yaml",
    "--imgsz", "416",
    "--task", "detect",
    "--out", "$ReportDir\yolov8s_box_metrics.json"
)

Run-Step "train_nsitnov" @(
    "scripts\train_segment_freeze.py",
    "--model", "weights\8024.pt",
    "--data", "data\detection_hierarchical_rectseg\hierarchical_detection_rectseg.yaml",
    "--imgsz", "320",
    "--batch", "1",
    "--epochs", "30",
    "--patience", "10",
    "--freeze", "20",
    "--workers", "0",
    "--name", $NSRun,
    "--no-amp"
)

Run-Step "eval_nsitnov" @(
    "scripts\eval_yolo_box.py",
    "--weights", "$SegmentRunRoot\$NSRun\weights\best.pt",
    "--data", "data\detection_hierarchical_rectseg\hierarchical_detection_rectseg.yaml",
    "--imgsz", "320",
    "--task", "segment",
    "--out", "$ReportDir\nsitnov8024_freeze20_box_metrics.json"
)

Write-Host "Comparison complete: $ReportDir"
Stop-Transcript | Out-Null
