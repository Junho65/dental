param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [int]$Tail = 40
)

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Log path not found: $Path"
}

$ansiPattern = "`e\[[0-?]*[ -/]*[@-~]"

Get-Content -LiteralPath $Path -Encoding UTF8 -Wait -Tail $Tail |
    ForEach-Object {
        $line = [regex]::Replace($_, $ansiPattern, "")
        if ($line.Trim().Length -gt 0) {
            $line
        }
    }
