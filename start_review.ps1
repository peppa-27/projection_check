param(
    [string]$Python = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DataPaths
)

# Prefer a project environment, then the active Python.
$reviewCandidates = @()
if ($Python) {
    $reviewCandidates += $Python
} else {
    $reviewCandidates += Join-Path $PSScriptRoot '.venv-review\Scripts\python.exe'
    $reviewCandidates += Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $reviewCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($reviewCommand) {
        $reviewCandidates += $reviewCommand.Source
    }
}

$reviewSelectedPython = $null
foreach ($reviewCandidate in $reviewCandidates) {
    if (Test-Path -LiteralPath $reviewCandidate) {
        & $reviewCandidate -c 'from PySide6.QtWidgets import QApplication; import numpy; from PIL import Image' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $reviewSelectedPython = $reviewCandidate
            break
        }
    }
}

if (-not $reviewSelectedPython) {
    Write-Host 'No usable PySide6 environment was found.'
    Write-Host 'Install: python -m pip install -r requirements.txt'
    Write-Host 'Or run: .\start_review.ps1 -Python C:\path\to\python.exe'
    exit 1
}

Write-Host "Python: $reviewSelectedPython"
& $reviewSelectedPython (Join-Path $PSScriptRoot 'review_data.py') @DataPaths
exit $LASTEXITCODE
