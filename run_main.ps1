$python = "C:\Users\whisk\anaconda3\python.exe"
$script = Join-Path $PSScriptRoot "main.py"
$env:MPLBACKEND = "Agg"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python"
    exit 1
}

if (-not (Test-Path $script)) {
    Write-Error "main.py not found at $script"
    exit 1
}

& $python $script
