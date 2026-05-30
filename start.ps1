$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

$env:CAMERA_SERVER_URL = "http://127.0.0.1:8765"

Write-Host "Starting camera server..." -ForegroundColor Green
$cameraProcess = Start-Process -FilePath "python" -ArgumentList "camera_server.py" -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden

try {
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        try {
            $health = Invoke-RestMethod -Uri "$env:CAMERA_SERVER_URL/healthz" -TimeoutSec 1
            if ($health.ok) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $ready) {
        throw "Camera server did not become healthy at $env:CAMERA_SERVER_URL/healthz"
    }

    Write-Host "Camera server ready. Starting Django/Waitress..." -ForegroundColor Green
    python run_server.py
}
finally {
    if ($cameraProcess -and -not $cameraProcess.HasExited) {
        Write-Host "Stopping camera server..." -ForegroundColor Yellow
        Stop-Process -Id $cameraProcess.Id -Force
    }
}
