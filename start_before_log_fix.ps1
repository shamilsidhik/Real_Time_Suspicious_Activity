#requires -Version 5.1

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

$ROOT = $PSScriptRoot

Set-Location $ROOT

$VENV_PY = Join-Path $ROOT ".venv\Scripts\python.exe"

$CAM_PORT = 8765
$DJ_PORT = 8000
$CAM_WAIT = 90

# A unique run ID prevents Windows log-file lock conflicts.
$RUN_ID = Get-Date -Format "yyyyMMdd_HHmmss"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

function Stop-ProcessOnPort {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $processIds = (
        netstat -ano 2>$null |
        Select-String ":$Port\s"
    ) |
    ForEach-Object {
        ($_ -split "\s+")[-1]
    } |
    Where-Object {
        $_ -match "^\d+$" -and $_ -ne "0"
    } |
    Sort-Object -Unique

    foreach ($processId in $processIds) {
        Write-Host (
            "  Stopping stale PID {0} on port {1}" -f
            $processId,
            $Port
        ) -ForegroundColor DarkYellow

        Stop-Process `
            -Id $processId `
            -Force `
            -ErrorAction SilentlyContinue
    }
}


function Show-LogTail {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [int]$Lines = 30
    )

    if (Test-Path $Path) {
        Write-Host ""
        Write-Host "--- $Path ---" -ForegroundColor Yellow

        Get-Content `
            -Path $Path `
            -Tail $Lines `
            -ErrorAction SilentlyContinue
    }
}


# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

if (-not (Test-Path $VENV_PY)) {
    Write-Host (
        "ERROR: Virtual-environment Python was not found at:`n{0}" -f
        $VENV_PY
    ) -ForegroundColor Red

    exit 1
}

if (-not (Test-Path (Join-Path $ROOT "manage.py"))) {
    Write-Host (
        "ERROR: manage.py was not found inside:`n{0}" -f
        $ROOT
    ) -ForegroundColor Red

    exit 1
}

if (-not (Test-Path (Join-Path $ROOT "camera_server.py"))) {
    Write-Host (
        "ERROR: camera_server.py was not found inside:`n{0}" -f
        $ROOT
    ) -ForegroundColor Red

    exit 1
}


# -----------------------------------------------------------------------------
# Stop applications that may be using the camera
# -----------------------------------------------------------------------------

Write-Host "Stopping camera applications..." -ForegroundColor Yellow

@(
    "ms-teams",
    "Teams",
    "WhatsApp",
    "WindowsCamera",
    "olk"
) |
ForEach-Object {
    Stop-Process `
        -Name $_ `
        -Force `
        -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2


# -----------------------------------------------------------------------------
# Stop stale project processes
# -----------------------------------------------------------------------------

Write-Host "Stopping stale project processes..." -ForegroundColor Yellow

Stop-ProcessOnPort -Port $CAM_PORT
Stop-ProcessOnPort -Port $DJ_PORT

Start-Sleep -Seconds 2


# -----------------------------------------------------------------------------
# Log setup
# -----------------------------------------------------------------------------

$LOG_DIR = Join-Path $ROOT "logs"

New-Item `
    -ItemType Directory `
    -Path $LOG_DIR `
    -Force |
Out-Null

# Camera logs.
$CAM_LOG = Join-Path $LOG_DIR "camera_server.log"
$CAM_ERR_LOG = Join-Path $LOG_DIR "camera_server_err.log"

# IMPORTANT:
# Django itself writes application logs to logs\django.log from settings.py.
# The PowerShell console must therefore use a different file.
$DJ_LOG = Join-Path (
    $LOG_DIR
) (
    "django_console_{0}.log" -f $RUN_ID
)

# Clear only the camera logs after stale camera processes were stopped.
Set-Content -Path $CAM_LOG -Value "" -ErrorAction SilentlyContinue
Set-Content -Path $CAM_ERR_LOG -Value "" -ErrorAction SilentlyContinue


# -----------------------------------------------------------------------------
# Clear expired Django sessions
# -----------------------------------------------------------------------------

Write-Host "Clearing expired Django sessions..." -ForegroundColor DarkGray

try {
    & $VENV_PY manage.py clearsessions 2>&1 |
    Out-Null

    Write-Host "  Sessions cleared." -ForegroundColor DarkGray
}
catch {
    Write-Host (
        "  Session cleanup warning: {0}" -f
        $_.Exception.Message
    ) -ForegroundColor DarkYellow
}


# -----------------------------------------------------------------------------
# Start camera server
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host (
    "Starting Camera Server (output -> {0})" -f
    $CAM_LOG
) -ForegroundColor Cyan

$cameraProcess = Start-Process `
    -FilePath $VENV_PY `
    -ArgumentList "camera_server.py" `
    -WorkingDirectory $ROOT `
    -RedirectStandardOutput $CAM_LOG `
    -RedirectStandardError $CAM_ERR_LOG `
    -PassThru `
    -WindowStyle Hidden

Write-Host (
    "  Camera server PID: {0}" -f
    $cameraProcess.Id
)

Write-Host (
    "  Waiting 5 seconds for models and camera..."
) -ForegroundColor DarkGray

Start-Sleep -Seconds 5


# -----------------------------------------------------------------------------
# Poll camera-server health
# -----------------------------------------------------------------------------

Write-Host (
    "Polling healthz for up to {0} seconds..." -f
    $CAM_WAIT
) -ForegroundColor Yellow

$cameraReady = $false

for ($second = 1; $second -le $CAM_WAIT; $second++) {

    $cameraProcess.Refresh()

    if ($cameraProcess.HasExited) {
        Write-Host ""
        Write-Host (
            "ERROR: Camera server exited early with code {0}." -f
            $cameraProcess.ExitCode
        ) -ForegroundColor Red

        Show-LogTail -Path $CAM_LOG
        Show-LogTail -Path $CAM_ERR_LOG

        exit 1
    }

    try {
        $response = Invoke-WebRequest `
            -Uri "http://127.0.0.1:$CAM_PORT/healthz" `
            -TimeoutSec 2 `
            -UseBasicParsing `
            -ErrorAction Stop

        $body = $response.Content.Trim().ToLowerInvariant()

        if ($body -eq "ok") {
            Write-Host (
                "  Camera READY after approximately {0} seconds." -f
                ($second + 5)
            ) -ForegroundColor Green

            $cameraReady = $true
            break
        }

        Write-Host (
            "  warming... ({0}/{1}) [healthz={2}]" -f
            $second,
            $CAM_WAIT,
            $body
        ) -ForegroundColor DarkYellow
    }
    catch {
        Write-Host (
            "  waiting... ({0}/{1}) [{2}]" -f
            $second,
            $CAM_WAIT,
            $_.Exception.Message
        ) -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds 1
}


if (-not $cameraReady) {
    Write-Host ""
    Write-Host (
        "WARNING: Camera server did not report ready in {0} seconds." -f
        $CAM_WAIT
    ) -ForegroundColor Red

    Show-LogTail -Path $CAM_LOG -Lines 20
    Show-LogTail -Path $CAM_ERR_LOG -Lines 20

    Write-Host ""

    $answer = Read-Host (
        "Continue and start Django anyway? (y/N)"
    )

    if ($answer -notmatch "^[Yy]$") {
        if (-not $cameraProcess.HasExited) {
            Stop-Process `
                -Id $cameraProcess.Id `
                -Force `
                -ErrorAction SilentlyContinue
        }

        exit 1
    }
}


# -----------------------------------------------------------------------------
# Start Django
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host (
    "Starting Django on http://0.0.0.0:{0}" -f
    $DJ_PORT
) -ForegroundColor Green

Write-Host (
    "Django console log -> {0}" -f
    $DJ_LOG
) -ForegroundColor DarkGray

Write-Host (
    "Django application log -> {0}" -f
    (Join-Path $LOG_DIR "django.log")
) -ForegroundColor DarkGray

Write-Host (
    "Open in browser: http://127.0.0.1:{0}" -f
    $DJ_PORT
) -ForegroundColor Cyan

Write-Host ""
Write-Host (
    "Press Ctrl+C to stop Django and the camera server."
) -ForegroundColor Yellow

try {
    & $VENV_PY run_server.py 2>&1 |
    Tee-Object `
        -FilePath $DJ_LOG
}
finally {
    Write-Host ""

    if (
        $null -ne $cameraProcess -and
        -not $cameraProcess.HasExited
    ) {
        Write-Host (
            "Stopping camera server PID {0}..." -f
            $cameraProcess.Id
        ) -ForegroundColor Yellow

        Stop-Process `
            -Id $cameraProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }

    Write-Host "Shutdown complete." -ForegroundColor Green
}
