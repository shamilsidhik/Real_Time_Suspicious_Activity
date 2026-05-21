Write-Host "Stopping camera apps..." -ForegroundColor Yellow
Stop-Process -Name "ms-teams" -ErrorAction SilentlyContinue
Stop-Process -Name "Teams" -ErrorAction SilentlyContinue
Stop-Process -Name "WhatsApp" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "Starting Camera HTTP Server on port 8765..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd 'C:\Users\USER\OneDrive\Desktop\suspicious_activity\Real_Time_Suspicious_Activity'; .\.venv\Scripts\Activate.ps1; python camera_server.py"

Start-Sleep -Seconds 4
Write-Host "Starting Django..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"
python manage.py runserver --noreload
