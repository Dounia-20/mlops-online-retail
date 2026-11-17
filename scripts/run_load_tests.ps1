$workersList = 1, 2, 4
$usersList = 10, 50, 100
$duration = "30s"
New-Item -ItemType Directory -Force docs\charge | Out-Null
Remove-Item docs\charge\w*_u*.csv -ErrorAction SilentlyContinue

foreach ($w in $workersList) {
    Write-Host "=== API avec $w worker(s) ==="
    $p = Start-Process -FilePath ".venv\Scripts\python.exe" `
        -ArgumentList "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", $w `
        -PassThru -WindowStyle Hidden
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        try {
            $h = Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2
            if ($h.status -eq "ok") { $ready = $true; break }
        } catch {}
    }
    if (-not $ready) {
        Write-Host "API non prete avec $w worker(s), configuration ignoree"
        taskkill /PID $p.Id /T /F | Out-Null
        continue
    }
    foreach ($u in $usersList) {
        Write-Host "  -> $u utilisateurs pendant $duration"
        $rate = [Math]::Min($u, 20)
        & .venv\Scripts\locust.exe -f tests\load\locustfile.py --headless -u $u -r $rate -t $duration `
            --host http://127.0.0.1:8000 --csv "docs\charge\w${w}_u${u}" 2>&1 | Out-Null
        Start-Sleep -Seconds 3
    }
    taskkill /PID $p.Id /T /F | Out-Null
    Start-Sleep -Seconds 3
}
Write-Host "Campagne terminee."
