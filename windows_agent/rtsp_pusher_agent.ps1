$ErrorActionPreference = "Stop"

# ---- Настройки ----
$Server           = "213.199.56.244"
$CamUrl           = "rtsp://admin:123456@192.168.1.104:554/media/live/1/1"
$PublishUrl       = "rtsp://$Server`:8554/cam1"
$RestartSignalUrl = "https://cv.pediatr.kg/camera/cam1/restart-signal?x_api_key=iYv3H8VpdmW7hu8WIu_Yq2kn0Mspvd8k"
$PollIntervalSec  = 5     # как часто проверять ffmpeg и опрашивать сервер
$StartupGraceSec  = 30    # не проверять зависание первые N секунд (ffmpeg только подключается)
$StallTimeoutSec  = 20    # если кадры не идут дольше этого - считаем что зависло
$LogPath          = Join-Path $PSScriptRoot "rtsp_pusher_agent.log"
$StderrTempPath   = Join-Path $PSScriptRoot "ffmpeg_stderr.tmp.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogPath -Value $line
}

$lastSeenRestartTs = 0.0

function Test-RemoteRestartRequested {
    try {
        $resp = Invoke-RestMethod -Uri $RestartSignalUrl -TimeoutSec 5
        if ($resp.requested_at -gt $script:lastSeenRestartTs) {
            $script:lastSeenRestartTs = $resp.requested_at
            return $true
        }
    } catch {
        # Не удалось достучаться до сервера - не страшно, попробуем на следующем цикле
    }
    return $false
}

Write-Log "=== RTSP pusher agent started ==="

while ($true) {
    Write-Log "Starting ffmpeg push -> $PublishUrl"

    $ffmpegArgs = @(
        "-rtsp_transport", "tcp",
        "-probesize", "100M", "-analyzeduration", "100M",
        "-fflags", "+nobuffer+igndts",
        "-err_detect", "ignore_err",
        "-i", $CamUrl,
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        "-muxdelay", "0.1",
        $PublishUrl
    )

    Remove-Item $StderrTempPath -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath "ffmpeg" -ArgumentList $ffmpegArgs `
        -NoNewWindow -PassThru -RedirectStandardError $StderrTempPath
    # Без обращения к .Handle сразу после старта $proc.ExitCode потом не читается (баг PowerShell/.NET)
    $procHandle = $proc.Handle

    $procStartTime  = Get-Date
    $lastFrameTime  = Get-Date
    $lastFrameCount = -1

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds $PollIntervalSec

        if (Test-RemoteRestartRequested) {
            Write-Log "Restart requested from website - stopping ffmpeg"
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            break
        }

        if (Test-Path $StderrTempPath) {
            $tail = Get-Content $StderrTempPath -Tail 5 -ErrorAction SilentlyContinue
            $frameLine = $tail | Where-Object { $_ -match "frame=\s*(\d+)" } | Select-Object -Last 1
            if ($frameLine -match "frame=\s*(\d+)") {
                $frameCount = [int]$Matches[1]
                if ($frameCount -ne $lastFrameCount) {
                    $lastFrameCount = $frameCount
                    $lastFrameTime = Get-Date
                }
            }
        }

        if (((Get-Date) - $procStartTime).TotalSeconds -gt $StartupGraceSec) {
            $stalledFor = ((Get-Date) - $lastFrameTime).TotalSeconds
            if ($stalledFor -gt $StallTimeoutSec) {
                Write-Log "No frame progress for $([int]$stalledFor)s - stream frozen, restarting"
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                break
            }
        }
    }

    if ($proc.HasExited) {
        Write-Log "ffmpeg exited with code $($proc.ExitCode)"
        $errorTail = Get-Content $StderrTempPath -Tail 12 -ErrorAction SilentlyContinue
        if ($errorTail) {
            Write-Log "---- ffmpeg last output ----"
            $errorTail | ForEach-Object { Write-Log $_ }
            Write-Log "-----------------------------"
        }
    }

    Write-Log "Restarting in 5s..."
    Start-Sleep -Seconds 5
}
