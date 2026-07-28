$ErrorActionPreference = "Stop"

$targetPorts = @(8765, 8766)
$serverProcessNames = @("python", "pythonw")
$dialogTitle = "Stop Host Server"

Add-Type -AssemblyName PresentationFramework

try {
    $portPattern = $targetPorts -join "|"
    $listenerPattern = (
        "^\s*TCP\s+\S+:(?:{0})\s+\S+\s+LISTENING\s+(\d+)\s*$" `
        -f $portPattern
    )
    $listenerProcessIds = @(
        netstat -ano -p TCP |
            ForEach-Object {
                if ($_ -match $listenerPattern) {
                    [int]$Matches[1]
                }
            } |
            Sort-Object -Unique
    )

    if ($listenerProcessIds.Count -eq 0) {
        [System.Windows.MessageBox]::Show(
            "The host server is not running.",
            $dialogTitle,
            "OK",
            "Information"
        ) | Out-Null
        exit 0
    }

    $stopped = @()
    $skipped = @()

    foreach ($processId in $listenerProcessIds) {
        $process = Get-Process -Id $processId -ErrorAction Stop
        if ($process.ProcessName -notin $serverProcessNames) {
            $skipped += "$($process.ProcessName) (PID $processId)"
            continue
        }

        Stop-Process -Id $processId -ErrorAction Stop
        $stopped += "$($process.ProcessName) (PID $processId)"
    }

    if ($skipped.Count -gt 0) {
        $message = (
            "For safety, these non-Python listeners were not stopped:`n`n" +
            ($skipped -join "`n")
        )
        [System.Windows.MessageBox]::Show(
            $message,
            $dialogTitle,
            "OK",
            "Warning"
        ) | Out-Null
        exit 1
    }

    [System.Windows.MessageBox]::Show(
        "The host server was stopped successfully.`n`n$($stopped -join "`n")",
        $dialogTitle,
        "OK",
        "Information"
    ) | Out-Null
}
catch {
    [System.Windows.MessageBox]::Show(
        "The host server could not be stopped:`n`n$($_.Exception.Message)",
        $dialogTitle,
        "OK",
        "Error"
    ) | Out-Null
    exit 1
}
