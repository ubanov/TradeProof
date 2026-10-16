<#
.SYNOPSIS
    Resident, visible console loop that runs `tradingagents archive-evidence`
    on a fixed America/New_York schedule, for debugging EvidenceArchive.

.DESCRIPTION
    This is a plain PowerShell script meant to be left running in a visible
    console window -- NOT a Windows service, NOT a Scheduled Task, NOT a
    hidden/background daemon. It never implements any ticker-list parsing,
    capture logic, or provider logic itself: every actual capture is done by
    invoking the real, already-implemented `tradingagents archive-evidence`
    CLI as an external process and waiting for it to exit.

    Profiles (see $Profiles below):
      MASSIVE       config/evidence-universe.txt, news=yfinance, social=stocktwits
                    09:00 / 13:00 / 16:15 ET
      REDDIT        config/evidence-reddit.txt, social only, provider=reddit
                    16:30 ET
      ALPHA_VANTAGE prepared but Enabled=$false by default (small free quota --
                    opt in explicitly by flipping Enabled and setting real times)

    Overlap / "missed slot" policy (deliberately simple and deterministic):
    runs are always sequential -- the loop never starts a new capture while a
    previous one is still running. After every run finishes (or when the loop
    wakes up with nothing due), the next action is always recomputed as the
    SOONEST schedule time that is still in the future AT THAT MOMENT, across
    every enabled profile. Any slot whose time already passed while another
    profile's run was in flight is simply skipped -- there is no catch-up
    queue and a missed slot is never run late. This falls out naturally from
    always re-deriving "next slot" from the live clock rather than from a
    precomputed calendar.

    Timezone: resolved via .NET's TimeZoneInfo against "America/New_York"
    (falling back to the Windows ID "Eastern Standard Time" if the IANA ID
    isn't resolvable on a given host), so EST/EDT (DST) is handled correctly
    by the OS/.NET timezone database -- no UTC offset or local-time
    assumption is hardcoded anywhere in this script.

.PARAMETER DryRun
    Show the resolved repo paths, per-profile logical commands, and the
    next-24h schedule in Eastern Time, then exit immediately. Never invokes
    the CLI.

.PARAMETER RunNow
    Run the MASSIVE profile once, immediately, synchronously -- then fall
    through into the normal scheduled loop.

.PARAMETER RunNowAll
    Run every ENABLED profile once, immediately, in order (MASSIVE, then
    REDDIT, then any other enabled profile) -- then fall through into the
    normal scheduled loop.

.PARAMETER FunctionsOnly
    Internal/testing hook: define this script's functions and return without
    running anything else (no DryRun output, no CLI resolution, no loop).
    Used to dot-source just the scheduling math for a quick manual check,
    e.g.:
        . .\scripts\archive-evidence-loop.ps1 -FunctionsOnly
        Get-NextOccurrenceEt -TimesEt @('09:00','13:00','16:15') -NowEt (Get-Date '2026-10-16 08:00')

.EXAMPLE
    # Check the schedule and resolved paths without running anything
    .\scripts\archive-evidence-loop.ps1 -DryRun

.EXAMPLE
    # Run MASSIVE immediately, then keep looping on the normal schedule
    .\scripts\archive-evidence-loop.ps1 -RunNow

.EXAMPLE
    # Just follow the normal schedule (09:00 / 13:00 / 16:15 / 16:30 ET)
    .\scripts\archive-evidence-loop.ps1

    Stop with Ctrl+C at any time -- the console Ctrl+C signal is delivered to
    this script and to any in-flight capture process together (they share
    the same console); the script's `finally` block logs a clean shutdown
    line before exiting.
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$RunNow,
    [switch]$RunNowAll,
    [switch]$FunctionsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths, resolved relative to the SCRIPT's own location, never the
# current working directory (the shell may be launched from anywhere). -----

$RepoRoot = Split-Path -Parent $PSScriptRoot

$EvidenceRoot = if ($env:TRADINGAGENTS_EVIDENCE_ROOT) {
    $env:TRADINGAGENTS_EVIDENCE_ROOT
} else {
    Join-Path $HOME '.tradingagents\evidence'
}
$LogDir = Join-Path $EvidenceRoot 'logs'

# How often the "waiting" status line refreshes while idle. Within the
# requested 30-60s band -- never a once-a-second line, never an hours-long
# single Sleep (so the loop can still react promptly to Ctrl+C and never
# needs duration arithmetic across a possible DST boundary).
$PollSeconds = 45

# --- Eastern Time resolution (DST-correct via .NET's own tz database) -----

function Get-EasternTimeZone {
    try {
        return [System.TimeZoneInfo]::FindSystemTimeZoneById('America/New_York')
    } catch {
        # Older/locked-down Windows hosts without ICU-based IANA lookup:
        # fall back to the equivalent Windows timezone ID. Same real zone,
        # same DST rules -- not a different, hardcoded offset.
        return [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    }
}

$EasternZone = Get-EasternTimeZone

function Get-EasternNow {
    [System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $EasternZone)
}

# --- Scheduling math (the only thing this script's own tests exercise) ----

function Get-NextOccurrenceEt {
    <#
    Smallest time-of-day in $TimesEt (each "HH:mm", Eastern local) that is
    strictly after $NowEt. If every one of today's times has already
    passed, returns the first (earliest) time on the FOLLOWING day. Pure
    wall-clock arithmetic, and the wait loop below compares this directly
    against a fresh Get-EasternNow on every tick -- both values come from
    the SAME Eastern-local frame, so the comparison never round-trips
    through UTC or the local machine's own timezone at all (a round trip
    through ConvertTimeToUtc + the local machine's clock used to live here;
    removed as unnecessary complexity, though it was NOT the cause of the
    real-run ~1h countdown jumps -- see the countdown formatting fix below
    for the actual root cause, a PowerShell [int] cast that rounds instead
    of truncating).
    #>
    param(
        [Parameter(Mandatory)][string[]]$TimesEt,
        [Parameter(Mandatory)][datetime]$NowEt
    )
    # @(...) forces array semantics even for a single-element $TimesEt (e.g.
    # REDDIT's lone '16:30') -- PowerShell otherwise unwraps a one-item
    # pipeline result (Sort-Object here) into a bare scalar, which would
    # make $sorted[0] below index the STRING's first CHARACTER instead of
    # its one element, and crash the "roll over to tomorrow" branch.
    $sorted = @($TimesEt | Sort-Object)
    $today = $NowEt.Date
    foreach ($t in $sorted) {
        $parts = $t -split ':'
        $candidate = $today.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
        if ($candidate -gt $NowEt) { return $candidate }
    }
    $tomorrow = $today.AddDays(1)
    $firstParts = $sorted[0] -split ':'
    return $tomorrow.AddHours([int]$firstParts[0]).AddMinutes([int]$firstParts[1])
}

# --- Profiles ---------------------------------------------------------------
# BuildArgs is a scriptblock taking the resolved universe-file path and
# returning the exact argv for the REAL CLI -- never a reimplementation of
# what those arguments do.

$Profiles = [ordered]@{
    MASSIVE = @{
        Enabled      = $true
        TimesEt      = @('09:00', '13:00', '16:15')
        UniverseFile = Join-Path $RepoRoot 'config\evidence-universe.txt'
        BuildArgs    = { param($File) @(
            'archive-evidence', '--tickers-file', $File,
            '--news-providers', 'yfinance', '--social-providers', 'stocktwits'
        ) }
    }
    REDDIT = @{
        Enabled      = $true
        TimesEt      = @('16:30')
        UniverseFile = Join-Path $RepoRoot 'config\evidence-reddit.txt'
        BuildArgs    = { param($File) @(
            'archive-evidence', '--tickers-file', $File,
            '--types', 'social', '--social-providers', 'reddit'
        ) }
    }
    ALPHA_VANTAGE = @{
        # Disabled by default: Alpha Vantage's free-tier quota is too small
        # for routine, scheduled, multi-ticker capture. To opt in: flip
        # Enabled to $true, pick real TimesEt, and make sure
        # ALPHA_VANTAGE_API_KEY is configured for the process this script
        # launches the CLI from.
        Enabled      = $false
        TimesEt      = @('12:00')
        UniverseFile = Join-Path $RepoRoot 'config\evidence-universe.txt'
        BuildArgs    = { param($File) @(
            'archive-evidence', '--tickers-file', $File,
            '--types', 'news', '--news-providers', 'alpha_vantage'
        ) }
    }
}

# --- Logging -----------------------------------------------------------------

function Get-LogPath {
    $dateStr = (Get-EasternNow).ToString('yyyy-MM-dd')
    Join-Path $LogDir "archive-loop-$dateStr.log"
}

function Write-LoopLog {
    <#
    Append-only, structured, ET-timestamped log line. Never passed any
    environment variable or secret -- only this script's own composed
    fields (profile name, the logical argv, exit code, duration).
    #>
    param([Parameter(Mandatory)][string]$Message)
    if (-not (Test-Path -LiteralPath $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    $stamp = (Get-EasternNow).ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -LiteralPath (Get-LogPath) -Value "[$stamp ET] $Message" -Encoding utf8
}

# --- CLI resolution ----------------------------------------------------------

function Resolve-TradingAgentsCli {
    <#
    Prefers the repo's own virtualenv console-script (works regardless of
    the launching shell's CWD or whether the venv is "activated"), then a
    `tradingagents` already on PATH, then the venv's python run as
    `-m cli.main`, then a bare `python` on PATH. Returns $null, never
    throws, if none of these exist -- the caller decides how to fail.
    #>
    $venvExe = Join-Path $RepoRoot '.venv\Scripts\tradingagents.exe'
    if (Test-Path -LiteralPath $venvExe) {
        return @{ Mode = 'exe'; Path = $venvExe }
    }
    $onPath = Get-Command 'tradingagents' -ErrorAction SilentlyContinue
    if ($onPath) {
        return @{ Mode = 'exe'; Path = $onPath.Source }
    }
    $venvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) {
        return @{ Mode = 'module'; Path = $venvPython }
    }
    $pythonOnPath = Get-Command 'python' -ErrorAction SilentlyContinue
    if ($pythonOnPath) {
        return @{ Mode = 'module'; Path = $pythonOnPath.Source }
    }
    return $null
}

function Invoke-ArchiveEvidenceCli {
    <#
    Runs the real CLI synchronously, with -NoNewWindow so its own per-
    ticker/provider output streams live into THIS console (the whole point
    of a "visible" loop) -- never captured/re-printed by this script, so it
    can never be a place secrets would leak through our own code. Returns
    the child process's exit code.
    #>
    param([Parameter(Mandatory)][string[]]$Arguments)
    if ($null -eq $Script:CliInfo) {
        throw 'tradingagents CLI could not be resolved (see the startup error above).'
    }
    if ($Script:CliInfo.Mode -eq 'exe') {
        $proc = Start-Process -FilePath $Script:CliInfo.Path -ArgumentList $Arguments `
            -NoNewWindow -Wait -PassThru -WorkingDirectory $RepoRoot
    } else {
        $fullArgs = @('-m', 'cli.main') + $Arguments
        $proc = Start-Process -FilePath $Script:CliInfo.Path -ArgumentList $fullArgs `
            -NoNewWindow -Wait -PassThru -WorkingDirectory $RepoRoot
    }
    return $proc.ExitCode
}

function Invoke-Profile {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][hashtable]$Profile
    )
    $universeFile = $Profile.UniverseFile
    $argv = & $Profile.BuildArgs $universeFile
    $logicalCommand = 'tradingagents ' + ($argv -join ' ')

    if (-not (Test-Path -LiteralPath $universeFile)) {
        # A thin EXISTENCE check only -- not ticker-list parsing/validation
        # (the CLI itself owns that). This just avoids spinning up a
        # process we already know cannot do anything useful, and gives an
        # immediate, readable message instead of waiting for the CLI's own
        # (also clear, but slower-to-appear) error.
        $msg = "Universe file not found: $universeFile"
        Write-Host "[$((Get-EasternNow).ToString('HH:mm:ss')) ET] $Name SKIPPED: $msg" -ForegroundColor Yellow
        Write-LoopLog "SKIPPED profile=$Name reason=`"$msg`""
        return
    }

    $startEt = Get-EasternNow
    $startedAtUtc = Get-Date
    Write-Host "[$($startEt.ToString('HH:mm:ss')) ET] Iniciando $Name..." -ForegroundColor Cyan
    Write-Host "  comando: $logicalCommand"
    Write-LoopLog "START profile=$Name command=`"$logicalCommand`""

    $exitCode = $null
    try {
        $exitCode = Invoke-ArchiveEvidenceCli -Arguments $argv
    } catch {
        Write-Host "  EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
        Write-LoopLog "EXCEPTION profile=$Name message=`"$($_.Exception.Message)`""
        $exitCode = -1
    }

    $endEt = Get-EasternNow
    $duration = (Get-Date) - $startedAtUtc
    # NOT [int]$duration.TotalHours: PowerShell's [int] cast on a double
    # ROUNDS to nearest (Convert.ToInt32 semantics), not truncates -- for a
    # fractional-hours value like 11.9933 (11h 59m 36s) that rounds UP to
    # 12, while .Minutes/.Seconds still correctly read 59:36, producing a
    # mismatched, too-high hours digit. Days*24 + Hours is the correct,
    # rounding-free decomposition (see the identical fix below for the
    # countdown display -- this was the real root cause of the ~1h-jump bug).
    $durationStr = [string]::Format('{0:D2}:{1:D2}:{2:D2}', ($duration.Days * 24 + $duration.Hours), $duration.Minutes, $duration.Seconds)
    $statusWord = if ($exitCode -eq 0) { 'OK' } else { 'FAILED' }
    $color = if ($exitCode -eq 0) { 'Green' } else { 'Red' }
    Write-Host "[$($endEt.ToString('HH:mm:ss')) ET] $Name terminado (exit=$exitCode, $durationStr)" -ForegroundColor $color
    Write-LoopLog "END profile=$Name exit=$exitCode duration=$durationStr status=$statusWord"
}

# --- Testing hook: define functions, do nothing else ------------------------

if ($FunctionsOnly) {
    return
}

# =============================================================================
# Main
# =============================================================================

Write-Host "EvidenceArchive loop -- repo root: $RepoRoot"
Write-Host "Eastern Time zone resolved: $($EasternZone.Id) ($($EasternZone.DisplayName))"
Write-Host "Log directory: $LogDir"
Write-Host ''

$Script:CliInfo = Resolve-TradingAgentsCli

if ($DryRun) {
    Write-Host '=== DRY RUN: planning only, nothing will be executed ==='
    Write-Host ''
    foreach ($name in $Profiles.Keys) {
        $p = $Profiles[$name]
        $exists = if (Test-Path -LiteralPath $p.UniverseFile) { 'exists' } else { 'MISSING' }
        $argv = & $p.BuildArgs $p.UniverseFile
        Write-Host "[$name] enabled=$($p.Enabled)"
        Write-Host "    universe file : $($p.UniverseFile) ($exists)"
        Write-Host "    times ET      : $($p.TimesEt -join ', ')"
        Write-Host "    command       : tradingagents $($argv -join ' ')"
        Write-Host ''
    }

    Write-Host '=== Next 24h schedule (ET) ==='
    $nowEt = Get-EasternNow
    $horizon = $nowEt.AddHours(24)
    $events = [System.Collections.Generic.List[object]]::new()
    foreach ($name in $Profiles.Keys) {
        $p = $Profiles[$name]
        if (-not $p.Enabled) { continue }
        $cursor = $nowEt
        while ($true) {
            $next = Get-NextOccurrenceEt -TimesEt $p.TimesEt -NowEt $cursor
            if ($next -gt $horizon) { break }
            $events.Add([PSCustomObject]@{ When = $next; Profile = $name })
            $cursor = $next
        }
    }
    if ($events.Count -eq 0) {
        Write-Host '  (no enabled profile has a slot in the next 24h)'
    } else {
        foreach ($evt in ($events | Sort-Object When)) {
            Write-Host ("  {0}  {1}" -f $evt.When.ToString('yyyy-MM-dd HH:mm'), $evt.Profile)
        }
    }

    Write-Host ''
    $cliText = if ($Script:CliInfo) { "$($Script:CliInfo.Mode) -> $($Script:CliInfo.Path)" } else { 'NOT FOUND' }
    Write-Host "CLI resolution: $cliText"
    exit 0
}

if (-not $Script:CliInfo) {
    Write-Host ''
    Write-Host 'ERROR: could not find the tradingagents CLI.' -ForegroundColor Red
    Write-Host "  Looked for: $(Join-Path $RepoRoot '.venv\Scripts\tradingagents.exe')"
    Write-Host '  Looked for: tradingagents on PATH'
    Write-Host "  Looked for: $(Join-Path $RepoRoot '.venv\Scripts\python.exe')"
    Write-Host '  Looked for: python on PATH'
    Write-Host '  Fix: create/activate the project virtualenv (pip install -e .) or add tradingagents/python to PATH.'
    exit 1
}

try {
    if ($RunNowAll) {
        foreach ($name in $Profiles.Keys) {
            if ($Profiles[$name].Enabled) {
                Invoke-Profile -Name $name -Profile $Profiles[$name]
            }
        }
    } elseif ($RunNow) {
        Invoke-Profile -Name 'MASSIVE' -Profile $Profiles['MASSIVE']
    }

    $Script:LastStatusLineLength = 0

    while ($true) {
        $nowEt = Get-EasternNow
        $candidates = foreach ($name in $Profiles.Keys) {
            $p = $Profiles[$name]
            if (-not $p.Enabled) { continue }
            [PSCustomObject]@{ Name = $name; NextEt = (Get-NextOccurrenceEt -TimesEt $p.TimesEt -NowEt $nowEt) }
        }
        if (-not $candidates) {
            Write-Host ''
            Write-Host 'No enabled profile has a schedule -- nothing to wait for. Exiting.' -ForegroundColor Yellow
            break
        }
        $chosen = $candidates | Sort-Object NextEt | Select-Object -First 1

        if ($nowEt -ge $chosen.NextEt) {
            # The chosen slot has already arrived -- end the overwritten
            # status line cleanly before Invoke-Profile prints its own,
            # ordinary (non-overwritten) lines.
            Write-Host ''
            Invoke-Profile -Name $chosen.Name -Profile $Profiles[$chosen.Name]
            # Loop back: the next iteration recomputes everything fresh
            # against the live clock, which is exactly what makes the "skip
            # anything that passed during the run, never catch up" policy
            # work with no extra bookkeeping.
            continue
        }

        # Single self-overwriting status line (requested over one new line
        # per tick): both sides of this countdown -- $nowEt and
        # $chosen.NextEt -- come from the SAME Eastern-local frame
        # (Get-EasternNow / Get-NextOccurrenceEt), so the subtraction below
        # never round-trips through UTC or the local machine's own
        # timezone, unlike the comparison this replaced.
        $remaining = $chosen.NextEt - $nowEt
        # NOT [int]$remaining.TotalHours -- see Invoke-Profile's identical
        # fix above for why that rounds instead of truncating and silently
        # produces an hours digit inconsistent with .Minutes/.Seconds. This
        # rounding bug, not timezone/UTC handling, was the real cause of the
        # real-run ~1h countdown jumps (confirmed: Get-NextOccurrenceEt
        # itself was correct in every case tested; only this cast was wrong).
        $remainingStr = [string]::Format('{0:D2}:{1:D2}:{2:D2}', ($remaining.Days * 24 + $remaining.Hours), $remaining.Minutes, $remaining.Seconds)
        $line = "[{0} ET] Esperando. Proxima ejecucion {1} a las {2} ET ({3})" -f `
            $nowEt.ToString('HH:mm:ss'), $chosen.Name, $chosen.NextEt.ToString('HH:mm'), $remainingStr
        $pad = [Math]::Max(0, $Script:LastStatusLineLength - $line.Length)
        Write-Host -NoNewline ("`r" + $line + (' ' * $pad))
        $Script:LastStatusLineLength = $line.Length
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    Write-Host ''
    Write-Host 'EvidenceArchive loop stopped.' -ForegroundColor Yellow
    Write-LoopLog 'LOOP_STOPPED (Ctrl+C or script exit)'
}
