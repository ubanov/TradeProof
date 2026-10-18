<#
.SYNOPSIS
    Resident, visible console orchestrator for EvidenceArchive capture AND
    the nightly TradeProof analysis batch, on a fixed America/New_York
    schedule.

.DESCRIPTION
    This is a plain PowerShell script meant to be left running in a visible
    console window -- NOT a Windows service, NOT a Scheduled Task, NOT a
    hidden/background daemon. It never implements any ticker-list parsing,
    capture logic, provider logic, or analysis logic itself: every actual
    piece of work is done by invoking the real, already-implemented CLIs
    (`tradingagents archive-evidence`, `tradingagents run`) as external
    processes and waiting for them to exit. Ticker-file parsing is done by
    shelling out to the project's own `cli.headless.parse_tickers` (see
    Get-AnalysisTickers below) rather than re-parsing the file in PowerShell.

    Jobs (see $Profiles below):
      MASSIVE       archive-evidence, config/evidence-universe.txt,
                    news=yfinance, social=stocktwits -- 09:00 / 13:00 / 16:15 ET
      REDDIT        archive-evidence, config/evidence-reddit.txt,
                    social only, provider=reddit -- 16:30 ET
      ALPHA_VANTAGE archive-evidence, prepared but Enabled=$false by default
                    (small free quota -- opt in explicitly)
      ANALYSIS      `tradingagents run`, once per ticker in
                    config/analysis-universe.txt, sequentially -- 17:00 ET

    Overlap / "missed slot" policy (deliberately simple and deterministic):
    jobs are always sequential -- the loop never starts a new job while a
    previous one (or one of ANALYSIS's own per-ticker runs) is still
    running. After every job finishes (or when the loop wakes up with
    nothing due), the next action is always recomputed as the SOONEST
    schedule time that is still in the future AT THAT MOMENT, across every
    enabled job. Any slot whose time already passed while another job was
    in flight is simply skipped -- there is no catch-up queue and a missed
    slot is never run late. This falls out naturally from always
    re-deriving "next slot" from the live clock rather than from a
    precomputed calendar -- and it is also what keeps ANALYSIS to "at most
    once per calendar day": once 17:00 ET has passed, the next computed
    ANALYSIS slot is tomorrow's 17:00, with no separate bookkeeping needed.

    ANALYSIS duplicate-avoidance (never "the directory exists" alone -- see
    Test-AnalysisAlreadyDone): before launching ticker/date, this checks (1)
    a local, auditable marker file (plain JSONL under the log directory)
    recording every CONFIRMED-successful analysis this orchestrator itself
    ran, cross-verified against that marker's own recorded ledger path
    still existing, then (2) falls back to scanning the REAL
    RecommendationRecord ledger (~/.tradingagents/ledger/recommendations/
    <ticker>/*.json, or $env:TRADINGAGENTS_LEDGER_ROOT) for a record whose
    own `run.analysis_date` matches -- the ledger is the authoritative
    source of truth (a record is written there only after a real,
    structurally-complete success; a failed `tradingagents run` never
    writes one), and a ledger hit backfills the marker for next time.

    Timezone: resolved via .NET's TimeZoneInfo against "America/New_York"
    (falling back to the Windows ID "Eastern Standard Time" if the IANA ID
    isn't resolvable on a given host), so EST/EDT (DST) is handled correctly
    by the OS/.NET timezone database -- no UTC offset or local-time
    assumption is hardcoded anywhere in this script.

    Analysis date (see Get-AnalysisDate): this project has NO existing
    NYSE-holiday-aware trading-calendar utility to reuse (confirmed by
    reading tradingagents/graph/trading_graph.py::_validate_trade_date,
    which only checks YYYY-MM-DD format and "not in the future" -- nothing
    about weekdays or holidays). This first version approximates "the last
    reasonable US market day" as: Monday-Friday -> today's ET date;
    Saturday/Sunday -> the preceding Friday. This does NOT yet account for
    NYSE holidays (Thanksgiving, Christmas, etc. still resolve to their own
    weekday date) -- a documented, accepted limitation, not a silent gap.
    No external trading-calendar package was added for this, per the
    explicit "no large external dependency just for this" constraint.

.PARAMETER DryRun
    Show the resolved repo paths, per-job logical commands (including the
    resolved ANALYSIS ticker list and analysis date), and the next-24h
    schedule in Eastern Time, then exit immediately. Never invokes a CLI.

.PARAMETER RunNow
    Kept for backward compatibility with the EvidenceArchive-only version of
    this script: equivalent to -RunNowEvidence (runs the MASSIVE job once,
    immediately). Prefer -RunNowEvidence / -RunNowAnalysis for clarity.

.PARAMETER RunNowEvidence
    Run the MASSIVE job once, immediately, synchronously -- then fall
    through into the normal scheduled loop.

.PARAMETER RunNowAnalysis
    Run the ANALYSIS job once, immediately, synchronously (every ticker in
    config/analysis-universe.txt not already analyzed for today's resolved
    analysis date) -- then fall through into the normal scheduled loop.

.PARAMETER RunNowAll
    Run every ENABLED job once, immediately, in order (MASSIVE, REDDIT,
    ALPHA_VANTAGE if enabled, ANALYSIS) -- then fall through into the
    normal scheduled loop.

.PARAMETER TraceLlm
    Forward --trace-llm to every `tradingagents run` call in the ANALYSIS
    job. OFF by default -- a normal daily run never traces (per-call traces
    are verbose and meant for debugging one run, not routine operation).

.PARAMETER FunctionsOnly
    Internal/testing hook: define this script's functions and return without
    running anything else (no DryRun output, no CLI resolution, no loop).
    Used to dot-source just the scheduling/date/dedup math for a quick
    manual check, e.g.:
        . .\scripts\archive-evidence-loop.ps1 -FunctionsOnly
        Get-NextOccurrenceEt -TimesEt @('09:00','13:00','16:15') -NowEt (Get-Date '2026-10-16 08:00')
        Get-AnalysisDate -NowEt (Get-Date '2026-10-17 08:00')   # a Saturday -> previous Friday

.EXAMPLE
    # Check the schedule, resolved paths, and ANALYSIS ticker list/date
    .\scripts\archive-evidence-loop.ps1 -DryRun

.EXAMPLE
    # Run the MASSIVE evidence round immediately, then keep looping
    .\scripts\archive-evidence-loop.ps1 -RunNowEvidence

.EXAMPLE
    # Run tonight's analysis batch immediately, then keep looping
    .\scripts\archive-evidence-loop.ps1 -RunNowAnalysis

.EXAMPLE
    # Just follow the normal schedule (09:00 / 13:00 / 16:15 / 16:30 / 17:00 ET)
    .\scripts\archive-evidence-loop.ps1

    Stop with Ctrl+C at any time -- the console Ctrl+C signal is delivered to
    this script and to any in-flight job process together (they share the
    same console); the script's `finally` block logs a clean shutdown line
    before exiting.
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$RunNow,
    [switch]$RunNowEvidence,
    [switch]$RunNowAnalysis,
    [switch]$RunNowAll,
    [switch]$TraceLlm,
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

# Same default convention as tradingagents.default_config's own
# `ledger_root` (TRADINGAGENTS_LEDGER_ROOT env var, else
# ~/.tradingagents/ledger) -- mirrored here only to LOCATE the ledger for
# the duplicate-avoidance check below, never to write to it or to
# reimplement anything about RecommendationRecord itself.
$LedgerRoot = if ($env:TRADINGAGENTS_LEDGER_ROOT) {
    $env:TRADINGAGENTS_LEDGER_ROOT
} else {
    Join-Path $HOME '.tradingagents\ledger'
}

# The local, auditable "already analyzed" marker (see Test-AnalysisAlreadyDone).
$AnalysisMarkerFile = Join-Path $LogDir 'analysis-completed.jsonl'

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
    through UTC or the local machine's own timezone at all.
    #>
    param(
        [Parameter(Mandatory)][string[]]$TimesEt,
        [Parameter(Mandatory)][datetime]$NowEt
    )
    # @(...) forces array semantics even for a single-element $TimesEt (e.g.
    # REDDIT's or ANALYSIS's lone slot) -- PowerShell otherwise unwraps a
    # one-item pipeline result (Sort-Object here) into a bare scalar, which
    # would make $sorted[0] below index the STRING's first CHARACTER instead
    # of its one element, and crash the "roll over to tomorrow" branch.
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

function Format-TimeSpanHms {
    <#
    "{H}:{mm}:{ss}" with NO rounding -- NOT [int]$ts.TotalHours, which
    PowerShell rounds to nearest (Convert.ToInt32 semantics) rather than
    truncates: a span like 11h59m36s has TotalHours=11.9933, which [int]
    rounds UP to 12 while .Minutes/.Seconds still correctly read 59:36,
    producing a too-high hours digit inconsistent with the rest (the real
    root cause of this script's own ~1h countdown-jump bug). Days*24+Hours
    is the correct, rounding-free decomposition.
    #>
    param([Parameter(Mandatory)][timespan]$TimeSpan)
    [string]::Format('{0:D2}:{1:D2}:{2:D2}', ($TimeSpan.Days * 24 + $TimeSpan.Hours), $TimeSpan.Minutes, $TimeSpan.Seconds)
}

function Get-AnalysisDate {
    <#
    Approximate "last reasonable US market day" for the nightly ANALYSIS
    job. Monday-Friday: today's ET date. Saturday/Sunday: the preceding
    Friday. Does NOT know about NYSE holidays -- see this script's own
    .DESCRIPTION for why (no existing utility to reuse, no new external
    dependency added just for this). A documented limitation, not a silent
    one: a holiday will still resolve to its own weekday date here.
    #>
    param([Parameter(Mandatory)][datetime]$NowEt)
    $daysBack = switch ($NowEt.DayOfWeek) {
        'Saturday' { 1 }
        'Sunday'   { 2 }
        default    { 0 }
    }
    return $NowEt.Date.AddDays(-$daysBack).ToString('yyyy-MM-dd')
}

# --- Jobs --------------------------------------------------------------------
# BuildArgs (Kind='Archive') is a scriptblock taking the resolved universe-
# file path and returning the exact argv for the REAL CLI -- never a
# reimplementation of what those arguments do. ANALYSIS (Kind='Analysis')
# has its own per-ticker loop in Invoke-AnalysisJob below instead, since
# `tradingagents run` takes exactly one --ticker (confirmed: it has no
# --tickers-file option at all -- only `batch` and `archive-evidence` do).

$Profiles = [ordered]@{
    MASSIVE = @{
        Kind         = 'Archive'
        Enabled      = $true
        TimesEt      = @('09:00', '13:00', '16:15')
        UniverseFile = Join-Path $RepoRoot 'config\evidence-universe.txt'
        BuildArgs    = { param($File) @(
            'archive-evidence', '--tickers-file', $File,
            '--news-providers', 'yfinance', '--social-providers', 'stocktwits'
        ) }
    }
    REDDIT = @{
        Kind         = 'Archive'
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
        Kind         = 'Archive'
        Enabled      = $false
        TimesEt      = @('12:00')
        UniverseFile = Join-Path $RepoRoot 'config\evidence-universe.txt'
        BuildArgs    = { param($File) @(
            'archive-evidence', '--tickers-file', $File,
            '--types', 'news', '--news-providers', 'alpha_vantage'
        ) }
    }
    ANALYSIS = @{
        Kind         = 'Analysis'
        Enabled      = $true
        TimesEt      = @('17:00')
        UniverseFile = Join-Path $RepoRoot 'config\analysis-universe.txt'
    }
}

# Fixed model/depth/language contract for the nightly ANALYSIS job. Not a
# config file by request scope -- change here if the models change.
$AnalysisRunDefaults = @{
    Analysts              = 'market,fundamentals,news,social'
    Depth                 = 'medium'
    Language              = 'Spanish'
    GenerationLlm         = 'deepseek-v4.1-flash:cloud'
    GenerationEffort      = 'high'
    ReviewLlm             = 'qwen3.8-flash-next:125b-a6b-q4_K_M'
    ReviewEffort          = 'medium'
    ValidatorLlm          = 'gemma4:26b'
}

# --- Logging -----------------------------------------------------------------

function Get-LogPath {
    $dateStr = (Get-EasternNow).ToString('yyyy-MM-dd')
    Join-Path $LogDir "orchestrator-$dateStr.log"
}

function Write-LoopLog {
    <#
    Append-only, structured, ET-timestamped log line. Never passed any
    environment variable or secret -- only this script's own composed
    fields (job name, ticker, analysis date, the logical argv, exit code,
    duration, the RecommendationRecord path when one was written).
    #>
    param([Parameter(Mandatory)][string]$Message)
    if (-not (Test-Path -LiteralPath $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    $stamp = (Get-EasternNow).ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -LiteralPath (Get-LogPath) -Value "[$stamp ET] $Message" -Encoding utf8
}

# --- CLI resolution & invocation ---------------------------------------------

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

function Resolve-PythonForModuleCalls {
    <#
    A bare python.exe usable for one-off `-c` calls (today: only
    Get-AnalysisTickers, to reuse cli.headless.parse_tickers without
    re-parsing the ticker file in PowerShell). When CliInfo.Mode is already
    'module' this IS that python. When CliInfo.Mode is 'exe', its sibling
    python.exe in the same venv's Scripts folder is used; falls back to a
    bare `python` on PATH. Returns $null, never throws, if none exist.
    #>
    if ($null -eq $Script:CliInfo) { return $null }
    if ($Script:CliInfo.Mode -eq 'module') { return $Script:CliInfo.Path }
    $sibling = Join-Path (Split-Path -Parent $Script:CliInfo.Path) 'python.exe'
    if (Test-Path -LiteralPath $sibling) { return $sibling }
    $pythonOnPath = Get-Command 'python' -ErrorAction SilentlyContinue
    if ($pythonOnPath) { return $pythonOnPath.Source }
    return $null
}

function Invoke-TradingAgentsCli {
    <#
    Runs the real CLI synchronously. Without -CaptureOutput, uses
    Start-Process -NoNewWindow so output streams live into THIS console
    with no intermediate buffering (the EvidenceArchive jobs' own
    behavior, unchanged). With -CaptureOutput (used by ANALYSIS, which
    needs to find the "Recommendation record: <path>" line the CLI prints),
    runs via the call operator piped through Tee-Object + Out-Host so the
    SAME live output still reaches the console while also landing in
    $Script:LastCliOutput. Returns the child process's exit code either way.

    Two real-run bugs fixed here, both specific to the -CaptureOutput path
    (Start-Process above was never affected by either):

    1. stdout is captured WITHOUT merging stderr (no `2>&1`): merging a
       native command's stderr into a PowerShell pipeline wraps each line
       as an ErrorRecord, and with $ErrorActionPreference='Stop' (set
       globally at the top of this script) that can abort the WHOLE
       pipeline over an ordinary warning the run would otherwise have
       continued past completely normally. Confirmed by a real run: a
       benign "no closing price yet, using the prior day's close" warning
       killed a ticker's analysis after a few seconds instead of letting
       it run to completion like the other tickers. stderr still reaches
       the console directly either way -- an unredirected child process's
       stderr goes straight to the parent's own console -- it is simply
       not captured into $Script:LastCliOutput, which only ever needs the
       CLI's own stdout (the "Recommendation record:" line is printed to
       stdout, never stderr).
    2. The pipeline's own output is piped to Out-Host, never left
       unconsumed: inside a PowerShell FUNCTION (unlike top-level script
       code), pipeline output nobody consumes becomes part of the
       function's own return value, silently concatenated with whatever is
       later `return`ed. Confirmed by a real run: $exitCode ended up being
       the ENTIRE multi-hour captured transcript with the real integer
       exit code buried at the very end, rendering "terminado
       exit=<thousands of characters of run output>" instead of a clean
       number. Out-Host forces live display (preserving "visible loop")
       while producing no further pipeline output of its own, so nothing
       leaks into $exitCode.
    #>
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$CaptureOutput
    )
    if ($null -eq $Script:CliInfo) {
        throw 'tradingagents CLI could not be resolved (see the startup error above).'
    }
    $exe = $Script:CliInfo.Path
    $fullArgs = if ($Script:CliInfo.Mode -eq 'exe') { $Arguments } else { @('-m', 'cli.main') + $Arguments }

    if (-not $CaptureOutput) {
        $proc = Start-Process -FilePath $exe -ArgumentList $fullArgs `
            -NoNewWindow -Wait -PassThru -WorkingDirectory $RepoRoot
        return $proc.ExitCode
    }

    $Script:LastCliOutput = @()
    $prevEap = $ErrorActionPreference
    # Defense in depth alongside fix 1 above: a native command's stderr
    # (even when not explicitly merged) can still occasionally surface as
    # a non-terminating error record under certain hosts/wrappers: this
    # guarantees it is never escalated to terminating for the duration of
    # THIS call only, restored immediately after.
    $ErrorActionPreference = 'Continue'
    Push-Location $RepoRoot
    try {
        & $exe @fullArgs | Tee-Object -Variable Script:LastCliOutput | Out-Host
        return $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEap
    }
}

function Get-AnalysisTickers {
    <#
    Reuses the project's OWN cli.headless.parse_tickers (comment/blank-line
    skip, dedupe, Yahoo-symbol normalization, path-safety validation) by
    shelling out to Python -- never a second, parallel ticker-file parser
    in PowerShell. Returns an array of tickers, or $null (with a clear
    console message) if the universe file is missing/empty/invalid, or if
    no Python interpreter could be resolved at all.

    Writes the Python one-off to a REAL temp .py file and runs
    `python <file> <universe>` rather than `python -c <code>`: passing a
    multi-line script (with embedded quotes) as a single `-c` argument
    goes through Windows' native command-line argument marshaling, which
    does not reliably preserve embedded quotes/newlines -- confirmed by a
    real failure (`python.exe : File "<string>", line 7` -- a syntax
    error from content that was never actually malformed, just mangled in
    transit). A real file sidesteps that whole class of problem.
    #>
    param([Parameter(Mandatory)][string]$UniverseFile)
    $python = Resolve-PythonForModuleCalls
    if (-not $python) {
        Write-Host 'ERROR: could not resolve a Python interpreter to parse the ticker file.' -ForegroundColor Red
        return $null
    }
    if (-not (Test-Path -LiteralPath $UniverseFile)) {
        return $null
    }
    $pyScript = @'
import sys
from pathlib import Path
from cli.headless import parse_tickers

try:
    tickers = parse_tickers(None, Path(sys.argv[1]))
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

print("\n".join(tickers))
'@
    $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) "parse_tickers_$([guid]::NewGuid().ToString('N')).py"
    Set-Content -LiteralPath $tempScript -Value $pyScript -Encoding utf8
    try {
        Push-Location $RepoRoot
        try {
            $output = & $python $tempScript $UniverseFile 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR parsing ${UniverseFile}: $output" -ForegroundColor Red
                return $null
            }
            return @($output | Where-Object { $_ -ne '' })
        } finally {
            Pop-Location
        }
    } finally {
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

# --- ANALYSIS duplicate-avoidance (never "directory exists" alone) ----------

function Add-AnalysisMarker {
    param(
        [Parameter(Mandatory)][string]$Ticker,
        [Parameter(Mandatory)][string]$AnalysisDate,
        [string]$LedgerRecordPath
    )
    if (-not (Test-Path -LiteralPath $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    $entry = [PSCustomObject]@{
        ticker             = $Ticker
        analysis_date      = $AnalysisDate
        ledger_record_path = $LedgerRecordPath
        recorded_at_et     = (Get-EasternNow).ToString('yyyy-MM-dd HH:mm:ss')
    }
    Add-Content -LiteralPath $AnalysisMarkerFile -Value ($entry | ConvertTo-Json -Compress)
}

function Test-AnalysisAlreadyDone {
    <#
    Explicit, auditable, two-layer policy -- NEVER "the run's own output
    directory exists" (a failed run still writes its own result.json with
    status=error into --output-dir, so that alone proves nothing):

    1. Fast path: this orchestrator's own local marker
       ($AnalysisMarkerFile, plain JSONL, human-readable) for an exact
       ticker+date match, cross-checked against that marker's OWN recorded
       ledger file still existing on disk (defends against a stale marker
       surviving a since-deleted ledger entry).
    2. Authoritative fallback: scan the REAL RecommendationRecord ledger
       (<ledger_root>/recommendations/<ticker>/*.json) for a record whose
       own `run.analysis_date` field matches -- a record is written there
       ONLY after a structurally-complete success (confirmed: a failed
       `tradingagents run` never writes one at all; it is a separate tree
       from the run's own --output-dir). A hit here backfills the marker.

    Returns $true (skip) or $false (run it).
    #>
    param(
        [Parameter(Mandatory)][string]$Ticker,
        [Parameter(Mandatory)][string]$AnalysisDate
    )
    if (Test-Path -LiteralPath $AnalysisMarkerFile) {
        foreach ($line in Get-Content -LiteralPath $AnalysisMarkerFile) {
            if (-not $line.Trim()) { continue }
            try { $entry = $line | ConvertFrom-Json } catch { continue }
            if ($entry.ticker -eq $Ticker -and $entry.analysis_date -eq $AnalysisDate) {
                if ($entry.ledger_record_path -and (Test-Path -LiteralPath $entry.ledger_record_path)) {
                    return $true
                }
            }
        }
    }

    $tickerDir = Join-Path (Join-Path $LedgerRoot 'recommendations') $Ticker
    if (Test-Path -LiteralPath $tickerDir) {
        foreach ($file in Get-ChildItem -LiteralPath $tickerDir -Filter '*.json' -File -ErrorAction SilentlyContinue) {
            try {
                $record = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
            } catch { continue }
            if ($record.run -and $record.run.analysis_date -eq $AnalysisDate) {
                Add-AnalysisMarker -Ticker $Ticker -AnalysisDate $AnalysisDate -LedgerRecordPath $file.FullName
                return $true
            }
        }
    }

    return $false
}

# --- ANALYSIS job: one `tradingagents run` call per ticker, sequential ------

function Invoke-AnalysisRunCommand {
    <#
    Thin wrapper around Invoke-TradingAgentsCli, isolated into its own
    function so a test can override/replace THIS function (plain
    PowerShell function redefinition after dot-sourcing -FunctionsOnly) to
    simulate a CLI success/failure without spawning a real process or
    calling an LLM.
    #>
    param([Parameter(Mandatory)][string[]]$Arguments)
    return Invoke-TradingAgentsCli -Arguments $Arguments -CaptureOutput
}

function Invoke-AnalysisJob {
    param([Parameter(Mandatory)][hashtable]$Profile)

    $analysisDate = Get-AnalysisDate -NowEt (Get-EasternNow)
    $universeFile = $Profile.UniverseFile

    if (-not (Test-Path -LiteralPath $universeFile)) {
        $msg = "Universe file not found: $universeFile"
        Write-Host "[$((Get-EasternNow).ToString('HH:mm:ss')) ET] ANALYSIS SKIPPED: $msg" -ForegroundColor Yellow
        Write-LoopLog "SKIPPED profile=ANALYSIS reason=`"$msg`""
        return
    }

    $tickers = Get-AnalysisTickers -UniverseFile $universeFile
    if (-not $tickers) {
        $msg = "no tickers resolved from $universeFile"
        Write-Host "[$((Get-EasternNow).ToString('HH:mm:ss')) ET] ANALYSIS SKIPPED: $msg" -ForegroundColor Yellow
        Write-LoopLog "SKIPPED profile=ANALYSIS reason=`"$msg`""
        return
    }

    foreach ($ticker in $tickers) {
        if (Test-AnalysisAlreadyDone -Ticker $ticker -AnalysisDate $analysisDate) {
            Write-Host "[$((Get-EasternNow).ToString('HH:mm:ss')) ET] ANALYSIS $ticker $analysisDate ya registrado -- omitido" -ForegroundColor Yellow
            Write-LoopLog "SKIPPED_DUPLICATE profile=ANALYSIS ticker=$ticker analysis_date=$analysisDate"
            continue
        }

        $argv = @(
            'run', '--ticker', $ticker, '--date', $analysisDate,
            '--analysts', $AnalysisRunDefaults.Analysts,
            '--depth', $AnalysisRunDefaults.Depth,
            '--language', $AnalysisRunDefaults.Language,
            '--generation-llm', $AnalysisRunDefaults.GenerationLlm,
            '--generation-reasoning-effort', $AnalysisRunDefaults.GenerationEffort,
            '--review-llm', $AnalysisRunDefaults.ReviewLlm,
            '--review-reasoning-effort', $AnalysisRunDefaults.ReviewEffort,
            '--validator-llm', $AnalysisRunDefaults.ValidatorLlm,
            '--validation'
        )
        if ($Script:TraceLlmEnabled) { $argv += '--trace-llm' }
        $logicalCommand = 'tradingagents ' + ($argv -join ' ')

        $startEt = Get-EasternNow
        $startedAtUtc = Get-Date
        Write-Host "[$($startEt.ToString('HH:mm:ss')) ET] ANALYSIS $ticker $analysisDate iniciado" -ForegroundColor Cyan
        Write-Host "  comando: $logicalCommand"
        Write-LoopLog "START profile=ANALYSIS ticker=$ticker analysis_date=$analysisDate command=`"$logicalCommand`""

        $exitCode = $null
        $recordPath = $null
        try {
            $exitCode = Invoke-AnalysisRunCommand -Arguments $argv
            $output = @($Script:LastCliOutput)
            $recordLine = $output | Where-Object { $_ -match 'Recommendation record:' } | Select-Object -Last 1
            if ($recordLine -and $recordLine -notmatch 'not written') {
                $recordPath = ($recordLine -split 'Recommendation record:', 2)[1].Trim()
                Add-AnalysisMarker -Ticker $ticker -AnalysisDate $analysisDate -LedgerRecordPath $recordPath
            }
        } catch {
            Write-Host "  EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
            Write-LoopLog "EXCEPTION profile=ANALYSIS ticker=$ticker message=`"$($_.Exception.Message)`""
            $exitCode = -1
        }

        $endEt = Get-EasternNow
        $duration = (Get-Date) - $startedAtUtc
        $durationStr = Format-TimeSpanHms -TimeSpan $duration
        $statusWord = if ($exitCode -eq 0) { 'OK' } else { 'FAILED' }
        $color = if ($exitCode -eq 0) { 'Green' } else { 'Red' }
        Write-Host "[$($endEt.ToString('HH:mm:ss')) ET] ANALYSIS $ticker terminado exit=$exitCode duration=$durationStr" -ForegroundColor $color
        Write-LoopLog "END profile=ANALYSIS ticker=$ticker analysis_date=$analysisDate exit=$exitCode duration=$durationStr status=$statusWord recommendation_record=`"$recordPath`""
        # A failing ticker never aborts the batch -- the foreach continues
        # to the next ticker unconditionally, regardless of $exitCode.
    }
}

# --- ARCHIVE job (MASSIVE / REDDIT / ALPHA_VANTAGE): unchanged behavior ----

function Invoke-ArchiveJob {
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
        $exitCode = Invoke-TradingAgentsCli -Arguments $argv
    } catch {
        Write-Host "  EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
        Write-LoopLog "EXCEPTION profile=$Name message=`"$($_.Exception.Message)`""
        $exitCode = -1
    }

    $endEt = Get-EasternNow
    $duration = (Get-Date) - $startedAtUtc
    $durationStr = Format-TimeSpanHms -TimeSpan $duration
    $statusWord = if ($exitCode -eq 0) { 'OK' } else { 'FAILED' }
    $color = if ($exitCode -eq 0) { 'Green' } else { 'Red' }
    Write-Host "[$($endEt.ToString('HH:mm:ss')) ET] $Name terminado (exit=$exitCode, $durationStr)" -ForegroundColor $color
    Write-LoopLog "END profile=$Name exit=$exitCode duration=$durationStr status=$statusWord"
}

function Invoke-Profile {
    <#
    Single dispatcher for every job kind -- the "one orchestrator" entry
    point the main loop and -RunNow*/-RunNowAll all call uniformly.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][hashtable]$Profile
    )
    if ($Profile.Kind -eq 'Analysis') {
        Invoke-AnalysisJob -Profile $Profile
    } else {
        Invoke-ArchiveJob -Name $Name -Profile $Profile
    }
}

# --- Testing hook: define functions, do nothing else ------------------------

if ($FunctionsOnly) {
    return
}

# =============================================================================
# Main
# =============================================================================

$Script:TraceLlmEnabled = [bool]$TraceLlm

Write-Host "TradeProof orchestrator -- repo root: $RepoRoot"
Write-Host "Eastern Time zone resolved: $($EasternZone.Id) ($($EasternZone.DisplayName))"
Write-Host "Log directory: $LogDir"
Write-Host "Ledger root (for ANALYSIS duplicate-avoidance): $LedgerRoot"
Write-Host ''

$Script:CliInfo = Resolve-TradingAgentsCli

if ($DryRun) {
    Write-Host '=== DRY RUN: planning only, nothing will be executed ==='
    Write-Host ''
    foreach ($name in $Profiles.Keys) {
        $p = $Profiles[$name]
        $exists = if (Test-Path -LiteralPath $p.UniverseFile) { 'exists' } else { 'MISSING' }
        Write-Host "[$name] enabled=$($p.Enabled) kind=$($p.Kind)"
        Write-Host "    universe file : $($p.UniverseFile) ($exists)"
        Write-Host "    times ET      : $($p.TimesEt -join ', ')"
        if ($p.Kind -eq 'Analysis') {
            $analysisDate = Get-AnalysisDate -NowEt (Get-EasternNow)
            Write-Host "    analysis date : $analysisDate (see .DESCRIPTION: weekday/weekend only, no NYSE holidays yet)"
            $tickers = if ($exists -eq 'exists') { Get-AnalysisTickers -UniverseFile $p.UniverseFile } else { $null }
            Write-Host "    tickers       : $(if ($tickers) { $tickers -join ', ' } else { '(none resolved)' })"
            Write-Host "    command (x N) : tradingagents run --ticker <TICKER> --date $analysisDate --analysts $($AnalysisRunDefaults.Analysts) --depth $($AnalysisRunDefaults.Depth) --language $($AnalysisRunDefaults.Language) --generation-llm $($AnalysisRunDefaults.GenerationLlm) --generation-reasoning-effort $($AnalysisRunDefaults.GenerationEffort) --review-llm $($AnalysisRunDefaults.ReviewLlm) --review-reasoning-effort $($AnalysisRunDefaults.ReviewEffort) --validator-llm $($AnalysisRunDefaults.ValidatorLlm) --validation$(if ($Script:TraceLlmEnabled) { ' --trace-llm' })"
        } else {
            $argv = & $p.BuildArgs $p.UniverseFile
            Write-Host "    command       : tradingagents $($argv -join ' ')"
        }
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
    } elseif ($RunNowEvidence) {
        Invoke-Profile -Name 'MASSIVE' -Profile $Profiles['MASSIVE']
    } elseif ($RunNowAnalysis) {
        Invoke-Profile -Name 'ANALYSIS' -Profile $Profiles['ANALYSIS']
    } elseif ($RunNow) {
        # Backward-compatible alias -- see .PARAMETER RunNow.
        Invoke-Profile -Name 'MASSIVE' -Profile $Profiles['MASSIVE']
    }

    $Script:LastStatusLineLength = 0

    # Generous slack over the normal poll cadence: real on-time arrivals
    # are only ever "late" by up to ~$PollSeconds (we only check this
    # often). Anything beyond this margin when we actually notice arrival
    # means the WAIT ITSELF stalled for a long stretch (Windows Console's
    # "QuickEdit Mode" pauses the ENTIRE process the instant someone
    # clicks/selects text in a legacy console window -- very easy to
    # trigger by accident -- or the machine slept) -- that slot is logged
    # as SLOT_SKIPPED and never run late, per "no catch-up".
    $GraceSeconds = $PollSeconds * 3

    while ($true) {
        # One "pass": pick the single nearest slot across every enabled
        # profile ONCE, then wait for exactly THAT fixed target -- never
        # recomputed while waiting. Recomputing it on every poll tick (an
        # earlier version of this script did) is a dead end: Get-
        # NextOccurrenceEt is BY CONSTRUCTION always strictly later than
        # whatever "now" it is given, so comparing a target just derived
        # from the current "now" against that SAME "now" can never show
        # arrival -- every single slot would silently (or, after the
        # SLOT_SKIPPED logging was added, LOUDLY and WRONGLY) report as
        # missed the moment its time passed, even by one second, because
        # the recomputed target had already jumped to the NEXT occurrence
        # before the comparison ever ran. Real-run regression confirmed
        # this: every scheduled slot showed SLOT SALTADO within one poll
        # tick of its own countdown reaching zero.
        $pickNowEt = Get-EasternNow
        $candidates = foreach ($name in $Profiles.Keys) {
            $p = $Profiles[$name]
            if (-not $p.Enabled) { continue }
            [PSCustomObject]@{ Name = $name; NextEt = (Get-NextOccurrenceEt -TimesEt $p.TimesEt -NowEt $pickNowEt) }
        }
        if (-not $candidates) {
            Write-Host ''
            Write-Host 'No enabled profile has a schedule -- nothing to wait for. Exiting.' -ForegroundColor Yellow
            break
        }
        $chosen = $candidates | Sort-Object NextEt | Select-Object -First 1

        while ((Get-EasternNow) -lt $chosen.NextEt) {
            # Single self-overwriting status line: both sides of this
            # countdown come from the SAME Eastern-local frame
            # (Get-EasternNow / Get-NextOccurrenceEt), so the subtraction
            # never round-trips through UTC or the local machine's own
            # timezone.
            $nowEt = Get-EasternNow
            $remaining = $chosen.NextEt - $nowEt
            $remainingStr = Format-TimeSpanHms -TimeSpan $remaining
            $line = "[{0} ET] Esperando. Proxima tarea {1} {2} ET ({3})" -f `
                $nowEt.ToString('HH:mm:ss'), $chosen.Name, $chosen.NextEt.ToString('HH:mm'), $remainingStr
            $pad = [Math]::Max(0, $Script:LastStatusLineLength - $line.Length)
            Write-Host -NoNewline ("`r" + $line + (' ' * $pad))
            $Script:LastStatusLineLength = $line.Length
            Start-Sleep -Seconds $PollSeconds
        }

        $arrivedAt = Get-EasternNow
        $lateness = ($arrivedAt - $chosen.NextEt).TotalSeconds
        Write-Host ''
        if ($lateness -gt $GraceSeconds) {
            Write-Host ("[{0} ET] {1} SLOT SALTADO: estaba previsto a las {2} ET y se detecto {3}s tarde -- no se ejecuta con retraso (revisa 'QuickEdit Mode' de la consola, o si el equipo se suspendio/durmio)" -f `
                $arrivedAt.ToString('HH:mm:ss'), $chosen.Name, $chosen.NextEt.ToString('HH:mm:ss'), [int]$lateness) -ForegroundColor Yellow
            Write-LoopLog "SLOT_SKIPPED profile=$($chosen.Name) scheduled=$($chosen.NextEt.ToString('yyyy-MM-dd HH:mm:ss')) detected_late_seconds=$([int]$lateness)"
        } else {
            Invoke-Profile -Name $chosen.Name -Profile $Profiles[$chosen.Name]
        }
        # Loop back: the next pass recomputes everything fresh against the
        # live clock, which is exactly what makes the "skip anything that
        # passed, never catch up" policy work with no extra bookkeeping
        # (ANALYSIS's own "once per calendar day" falls out of this too).
    }
} finally {
    Write-Host ''
    Write-Host 'TradeProof orchestrator stopped.' -ForegroundColor Yellow
    Write-LoopLog 'LOOP_STOPPED (Ctrl+C or script exit)'
}
