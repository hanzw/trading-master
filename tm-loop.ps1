# Trading Master Evolution Loop
# Usage: powershell -ExecutionPolicy Bypass -File "D:\trading-master\tm-loop.ps1"
#
# A disciplined iteration loop: each cycle runs a 3-agent judge panel,
# then executes the highest-priority fix/feature, tests, and commits.

param(
    [string]$Model = "claude-opus-4-6",
    [int]$MaxIterations = 999,
    [int]$MaxTurns = 25,
    [int]$CycleSeconds = 180          # 3 minutes between iterations
)

$ErrorActionPreference = "Continue"
$ProjectDir = "D:\trading-master"
Set-Location $ProjectDir

# --- Force UTF-8 ---
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# --- Logging ---
$LogDir = Join-Path $ProjectDir "logs\loop"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "tm_$timestamp.log"
[System.IO.File]::WriteAllText($LogFile, "", $Utf8NoBom)

function Log {
    param([string]$Msg, [string]$Color = "White")
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $Msg"
    Write-Host $line -ForegroundColor $Color
    try { [System.IO.File]::AppendAllText($LogFile, "$line`n", $Utf8NoBom) } catch {}
}

function LogSection {
    param([string]$Msg)
    Log ("=" * 60) "DarkGray"
    Log $Msg "Cyan"
    Log ("=" * 60) "DarkGray"
}

# --- The Prompt ---
# This is deliberately constrained: ONE concrete change per iteration,
# judged by 3 specialists, then implemented and tested.
$PROMPT = @"
=== TRADING MASTER EVOLUTION ENGINE ===

You are evolving a multi-agent portfolio management system.
Working directory: D:\trading-master

=== PHASE 1: ASSESS (read state) ===
1. Run: git log --oneline -5
2. Run: python -m pytest tests/ -q
3. Read pyproject.toml for current version
4. Skim src/trading_master/quant/__init__.py and any recent changes

=== PHASE 2: JUDGE PANEL (3 experts, each gives exactly 1 verdict) ===
Think as three independent experts. Each MUST identify a real, specific
weakness or gap — not vague suggestions. Cite file paths and line numbers.

Expert 1 — Portfolio Risk Manager:
  "What is the single biggest risk-management gap that could lose money?"
  Look at: sizing, correlation, CVaR, circuit breaker, stop loss, regime logic.

Expert 2 — Quant Researcher:
  "What is the most impactful quantitative model or analysis still missing?"
  Consider: Risk Parity, Sector Rotation, Options Sentiment, Multi-Timeframe
  Technical, Pairs Trading, Regime Switching (Markov), Tail Risk (EVT), etc.
  Check what quant/ already has vs what's missing.

Expert 3 — Software Engineer:
  "What is the most important code quality / reliability issue?"
  Look at: test coverage gaps, error handling, edge cases, performance,
  missing CLI integration for existing modules.

=== PHASE 3: EXECUTE (pick the top-priority item and implement it) ===
From the 3 verdicts, pick the ONE with highest impact-to-effort ratio.
Then implement it completely:
1. Write the code (new module or fix)
2. Write thorough tests (aim for 15+ test cases for new modules)
3. Integrate into CLI if applicable
4. Run: python -m pytest tests/ -q  (must pass)
5. Update version in pyproject.toml (bump patch)
6. git add the specific changed files (never git add -A)
7. git commit with descriptive message
8. git push

=== RULES ===
- Every iteration MUST produce exactly ONE git commit with ONE focused change
- Tests MUST pass before committing — if they fail, fix them
- Never skip tests, never force-push, never --no-verify
- Be honest: if a judge finds nothing real, say so — don't fabricate issues
- Prefer depth over breadth: one solid module > three half-baked ones
"@

# --- Main Loop ---
LogSection "TRADING MASTER EVOLUTION LOOP"
Log "Model: $Model | MaxTurns: $MaxTurns | Cycle: ${CycleSeconds}s | MaxIter: $MaxIterations" "Yellow"
Log "Log: $LogFile" "DarkGray"
Log "Started: $(Get-Date)" "Green"

$iteration = 0
$consecutiveFailures = 0
$totalCost = 0.0

while ($iteration -lt $MaxIterations) {
    $iteration++
    LogSection "ITERATION $iteration / $MaxIterations"

    # Git state before
    try {
        $lastCommit = git log --oneline -1 2>&1
        Log "Last commit: $lastCommit" "DarkGray"
    } catch {
        Log "Git log failed: $_" "DarkYellow"
    }

    $iterStart = Get-Date
    $exitCode = -1
    $resultText = ""
    $iterCost = 0.0
    $hadError = $false
    $toolUseCount = 0
    $subagentMessages = 0

    try {
        Log "Launching claude (model=$Model, max-turns=$MaxTurns)..." "Yellow"

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "claude"
        $psi.Arguments = "-p `"$($PROMPT -replace '"','\"')`" --output-format stream-json --verbose --model $Model --max-turns $MaxTurns --dangerously-skip-permissions"
        $psi.WorkingDirectory = $ProjectDir
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $psi

        $stderrBuilder = [System.Text.StringBuilder]::new()
        $stderrEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
            if ($EventArgs.Data) { $stderrBuilder.AppendLine($EventArgs.Data) | Out-Null }
        }

        $process.Start() | Out-Null
        $process.BeginErrorReadLine()

        $reader = $process.StandardOutput

        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($line)) { continue }

            try {
                $json = $line | ConvertFrom-Json

                switch ($json.type) {
                    "system" {
                        if ($json.subtype -eq "init") {
                            Log "  [INIT] session=$($json.session_id) model=$($json.model)" "DarkGray"
                        }
                    }
                    "assistant" {
                        $msg = $json.message
                        if ($msg -and $msg.content) {
                            foreach ($block in $msg.content) {
                                if ($block.type -eq "text" -and $block.text) {
                                    $preview = $block.text
                                    if ($preview.Length -gt 200) { $preview = $preview.Substring(0, 200) + "..." }
                                    Log "  [ASSISTANT] $preview" "White"
                                }
                                elseif ($block.type -eq "tool_use") {
                                    $toolUseCount++
                                    $toolName = $block.name
                                    $inputPreview = ""
                                    if ($block.input) {
                                        $inputStr = $block.input | ConvertTo-Json -Compress -Depth 2
                                        if ($inputStr.Length -gt 150) { $inputStr = $inputStr.Substring(0, 150) + "..." }
                                        $inputPreview = " | $inputStr"
                                    }
                                    Log "  [TOOL #$toolUseCount] $toolName$inputPreview" "Magenta"
                                }
                            }
                        }
                        if ($json.parent_tool_use_id) {
                            $subagentMessages++
                        }
                    }
                    "result" {
                        $resultText = $json.result
                        if ($json.total_cost_usd) { $iterCost = $json.total_cost_usd }
                        $duration = $json.duration_ms
                        $turns = $json.num_turns
                        $stopReason = $json.stop_reason
                        Log "  [RESULT] turns=$turns cost=`$$([math]::Round($iterCost, 4)) duration=$($duration)ms stop=$stopReason" "Green"
                        if ($json.is_error) {
                            $hadError = $true
                            Log "  [RESULT ERROR] $resultText" "Red"
                        }
                    }
                    "rate_limit_event" {
                        $rlStatus = $json.rate_limit_info.status
                        if ($rlStatus -ne "allowed") {
                            $resetsAt = $json.rate_limit_info.resetsAt
                            $resetTime = [DateTimeOffset]::FromUnixTimeSeconds($resetsAt).LocalDateTime
                            Log "  [RATE LIMIT] status=$rlStatus resets=$resetTime" "Red"
                        }
                    }
                }
            } catch {
                $preview = $line
                if ($preview.Length -gt 200) { $preview = $preview.Substring(0, 200) + "..." }
                Log "  [RAW] $preview" "DarkYellow"
            }
        }

        $process.WaitForExit()
        $exitCode = $process.ExitCode

        Unregister-Event -SourceIdentifier $stderrEvent.Name -ErrorAction SilentlyContinue
        Remove-Job -Job $stderrEvent -Force -ErrorAction SilentlyContinue

        $stderrText = $stderrBuilder.ToString()
        if ($stderrText.Trim()) {
            Log "  [STDERR] $($stderrText.Trim().Substring(0, [Math]::Min(300, $stderrText.Trim().Length)))" "DarkYellow"
        }

    } catch {
        $hadError = $true
        Log "PROCESS ERROR: $_" "Red"
        Log "Stack: $($_.ScriptStackTrace)" "DarkRed"
    }

    # --- Summary ---
    $elapsed = (Get-Date) - $iterStart
    $totalCost += $iterCost

    Log "" "White"
    Log "--- Iteration $iteration Summary ---" "Cyan"
    Log "  Exit=$exitCode Error=$hadError Tools=$toolUseCount Subagents=$subagentMessages" "Cyan"
    Log "  Time=$([math]::Round($elapsed.TotalSeconds))s Cost=`$$([math]::Round($iterCost, 4)) TotalCost=`$$([math]::Round($totalCost, 4))" "Cyan"

    try {
        $newCommit = git log --oneline -1 2>&1
        if ($newCommit -ne $lastCommit) {
            Log "  NEW COMMIT: $newCommit" "Green"
        } else {
            Log "  No new commit this iteration" "DarkYellow"
        }
    } catch {}

    if ($resultText) {
        $snippet = $resultText
        if ($snippet.Length -gt 400) { $snippet = $snippet.Substring(0, 400) + "..." }
        Log "  Result: $snippet" "White"
    }

    # --- Backoff on failure, otherwise wait CycleSeconds ---
    if ($hadError -or $exitCode -ne 0) {
        $consecutiveFailures++
        $backoff = [Math]::Min(600, $CycleSeconds * [Math]::Pow(2, $consecutiveFailures - 1))
        Log "FAILURE #$consecutiveFailures - backing off ${backoff}s..." "Red"
        Start-Sleep -Seconds $backoff
    } else {
        $consecutiveFailures = 0
        Log "Next iteration in ${CycleSeconds}s..." "DarkGray"
        Start-Sleep -Seconds $CycleSeconds
    }
}

LogSection "EVOLUTION LOOP ENDED"
Log "Total iterations: $iteration | Total cost: `$$([math]::Round($totalCost, 4))" "Yellow"
Log "Ended: $(Get-Date)" "Yellow"
