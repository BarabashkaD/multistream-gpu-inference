# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.

<#
.SYNOPSIS
Runs launch_instances.py with the project's virtual environment Python, or with the Python found on PATH.

.DESCRIPTION
Picks .venv\Scripts\python.exe next to this script when it exists and otherwise falls back to the first python on
PATH (an error if there is none). It then runs launch_instances.py from the same folder, passing every argument
through unchanged, and exits with that script's exit code.

.PARAMETER Arguments
Arguments forwarded as they are to launch_instances.py, for example --instances 2 --headless.

.EXAMPLE
.\launch_instances.ps1 --instances 2 --headless
Starts two headless demo processes.
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
& $Python (Join-Path $ProjectRoot "launch_instances.py") @Arguments
exit $LASTEXITCODE
