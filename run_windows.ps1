# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.

<#
.SYNOPSIS
Runs launch_instances.py with the virtual environment created by setup_windows.ps1.

.DESCRIPTION
Requires .venv\Scripts\python.exe next to this script and throws, asking for setup_windows.ps1 to be run first, when
it is missing; unlike launch_instances.ps1 it never falls back to another Python. It passes every argument through
unchanged to launch_instances.py and exits with that script's exit code.

.PARAMETER Arguments
Arguments forwarded as they are to launch_instances.py, for example --instances 1 --headless.

.EXAMPLE
.\run_windows.ps1 --instances 1 --headless
Runs one headless demo process.
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup_windows.ps1 first; native venv not found at $Python"
}
& $Python (Join-Path $ProjectRoot "launch_instances.py") @Arguments
exit $LASTEXITCODE
