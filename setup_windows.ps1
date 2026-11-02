# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.

<#
.SYNOPSIS
Creates the project's virtual environment on Windows and installs its dependencies.

.DESCRIPTION
Creates .venv next to this script with the py launcher when the folder does not exist, then upgrades pip and installs
requirements-windows-plan.txt into it. Stops with an error if the py launcher is missing or the venv has no
python.exe. Writes to the .venv folder and, unless installing is skipped, downloads packages.

.PARAMETER SkipInstall
Creates the virtual environment but does not upgrade pip or install the requirements.

.EXAMPLE
.\setup_windows.ps1
Creates the environment and installs everything.

.EXAMPLE
.\setup_windows.ps1 -SkipInstall
Only makes sure the environment exists.
#>

param(
    [switch] $SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Venv)) {
    $PyLauncher = Get-Command py -ErrorAction Stop
    & $PyLauncher.Source -3 -m venv $Venv
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Native Windows venv was not created: $Python"
}
if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-windows-plan.txt")
}
Write-Host "Windows project prepared at $ProjectRoot"
Write-Host "Use .\run_windows.ps1 --instances 1 --headless"
