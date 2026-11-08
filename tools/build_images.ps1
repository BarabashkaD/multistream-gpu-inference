# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.

#requires -Version 7.3

<#
.SYNOPSIS
Prepares and builds the self-contained v3 images traffic-v3:instance and traffic-v3:server on Windows.

.DESCRIPTION
Needs only Docker Desktop (WSL2 backend) and an NVIDIA driver. Each step runs only when needed:
  1. checks that Docker works;
  2. downloads the clip to <repository>/data if it is missing or its SHA-256 differs;
  3. exports the missing ONNX models (models/b<size>.onnx) in a throwaway CPU-only container;
  4. builds both images with the clip as a named build context;
  5. smoke-tests them (unless -SkipTests): the clip in the instance image, the models and the CUDA provider in the
     server image;
  6. with -SaveTo, writes both images to one tar file for `docker load` on another machine.
tools/build_images.sh does the same on Ubuntu 24.04.

.PARAMETER Sizes
Batch sizes of the models that must exist (default 1,8,16).

.PARAMETER SaveTo
Write both images to this tar file (or to traffic-v3-images.tar in this folder) with docker save.

.PARAMETER SkipTests
Skip the smoke test.

.EXAMPLE
pwsh tools/build_images.ps1
pwsh tools/build_images.ps1 -SaveTo traffic-v3-images.tar
#>
[CmdletBinding()]
param(
    [ValidatePattern('^[1-9][0-9]*(,[1-9][0-9]*)*$')][string]$Sizes = '1,8,16',
    [string]$SaveTo,
    [switch]$SkipTests
)

# A comma list given with -File arrives as one string, so it is split here.
$SizeList = [int[]]($Sizes -split ',')

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$ProgressPreference = 'SilentlyContinue'

$V3 = Split-Path -Parent $PSScriptRoot
$Repo = $V3
$ClipName = 'pexels-36304379.mp4'
$ClipDir = Join-Path $Repo 'data'
$Clip = Join-Path $ClipDir $ClipName
$ClipUrl = 'https://videos.pexels.com/video-files/36304379/15396218_1920_1080_25fps.mp4'
$ClipSha256 = 'D105FB6777E82CAAFADC3763098F533864CF294448D41AC195E17C69E068B278'
$ClipBytes = 10603143
$UserAgent = 'traffic-v3-build-script/1.0'
$Images = @('traffic-v3:instance', 'traffic-v3:server')

# Prints a progress line.
function Say([string]$Message) { Write-Host $Message }

# Prints the message in red and ends the script with exit code 1, so every failure reads the same way.
function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

# The base image comes from the Dockerfile, the one place that pins its digest.
$match = Select-String -Path (Join-Path $V3 'Dockerfile') -Pattern '^ARG PYTHON_IMAGE=(\S+)' | Select-Object -First 1
if (-not $match) { Fail "no 'ARG PYTHON_IMAGE=' line in $(Join-Path $V3 'Dockerfile')" }
$BaseImage = $match.Matches[0].Groups[1].Value

# 1. Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail 'docker is not on PATH: install Docker Desktop (WSL2 backend), start it and run again.'
}
try {
    $serverVersion = docker version --format '{{.Server.Version}}' 2>$null
} catch {
    Fail 'Docker is not reachable: start Docker Desktop and run again.'
}
Say "[1/6] Docker: server $serverVersion"

# 2. Clip
# True when the file's SHA-256 is the expected one, which also catches a truncated download.
function Test-Clip([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $ClipSha256 }

if ((Test-Path -LiteralPath $Clip) -and (Test-Clip $Clip)) {
    Say "[2/6] Clip: $Clip present and verified; download skipped"
} else {
    $reason = if (Test-Path -LiteralPath $Clip) { 'has the wrong SHA-256' } else { 'missing' }
    Say "[2/6] Clip: $Clip $reason; downloading $ClipUrl"
    New-Item -ItemType Directory -Force -Path $ClipDir | Out-Null
    $part = Join-Path $ClipDir ".$ClipName.part"
    try {
        for ($attempt = 1; ; $attempt++) {
            try {
                Invoke-WebRequest -Uri $ClipUrl -OutFile $part -UserAgent $UserAgent
                break
            } catch {
                if ($attempt -gt 3) { Fail "download failed: $($_.Exception.Message)" }
                Say "[2/6] Clip: download failed ($($_.Exception.Message)); retrying"
                Start-Sleep -Seconds $attempt
            }
        }
        if (-not (Test-Clip $part)) {
            Fail "the downloaded clip's SHA-256 is not $ClipSha256; deleted it"
        }
        Move-Item -LiteralPath $part -Destination $Clip -Force
    } finally {
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    }
    Say '[2/6] Clip: downloaded and verified'
}

# 3. Models
$names = $SizeList | ForEach-Object { "b$_" }
$missing = @($SizeList | Where-Object {
    $file = Join-Path $V3 "models/b$_.onnx"
    -not ((Test-Path -LiteralPath $file) -and (Get-Item -LiteralPath $file).Length -gt 0)
})
if ($missing.Count -eq 0) {
    Say "[3/6] Models: $($names -join ' ') present; export skipped"
} else {
    $missingNames = ($missing | ForEach-Object { "b$_" }) -join ' '
    Say "[3/6] Models: exporting $missingNames in a throwaway container (CPU only, a few minutes)"
    # Single quotes: the container's sh expands $SIZES, $HOST_UID and $rc, not PowerShell. On Linux the same
    # command gives the files back to the host user; on Windows HOST_UID is unset and ownership does not apply.
    $export = 'apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 >/dev/null' +
        ' && pip install -q --no-cache-dir --root-user-action=ignore -r requirements-export.txt' +
        ' && python tools/export_models.py --sizes "$SIZES"; rc=$?;' +
        ' if [ -n "${HOST_UID:-}" ]; then chown -R "$HOST_UID:$HOST_GID" /work/models; fi; exit $rc'
    try {
        docker run --rm -v "${V3}:/work" -w /work -e "SIZES=$($missing -join ',')" $BaseImage sh -c $export
    } catch {
        Fail "the model export failed (see the output above)"
    }
    foreach ($size in $missing) {
        $path = Join-Path $V3 "models/b$size.onnx"
        if (-not ((Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -gt 0)) { Fail "models/b$size.onnx was not written" }
    }
    Say "[3/6] Models: exported $missingNames"
}

# 4. Build
foreach ($image in $Images) {
    $target = $image.Split(':')[1]
    Say "[4/6] Build: $image"
    try {
        docker build --build-context "clip=$ClipDir" --target $target -t $image $V3
    } catch {
        Fail "building $image failed (see the output above)"
    }
}
Say '[4/6] Build: both images built'

# 5. Smoke test
if ($SkipTests) {
    Say '[5/6] Smoke test: skipped (-SkipTests)'
} else {
    try {
        $bytes = docker run --rm traffic-v3:instance python -c "import os; print(os.path.getsize('/data/$ClipName'))"
    } catch {
        Fail "the clip is missing from traffic-v3:instance"
    }
    if ([int64]$bytes -ne $ClipBytes) { Fail "the clip in traffic-v3:instance has $bytes bytes, expected $ClipBytes" }
    Say "[5/6] Smoke test: clip in traffic-v3:instance ok ($bytes bytes)"
    $probe = 'import os; from pathlib import Path; from server.detector import BatchDetector; ' +
        "BatchDetector(Path('models'), tuple(int(s) for s in os.environ['SIZES'].split(',')), 0); " +
        "print('models and CUDA provider ok')"
    try {
        docker run --rm --gpus all -e "SIZES=$($SizeList -join ',')" traffic-v3:server python -c $probe
    } catch {
        Fail "traffic-v3:server could not load $($names -join ' ') with the CUDA provider (see the output above)"
    }
    Say "[5/6] Smoke test: $($names -join ' ') and CUDA provider ok in traffic-v3:server"
}

# 6. Save
if (-not $SaveTo) {
    Say '[6/6] Save: skipped (no -SaveTo)'
} else {
    $file = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SaveTo)
    if (Test-Path -LiteralPath $file -PathType Container) { $file = Join-Path $file 'traffic-v3-images.tar' }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $file) | Out-Null
    Say "[6/6] Save: writing $file"
    try {
        docker save @Images -o $file
    } catch {
        Fail "docker save failed (see the output above)"
    }
    $gb = ((Get-Item -LiteralPath $file).Length / 1e9).ToString('N1', [cultureinfo]::InvariantCulture)
    Say "[6/6] Save: wrote $file ($gb GB)"
    Say "[6/6] Save: on the other machine run: docker load -i $(Split-Path -Leaf $file)"
}

Say "Done: $($Images -join ' and ') are ready."
