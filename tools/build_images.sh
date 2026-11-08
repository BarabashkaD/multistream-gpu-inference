#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
# Prepares and builds the self-contained v3 images traffic-v3:instance and traffic-v3:server on Ubuntu 24.04.
#
# Needs Docker/buildx, curl and SHA-256 tools. GPU mode also needs the NVIDIA driver and Container Toolkit.
#   1. checks Docker/buildx (and GPU access unless --cpu-only);
#   2. downloads and verifies the clip;
#   3. exports hash-verified ONNX models in a throwaway CPU-only container;
#   4. builds both images with the clip as a named build context;
#   5. always verifies contents, then runs the GPU smoke test unless disabled;
#   6. optionally writes both images to a tar file for docker load.
# CPU-only mode is currently supported by this Linux script, not build_images.ps1.
#
# Usage: tools/build_images.sh [--sizes 1,8,16] [--save-to PATH] [--skip-tests] [--cpu-only]
set -euo pipefail

v3="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$v3"
clip_name="pexels-36304379.mp4"
clip_dir="$repo/data"
clip="$clip_dir/$clip_name"
clip_url="https://videos.pexels.com/video-files/36304379/15396218_1920_1080_25fps.mp4"
clip_sha256="d105fb6777e82caafadc3763098f533864cf294448d41ac195e17c69e068b278"
user_agent="traffic-v3-build-script/1.0"
images=(traffic-v3:instance traffic-v3:server)

sizes_arg="1,8,16"
save_to=""
skip_tests=0
cpu_only=0

# Prints a progress line.
say() { printf '%s\n' "$*"; }

# Prints the message on standard error and ends the script with exit code 1, so every failure reads the same way.
fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# Prints the command-line help; shared by --help and the unknown-argument error.
usage() {
    say "Usage: $0 [--sizes 1,8,16] [--save-to PATH] [--skip-tests] [--cpu-only]"
    say "  --sizes       batch sizes of the models that must exist (default 1,8,16)"
    say "  --save-to     write both images to this tar file (or to traffic-v3-images.tar in this folder)"
    say "  --skip-tests  skip the GPU smoke test (contents checks always run)"
    say "  --cpu-only    build and check contents without GPU access; GPU tests NOT RUN"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sizes) [ $# -ge 2 ] || fail "--sizes needs a value"; sizes_arg="$2"; shift 2 ;;
        --save-to) [ $# -ge 2 ] || fail "--save-to needs a value"; save_to="$2"; shift 2 ;;
        --skip-tests) skip_tests=1; shift ;;
        --cpu-only) cpu_only=1; skip_tests=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) usage >&2; fail "unknown argument: $1" ;;
    esac
done
[[ "$sizes_arg" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || fail "--sizes must be a comma-separated list of positive numbers"
IFS=, read -r -a sizes <<< "$sizes_arg"
names=""
for size in "${sizes[@]}"; do names="${names:+$names }b$size"; done

# The base image comes from the Dockerfile, the one place that pins its digest.
base_image="$(sed -n 's/^ARG PYTHON_IMAGE=\([^[:space:]]*\).*/\1/p' "$v3/Dockerfile" | head -n 1)"
[ -n "$base_image" ] || fail "no 'ARG PYTHON_IMAGE=' line in $v3/Dockerfile"

# 1. Docker, buildx and GPU access
command -v docker > /dev/null || fail "docker is not installed: see https://docs.docker.com/engine/install/ubuntu/"
server_version="$(docker version --format '{{.Server.Version}}' 2> /dev/null)" \
    || fail "Docker is not reachable: start it (sudo systemctl start docker) and make sure your user may use it (the docker group), then run again."
say "[1/6] Docker: server $server_version"

if ! buildx_version="$(docker buildx version 2> /dev/null)"; then
    cat >&2 << 'EOF'
docker buildx is missing; the builds need it for --build-context. Install it, then run this script again:
  with Docker's apt repository (docker-ce):  sudo apt-get install docker-buildx-plugin
  with Ubuntu's docker.io package:           sudo apt-get install docker-buildx
EOF
    exit 1
fi
say "[1/6] Docker: buildx ${buildx_version}"

docker pull "$base_image" > /dev/null || fail "pulling $base_image failed: check the network and that Docker can reach its registry"
if [ "$cpu_only" -eq 0 ]; then
if ! docker run --rm --gpus all "$base_image" nvidia-smi > /dev/null; then
    cat >&2 << 'EOF'
GPU access in containers failed. First check the NVIDIA driver on the host (nvidia-smi). Then install the
NVIDIA Container Toolkit from NVIDIA's apt repository and run this script again:
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
Guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
EOF
    exit 1
fi
say "[1/6] Docker: GPU access ok"
else
    say "[1/6] GPU access: NOT TESTED (--cpu-only)"
fi

# 2. Clip
# Succeeds when the file's SHA-256 is the expected one, which also catches a truncated download.
clip_ok() { [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$clip_sha256" ]; }

part="$clip_dir/.$clip_name.part"
if [ -f "$clip" ] && clip_ok "$clip"; then
    say "[2/6] Clip: $clip present and verified; download skipped"
else
    if [ -f "$clip" ]; then reason="has the wrong SHA-256"; else reason="missing"; fi
    say "[2/6] Clip: $clip $reason; downloading $clip_url"
    command -v curl > /dev/null || fail "curl is missing: sudo apt-get install curl"
    mkdir -p "$clip_dir"
    trap 'rm -f "$part"' EXIT
    curl -fL --retry 3 --retry-all-errors -A "$user_agent" -o "$part" "$clip_url" || fail "download failed"
    clip_ok "$part" || fail "the downloaded clip's SHA-256 is not $clip_sha256; deleted it"
    mv -f "$part" "$clip"
    say "[2/6] Clip: downloaded and verified"
fi

# 3. Models
# Re-export all requested sizes; stale/nonempty models are not trusted.
missing=("${sizes[@]}")
{
    missing_names=""
    for size in "${missing[@]}"; do missing_names="${missing_names:+$missing_names }b$size"; done
    missing_csv="$(IFS=,; printf '%s' "${missing[*]}")"
    say "[3/6] Models: exporting $missing_names in a throwaway container (CPU only, a few minutes)"
    # Single quotes: the container's sh expands these variables. The same shell gives the files back to the host
    # user (they are created by root in the container) and keeps the export's exit code.
    # shellcheck disable=SC2016
    export_cmd='apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 >/dev/null'
    export_cmd+=' && pip install -q --no-cache-dir --root-user-action=ignore -r requirements-export.txt'
    # shellcheck disable=SC2016
    export_cmd+=' && python tools/export_models.py --sizes "$SIZES"; rc=$?;'
    # shellcheck disable=SC2016
    export_cmd+=' if [ -n "${HOST_UID:-}" ]; then chown -R "$HOST_UID:$HOST_GID" /work/models; fi; exit $rc'
    docker run --rm -v "$v3:/work" -w /work -e "SIZES=$missing_csv" -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
        "$base_image" sh -c "$export_cmd" || fail "the model export failed (see the output above)"
    for size in "${missing[@]}"; do
        [ -s "$v3/models/b$size.onnx" ] || fail "models/b$size.onnx was not written"
    done
    say "[3/6] Models: exported $missing_names"
}

# 4. Build
revision="${VCS_REF:-$(git -C "$v3" rev-parse HEAD 2>/dev/null || printf unknown)}"
for image in "${images[@]}"; do
    say "[4/6] Build: $image"
    docker build --build-arg "VCS_REF=$revision" --build-context "clip=$clip_dir" --target "${image#*:}" -t "$image" "$v3" \
        || fail "building $image failed (see the output above)"
done
say "[4/6] Build: both images built"

# 5. Mandatory CPU contents checks, followed by the optional GPU smoke test.
expected_documents="{"
separator=""
for document in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do
    hash="$(sha256sum "$v3/$document" | cut -d ' ' -f 1)"
    expected_documents+="$separator\"$document\":\"$hash\""
    separator=","
done
expected_documents+="}"
for image in "${images[@]}"; do
    docker run --rm --network none "$image" python tools/check_image.py \
        --target "${image#*:}" --sizes "$sizes_arg" --expected-documents "$expected_documents" \
        || fail "CPU contents check failed for $image"
done
if [ "$skip_tests" -eq 1 ]; then
    say "[5/6] GPU smoke test: NOT RUN (CPU contents checks passed)"
else
    probe="import os; from pathlib import Path; from server.detector import BatchDetector; "
    probe+="BatchDetector(Path('models'), tuple(int(s) for s in os.environ['SIZES'].split(',')), 0); "
    probe+="print('models and CUDA provider ok')"
    docker run --rm --gpus all -e "SIZES=$sizes_arg" traffic-v3:server python -c "$probe" \
        || fail "traffic-v3:server could not load $names with the CUDA provider (see the output above)"
    say "[5/6] Smoke test: $names and CUDA provider ok in traffic-v3:server"
fi

# 6. Save
if [ -z "$save_to" ]; then
    say "[6/6] Save: skipped (no --save-to)"
else
    if [ -d "$save_to" ]; then save_to="$save_to/traffic-v3-images.tar"; fi
    mkdir -p "$(dirname -- "$save_to")"
    file="$(cd -- "$(dirname -- "$save_to")" && pwd)/$(basename -- "$save_to")"
    say "[6/6] Save: writing $file"
    docker save "${images[@]}" -o "$file" || fail "docker save failed (see the output above)"
    gb="$(awk -v bytes="$(stat -c %s "$file")" 'BEGIN { printf "%.1f", bytes / 1e9 }')"
    say "[6/6] Save: wrote $file ($gb GB)"
    say "[6/6] Save: on the other machine run: docker load -i $(basename -- "$file")"
fi

say "Done: ${images[0]} and ${images[1]} built and contents checked. Redistribution review is separate."
