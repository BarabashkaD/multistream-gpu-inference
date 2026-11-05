# syntax=docker/dockerfile:1
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
# Two targets: "instance" (instances and the monitor) and "server" (the server, the single-container fleet, tests).
# The clip is inside both images and the ONNX models inside the server image; a mount over /data or /app/models
# replaces them. Build with the clip as a named context: docker build --build-context clip=./data --target <name> .
# (compose does this itself). The models must exist in models/ before building the server image.
# Results are mounted at run time. See README.md, "Containers".
ARG PYTHON_IMAGE=python:3.14-slim@sha256:930e1abba7c4e8f96ed93e8012c35d174d7732752731bbd5e8e757d7ec6f3966

# Build stage: a virtual environment with the instance requirements. Only the environment is copied on, so the
# final images carry no build leftovers.
FROM ${PYTHON_IMAGE} AS build-instance
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
COPY requirements-instance.txt /tmp/
RUN /opt/venv/bin/pip install -r /tmp/requirements-instance.txt

# Build stage: the instance environment plus the server requirements.
FROM build-instance AS build-server
COPY requirements-server.txt /tmp/
RUN /opt/venv/bin/pip install -r /tmp/requirements-server.txt

# Shared base of both targets: the environment on PATH, unbuffered output so logs appear at once, one BLAS thread
# per process, and the NVIDIA capabilities (compute and video decode) the container needs.
FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
WORKDIR /app

# Target "instance": the client pipeline and the monitor, with the clip.
FROM runtime AS instance
COPY --from=build-instance /opt/venv /opt/venv
COPY pipeline/ pipeline/
COPY run_instance.py monitor.py ./
COPY --from=clip pexels-36304379.mp4 /data/pexels-36304379.mp4
# The image description. Keep it in step with the server target's label.
LABEL org.opencontainers.image.description="Multi-stream YOLO26n detection and ByteTrack tracking on one NVIDIA GPU. Source, build instructions and license: https://github.com/BarabashkaD/multistream-gpu-inference"

# Target "server": the inference server, the fleet and benchmark scripts, the tests, the models and the clip.
FROM runtime AS server
COPY --from=build-server /opt/venv /opt/venv
COPY pipeline/ pipeline/
COPY server/ server/
COPY tests/ tests/
COPY tools/ tools/
COPY compose.yaml compose.mounts.yaml ./
COPY run_server.py run_instance.py run_fleet.py bench.py monitor.py pytest.ini ./
COPY models/ models/
COPY --from=clip pexels-36304379.mp4 /data/pexels-36304379.mp4
# The image description. Keep it in step with the instance target's label.
LABEL org.opencontainers.image.description="Multi-stream YOLO26n detection and ByteTrack tracking on one NVIDIA GPU. Source, build instructions and license: https://github.com/BarabashkaD/multistream-gpu-inference"
