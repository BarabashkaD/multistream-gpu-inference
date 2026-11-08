# Candidate verification and deferred release

This PR prepares GPU verification. **It does not publish candidate images, stable
images, archives, releases or tags.** GPU verification is disabled by the checked-in
policy and also requires an explicit repository variable. No GPU has been tested
merely because CPU CI is green. Passing runtime tests is not license clearance.

## What CI retains

The CPU workflow builds both targets, runs mandatory tests, and retains selected
JSON/text evidence for 14 days: image IDs/source revision, installed-package
inventory, model provenance/hashes, project license/notices, and the bundled
FFmpeg build README. These are review aids, not a complete SBOM, source offer,
or proof that every proprietary component is redistributable. Locally built image
IDs are not registry manifest digests. No image layers, weights or source archives
are uploaded as artifacts.

## GPU verification on an authorized host

An owner must explicitly authorize a disposable Linux x86-64 NVIDIA host and
access to the specific candidate images. This change does not provision a host,
spend money, install Docker/drivers, use a personal computer, or create registry
credentials. Requirements: Python 3.10+, Docker Engine, Compose v2, NVIDIA
Container Toolkit, a suitable GPU/driver, and permission to pull the two existing
private images. Credentials, if needed, should be short-lived and pull-only.
Candidate images must include this PR's mandatory GPU test runner.

The portable command is:

```sh
python tools/gpu_smoke.py \
  --instance ghcr.io/OWNER/INSTANCE@sha256:INSTANCE_DIGEST \
  --server ghcr.io/OWNER/SERVER@sha256:SERVER_DIGEST \
  --revision FULL_40_CHARACTER_SOURCE_SHA --output gpu-report
```

Replace placeholders with real lowercase immutable references. Never use a mutable
tag. The script checks both source labels, pulls exact digests, records local IDs,
warms every model in the manifest using the CUDA provider, runs mandatory CUDA IPC
tests, then processes the fixed sample clip with two separate instance containers
and one server. It requires 611 tracked frames per stream, real detections, all
buffers returned, 1222 server inferences and no dead streams. Skipped tests fail.
Containers have no network; instances share only the server PID namespace and a
job-specific IPC volume. Server SYS_PTRACE matches the existing CUDA IPC design.
The script removes its own containers/volume. It writes a digest-bound JSON report
and logs, including a failed report when Docker/GPU is unavailable.

This is a headless functional smoke test, not GUI coverage, a benchmark, driver
matrix coverage, multi-GPU qualification, or proof that no ONNX node falls back to
CPU. The full-GPU CUDA provider check does not establish the last claim.

For GitHub execution, BEFORE enabling anything:

1. Use a dedicated disposable runner labeled `mgi-gpu-ephemeral`. Never expose this
   runner to pull-request jobs or unrelated repositories. Tear it down after use.
2. Configure environment `gpu-verification` with required owner reviewers,
   main-only deployment and no self-approval. GitHub YAML cannot create or prove
   these protections; verify they exist. Do not enable on a plan lacking them.
3. In a reviewed change on main, set `compliance/candidate-policy.json` to the exact
   two digests/source SHA, a specific owner authorization record, and enabled true.
   This is authorization to test those images, NOT redistribution approval.
4. Explicitly set repository variable `ENABLE_GPU_VERIFICATION=true`, then manually
   dispatch from main with exactly those inputs. The workflow has read-only token
   permissions, no registry login, build, tagging, or push step. Pull access must
   already be configured on the approved disposable runner.
5. Review the same-run report and artifact, then disable the variable/policy again
   and dispose of the runner. A skipped policy job is not a GPU pass.

## Release remains blocked

There is deliberately no stable-promotion workflow in this PR. Before any
publication or distribution, resolve and record at least:

- Exact shipped binary inventory, including nested/renamed CUDA libraries inside
  PyNvVideoCodec and ONNX Runtime providers, not only Python distribution names.
- Applicable NVIDIA redistribution grants/notices for each actual payload and
  version. PyNvVideoCodec's bundled CUDA 12 runtime is not covered merely by a
  separate CUDA 13 package inventory.
- FFmpeg corresponding source/version, patches, actual build/link flags, notices
  and any relinking materials required for the shipped combined binary. Embedded
  LGPL-2.1-or-later identification and the wheel's LGPL3 notice/election must be
  described accurately, not conflated with an LGPL3-only upstream binary.
- ONNX Runtime compiled third-party components, including Eigen MPL2 source and
  notice obligations, as well as provider binaries actually shipped.
- Ultralytics-derived code and model/weight rights. The project author's NVIDIA
  exception cannot relicense another copyright holder's material.
- Reproducible dependency resolution and provenance for the actual release build.

A future release implementation must build authorized candidates once, retain
source/license evidence, test those exact immutable digests, and promote the same
digests without rebuilding. Bind promotion to successful evidence from the same
trusted run, protect the release environment, scope `packages: write` only to the
promotion job, refuse existing version-tag overwrite, and verify the destination
manifest digest after each promotion. Define recovery for partial two-image
promotion before exposing a stable/latest pair. No existing report or boolean
license flag alone is sufficient legal or technical authorization.
