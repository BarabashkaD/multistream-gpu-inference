# Container CI

The PR/main workflow builds both `instance` and `server` on a CPU runner without
publishing images, Docker archives, or registry credentials. It retains the
existing Linux dependency pins; full dependency/asset locking and distribution
review remain separate work before release.

Run `bash tools/build_images.sh --cpu-only` to prepare the clip/models, build both
targets and verify their contents. This needs Docker/buildx and network access,
but no GPU. Normal execution still performs the existing NVIDIA preflight and
GPU model smoke test. `--skip-tests` skips only the GPU smoke test, not contents
checks; use `--cpu-only` when no NVIDIA runtime is available.

The CPU checks compare LICENSE, NOTICE and THIRD_PARTY_NOTICES.md with the source
files by hash, verify the video hash, check expected model files, and check the
instance image has no Torch. They do not prove model provenance or third-party
license compliance. Original wheel/base-image license files are not removed.
Project legal documents live at `/app/`; `/app/data` points to `/data` so the
application and test defaults find the packaged video.

`docker run --rm --network none traffic-v3:server python tools/run_cpu_tests.py`
runs explicitly required CPU suites and fails if a suite disappears or any
mandatory test is skipped. Other tests, including optional comparison/display
and GPU tests, are not covered by this small suite. GPU tests must be run
separately on an authorized GPU machine against the actual release candidate.

Before release: inspect real bundled components and corresponding-source duties,
verify model sources/hashes and NVIDIA/Ultralytics rights, lock dependencies,
record inventories, then select a candidate transfer/GPU-test/publication flow.
Do not interpret green CPU CI as GPU validation or redistribution approval.
