# GHCR images and repository releases

One manual workflow, **Publish checked images to GHCR and release**, handles both
images and a GitHub prerelease. There is no activation policy, marker bootstrap,
new PAT, or follow-up configuration PR. Merging this change publishes nothing.

## Run after merging

1. Review redistribution obligations for the actual bundled components below.
   **Dispatch uploads real binaries, including a temporary Actions artifact
   visible to this public repository's readers.** A private GHCR package does
   not make this operation private.
2. In Actions, select **Publish checked images to GHCR and release**, choose
   **Run workflow** on `main`. The exact dispatched main commit is built; a stale
   dispatch is rejected before building. Standard `GITHUB_TOKEN` is sufficient.
3. CI builds and checks both targets, runs mandatory CPU suites, pushes them to:
   - `ghcr.io/barabashkad/multistream-gpu-inference-instance`
   - `ghcr.io/barabashkad/multistream-gpu-inference-server`
4. Open the resulting GitHub prerelease. It contains immutable `docker pull
   ...@sha256:...` commands, package links, source revision, and license/inventory
   evidence. Docker archives are not attached to the release.
5. **First push creates private GHCR packages by default**, even for this public
   repository. For anonymous pulls, open each package link → Package settings →
   Change visibility → Public. This is a one-time owner setting, not another PR,
   workflow or rebuild. GitHub warns that public packages cannot be made private
   again. Inspect existing package permissions before changing visibility.
   The release records visibility at creation time, not a live status.

Subsequent runs use the same packages and their existing visibility. Repository
visibility and package visibility are independent. If a pre-existing package
rejects the push, its owner must grant this repository Actions access in Package
settings; do not create a new personal token as a workaround.

## What CI verifies

A read-only runner executes source/build/test tools and saves both checked images.
A fresh packages-write runner loads, but does not execute, the images. It checks
image IDs and revision labels, pushes unique `ci-<sha12>-<run>-<attempt>` tags,
pulls by registry digest and verifies config identity and repository association.
A separate contents-write runner creates the source-bound tag and draft release,
verifies downloaded assets, and publishes the prerelease only after both targets
succeed. No `latest` or stable tag is assigned. Consumers should use digests.

A failed run can leave one pushed image, a source tag, or a draft release.
There is no cross-package rollback. Rerunning creates a new attempt-specific tag;
inspect partial state rather than deleting versions automatically. Image transfer
uses a one-day Actions artifact; storage quotas can block the run. No fallback
upload is attempted. Main protection and read-only PR CI are unchanged.

## License and GPU boundaries

The project is AGPL-3.0-or-later with NOTICE terms. Dependencies, model weights,
media and binary components retain their own licenses. Inventory and attached
notices do **not** certify complete redistribution clearance. Review the concrete
Ultralytics/AGPL and NVIDIA terms and the matching-source/relink obligations for
bundled FFmpeg before distributing these images. The project's linking exception
cannot grant rights over other contributors' code. This workflow does not replace
that assessment with approval-shaped strings in JSON.

Images are Linux/amd64 with CPU-exported ONNX batches 1/8/16. The published digests
are **not GPU qualified** by CPU CI. Prior local GPU success is not validation of
these registry bytes; use the existing digest-bound GPU smoke harness separately.

Private pulls require GHCR authentication with read:packages; public pulls do not.
Do not commit tokens or put them on a command line; use password-stdin.

References:
- https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
- https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility
