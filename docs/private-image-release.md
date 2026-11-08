# Private CI image archive

The ordinary Container CI workflow retains text evidence only. Its temporary Docker images are not recoverable after the runner is destroyed.

After this workflow is reviewed and merged, manually run **Save private CI image release** from `main` in Actions. It only runs in the existing private repository. It builds both targets anew, runs mandatory CPU tests, packages those same image IDs and preserves them as a **prerelease**, never `latest`. It does not alter repository visibility or push to GHCR.

The combined Docker archive shares layers and is compressed/split into assets smaller than 2 GiB. The release includes SHA256SUMS, a manifest, restoration instructions and license evidence. The release job downloads uploaded assets and verifies all hashes before publishing the draft. Failed uploads leave a draft for inspection; never delete/retry blindly. Each run attempt has a distinct tag.

No new GPU test or legal clearance is implied. Private release assets are available to people already able to read the private repository; private is not the same as personal-only. Do not add recipients or redistribute outside that audience before resolving docs/distribution-review.md. If the repository becomes public, this workflow refuses new runs, but already published assets follow repository access: reassess before changing visibility.

Only the release job receives `contents: write`; build and PR jobs do not. No PAT, package permission, GPU runner, automatic PR publishing or direct main write is required. Workflow dispatch normally becomes available after merge to the default branch; creating the draft PR alone does not create an image release.

The archive contains the mutable local tags `traffic-v3:instance/server`. Loading it replaces those tags in the destination daemon, so preserve any local tags you need first. Verify manifest image IDs after loading. Full dependency locks and public redistribution review remain separate work.
