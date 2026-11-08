# GHCR images and repository releases

This replaces the closed, unmerged archive proposal in PR #6.
Existing Container CI still does not push images. This new manual workflow is
**disabled** by `compliance/ghcr-release-policy.json`. Merging it alone publishes
nothing. The merged PR #7 corrected the GPU harness, not registry publication.

## Destinations and behavior

- `ghcr.io/barabashkad/multistream-gpu-inference-instance`
- `ghcr.io/barabashkad/multistream-gpu-inference-server`

Existing packages linked to this exact repository are accepted only when their
visibility and the repository visibility both match the explicit policy
(`private` or `public`). The default remains disabled and private. A 403/404 is a blocker, not proof that a name is unused. This intentionally
requires an authorized first-package bootstrap outside this workflow if packages
do not exist. Do not bootstrap, change visibility or grant access implicitly.
Review package collaborators, inherited repository access, and Actions access
before enabling. The API visibility/link check cannot enumerate all recipients.

A maintainer must separately review the intended audience and exact
source's distribution obligations, record the approval/review references and
40-character source commit in the policy, and change `enabled` via reviewed PR.
That commit must be an ancestor of the dispatch's main commit. Do not use the
policy-edit commit's own SHA (self-reference); authorize an existing source commit.
The source checkout is detached and exact. It must contain the build, tests and
evidence tooling. Workflows only run on manual main dispatch. Public mode additionally requires a
nonempty `public_distribution_review` reference; private review is not reusable
as public clearance. The workflow checks live repository and package visibility
before upload and repository visibility again before release promotion.
No secrets, new PAT, package permission changes or security settings are added.

Read-only build runner builds both images, verifies contents, runs mandatory CPU
suites in server, extracts evidence and transfers the exact Docker image archive
through a one-day Actions artifact (readable by the repository audience). A fresh publishing runner has only
`contents:read` and `packages:write`; it loads but does not execute containers or
repository scripts. It checks image IDs/revisions, pushes unique run-attempt tags,
resolves registry digests, pulls by digest and compares config IDs. A separate
`contents:write` runner creates the source-bound GitHub prerelease only after both
images succeed, attaches evidence and verifies downloaded assets. No archives are
attached to releases, and no `latest` or stable tag is assigned. Releases have the
repository's current audience; they are not a registry themselves.

If a push partly succeeds, the completed package version remains; there is no
cross-package transaction. The release job atomically creates a lightweight source tag first. A failed release
creation may leave that tag; a failed upload may leave a draft. Inspect these
states manually. Retrying produces a new run-attempt tag and never declares the
partial attempt a complete release. Do not automatically delete existing versions.
Storage quotas can block the large intermediate artifact; no hidden fallback
publication or rebuild is performed. Tag names are mutable at the registry:
consumers should use the immutable digest references in the release.

## License and GPU boundaries

The project is AGPL-3.0-or-later with NOTICE terms. Bundled libraries, weights,
media and binary components retain their own terms; the OCI project-license label
is not a blanket license for them. Inventory/notices do not by themselves resolve
Ultralytics/NVIDIA compatibility, component redistribution permissions or complete
matching source/relink obligations (including bundled FFmpeg). Resolve the specific
scope before authorizing it. Private publication also shares with authorized
package recipients, so it is not automatically exempt from review.

The images are Linux/amd64, CPU-exported ONNX batches 1/8/16. These release bytes
are **not GPU qualified** by this workflow. The successful prior local WSL test
is not a test of the registry digest. The merged PR #7 corrects the GPU harness; perform explicit digest-bound verification.

## Using the result

Open the GitHub release for `docker pull ghcr.io/...@sha256:...` for both targets
and links to their package pages. Private pulls require appropriate GHCR login;
GitHub documents a classic PAT with `read:packages` for local users. Supply tokens
through password-stdin, never commit them. Actions publication uses GITHUB_TOKEN,
not a new personal credential. Package creation defaults to private, but an
existing package can have independent visibility/access: never infer it from the
source repository alone.

References (checked 2026-11-08):
- https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
- https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility

## Moving to public distribution

1. Review the exact source, repository history, Actions artifacts/logs and bundled
   component obligations before making anything public. Changing repository
   visibility does not change existing package visibility.
2. Separately authorize and provision both packages, link them to this repository,
   grant this repository Actions access, and set the intended visibility. Missing
   or inaccessible packages remain a hard blocker; this workflow never creates a
   first package implicitly or changes visibility.
3. In a reviewed policy PR set `visibility: "public"`, an existing source SHA,
   approval/audience/distribution references, and `public_distribution_review`
   documenting public redistribution clearance. Set `enabled: true` only after
   those reviews. These strings are audit references, not automatic legal checks.
4. Dispatch from main and inspect the resulting prerelease and both digest links.
   Public package pulls need no login. No publication occurs from this preparation
   PR alone. GPU qualification is still a separate step.

Visibility can be changed concurrently by administrators; API checks reduce
accidental audience mismatch but cannot lock those settings transactionally.
