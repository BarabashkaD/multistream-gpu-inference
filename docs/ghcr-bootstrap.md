# Standalone empty-package bootstrap

The manual **Bootstrap empty private GHCR packages** workflow creates only:
- `ghcr.io/barabashkad/multistream-gpu-inference-instance`
- `ghcr.io/barabashkad/multistream-gpu-inference-server`

It is separate from the release workflow: no Dockerfile build, model download,
application archive, artifact upload, or GitHub Release is possible in this workflow.
No repository checkout or repository script runs with a write token.

## Enable and run

1. Review the bootstrap workflow and intended package access. In a reviewed PR,
   set only `bootstrap_enabled: true` and `bootstrap_review` to the real
   authorization/review record in `compliance/ghcr-release-policy.json`.
   The record must authorize creation of these two private namespaces and empty
   marker versions. This change does not authorize application distribution.
2. Keep release `enabled: false` and all unfinished distribution review fields
   empty. This preparation PR does not alter any policy field or assert approval.
3. After that PR is merged, dispatch this workflow **from main**, explicitly
   checking `confirm_empty_markers`. Policy is fetched at the dispatch commit;
   the workflow checks that commit still equals current main before writes.
4. Inspect both package settings: repository link, inherited access, collaborators
   and Actions access. API checks verify private visibility and repository link,
   not a complete list of recipients. Package visibility is never changed here.

HTTP 200 requires an already private, correctly linked package and writes no marker.
Only HTTP 404 attempts an empty marker, because 404 does not prove absence.
Other statuses fail closed. GitHub creates new packages privately by default.
The workflow verifies their visibility and link after a push; unexpected state
stops it. A hidden existing package may receive an **empty** marker before that
check, but never project files, model weights or dependencies.

The marker uses an empty tar filesystem and only a public repository-source label.
It is not a runnable application. Tags are `bootstrap-empty-<run>-<attempt>`.
The workflow leaves markers for inspection; it deletes nothing. A failed run can
leave one or both packages; retries handle existing private packages safely.
No new personal token is required: the runner uses its scoped `GITHUB_TOKEN`.

## Application publication remains a separate gate

This bootstrap needs no claim of third-party binary redistribution clearance
because it distributes no third-party binary or project code. It does **not**
waive or satisfy the release workflow's exact-source, audience and distribution
reviews. Do not fill those fields with bootstrap approval.

For later public images, review redistribution obligations first, separately
authorize the irreversible package-public visibility changes, and activate the
release policy through a reviewed PR. The original release workflow's own
bootstrap option remains available for already reviewed distributions, but that
workflow builds and transfers application archives before package provisioning.
Use this standalone workflow when only empty namespace creation is intended.

Both workflows share the `ghcr-release` concurrency group. Administrator changes
to main or package settings are not transactionally locked; fail-closed checks
reduce races but do not prevent every concurrent external change.
