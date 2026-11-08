# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Guard the disabled, manual GHCR release's trust boundaries without Docker."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/ghcr-release.yml').read_text()


class ReleaseContract(unittest.TestCase):
    """Static workflow safety checks; run on the source checkout, not in images."""
    def test_publication_policy_defaults_closed(self):
        """A merged default policy must not authorize publication."""
        policy = json.loads((ROOT / 'compliance/ghcr-release-policy.json').read_text())
        assert policy['enabled'] is False
        assert policy['visibility'] == 'private'
        assert not policy['source_revision']
        assert not policy['authorization_record']
        assert not policy['distribution_review']
        assert not policy['audience_review']
        assert not policy['public_distribution_review']


    def test_privileged_jobs_do_not_execute_repository_or_images(self):
        """Fresh privileged runners only transport validated image data."""
        privileged = WORKFLOW.split('  publish:', 1)[1]
        assert 'actions/checkout@' not in privileged
        assert 'docker run' not in privileged
        assert 'docker build' not in privileged
        assert 'bash tools/' not in privileged
        assert 'packages: write' in privileged
        assert 'contents: write' in privileged
        assert 'needs: [build, publish]' in privileged
        assert 'docker pull "$digest"' in privileged
        assert 'test "$(docker inspect --format' in privileged


    def test_source_and_publication_are_fail_closed(self):
        """Require manual exact-source approval, matching package audience and no latest."""
        assert 'workflow_dispatch:' in WORKFLOW
        assert 'pull_request:' not in WORKFLOW
        assert 'git merge-base --is-ancestor "$revision" HEAD' in WORKFLOW
        assert 'git checkout --detach "$revision"' in WORKFLOW
        assert '.enabled == true' in WORKFLOW
        assert '.visibility == $visibility and .repository.full_name == $repo' in WORKFLOW
        assert '--latest=false' in WORKFLOW
        assert ':latest' not in WORKFLOW
        assert 'sha256sum -c SHA256SUMS' in WORKFLOW


    def test_release_tag_is_verified_before_draft_promotion(self):
        """An existing tag must never redirect a source-bound release."""
        assert 'git/matching-refs/tags/$RELEASE_TAG' in WORKFLOW
        assert WORKFLOW.index('gh api --method POST') < WORKFLOW.index('gh release create')
        assert '--verify-tag' in WORKFLOW
        assert '-f "sha=$SOURCE_SHA"' in WORKFLOW
        assert '.object.type == "commit" and .object.sha == $sha' in WORKFLOW
        assert WORKFLOW.index('git/ref/tags/$RELEASE_TAG') < WORKFLOW.index('gh release edit')

    def test_public_requires_explicit_distribution_review(self):
        """Public mode must require extra review and propagate exact audience."""
        assert '.visibility == "private" or .visibility == "public"' in WORKFLOW
        assert 'if .visibility == "public" then (.public_distribution_review' in WORKFLOW
        assert 'visibility: ${{ steps.policy.outputs.visibility }}' in WORKFLOW
        assert WORKFLOW.count('EXPECTED_VISIBILITY: ${{ needs.build.outputs.visibility }}') == 2
        assert WORKFLOW.count('--jq .visibility)" = "$EXPECTED_VISIBILITY"') == 3
        assert WORKFLOW.count('.visibility == $visibility and .repository.full_name == $repo') == 2
        assert 'github.event.repository.private == true' not in WORKFLOW
        assert '--method PATCH' not in WORKFLOW

    @unittest.skipUnless(shutil.which("jq"), "jq is required for policy execution")
    def test_actual_policy_predicate(self):
        """Execute the workflow predicate for approved and denied audiences."""
        predicate = WORKFLOW.split("jq -e '", 1)[1].split("' \"$policy\"", 1)[0]
        private = dict(schema=1, enabled=True, visibility="private",
                       authorization_record="approval", distribution_review="review",
                       audience_review="audience")
        cases = [
            (private, True),
            (dict(private, visibility="public"), False),
            (dict(private, visibility="public", public_distribution_review=""), False),
            (dict(private, visibility="public", public_distribution_review="review"), True),
            (dict(private, visibility="internal"), False),
            (dict(private, enabled=False), False),
            (dict(private, authorization_record=""), False),
        ]
        for policy, allowed in cases:
            with self.subTest(policy=policy):
                result = subprocess.run(["jq", "-e", predicate],
                                        input=json.dumps(policy), text=True,
                                        capture_output=True, check=False)
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
