# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Guard the disabled, manual GHCR release's trust boundaries without Docker."""
import json
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
        """Require manual exact-source approval, private packages and no latest."""
        assert 'workflow_dispatch:' in WORKFLOW
        assert 'pull_request:' not in WORKFLOW
        assert 'git merge-base --is-ancestor "$revision" HEAD' in WORKFLOW
        assert 'git checkout --detach "$revision"' in WORKFLOW
        assert '.enabled == true' in WORKFLOW
        assert '.visibility == "private" and .repository.full_name == $repo' in WORKFLOW
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
