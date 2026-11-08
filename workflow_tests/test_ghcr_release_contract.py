# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Exercise release transport failures without credentials, Docker or network."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/ghcr-release.yml').read_text()
SHA = 'a' * 40
REPO = 'BarabashkaD/multistream-gpu-inference'


def script_after(marker):
    """Extract a workflow's literal bash block without a YAML dependency."""
    tail = WORKFLOW.split(marker, 1)[1].split('        run: |\n', 1)[1]
    lines = []
    for line in tail.splitlines():
        if line and not line.startswith('          '):
            break
        lines.append(line[10:])
    return '\n'.join(lines)


class ReleaseContract(unittest.TestCase):
    """Check runnable transport and least-privilege workflow structure."""

    def test_manual_exact_main_only(self):
        """Merges and PRs never publish; dispatch pins the checked source."""
        self.assertIn('workflow_dispatch:', WORKFLOW)
        self.assertNotIn('pull_request:', WORKFLOW)
        self.assertNotIn('\n  push:', WORKFLOW)
        self.assertIn("if: github.ref == 'refs/heads/main'", WORKFLOW)
        self.assertIn('git rev-parse HEAD', WORKFLOW)
        self.assertIn('git/ref/heads/main', WORKFLOW)
        self.assertNotIn('ghcr-release-policy', WORKFLOW)
        self.assertNotIn('bootstrap', WORKFLOW)
        self.assertIn('cancel-in-progress: false', WORKFLOW)

    def test_privileged_jobs_only_transport(self):
        """Untrusted builds/tests run without package or content-write grants."""
        build, privileged = WORKFLOW.split('  publish:', 1)
        self.assertNotIn('packages: write', build)
        self.assertNotIn('contents: write', build)
        for forbidden in ('actions/checkout@', 'docker run ', 'docker build ', 'bash tools/'):
            self.assertNotIn(forbidden, privileged)
        self.assertIn('packages: write', privileged)
        self.assertIn('contents: write', privileged)
        self.assertIn('needs: [build, publish]', privileged)
        self.assertIn('persist-credentials: false', build)
        self.assertIn('tools/run_cpu_tests.py', build)
        self.assertIn('tools/collect_image_evidence.py', build)

    def test_release_is_source_bound_and_not_gpu_qualified(self):
        """Verify tag/assets before promotion; provide digest pulls, not archives."""
        self.assertIn('docker pull "$digest"', WORKFLOW)
        self.assertIn('sha256sum -c SHA256SUMS', WORKFLOW)
        self.assertIn('--verify-tag', WORKFLOW)
        self.assertIn('-f "sha=$SOURCE_SHA"', WORKFLOW)
        self.assertIn('.object.type == "commit" and .object.sha == $sha', WORKFLOW)
        self.assertLess(WORKFLOW.index('git/ref/tags/$RELEASE_TAG'), WORKFLOW.index('gh release edit'))
        self.assertIn('--latest=false', WORKFLOW)
        self.assertNotIn(':latest', WORKFLOW)
        self.assertIn('GPU runtime for these exact published images: NOT TESTED', WORKFLOW)
        self.assertIn('instance=$instance_visibility; server=$server_visibility', WORKFLOW)
        create = next(line for line in WORKFLOW.splitlines() if 'gh release create' in line)
        self.assertNotIn('images.tar', create)

    def test_all_literal_scripts_parse(self):
        """Catch shell syntax errors even without running Docker operations."""
        for block in WORKFLOW.split('        run: |\n')[1:]:
            lines = []
            for line in block.splitlines():
                if line and not line.startswith('          '):
                    break
                lines.append(line[10:])
            result = subprocess.run(['bash', '-n'], input='\n'.join(lines),
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_source_selection(self):
        """Reject dispatch/repository/source mismatches before a build."""
        script = script_after('      - name: Select exact dispatched main source')
        mocks = 'git() { echo "$CHECKOUT"; }; gh() { echo "$MAIN"; };\n'
        for checkout, main, repo, allowed in (
                (SHA, SHA, REPO, True), ('bad', SHA, REPO, False),
                (SHA, 'bad', REPO, False), (SHA, SHA, 'other/repo', False)):
            with self.subTest(checkout=checkout, main=main, repo=repo), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'output'
                env = dict(os.environ, CHECKOUT=checkout, MAIN=main, GH_REPO=repo,
                           GITHUB_SHA=SHA, GITHUB_OUTPUT=str(output), GITHUB_RUN_ID='1',
                           GITHUB_RUN_ATTEMPT='2')
                result = subprocess.run(['bash', '-c', mocks + script], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
                if allowed:
                    self.assertIn(f'revision={SHA}', output.read_text())
                    self.assertIn(f'tag=ci-{SHA[:12]}-1-2', output.read_text())

    @unittest.skipUnless(shutil.which('jq'), 'jq required for shell checks')
    def test_actual_publish_script(self):
        """First push needs no existing namespace; failures prevent completion."""
        script = script_after('      - name: Load checked images, push and verify registry digests')
        mocks = r'''
docker() {
  printf '%s\n' "$*" >> "$TRACE"
  case "$1" in
    login) cat >/dev/null;;
    push) [[ "$CASE" != push_failure ]];;
    inspect)
      case "$3" in
        *RepoDigests*)
          if [[ "$CASE" = bad_digest ]]; then echo invalid; else printf '%s@sha256:%064d\n' "${4%:*}" 1; fi;;
        *revision*) if [[ "$CASE" = wrong_revision ]]; then echo wrong; else echo "$SOURCE_SHA"; fi;;
        *) if [[ "$4" = *@* && "$CASE" = wrong_image ]]; then echo wrong; else echo sha256:checked; fi;;
      esac;;
  esac
}
gh() {
  printf 'api %s\n' "$*" >> "$TRACE"
  [[ "$CASE" != api_failure ]] || return 1
  repo="$GH_REPO"; visibility=private
  [[ "$CASE" != wrong_repo ]] || repo=another/repo
  [[ "$CASE" != public ]] || visibility=public
  [[ "$CASE" != bad_visibility ]] || visibility=internal
  printf '{"repository":{"full_name":"%s"},"visibility":"%s"}\n' "$repo" "$visibility"
}
'''
        cases = ('first_push', 'public', 'push_failure', 'bad_digest', 'wrong_revision',
                 'wrong_image', 'api_failure', 'wrong_repo', 'bad_visibility', 'bad_archive')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                transfer = root / 'transfer'
                evidence = transfer / 'image-evidence'
                evidence.mkdir(parents=True)
                (evidence / 'images.json').write_text(json.dumps(dict(
                    revision=SHA, images={target: dict(id='sha256:checked') for target in ('instance', 'server')})))
                (transfer / 'images.tar').write_bytes(b'test archive')
                checksum = hashlib.sha256(b'test archive').hexdigest()
                if case == 'bad_archive':
                    checksum = '0' * 64
                (transfer / 'SHA256SUMS').write_text(f'{checksum}  images.tar\n')
                trace = root / 'trace'
                env = dict(os.environ, CASE=case, TRACE=str(trace), GH_REPO=REPO,
                           GH_TOKEN='test', GITHUB_ACTOR='test', SOURCE_SHA=SHA, RELEASE_TAG='ci-test')
                result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + mocks + script],
                                        cwd=root, env=env, capture_output=True, text=True)
                allowed = case in ('first_push', 'public')
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
                self.assertEqual((root / 'release-assets/SHA256SUMS').exists(), allowed)
                lines = trace.read_text().splitlines()
                self.assertIn('logout ghcr.io', lines)
                if allowed:
                    self.assertEqual(sum(line.startswith('push ') for line in lines), 2)
                    self.assertLess(next(i for i, line in enumerate(lines) if line.startswith('push ')),
                                    next(i for i, line in enumerate(lines) if line.startswith('api ')))
                    for target in ('instance', 'server'):
                        self.assertRegex((root / f'release-assets/{target}.ref').read_text(),
                                         rf'^ghcr.io/barabashkad/multistream-gpu-inference-{target}@sha256:[0-9a-f]{{64}}\n$')
                if case in ('wrong_revision', 'bad_archive'):
                    self.assertFalse(any(line.startswith('push ') for line in lines))


if __name__ == '__main__':
    unittest.main()
