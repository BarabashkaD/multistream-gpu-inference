# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Exercise the actual standalone bootstrap shell without network or Docker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/ghcr-bootstrap.yml").read_text()


def scripts():
    # No YAML dependency required in the stock CI Python.
    result = []
    for part in WORKFLOW.split("        run: |\n")[1:]:
        lines = []
        for line in part.splitlines():
            if line and not line.startswith("          "):
                break
            lines.append(line[10:])
        result.append("\n".join(lines))
    return result


MOCKS = r"""
gh() {
  if [[ "$*" == *commits/main* ]]; then printf '%s' "$LIVE_SHA"
  elif [[ "$*" == *contents/compliance* ]]; then printf '%s' "$POLICY" | base64
  else printf '{"visibility":"%s","repository":{"full_name":"%s"}}' "$AFTER" "$LINK"
  fi
}
docker() {
  printf '%s\n' "$*" >> "$TRACE"
  if [[ "$1" = import || "$1" = login ]]; then cat >/dev/null; fi
}
curl() {
  local output=""
  while (($#)); do
    if [[ "$1" = --output ]]; then output="$2"; shift; fi
    shift
  done
  printf '{"visibility":"%s","repository":{"full_name":"%s"}}' "$BEFORE" "$LINK" > "$output"
  if [[ "$STATUS" = mixed ]]; then
    if [[ "$output" = *server* ]]; then printf 404; else printf 200; fi
  else printf '%s' "$STATUS"; fi
}
"""


@unittest.skipUnless(shutil.which("jq"), "jq required")
class BootstrapContract(unittest.TestCase):
    def run_script(self, script, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace"
            env = dict(os.environ, GH_REPO="BarabashkaD/multistream-gpu-inference",
                       GH_TOKEN="test", GITHUB_ACTOR="test", GITHUB_REF="refs/heads/main",
                       GITHUB_SHA="a"*40, LIVE_SHA="a"*40, GITHUB_RUN_ID="123",
                       GITHUB_RUN_ATTEMPT="1", GITHUB_OUTPUT=str(Path(directory)/"output"),
                       GITHUB_STEP_SUMMARY=str(Path(directory)/"summary"),
                       STATUS="404", BEFORE="private", AFTER="private",
                       LINK="BarabashkaD/multistream-gpu-inference", TRACE=str(trace),
                       POLICY=json.dumps(dict(schema=1, enabled=False,
                                              bootstrap_enabled=True, bootstrap_review="record")))
            env.update(overrides)
            result = subprocess.run(["bash", "-c", MOCKS + script], env=env,
                                    cwd=directory, capture_output=True, text=True)
            lines = trace.read_text().splitlines() if trace.exists() else []
            return result, lines

    def test_authorization(self):
        for policy, allowed in [
            (dict(schema=1, enabled=False, bootstrap_enabled=True, bootstrap_review="record"), True),
            (dict(schema=1, enabled=True, bootstrap_enabled=False, bootstrap_review="record"), False),
            (dict(schema=1, bootstrap_enabled=True, bootstrap_review=""), False),
            (dict(schema=1, bootstrap_enabled=True, bootstrap_review="  "), False),
            (dict(schema=1, bootstrap_enabled=True, bootstrap_review=None), False),
            (dict(schema=2, bootstrap_enabled=True, bootstrap_review="record"), False),
        ]:
            with self.subTest(policy=policy):
                result, _ = self.run_script(scripts()[0], POLICY=json.dumps(policy))
                self.assertEqual(result.returncode == 0, allowed, result.stderr)

    def test_transport(self):
        for status, before, after, linked, allowed, pushes in [
            ("404", "private", "private", True, True, 2),
            ("mixed", "private", "private", True, True, 1),
            ("200", "private", "private", True, True, 0),
            ("403", "private", "private", True, False, 0),
            ("500", "private", "private", True, False, 0),
            ("200", "public", "public", True, False, 0),
            ("200", "private", "private", False, False, 0),
            ("404", "private", "public", True, False, 1),
            ("404", "private", "private", False, False, 1),
        ]:
            with self.subTest(status=status, before=before, after=after, linked=linked):
                result, lines = self.run_script(scripts()[1], STATUS=status, BEFORE=before,
                                               AFTER=after, LINK="BarabashkaD/multistream-gpu-inference"
                                               if linked else "other/repo")
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
                pushed = [line for line in lines if line.startswith("push ")]
                self.assertEqual(len(pushed), pushes)
                self.assertTrue(all(":bootstrap-empty-123-1" in line for line in pushed))
                self.assertTrue(all(line.split()[0] in ("login", "logout", "import", "push")
                                    for line in lines))

    def test_source_gates_before_write(self):
        for override in [dict(LIVE_SHA="b"*40), dict(GITHUB_SHA="bad"),
                         dict(GITHUB_REF="refs/heads/dev"), dict(GH_REPO="other/repo")]:
            for script in scripts():
                result, lines = self.run_script(script, **override)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(lines)

    def test_no_application_publication_path(self):
        for forbidden in ["actions/checkout", "upload-artifact", "download-artifact",
                          "docker build", "docker run", "docker load", "docker save",
                          "gh release", "contents: write", "--method PATCH"]:
            self.assertNotIn(forbidden, WORKFLOW)
        self.assertIn("group: ghcr-release", WORKFLOW)
        self.assertIn("github.ref == 'refs/heads/main' && inputs.confirm_empty_markers", WORKFLOW)
        self.assertIn("if: needs.authorize.outputs.approved == 'true'", WORKFLOW)
        self.assertIn("tar -cf - --files-from /dev/null | docker import", WORKFLOW)
        self.assertEqual(WORKFLOW.count("packages: write"), 1)
        self.assertIn("default: false", WORKFLOW)
        policy = json.loads((ROOT / "compliance/ghcr-release-policy.json").read_text())
        self.assertFalse(policy["enabled"])
        self.assertFalse(policy["bootstrap_enabled"])
        self.assertFalse(policy["bootstrap_review"])
