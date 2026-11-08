# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Exercise archive splitting and checksums without Docker or publication."""
import gzip
from tools.package_image_release import PartsWriter, sha256, write_checksums


def test_bounded_parts_restore_exact_stream(tmp_path):
    """Parts preserve gzip bytes and never exceed the configured limit."""
    payload = bytes(range(256)) * 100
    writer = PartsWriter(tmp_path, limit=50)
    with gzip.GzipFile(fileobj=writer, mode='wb', mtime=0) as stream:
        stream.write(payload)
    writer.close()
    parts = sorted(tmp_path.glob('docker-images.tar.gz.part*'))
    assert len(parts) > 1
    assert all(0 < p.stat().st_size <= 50 for p in parts)
    assert gzip.decompress(b''.join(p.read_bytes() for p in parts)) == payload


def test_checksums_include_every_asset(tmp_path):
    """Checksums cover saved files but not themselves."""
    (tmp_path / 'a').write_bytes(b'abc')
    (tmp_path / 'b').write_bytes(b'def')
    write_checksums(tmp_path)
    assert (tmp_path / 'SHA256SUMS').read_text() == ''.join(f'{sha256(tmp_path / n)}  {n}\n' for n in ('a', 'b'))


def test_invalid_part_limit(tmp_path):
    """Invalid chunk sizes fail rather than looping forever."""
    import pytest
    with pytest.raises(ValueError):
        PartsWriter(tmp_path, limit=0)


def test_package_checks_identity_and_emits_verified_manifest(tmp_path, monkeypatch):
    """Mock Docker transport while testing the complete local packaging path."""
    import io
    import json
    from tools import package_image_release as release
    revision = 'a' * 40
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'images.json').write_text(json.dumps({'revision': revision, 'images': {
        target: {'id': 'sha256:' + target} for target in ('instance', 'server')}}))
    monkeypatch.setenv('EXPECTED_REVISION', revision)
    monkeypatch.setenv('CI_RUN_URL', 'https://github.com/example/repo/actions/runs/1')

    def inspect(command, text):
        """Return distinct Docker image identities with matching labels."""
        target = command[-1].split(':')[-1]
        return json.dumps([{'Id': 'sha256:' + target, 'Os': 'linux', 'Architecture': 'amd64',
                            'Config': {'Labels': {'org.opencontainers.image.revision': revision}}}])

    class SavedProcess:
        """Stand in for a successful Docker save stream."""
        def __init__(self, *args, **kwargs):
            """Prepare an in-memory archive transport."""
            self.stdout = io.BytesIO(b'docker archive test data')
        def __enter__(self):
            """Return this managed mock process."""
            return self
        def __exit__(self, *args):
            """Do not suppress errors."""
            return False
        def wait(self):
            """Report successful export."""
            return 0

    monkeypatch.setattr(release.subprocess, 'check_output', inspect)
    monkeypatch.setattr(release.subprocess, 'Popen', SavedProcess)
    output = tmp_path / 'assets'
    release.package(output, evidence)
    manifest = json.loads((output / 'release-manifest.json').read_text())
    assert manifest['source_revision'] == revision
    assert manifest['images']['server']['id'] == 'sha256:server'
    parts = manifest['parts']
    assert all(sha256(output / p['name']) == p['sha256'] for p in parts)
    assert gzip.decompress(b''.join((output / p['name']).read_bytes() for p in parts)) == b'docker archive test data'
    import pytest
    with pytest.raises(ValueError, match='already exist'):
        release.package(output, evidence)
    monkeypatch.setenv('EXPECTED_REVISION', 'b' * 40)
    with pytest.raises(ValueError, match='revision mismatch'):
        release.package(tmp_path / 'wrong', evidence)
