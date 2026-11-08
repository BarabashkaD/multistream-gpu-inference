# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
import copy
import json
from pathlib import Path
import pytest
from tools.gpu_smoke import validate_images, compose_spec, check_summaries, smoke
from tools.verify_candidate_policy import validate

I = 'ghcr.io/example/instance@sha256:' + 'a' * 64
S = 'ghcr.io/example/server@sha256:' + 'b' * 64
R = 'c' * 40


def test_immutable_refs_and_revision_required():
    validate_images(I, S, R)
    for values in [(I + ':latest', S, R), (I, S, 'main'), (I, I, R), ('ubuntu:latest', S, R)]:
        with pytest.raises(ValueError):
            validate_images(*values)


def test_compose_uses_candidates_without_host_namespace_or_network():
    spec = compose_spec(I, S)
    assert spec['services']['server']['image'] == S
    assert spec['services']['server']['cap_add'] == ['SYS_PTRACE']
    for name, service in spec['services'].items():
        assert service['network_mode'] == 'none'
        assert service['pull_policy'] == 'never'
        assert service['volumes'] == ['ipc:/run/traffic']
        assert 'build' not in service and 'privileged' not in service
        if name != 'server':
            assert service['image'] == I
            assert service['pid'] == 'service:server'
            assert '--no-display' in service['command']


def test_frame_detection_and_pool_contract():
    results = {name: {'counts': {'tracked': 611, 'rows': 1834}, 'pool': {'free': 4, 'size': 4}}
               for name in ('instance0', 'instance1')}
    results['server'] = {'counts': {'inferred': 1222}, 'dead': [], 'streams': [0, 1]}
    check_summaries(results)
    for path, value in [(('instance0', 'counts', 'tracked'), 610),
                        (('instance1', 'counts', 'rows'), 1833),
                        (('instance1', 'pool', 'free'), 3),
                        (('server', 'counts', 'inferred'), 1221)]:
        changed = copy.deepcopy(results)
        changed[path[0]][path[1]][path[2]] = value
        with pytest.raises(ValueError):
            check_summaries(changed)


def test_policy_disabled_and_digest_bound():
    policy = json.loads((Path(__file__).resolve().parents[1] / 'compliance/candidate-policy.json').read_text())
    with pytest.raises(ValueError, match='disabled'):
        validate(policy, I, S, R)
    approved = {'schema': 1, 'gpu_verification_enabled': True, 'images': {'instance': I, 'server': S},
                'revision': R, 'authorization_record': 'owner-approved disposable host and candidates'}
    validate(approved, I, S, R)
    with pytest.raises(ValueError):
        validate(approved, I, S.replace('b' * 64, 'd' * 64), R)
    approved['authorization_record'] = ''
    with pytest.raises(ValueError):
        validate(approved, I, S, R)


def test_failure_writes_report_even_without_docker(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError('docker unavailable')
    monkeypatch.setattr('tools.gpu_smoke.subprocess.check_output', missing)
    monkeypatch.setattr('tools.gpu_smoke.subprocess.run', missing)
    with pytest.raises(FileNotFoundError):
        smoke(I, S, R, tmp_path)
    report = json.loads((tmp_path / 'gpu-report.json').read_text())
    assert report['status'] == 'failed' and report['images'] == {'instance': I, 'server': S}
    assert 'cleanup_error' in report


def test_evidence_collector_copies_only_selected_text(monkeypatch, tmp_path):
    from tools import collect_image_evidence as evidence
    calls = []
    def fake(*args):
        calls.append(args)
        if args[1:3] == ('image', 'inspect'):
            return json.dumps([{'Id': 'sha256:' + 'e' * 64, 'Config': {'Labels': {
                'org.opencontainers.image.revision': R}}, 'RepoDigests': []}])
        if args[1] == 'create':
            return 'stopped-container'
        return ''
    monkeypatch.setenv('EXPECTED_REVISION', R)
    monkeypatch.setattr(evidence, 'run', fake)
    evidence.collect(tmp_path)
    manifest = json.loads((tmp_path / 'images.json').read_text())
    assert set(manifest['images']) == {'instance', 'server'}
    assert manifest['revision'] == R
    copied = [call[2] for call in calls if call[1] == 'cp']
    assert len(copied) == 12
    assert sum(path.endswith('/manifest.json') for path in copied) == 1
    assert all(path.endswith(('.json', '.md', '.txt', '/LICENSE', '/NOTICE')) for path in copied)
    assert all(call[1] not in ('run', 'push', 'save') for call in calls)


def test_standalone_probe_matches_compose_capability(monkeypatch, tmp_path):
    """Exercise the real probe command without starting Docker or a GPU."""
    calls = []

    def fake_run(args, **kwargs):
        """Supply image metadata and stop after capturing the probe command."""
        calls.append(args)
        if args[:3] == ['docker', 'image', 'inspect']:
            return json.dumps([{'Id': 'sha256:' + 'e' * 64, 'Config': {
                'Labels': {'org.opencontainers.image.revision': R}}}])
        if args[:2] == ['docker', 'run']:
            raise RuntimeError('probe captured')
        return ''

    def fake_cleanup(*args, **kwargs):
        """Avoid invoking Docker during the harness cleanup."""
        return None

    monkeypatch.setattr('tools.gpu_smoke.run', fake_run)
    monkeypatch.setattr('tools.gpu_smoke.subprocess.run', fake_cleanup)
    with pytest.raises(RuntimeError, match='probe captured'):
        smoke(I, S, R, tmp_path)
    probe, = [args for args in calls if args[:2] == ['docker', 'run']]
    assert probe[-5:] == ['--cap-add', 'SYS_PTRACE', S, 'python', 'tools/run_gpu_tests.py']
    assert probe.count('--cap-add') == 1
    assert probe[probe.index('--network') + 1] == 'none'
    assert probe[probe.index('--gpus') + 1] == 'all'
    assert '--rm' in probe
    assert not {'--privileged', '--security-opt', '--pid', '--ipc', '--volume', '-v', '--mount'} & set(probe)
