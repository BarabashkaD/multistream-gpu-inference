# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
from email.message import Message
from pathlib import Path
import pytest
from tools import image_inventory as inventory


class Distribution:
    def __init__(self, root, name, version, files):
        self.root, self.version, self.files = root, version, list(files)
        self.metadata = Message()
        self.metadata['Name'] = name
        for path, data in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def locate_file(self, entry):
        return self.root / entry


def test_inline_bsd_and_renamed_native_binary(tmp_path):
    dist = Distribution(tmp_path, 'nvidia-ml-py', '13.615.71', {
        'pynvml.py': b'# Redistribution and use in source and binary forms\n',
        'vendor.libs/libcudart-hash.so.12.3.101': b'\x7fELFtest'})
    result = inventory.package_inventory(dist)
    assert result['evidence'][0]['kind'] == 'inline-license-header'
    assert result['evidence'][0]['sha256'] == inventory.digest(tmp_path / 'pynvml.py')
    assert result['native_binaries'][0]['size_bytes'] == 8
    assert result['payload_classification'] == 'contains-payload'


def test_metadata_only_is_based_on_actual_files(tmp_path):
    dist = Distribution(tmp_path, 'cuda-toolkit', '13.0.3.0', {
        'cuda_toolkit-13.0.3.0.dist-info/METADATA': b'Name: cuda-toolkit\n',
        'cuda_toolkit-13.0.3.0.dist-info/INSTALLER': b'pip'})
    assert inventory.package_inventory(dist)['payload_classification'] == 'metadata-only'
    (tmp_path / 'tool').write_bytes(b'payload')
    dist.files.append('tool')
    assert inventory.package_inventory(dist)['payload_classification'] == 'contains-payload'


def test_flatbuffers_full_text_required_only_for_audited_version(tmp_path):
    dist = Distribution(tmp_path, 'flatbuffers', '25.12.19', {
        'flatbuffers/__init__.py': b'# https://www.apache.org/licenses/LICENSE-2.0\n'})
    text = tmp_path / 'Apache-2.0.txt'
    with pytest.raises(FileNotFoundError):
        inventory.package_inventory(dist, text)
    text.write_text('Apache License full text test fixture')
    result = inventory.package_inventory(dist, text)
    assert len(result['evidence']) == 2
    assert result['evidence'][1]['kind'] == 'supplemental-Apache-2.0-full-text'
    dist.version = 'different-version'
    assert len(inventory.package_inventory(dist, text)['evidence']) == 1


def test_debian_source_and_notice_hashes(tmp_path):
    copyright_file = tmp_path / 'example/copyright'
    copyright_file.parent.mkdir()
    copyright_file.write_text('copyright fixture')
    result = inventory.debian_inventory('example:amd64\t1.2-3\tsource-example\t1.2-3\nmissing\t1\tmissing\t1\n', tmp_path)
    assert result[0]['source_package'] == 'source-example'
    assert result[0]['source_version'] == '1.2-3'
    assert result[0]['copyright_evidence']['sha256'] == inventory.digest(copyright_file)
    assert result[1]['copyright_evidence'] is None


def test_extensionless_bundled_elf_tool(tmp_path):
    dist = Distribution(tmp_path, 'triton', '3.8.0', {'triton/backends/nvidia/bin/ptxas': b'\x7fELFtool'})
    assert len(inventory.package_inventory(dist)['native_binaries']) == 1


def test_apache_text_is_official_unmodified():
    path = Path(__file__).resolve().parents[1] / 'compliance/licenses/Apache-2.0.txt'
    assert inventory.digest(path) == 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'


def test_missing_file_record_is_not_classified_as_metadata_only(tmp_path):
    dist = Distribution(tmp_path, 'unknown', '1', {})
    dist.files = None
    assert inventory.package_inventory(dist)['payload_classification'] == 'unknown-no-file-record'


def test_missing_declared_payload_and_empty_record_are_not_metadata_only(tmp_path):
    dist = Distribution(tmp_path, 'partial', '1', {'partial-1.dist-info/METADATA': b'Name: partial'})
    dist.files.append('missing.py')
    result = inventory.package_inventory(dist)
    assert result['payload_classification'] == 'unknown-missing-recorded-files'
    assert result['missing_files'] == ['missing.py']
    dist.files = []
    assert inventory.package_inventory(dist)['payload_classification'] == 'unknown-empty-file-record'
