#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Record installed packages and original license/source evidence, not legal approval."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def file_evidence(path, kind):
    return {'path': str(path.resolve()), 'sha256': digest(path), 'kind': kind}


def package_inventory(dist, supplemental=Path('/app/compliance/licenses/Apache-2.0.txt')):
    evidence, binaries, payload, metadata_evidence, missing = [], [], [], [], []
    recorded_files = dist.files
    for entry in recorded_files or ():
        name = str(entry).lower()
        path = Path(dist.locate_file(entry))
        if '.dist-info/' not in name:
            payload.append(str(entry))
        if not path.is_file():
            missing.append(str(entry))
            continue
        if name.endswith('.dist-info/metadata'):
            metadata_evidence.append(file_evidence(path, 'package-metadata-not-license-clearance'))
        if any(word in name for word in ('license', 'licence', 'notice', 'copying', 'copyright')):
            evidence.append(file_evidence(path, 'named-license-or-notice'))
        elif name.endswith(('.py', '.h', '.hpp')):
            with path.open('rb') as handle:
                header = handle.read(8192).decode('utf-8', errors='replace')
            if ('Redistribution and use in source and binary forms' in header
                    or 'apache.org/licenses/LICENSE-2.0' in header):
                evidence.append(file_evidence(path, 'inline-license-header'))
        # Include native libraries and extensionless bundled tools (e.g. Triton).
        if '.so' in path.name or '/bin/' in name or name.endswith(('.dll', '.exe', '.dylib')):
            with path.open('rb') as handle:
                magic = handle.read(4)
            if magic == b'\x7fELF' or magic[:2] == b'MZ' or name.endswith('.dylib'):
                item = file_evidence(path, 'native-binary')
                item['size_bytes'] = path.stat().st_size
                binaries.append(item)
    if dist.metadata['Name'].lower() == 'flatbuffers' and dist.version == '25.12.19':
        evidence.append(file_evidence(supplemental, 'supplemental-Apache-2.0-full-text'))
    return {'name': dist.metadata['Name'], 'version': dist.version,
            'license_expression': dist.metadata.get('License-Expression'),
            'license_metadata': dist.metadata.get('License'),
            'classifiers': dist.metadata.get_all('Classifier') or [],
            'payload_classification': ('unknown-no-file-record' if recorded_files is None else
                                       'unknown-empty-file-record' if not recorded_files else
                                       'unknown-missing-recorded-files' if missing else
                                       'metadata-only' if not payload else 'contains-payload'),
            'metadata_evidence': metadata_evidence, 'missing_files': missing,
            'evidence': evidence, 'native_binaries': binaries,
            'review_status': 'not-cleared; evidence and binary presence do not establish rights'}


def debian_inventory(output, doc_root=Path('/usr/share/doc')):
    packages = []
    for line in output.splitlines():
        binary, version, source, source_version = line.split('\t')
        notice = doc_root / binary.split(':')[0] / 'copyright'
        packages.append({'package': binary, 'version': version, 'source_package': source,
                         'source_version': source_version,
                         'copyright_evidence': file_evidence(notice, 'debian-copyright') if notice.is_file() else None})
    return packages


def interpreter_inventory():
    executable = Path(sys.executable).resolve()
    license_file = Path(sysconfig.get_path('stdlib')) / 'LICENSE.txt'
    return {'version': sys.version, 'prefix': sys.base_prefix,
            'executable': file_evidence(executable, 'python-executable'),
            'license_evidence': file_evidence(license_file, 'python-license') if license_file.is_file() else None,
            'scope': 'Interpreter identity only; not a complete inventory of base-image /usr/local components'}


def inventory():
    packages = [package_inventory(dist) for dist in importlib.metadata.distributions()]
    sources = []
    for path in Path('/opt/venv/external/ffmpeg/src').glob('*'):
        if path.is_file():
            sources.append({'path': str(path), 'sha256': digest(path)})
    if not any(item['path'].endswith('.tar.xz') for item in sources):
        raise RuntimeError('Expected bundled FFmpeg source archive not found')
    readme = Path('/opt/venv/external/ffmpeg/readme.txt')
    if not readme.is_file():
        raise RuntimeError('Expected bundled FFmpeg readme not found')
    sources.append({'path': str(readme), 'sha256': digest(readme)})
    os_packages = subprocess.check_output(['dpkg-query', '-W', '-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n'], text=True)
    return {'schema': 2, 'scope': 'Installed Python/OS packages and selected bundled evidence; not a complete SBOM or redistribution approval',
            'python': sorted(packages, key=lambda p: p['name'].lower()),
            'ffmpeg_source_evidence': sources, 'debian_packages': debian_inventory(os_packages),
            'python_interpreter': interpreter_inventory(),
            'debian_notices': '/usr/share/doc/*/copyright'}


if __name__ == '__main__':
    output = Path('/app/compliance/inventory.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory(), indent=2) + '\n')
