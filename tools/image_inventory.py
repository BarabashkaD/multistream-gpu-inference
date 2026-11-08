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


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inventory():
    packages = []
    for dist in importlib.metadata.distributions():
        evidence = []
        for entry in dist.files or ():
            name = str(entry).lower()
            if any(word in name for word in ('license', 'licence', 'notice', 'copying', 'copyright')):
                path = Path(dist.locate_file(entry))
                if path.is_file():
                    evidence.append({'path': str(path.resolve()), 'sha256': digest(path)})
        packages.append({'name': dist.metadata['Name'], 'version': dist.version,
                         'license_expression': dist.metadata.get('License-Expression'),
                         'evidence': evidence})
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
    os_packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package}\t${Version}\n'], text=True)
    return {'schema': 1, 'scope': 'Installed Python/OS packages and selected bundled evidence; not a complete SBOM or redistribution approval',
            'python': sorted(packages, key=lambda p: p['name'].lower()),
            'ffmpeg_source_evidence': sources, 'debian_packages': os_packages.splitlines(),
            'debian_notices': '/usr/share/doc/*/copyright'}


if __name__ == '__main__':
    output = Path('/app/compliance/inventory.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory(), indent=2) + '\n')
