#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Copy selected text/JSON evidence from stopped local images; never export images."""
import json
import os
from pathlib import Path
import subprocess


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'schema': 1, 'revision': os.environ['EXPECTED_REVISION'], 'images': {}}
    for target in ('instance', 'server'):
        image = 'traffic-v3:' + target
        info = json.loads(run('docker', 'image', 'inspect', image))[0]
        revision = (info['Config']['Labels'] or {}).get('org.opencontainers.image.revision')
        if revision != metadata['revision']:
            raise ValueError(f'{image}: revision mismatch')
        metadata['images'][target] = {'id': info['Id'], 'revision': revision,
                                      'repo_digests': info.get('RepoDigests', [])}
        folder = output / target
        folder.mkdir(exist_ok=True)
        cid = run('docker', 'create', image)
        try:
            files = ['/app/compliance/inventory.json', '/app/LICENSE', '/app/NOTICE',
                     '/app/THIRD_PARTY_NOTICES.md', '/opt/venv/external/ffmpeg/readme.txt']
            if target == 'server':
                files.append('/app/models/manifest.json')
                files.append('/app/compliance/licenses/Apache-2.0.txt')
            for path in files:
                run('docker', 'cp', f'{cid}:{path}', str(folder / Path(path).name))
        finally:
            run('docker', 'rm', cid)
    (output / 'images.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    collect(Path('image-evidence'))
