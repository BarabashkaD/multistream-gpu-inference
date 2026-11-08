#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Save both tested local images as a checksummed, split release archive."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

PART_SIZE = 1900 * 1024 * 1024


class PartsWriter:
    """Write a stream into bounded, lexically ordered release assets."""

    def __init__(self, directory, limit=PART_SIZE):
        """Start a new stream in an existing empty output directory."""
        if limit <= 0:
            raise ValueError('part limit must be positive')
        self.directory, self.limit = Path(directory), limit
        self.stream, self.size, self.number = None, 0, 0

    def write(self, data):
        """Write bytes, opening the next part at the exact size boundary."""
        total = len(data)
        while data:
            if self.stream is None or self.size == self.limit:
                self.close()
                self.number += 1
                if self.number > 900:
                    raise ValueError('too many release parts')
                self.stream = (self.directory / f'docker-images.tar.gz.part{self.number:04d}').open('xb')
                self.size = 0
            count = min(len(data), self.limit - self.size)
            self.stream.write(data[:count])
            self.size += count
            data = data[count:]
        return total

    def flush(self):
        """Flush an open part when gzip requests it."""
        if self.stream:
            self.stream.flush()

    def close(self):
        """Close the current part without removing any saved bytes."""
        if self.stream:
            self.stream.close()
            self.stream = None


def sha256(path):
    """Hash a file without loading a multi-gigabyte asset into memory."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_checksums(directory):
    """Write checksums of all assets except the checksum file itself."""
    files = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (Path(directory) / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files))


def package(output=Path('release-assets'), evidence=Path('image-evidence')):
    """Package exactly the revision-checked images without publishing them."""
    revision = os.environ['EXPECTED_REVISION']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('full source SHA required')
    output = Path(output)
    if output.exists():
        raise ValueError('output must not already exist')
    metadata = json.loads((Path(evidence) / 'images.json').read_text())
    if metadata['revision'] != revision:
        raise ValueError('evidence revision mismatch')
    images = {}
    for target in ('instance', 'server'):
        tag = f'traffic-v3:{target}'
        info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', tag], text=True))[0]
        if info['Id'] != metadata['images'][target]['id']:
            raise ValueError('image changed after evidence collection')
        if info['Config']['Labels']['org.opencontainers.image.revision'] != revision:
            raise ValueError('image revision mismatch')
        if (info['Os'], info['Architecture']) != ('linux', 'amd64'):
            raise ValueError('unsupported image platform')
        images[target] = {'tag': tag, 'id': info['Id'], 'platform': 'linux/amd64'}
    output.mkdir()
    writer = PartsWriter(output)
    try:
        with subprocess.Popen(['docker', 'save', 'traffic-v3:instance', 'traffic-v3:server'], stdout=subprocess.PIPE) as process:
            with gzip.GzipFile(fileobj=writer, mode='wb', mtime=0, compresslevel=1) as compressed:
                while chunk := process.stdout.read(1024 * 1024):
                    compressed.write(chunk)
            if process.wait() != 0:
                raise RuntimeError('docker save failed; release must not be published')
    finally:
        writer.close()
    with tarfile.open(output / 'image-evidence.tar.gz', 'w:gz') as archive:
        archive.add(evidence, arcname='image-evidence')
    manifest = {'schema': 1, 'source_revision': revision, 'images': images,
                'ci_run_url': os.environ['CI_RUN_URL'], 'gpu_runtime': 'not tested for these images',
                'distribution_review': 'incomplete; private archival prerelease only',
                'parts': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                          for p in sorted(output.glob('docker-images.tar.gz.part*'))]}
    (output / 'release-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'RESTORE.md').write_text('''# Private CI image archive\n\nCPU checks passed for these images. GPU execution and third-party redistribution review are not complete. Do not treat this archive as a publicly cleared release.\n\nDownload ALL assets into one directory. On Linux/WSL:\n\n```bash\nset -euo pipefail\nsha256sum -c SHA256SUMS\ncat docker-images.tar.gz.part* | gzip -dc | docker load\n```\n\nCompare `docker image inspect traffic-v3:instance traffic-v3:server` IDs with release-manifest.json. These tags can replace existing local tags; preserve any local images/tags you need before loading. Shared layers occur once in the combined archive.\n\nOriginal licenses and source evidence are inside the images and image-evidence.tar.gz. This is private archival storage, not permission to redistribute proprietary/third-party components externally. See docs/distribution-review.md in the exact source revision for unresolved questions.\n''')
    write_checksums(output)


if __name__ == '__main__':
    package()
