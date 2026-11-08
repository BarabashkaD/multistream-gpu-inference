# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Fail closed unless a reviewed policy authorizes these exact test candidates."""
import json
import os
from pathlib import Path
from tools.gpu_smoke import validate_images


def validate(policy, instance, server, revision):
    validate_images(instance, server, revision)
    if policy.get('schema') != 1 or policy.get('gpu_verification_enabled') is not True:
        raise ValueError('GPU verification is disabled pending owner review')
    expected = {'instance': instance, 'server': server}
    if policy.get('images') != expected or policy.get('revision') != revision:
        raise ValueError('Candidate digests/revision differ from reviewed policy')
    if not policy.get('authorization_record', '').strip():
        raise ValueError('Missing owner authorization record for host and candidate access')
    # This authorizes testing only; it is never a distribution-license clearance.


if __name__ == '__main__':
    validate(json.loads(Path('compliance/candidate-policy.json').read_text()),
             os.environ['INSTANCE_IMAGE'], os.environ['SERVER_IMAGE'], os.environ['REVISION'])
