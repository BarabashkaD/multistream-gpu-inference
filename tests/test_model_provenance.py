# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
import hashlib
import pytest
from tools import export_models


def test_existing_weights_must_match(tmp_path, monkeypatch):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'good')
    monkeypatch.setattr(export_models, 'WEIGHTS_SHA256', hashlib.sha256(b'good').hexdigest())
    assert export_models.verified_weights(path) == path
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='SHA-256'):
        export_models.verified_weights(path)


def test_download_verified_before_install(tmp_path, monkeypatch):
    path = tmp_path / 'model.pt'
    monkeypatch.setattr(export_models, 'WEIGHTS_SHA256', hashlib.sha256(b'good').hexdigest())
    monkeypatch.setattr(export_models.urllib.request, 'urlretrieve', lambda url, out: out.write_bytes(b'bad'))
    with pytest.raises(ValueError, match='SHA-256'):
        export_models.verified_weights(path)
    assert not path.exists()
    assert not path.with_suffix('.download').exists()


def test_download_valid(tmp_path, monkeypatch):
    path = tmp_path / 'model.pt'
    monkeypatch.setattr(export_models, 'WEIGHTS_SHA256', hashlib.sha256(b'good').hexdigest())
    monkeypatch.setattr(export_models.urllib.request, 'urlretrieve', lambda url, out: out.write_bytes(b'good'))
    assert export_models.verified_weights(path).read_bytes() == b'good'
