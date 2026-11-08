# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Negative checks for the CPU-only image contents checker."""
import hashlib
import pytest
from tools import check_image


@pytest.fixture
def image(tmp_path, monkeypatch):
    root = tmp_path / "app"
    data = tmp_path / "data"
    root.mkdir()
    data.mkdir()
    expected = {}
    for name in check_image.DOCUMENTS:
        content = name.encode()
        (root / name).write_bytes(content)
        expected[name] = hashlib.sha256(content).hexdigest()
    (data / "pexels-36304379.mp4").write_bytes(b"test video")
    monkeypatch.setattr(check_image, "CLIP_SHA256", hashlib.sha256(b"test video").hexdigest())
    (root / "models").mkdir()
    (root / "models/b1.onnx").write_bytes(b"model placeholder; shape checked during export")
    return root, data, expected


def test_valid_contents(image):
    root, data, expected = image
    check_image.check(root, data, "server", [1], expected)


def test_missing_notice_fails(image):
    root, data, expected = image
    (root / "NOTICE").unlink()
    with pytest.raises(FileNotFoundError):
        check_image.check(root, data, "server", [1], expected)


def test_changed_notice_fails(image):
    root, data, expected = image
    (root / "NOTICE").write_text("wrong")
    with pytest.raises(ValueError, match="legal document"):
        check_image.check(root, data, "server", [1], expected)


def test_corrupt_video_fails(image):
    root, data, expected = image
    (data / "pexels-36304379.mp4").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Video"):
        check_image.check(root, data, "server", [1], expected)


def test_missing_model_fails(image):
    root, data, expected = image
    with pytest.raises(ValueError, match="Missing model"):
        check_image.check(root, data, "server", [1, 8], expected)


def test_incomplete_expected_documents_fails(image):
    root, data, expected = image
    del expected["NOTICE"]
    with pytest.raises(ValueError, match="all project legal documents"):
        check_image.check(root, data, "server", [1], expected)


def test_empty_model_fails(image):
    root, data, expected = image
    (root / "models/b1.onnx").write_bytes(b"")
    with pytest.raises(ValueError, match="Missing model"):
        check_image.check(root, data, "server", [1], expected)


def test_instance_rejects_torch(image, monkeypatch):
    root, data, expected = image
    monkeypatch.setattr(check_image.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(ValueError, match="Torch"):
        check_image.check(root, data, "instance", [1], expected)


def test_instance_without_torch_passes(image, monkeypatch):
    root, data, expected = image
    monkeypatch.setattr(check_image.importlib.util, "find_spec", lambda name: None)
    check_image.check(root, data, "instance", [1], expected)
