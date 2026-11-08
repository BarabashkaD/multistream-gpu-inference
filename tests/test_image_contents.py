# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Negative checks for the CPU-only image contents checker."""
import hashlib
import json
import pytest
from tools import check_image, export_models


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
    (root / "models/manifest.json").write_text(json.dumps({"schema": 1, "weights": {"url": export_models.WEIGHTS_URL, "sha256": export_models.WEIGHTS_SHA256},
        "export": {"width": 640, "height": 384, "opset": 12, "dynamic": False, "simplify": False, "nms": False, "device": "cpu"},
        "packages": {"ultralytics": "test", "onnx": "test", "torch": "test"}, "models": {
        "b1.onnx": {"sha256": check_image.digest(root / "models/b1.onnx"), "shape": [1, 300, 6]}}}))
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
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        check_image.check(root, data, "server", [1, 8], expected)


def test_incomplete_expected_documents_fails(image):
    root, data, expected = image
    del expected["NOTICE"]
    with pytest.raises(ValueError, match="all project legal documents"):
        check_image.check(root, data, "server", [1], expected)


def test_empty_model_fails(image):
    root, data, expected = image
    (root / "models/b1.onnx").write_bytes(b"")
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
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


def test_model_hash_mismatch(image):
    root, data, expected = image
    (root / "models/b1.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        check_image.check(root, data, "server", [1], expected)


def test_extra_model_rejected(image):
    root, data, expected = image
    (root / "models/b4.onnx").write_bytes(b"stale")
    with pytest.raises(ValueError, match="Unrecorded"):
        check_image.check(root, data, "server", [1], expected)


@pytest.mark.parametrize("field", ["weights", "export", "schema", "packages"])
def test_missing_provenance_rejected(image, field):
    root, data, expected = image
    path = root / "models/manifest.json"
    manifest = json.loads(path.read_text())
    del manifest[field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        check_image.check(root, data, "server", [1], expected)


@pytest.mark.parametrize("packages", [{}, {"ultralytics": "test"}, {"ultralytics": "test", "onnx": "test", "torch": None}])
def test_incomplete_package_provenance(image, packages):
    root, data, expected = image
    path = root / "models/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["packages"] = packages
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="export provenance"):
        check_image.check(root, data, "server", [1], expected)
