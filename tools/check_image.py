# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""CPU-only image contents smoke check, not a third-party license approval."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

DOCUMENTS = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
CLIP_SHA256 = "d105fb6777e82caafadc3763098f533864cf294448d41ac195e17c69e068b278"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check(root, data, target, sizes, expected_documents):
    if set(expected_documents) != set(DOCUMENTS):
        raise ValueError("Expected hashes must cover all project legal documents")
    for name in DOCUMENTS:
        if digest(root / name) != expected_documents[name]:
            raise ValueError(f"Missing or mismatched legal document: {name}")
    if digest(data / "pexels-36304379.mp4") != CLIP_SHA256:
        raise ValueError("Video SHA-256 mismatch")
    if target == "server":
        for size in sizes:
            model = root / "models" / f"b{size}.onnx"
            if not model.is_file() or model.stat().st_size == 0:
                raise ValueError(f"Missing model: {model}")
    elif importlib.util.find_spec("torch") is not None:
        raise ValueError("Instance image unexpectedly contains Torch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("instance", "server"), required=True)
    parser.add_argument("--sizes", default="1,8,16")
    parser.add_argument("--expected-documents", required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(size <= 0 for size in sizes):
        parser.error("Sizes must be positive")
    check(Path("/app"), Path("/data"), args.target, sizes, json.loads(args.expected_documents))
    print(f"{args.target}: project documents and contents verified; GPU not tested")
    print("Third-party redistribution review remains a separate release requirement")


if __name__ == "__main__":
    main()
