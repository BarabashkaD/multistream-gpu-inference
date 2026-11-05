# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of the compose file generator in tools/make_compose.py and of the compose files committed from it."""
from pathlib import Path
import sys

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3 / "tools"))

from make_compose import render, render_mounts  # noqa: E402

KEY_LINE = "${TRAFFIC_INFERENCE_KEY:?set TRAFFIC_INFERENCE_KEY to a random value}"


def test_the_committed_file_is_the_generated_one():
    """compose.yaml equals what render(16) produces, so it was not edited by hand."""
    assert (V3 / "compose.yaml").read_text(encoding="utf-8") == render(16)


def test_the_committed_mounts_file_is_the_generated_one():
    """compose.mounts.yaml equals what render_mounts(16) produces, so it was not edited by hand."""
    assert (V3 / "compose.mounts.yaml").read_text(encoding="utf-8") == render_mounts(16)


def test_two_streams_give_two_instances_that_share_the_server_pid_namespace():
    """Two streams render exactly two instance services, each joining the server's PID namespace with ptrace."""
    text = render(2)
    assert "  instance0:" in text and "  instance1:" in text and "  instance2:" not in text
    assert text.count('pid: "service:server"') == 2
    assert "cap_add: [SYS_PTRACE]" in text
    assert '"--clients", "2"' in text and '"--streams", "2"' in text


def test_the_main_file_has_no_model_or_clip_mounts_but_adds_the_clip_build_context():
    """The main file mounts no models or clip and gives the server and each instance the clip build context."""
    text = render(2)
    assert "/app/models" not in text and "/data:ro" not in text
    context = "additional_contexts: {clip: ./data}"
    assert text.count(context) == 2
    server = text.split("  server:")[1].split("  instance0:")[0]
    instance0 = text.split("  instance0:")[1].split("  instance1:")[0]
    assert context in server and context in instance0


def test_the_mounts_file_mounts_only_models_and_the_clip():
    """The mounts file adds the models mount to the server and the clip mount to each instance, and nothing else."""
    text = render_mounts(2)
    assert text.count("./models:/app/models:ro") == 1
    assert text.count("./data:/data:ro") == 2
    assert text.count("- ") == 3
    server = text.split("  server:")[1].split("  instance0:")[0]
    assert "./models:/app/models:ro" in server
    assert "./data:/data:ro" in text.split("  instance0:")[1].split("  instance1:")[0]
    assert "./data:/data:ro" in text.split("  instance1:")[1]


def test_no_key_and_no_absolute_path_in_the_files():
    """The generated files hold no key value and no machine path, and name the key only as a required variable."""
    assert KEY_LINE in render(16)
    for text in (render(16), render_mounts(16)):
        assert ":\\" not in text and "/home/" not in text and "/Users/" not in text
        assert "TRAFFIC_INFERENCE_KEY" not in text.replace("TRAFFIC_INFERENCE_KEY: " + KEY_LINE, "")
