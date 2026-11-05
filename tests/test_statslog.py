# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.statslog: the JSON-lines stats log and the instance and server line builders."""
import json
import threading

from pipeline.metrics import Metrics
from pipeline.statslog import StatsLog, elapsed_s, instance_line, server_line

INSTANCE_KEYS = {"kind", "role", "stream", "time", "period_s", "current", "fps", "counts", "period_ms", "pool",
                 "memory_mb"}


def test_lines_go_to_stdout_and_the_file(tmp_path, capsys):
    """Each written line appears identically on stdout and in the log file."""
    path = tmp_path / "instance3.log"
    log = StatsLog(path)
    log.write({"kind": "period", "a": 1})
    log.write({"kind": "final", "b": 2})
    log.close()
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert printed == written == [{"kind": "period", "a": 1}, {"kind": "final", "b": 2}]


def test_the_file_is_truncated_on_open(tmp_path):
    """Opening a StatsLog on an existing file discards its old content."""
    path = tmp_path / "instance3.log"
    path.write_text('{"kind": "final", "old": true}\n', encoding="utf-8")
    log = StatsLog(path)
    log.write({"kind": "period"})
    log.close()
    assert path.read_text(encoding="utf-8").splitlines() == ['{"kind": "period"}']


def test_concurrent_writes_never_interleave(tmp_path, capsys):
    """Four threads writing 50 long lines each leave 200 complete JSON lines in both the file and stdout."""
    log = StatsLog(tmp_path / "x.log")
    threads = [threading.Thread(target=lambda n=n: [log.write({"n": n, "pad": "x" * 500}) for _ in range(50)])
               for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.close()
    written = (tmp_path / "x.log").read_text(encoding="utf-8").splitlines()
    printed = capsys.readouterr().out.splitlines()
    assert len(written) == len(printed) == 200
    for line in written + printed:
        json.loads(line)


def test_a_write_after_the_final_line_is_dropped(tmp_path, capsys):
    """Once write_final has run, a later write is dropped, so the final line is always the last."""
    path = tmp_path / "x.log"
    log = StatsLog(path)
    log.write({"kind": "period"})
    log.write_final({"kind": "final"})
    log.write({"kind": "period", "late": True})
    log.close()
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert printed == written == [{"kind": "period"}, {"kind": "final"}]


def test_instance_line_has_the_exact_keys():
    """An instance line has exactly the documented keys, rounded memory, and consumes its period samples."""
    metrics = Metrics()
    metrics.count("tracked", 3)
    metrics.count("dropped", 1)
    metrics.record_ms("decode", 0.9)
    line = instance_line(metrics, {"size": 6, "free": 4}, stream=3, period_s=1.0, memory_mb=339.6)
    assert set(line) == INSTANCE_KEYS
    assert (line["kind"], line["role"], line["stream"], line["period_s"], line["memory_mb"]) == \
        ("period", "instance", 3, 1.0, 340)
    assert line["counts"] == {"tracked": 3, "dropped": 1}
    assert line["period_ms"] == {"decode": {"p50": 0.9, "p95": 0.9}}
    assert instance_line(metrics, {}, 3, 1.0, 1.0)["period_ms"] == {}  # each sample once


def test_elapsed_is_seconds_since_the_process_started():
    """elapsed_s returns a non-negative number of seconds under an hour for a freshly started test process."""
    assert 0.0 <= elapsed_s() < 3600.0


SERVER_KEYS = {"kind", "role", "time", "period_s", "current", "queued", "batches", "available", "period_ms",
               "memory_mb"}


def test_server_line_has_the_exact_keys():
    """A server line has exactly the documented keys, rounded memory, and carries the batch and availability counts."""
    metrics = Metrics()
    metrics.count("inferred", 8)
    metrics.record_ms("infer", 1.7)
    line = server_line(metrics, {"8": 3}, queued=2, period_s=1.0, memory_mb=2646.2, available={"3": 1, "8+": 2})
    assert set(line) == SERVER_KEYS
    assert line["available"] == {"3": 1, "8+": 2}
    assert (line["kind"], line["role"], line["queued"], line["batches"], line["memory_mb"]) == \
        ("period", "server", 2, {"8": 3}, 2646)
    assert line["period_ms"] == {"infer": {"p50": 1.7, "p95": 1.7}}
