# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests of pipeline.monitor: reading the last stats line of each log and rendering the live table."""
import io
import json
import re

from rich.console import Console

from pipeline.monitor import Monitor, read_last, run


def period(stream, time, tracked=25.0, dropped=0, period_s=1.0):
    """Build an instance stats line of kind "period" for one stream, with fixed timings and pool figures."""
    return {"kind": "period", "role": "instance", "stream": stream, "time": time, "period_s": period_s,
            "current": {"tracked": tracked}, "fps": {"tracked": tracked}, "counts": {"dropped": dropped},
            "period_ms": {"decode": {"p50": 0.9, "p95": 1.2}, "remote": {"p50": 38.0, "p95": 52.0},
                          "track": {"p50": 0.85, "p95": 1.0}, "latency": {"p50": 44.0, "p95": 48.0 + 3 * stream}},
            "pool": {"size": 6, "free": 4}, "memory_mb": 340}


def server(time, kind="period"):
    """Build a server stats line of the given kind ("period" or "final") with fixed timings and queue figures."""
    return {"kind": kind, "role": "server", "time": time, "period_s": 1.0, "current": {"inferred": 50.0},
            "queued": 3, "batches": {"8": 10}, "period_ms": {"prepare": {"p50": 0.4, "p95": 0.5},
                                                              "infer": {"p50": 1.7, "p95": 2.0}}, "memory_mb": 2646}


def write(folder, name, *lines, tail=""):
    """Write the given dicts as JSON lines to a log file in the folder, followed by any raw tail text."""
    (folder / name).write_text("".join(json.dumps(line) + "\n" for line in lines) + tail, encoding="utf-8")


def text_of(table):
    """Render a rich table to plain text, 200 columns wide, without control codes."""
    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    console.print(table)
    return console.file.getvalue()


def test_last_complete_line_wins_and_a_partial_line_is_ignored(tmp_path):
    """read_last returns the newest complete JSON line and skips a half-written line at the end."""
    write(tmp_path, "instance0.log", period(0, 1.0), period(0, 2.0), tail='{"kind": "per')
    assert read_last(tmp_path / "instance0.log").line["time"] == 2.0


def test_a_line_longer_than_the_first_tail_is_found(tmp_path):
    """read_last still finds the last line when it is much longer than the first chunk read from the end."""
    big = server(5.0, kind="final") | {"streams": [{"pad": "x" * 100_000}]}
    write(tmp_path, "server.log", server(1.0), big)
    assert read_last(tmp_path / "server.log").line["kind"] == "final"


def test_plain_text_and_malformed_lines_are_skipped(tmp_path):
    """Non-JSON lines after the last stats line are counted as malformed but do not mark the log failed."""
    write(tmp_path, "instance0.log", period(0, 1.0), tail="some warning\n{not json\n")
    state = read_last(tmp_path / "instance0.log")
    assert state.line["time"] == 1.0
    assert state.malformed == 1
    assert not state.failed


def test_a_traceback_after_the_last_line_means_failed(tmp_path):
    """A Python traceback after the last stats line makes the monitor report that stream as failed."""
    write(tmp_path, "instance1.log", period(1, 3.0), tail="Traceback (most recent call last):\n  boom\n")
    monitor = Monitor(tmp_path, period_s=2.0)
    monitor.table()
    assert monitor.states() == {1: "failed"}


def test_rows_are_sorted_with_totals_and_the_server_header(tmp_path):
    """Rows come out in stream order, the totals add up, and the server header shows "no data" until it logs."""
    write(tmp_path, "instance1.log", period(1, 5.0, tracked=24.0, dropped=2))
    write(tmp_path, "instance0.log", period(0, 5.0))
    text = text_of(Monitor(tmp_path, period_s=2.0).table())
    assert "server: no data" in text
    write(tmp_path, "server.log", server(5.0))
    text = text_of(Monitor(tmp_path, period_s=2.0).table())
    rows = [line for line in text.splitlines() if "running" in line]
    assert [re.search(r"\d+", row).group() for row in rows] == ["0", "1"]  # the first number in a row is its stream
    assert "49.0" in text  # total FPS, 25 + 24
    assert "51.0" in text  # worst latency p95, 48 + 3
    assert "1.7" in text and "8 x 10" in text and "queue 3" in text


def test_a_final_server_line_shows_the_same_figures_as_a_period_line(tmp_path):
    """A final server line is rendered as "done" with its steady figures, and the instance row shows its memory."""
    final = {"kind": "final", "role": "server", "elapsed_s": 49.3, "steady_fps": {"inferred": 100.34},
             "median_ms": {"prepare": 1.87, "infer": 6.89}, "batches": {"8": 10}, "memory_mb": 2646}
    write(tmp_path, "server.log", final)
    write(tmp_path, "instance0.log", period(0, 1.0) | {"kind": "final", "memory_mb": 340})
    text = text_of(Monitor(tmp_path, period_s=2.0).table())
    for part in ("server done 49 s", "100.3 frames/s", "batches 8 x 10", "prepare 1.87 ms", "infer 6.89 ms",
                 "queue -", "2646 MB"):
        assert part in text
    assert "340" in [line for line in text.splitlines() if "done" in line and "server" not in line][0]


CAPTION = "frames available per batch: 1×34 2×5 8+×615"
AVAILABLE = {"1": 34, "2": 5, "8+": 615}


def test_a_period_server_line_shows_the_frames_available_per_batch(tmp_path):
    """A period server line with an "available" histogram adds the frames-available caption to the table."""
    write(tmp_path, "server.log", server(5.0) | {"available": AVAILABLE})
    write(tmp_path, "instance0.log", period(0, 5.0))
    assert CAPTION in text_of(Monitor(tmp_path, period_s=2.0).table())


def test_a_final_server_line_shows_the_frames_available_per_batch(tmp_path):
    """A final server line with an "available" histogram shows the same caption."""
    write(tmp_path, "server.log", server(5.0, kind="final") | {"available": AVAILABLE, "elapsed_s": 5.0})
    write(tmp_path, "instance0.log", period(0, 5.0))
    assert CAPTION in text_of(Monitor(tmp_path, period_s=2.0).table())


def test_without_available_there_is_no_caption(tmp_path):
    """When the server line has no "available" field, no caption is drawn."""
    write(tmp_path, "server.log", server(5.0))
    write(tmp_path, "instance0.log", period(0, 5.0))
    assert "frames available" not in text_of(Monitor(tmp_path, period_s=2.0).table())


def test_a_final_line_means_done(tmp_path):
    """A stream whose last line is of kind "final" is reported as done."""
    write(tmp_path, "instance0.log", period(0, 1.0), {"kind": "final", "role": "instance", "stream": 0,
                                                        "steady_fps": {"tracked": 25.0}, "counts": {"dropped": 0},
                                                        "median_ms": {}, "p95_ms": {}})
    monitor = Monitor(tmp_path, period_s=2.0)
    monitor.table()
    assert monitor.states() == {0: "done"}


def test_a_line_whose_time_stops_advancing_becomes_stale(tmp_path):
    """A stream whose line time stays unchanged for more than three periods is reported as stale, not before."""
    now = [100.0]
    write(tmp_path, "instance0.log", period(0, 7.0, period_s=1.0))
    monitor = Monitor(tmp_path, period_s=2.0, clock=lambda: now[0])
    monitor.table()
    now[0] = 102.9
    monitor.table()
    assert monitor.states() == {0: "running"}
    now[0] = 103.1  # more than 3 x period_s since the time last changed
    monitor.table()
    assert monitor.states() == {0: "stale"}


def test_finished_needs_every_stream_and_the_server(tmp_path):
    """finished() is true only once every expected stream and the server have logged a final line."""
    final = {"kind": "final", "role": "instance", "steady_fps": {}, "counts": {}, "median_ms": {}, "p95_ms": {}}
    write(tmp_path, "instance0.log", final | {"stream": 0})
    monitor = Monitor(tmp_path, period_s=2.0, streams=2)
    monitor.table()
    assert not monitor.finished()
    write(tmp_path, "instance1.log", final | {"stream": 1})
    write(tmp_path, "server.log", server(9.0, kind="final"))
    monitor.table()
    assert monitor.finished()


def test_once_prints_one_table_without_control_codes(tmp_path):
    """run(..., once=True) prints a single plain table with no terminal control codes and returns 0."""
    write(tmp_path, "instance0.log", period(0, 1.0))
    output = io.StringIO()
    assert run(tmp_path, 2.0, once=True, console=Console(file=output, width=200, force_terminal=False)) == 0
    assert "\x1b[" not in output.getvalue()
    assert "running" in output.getvalue()


def test_without_a_terminal_it_prints_until_finished(tmp_path):
    """On a console that is not a terminal, run keeps printing tables until everything is done, then returns 0."""
    final = {"kind": "final", "role": "instance", "stream": 0, "steady_fps": {}, "counts": {}, "median_ms": {},
             "p95_ms": {}}
    write(tmp_path, "instance0.log", final)
    write(tmp_path, "server.log", server(1.0, kind="final"))
    output = io.StringIO()
    assert run(tmp_path, 0.01, streams=1, console=Console(file=output, width=200, force_terminal=False)) == 0
    assert output.getvalue().count("done") >= 1


def test_the_terminal_branch_exits_when_everything_is_finished(tmp_path):
    """On a terminal console, the live display also stops and returns 0 once everything is finished."""
    final = {"kind": "final", "role": "instance", "stream": 0, "steady_fps": {}, "counts": {}, "median_ms": {},
             "p95_ms": {}}
    write(tmp_path, "instance0.log", final)
    write(tmp_path, "server.log", server(1.0, kind="final"))
    output = io.StringIO()
    assert run(tmp_path, 0.05, streams=1, console=Console(file=output, force_terminal=True, width=200)) == 0
    assert "done" in output.getvalue()
