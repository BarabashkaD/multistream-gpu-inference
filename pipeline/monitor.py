# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Live table of every process's statistics, read from the log files they write.

The only module that imports rich. It reads; it never talks to the processes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table

_TAIL_START = 64 * 1024
_TAIL_MAX = 1024 * 1024
STALE_PERIODS = 3
_TRACEBACK = "Traceback (most recent call last)"
_PLAIN_WIDTH = 160


@dataclass
class FileState:
    """What the monitor learned from one log file: its last usable statistics line and how trustworthy it is."""

    line: dict | None = None
    failed: bool = False  # a traceback follows the last line
    malformed: int = 0


def _usable(line: dict) -> bool:
    """A periodic or final statistics line. Lines without a kind (other JSON a process prints) are not."""
    return line.get("kind") in ("period", "final") and "role" in line


def read_last(path: Path) -> FileState:
    """Finds the newest usable statistics line in a log file without reading the whole file.

    Reads a tail of the file, doubling it (up to 1 MiB) until a usable line is found. Skips a half-written last
    line and a possibly cut first line; counts unparsable JSON lines in "malformed", and sets "failed" when a
    traceback follows the usable line. Raises OSError if the file cannot be read; "line" is None if none found."""
    size = path.stat().st_size
    tail = _TAIL_START
    while True:
        with path.open("rb") as handle:
            handle.seek(max(size - tail, 0))
            text = handle.read().decode("utf-8", errors="replace")
        lines = text.split("\n")[:-1]  # the last piece has no newline yet: still being written
        if size > tail and lines:
            lines = lines[1:]  # the first piece may start mid-line
        state = FileState()
        trailing = []
        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    line = json.loads(raw)
                except ValueError:
                    state.malformed += 1
                    continue
                if isinstance(line, dict) and _usable(line):
                    state.line = line
                    state.failed = any(_TRACEBACK in item for item in trailing)
                    return state
                continue
            trailing.append(raw)
        if size <= tail or tail >= _TAIL_MAX:
            return state
        tail *= 2


def _is_final(line: dict) -> bool:
    """True for a process's final summary line, False for a periodic one."""
    return line["kind"] == "final"


def _cell(value, digits: int = 1) -> str:
    """A number with a fixed count of decimals for a table cell, or "-" when there is no value."""
    return "-" if value is None else f"{value:.{digits}f}"


def _figures(line: dict) -> dict:
    """The numbers one instance row shows, taken from its periodic or final line; absent values are None.

    A final line has no instantaneous rate or pool state, so "now" is None and "pool" is "-"."""
    if _is_final(line):
        p50, p95 = line.get("median_ms", {}), line.get("p95_ms", {})
        now, average = None, line.get("steady_fps", {}).get("tracked")
        pool, memory = "-", line.get("memory_mb")
    else:
        timings = line.get("period_ms", {})
        p50 = {name: value["p50"] for name, value in timings.items()}
        p95 = {name: value["p95"] for name, value in timings.items()}
        now, average = line.get("current", {}).get("tracked"), line.get("fps", {}).get("tracked")
        stats = line.get("pool", {})
        pool = f"{stats['size'] - stats['free']}/{stats['size']}" if {"size", "free"} <= stats.keys() else "-"
        memory = line.get("memory_mb")
    return {"now": now, "average": average, "decode": p50.get("decode"), "rtt50": p50.get("remote"),
            "rtt95": p95.get("remote"), "track": p50.get("track"), "latency95": p95.get("latency"),
            "dropped": line.get("counts", {}).get("dropped", 0), "pool": pool, "memory": memory}


class Monitor:
    """Builds a table of the server and every instance from the statistics lines in a folder of log files.

    It only reads files. A stream is "stale" when its line has not changed for STALE_PERIODS reporting periods,
    "failed" when a traceback follows its last line, and "done" after its final line. clock is injectable so tests
    need not wait."""

    def __init__(self, folder: Path, period_s: float, streams: int | None = None, clock=time.monotonic):
        """folder holds the *.log files; streams, when known, lets finished() tell that the run is over."""
        self.folder = Path(folder)
        self.period_s = period_s
        self.streams = streams
        self.clock = clock
        self._seen: dict[Path, tuple[float | None, float]] = {}  # last "time" value and when it changed
        self._rows: dict[int, tuple[dict, str]] = {}
        self._server: dict | None = None
        self._malformed = 0

    def _state(self, path: Path, state: FileState, now: float) -> str:
        """done, failed, stale or running for one file; remembers when its "time" value last changed."""
        line = state.line
        if _is_final(line):
            return "done"
        if state.failed:
            return "failed"
        previous = self._seen.get(path)
        if previous is None or previous[0] != line.get("time"):
            self._seen[path] = (line.get("time"), now)
            return "running"
        return "stale" if now - previous[1] > STALE_PERIODS * line.get("period_s", 1.0) else "running"

    def poll(self) -> None:
        """Re-reads every log in the folder and replaces the stored server line, instance rows and bad-line count."""
        now = self.clock()
        rows, server, malformed = {}, None, 0
        for path in sorted(self.folder.glob("*.log")):
            try:
                state = read_last(path)
            except OSError:
                continue  # removed or not readable this moment; the next period tries again
            malformed += state.malformed
            if state.line is None:
                continue
            role = state.line["role"]
            if role == "server":
                server = state.line
            elif role == "instance" and "stream" in state.line:
                rows[int(state.line["stream"])] = (state.line, self._state(path, state, now))
        self._rows, self._server, self._malformed = rows, server, malformed

    def states(self) -> dict[int, str]:
        """The state of each stream seen at the last poll, ordered by stream number."""
        return {stream: state for stream, (_, state) in sorted(self._rows.items())}

    def finished(self) -> bool:
        """The server writes its final line only after every instance has left, so a stream that is not running by
        then has ended, with or without a final line of its own (a killed instance writes none)."""
        if self.streams is None or self._server is None or not _is_final(self._server):
            return False
        ended = [state for state in self.states().values() if state != "running"]
        return len(ended) >= self.streams

    def _title(self) -> str:
        """The table title: the server's elapsed time, rate, batch counts, per-frame timings, queue and memory."""
        line = self._server
        if line is None:
            return "server: no data"
        batches = ", ".join(f"{size} x {count}" for size, count in line.get("batches", {}).items()) or "-"
        if _is_final(line):
            timings = line.get("median_ms", {})
            head, seconds = "server done", line.get("elapsed_s")
            fps, queued = line.get("steady_fps", {}).get("inferred"), "-"
            prepare, infer = timings.get("prepare"), timings.get("infer")
        else:
            timings = line.get("period_ms", {})
            head, seconds = "server", line.get("time", 0)
            fps, queued = line.get("current", {}).get("inferred"), line.get("queued", 0)
            prepare, infer = timings.get("prepare", {}).get("p50"), timings.get("infer", {}).get("p50")
        return (f"{head} {_cell(seconds, 0)} s | {_cell(fps)} frames/s | batches {batches}"
                f" | prepare {_cell(prepare, 2)} ms | infer {_cell(infer, 2)} ms per frame | queue {queued}"
                f" | {line.get('memory_mb', '-')} MB")

    def _caption(self) -> str | None:
        """The table caption listing frames available per batch size, or None before the server reports any."""
        available = (self._server or {}).get("available")
        if not available:
            return None
        return "frames available per batch: " + " ".join(f"{label}×{count}" for label, count in available.items())

    def table(self) -> Table:
        """Polls the logs and returns a rich table with one row per stream and a footer of totals.

        The total rate uses each stream's current rate, or its average once it has finished; the footer latency is
        the worst p95 and the state cell reports unparsable log lines."""
        self.poll()
        table = Table(title=self._title(), title_justify="left", show_footer=True, caption=self._caption(),
                      caption_justify="left")
        figures = [(stream, _figures(line), state) for stream, (line, state) in sorted(self._rows.items())]
        total_fps = sum(f["now"] if f["now"] is not None else (f["average"] or 0.0) for _, f, _ in figures)
        worst = max((f["latency95"] for _, f, _ in figures if f["latency95"] is not None), default=None)
        dropped = sum(f["dropped"] for _, f, _ in figures)
        footer = {"stream": "total", "fps": _cell(total_fps), "latency": _cell(worst), "dropped": str(dropped),
                  "state": f"{self._malformed} bad lines" if self._malformed else ""}
        for key, title in (("stream", "stream"), ("fps", "fps"), ("avg", "avg fps"), ("decode", "decode ms"),
                           ("rtt", "round trip p50/p95 ms"), ("track", "track ms"), ("latency", "latency p95 ms"),
                           ("dropped", "dropped"), ("pool", "pool"), ("memory", "MB"), ("state", "state")):
            table.add_column(title, footer=footer.get(key, ""), justify="left" if key == "state" else "right")
        for stream, f, state in figures:
            table.add_row(str(stream), _cell(f["now"]), _cell(f["average"]), _cell(f["decode"], 2),
                          f"{_cell(f['rtt50'])} / {_cell(f['rtt95'])}", _cell(f["track"], 2), _cell(f["latency95"]),
                          str(f["dropped"]), f["pool"], "-" if f["memory"] is None else str(f["memory"]), state)
        return table


def _default_console() -> Console:
    """Without a terminal rich assumes 80 columns, which wraps the table; logs can be wider."""
    console = Console()
    return console if console.is_terminal else Console(width=_PLAIN_WIDTH)


def run(folder: Path, period_s: float, streams: int | None = None, once: bool = False,
        console: Console | None = None, stop: threading.Event | None = None) -> int:
    """Shows the table for a log folder and returns 0 when it is done.

    once prints a single table. On a terminal the table redraws in place every period; otherwise a timestamp and a
    new table are printed each period. Ends when the monitor reports the run finished (needs streams) or when
    stop is set, and never raises for missing or unreadable logs."""
    console = console or _default_console()
    stop = stop or threading.Event()
    monitor = Monitor(folder, period_s, streams)
    if once:
        console.print(monitor.table())
        return 0
    if console.is_terminal:
        # No redirection: Live would otherwise capture the process's stdout, where the fleet prints its final JSON.
        with Live(monitor.table(), console=console, auto_refresh=False, redirect_stdout=False,
                  redirect_stderr=False) as live:
            while not monitor.finished() and not stop.wait(period_s):
                live.update(monitor.table(), refresh=True)
        return 0
    while True:  # docker compose logs or a pipe: no redraw, one table per period
        console.print(time.strftime("%H:%M:%S"))
        console.print(monitor.table())
        if monitor.finished() or stop.wait(period_s):
            return 0
