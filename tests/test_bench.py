# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The pure parts of bench.py: log parsing, point summaries, table, naming and paths (no GPU, media or processes)."""
import json
from pathlib import Path
import tempfile
import unittest

import bench


def loop(number, fps, p50, p95, dropped=0, read_p95=None, max_lag_ms=None, frames=611):
    """One per-loop entry as run_demo.py prints it."""
    return dict(loop=number, frames_read=frames, frames_processed=frames - dropped, processing_fps=fps,
                due_to_tracked_ms=dict(p50=p50, p95=p95),
                read_to_batch_complete_ms=dict(p50=p50, p95=p95 if read_p95 is None else read_p95),
                frames_dropped=dropped, max_lag_ms=max_lag_ms, stopped_by_user=False)


def report(instance, loops):
    """A run_demo.py summary with the given loops."""
    return dict(instance_id=instance, loops_completed=len(loops), loops=loops, stopped_by_user=False)


def sample(t, sm, dec=5, mem_mb=1000.0, free_mb=7000.0, cpu=50.0, host_free_gb=20.0):
    """One 1-second load sample."""
    return dict(t=t, sm=sm, dec=dec, mem_mb=mem_mb, free_mb=free_mb, cpu=cpu, host_free_gb=host_free_gb)


class LogTests(unittest.TestCase):
    def test_final_json_in_a_log_mixing_utf8_and_utf16(self):
        summary = report('instance-01', [loop(1, 20.0, 10, 20)])
        data = (b"[loop 1] start frames=611\r\n" + "onnxruntime warning\r\n".encode('utf-16-le')
                + b"[loop 2] start\r\n" + json.dumps(summary, indent=2).encode('utf-8') + b"\r\n")
        text = bench.log_text(data)
        self.assertIn('onnxruntime warning', text)  # UTF-16 ASCII reads as text once NULs are dropped
        self.assertIn('[loop 2]', text)
        self.assertEqual(bench.final_json(text), summary)

    def test_final_json_skips_objects_that_are_not_the_summary(self):
        summary = report('instance-01', [])
        text = "x\n" + json.dumps(summary, indent=2) + '\n{"other": 1}\n{not json\n'
        self.assertEqual(bench.final_json(text), summary)

    def test_final_json_without_summary(self):
        with self.assertRaises(ValueError):
            bench.final_json('[loop 1] start\n{"other": 1}\n')


class SummaryTests(unittest.TestCase):
    def reports(self):
        fast = report('instance-01', [loop(1, 10.0, 900, 2000, dropped=7), loop(2, 25.0, 10, 30, max_lag_ms=44.0),
                                      loop(3, 25.0, 12, 40, max_lag_ms=44.0)])
        slow = report('instance-02', [loop(1, 10.0, 900, 2000), loop(2, 22.0, 50, 300, dropped=3, max_lag_ms=44.0),
                                      loop(3, 24.0, 70, 200, dropped=2, read_p95=90, max_lag_ms=44.0)])
        return [fast, slow]

    def test_steady_state_leaves_loop_one_out(self):
        samples = [sample(0, 10, mem_mb=900), sample(5, 80, mem_mb=1700), sample(6, 90, mem_mb=1800),
                   sample(9, 20, mem_mb=1100)]
        point = bench.summarize_point(2, self.reports(), samples, steady_from=4.5, steady_to=6.5, base_mem_mb=1000)
        self.assertEqual(point['total_fps'], 48.0)  # (25 + 25) / 2 + (22 + 24) / 2
        self.assertEqual(point['slowest_fps'], 23.0)
        self.assertEqual(point['latency_p50_ms'], 60.0)  # slowest process: mean of 50 and 70
        self.assertEqual(point['latency_p95_ms'], 300.0)  # slowest process: max of 300 and 200
        self.assertEqual(point['read_to_tracked_p95_ms'], 300.0)
        self.assertEqual(point['dropped'], 5)
        self.assertEqual(point['loop1_dropped'], 7)
        self.assertEqual(point['max_lag_ms'], [44.0])
        self.assertEqual(point['gpu_busy_pct'], 85)  # the two samples in the steady window
        self.assertEqual(point['load_window'], 'steady')
        self.assertEqual(point['gpu_mem_app_mb'], 800)  # peak 1800 - baseline 1000
        self.assertEqual(point['min_free_gpu_mb'], 7000)
        self.assertEqual([p['fps'] for p in point['per_process']], [25.0, 23.0])

    def test_load_falls_back_to_the_whole_run_without_a_window(self):
        samples = [sample(0, 10), sample(1, 30)]
        point = bench.summarize_point(2, self.reports(), samples, steady_from=None, steady_to=None, base_mem_mb=500)
        self.assertEqual(point['gpu_busy_pct'], 20)
        self.assertEqual(point['load_window'], 'whole run')
        self.assertIsNone(point['steady_window_s'])

    def test_missing_nvml_metrics_are_none(self):
        samples = [sample(0, None, dec=None, mem_mb=None, free_mb=None)]
        point = bench.summarize_point(2, self.reports(), samples, None, None, base_mem_mb=None)
        self.assertIsNone(point['gpu_busy_pct'])
        self.assertIsNone(point['gpu_mem_app_mb'])
        self.assertIsNone(point['min_free_gpu_mb'])

    def test_a_report_without_loop_two_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'no loop 2'):
            bench.summarize_point(1, [report('instance-01', [loop(1, 25.0, 10, 20)])], [sample(0, 1)], None, None, 0)
        with self.assertRaisesRegex(ValueError, 'expected 2 reports'):
            bench.summarize_point(2, self.reports()[:1], [sample(0, 1)], None, None, 0)


class TableTests(unittest.TestCase):
    def test_table_rows_errors_and_stop_reason(self):
        good = dict(streams=1, total_fps=25.0, slowest_fps=25.0, dropped=0, latency_p50_ms=9.1, latency_p95_ms=12.0,
                    gpu_busy_pct=30, nvdec_pct=4, cpu_pct=20, gpu_mem_app_mb=None)
        table = bench.render_table([good, dict(streams=2, error='timeout after 600 s')], 'stopped here')
        lines = table.splitlines()
        self.assertTrue(lines[0].startswith('| Processes | Total FPS | Slowest process FPS | Dropped |'))
        self.assertEqual(lines[1].count('---'), len(bench.TABLE_COLUMNS))
        self.assertEqual(lines[2], '| 1 | 25.0 | 25.0 | 0 | 9.1 | 12.0 | 30 | 4 | 20 | - |')
        self.assertIn('- 2 processes: not measured, timeout after 600 s', lines)
        self.assertIn('- Stopped: stopped here', lines)

    def test_table_without_notes(self):
        self.assertEqual(len(bench.render_table([]).splitlines()), 2)


class NamingAndCommandTests(unittest.TestCase):
    def test_out_dir_name(self):
        self.assertEqual(bench.out_dir_name('realtime', 'oldest', (384, 640), 8), 'v1-realtime-drop-640x384-b8')
        self.assertEqual(bench.out_dir_name('realtime', 'none', [384, 640], 1), 'v1-realtime-640x384-b1')
        self.assertEqual(bench.out_dir_name('max', 'none', 640, 16), 'v1-max-640x640-b16')

    def test_parse_instances(self):
        self.assertEqual(bench.parse_instances('1,2, 3,8'), [1, 2, 3, 8])
        for bad in ('', '0', '1,-2', 'a'):
            with self.assertRaises(ValueError):
                bench.parse_instances(bad)

    def test_launcher_is_found_next_to_the_script(self):
        self.assertEqual(bench.LAUNCHER, Path(bench.__file__).resolve().parent / 'launch_instances.py')
        self.assertTrue(bench.LAUNCHER.is_file())

    def test_launcher_command_forwards_pacing_and_lag_only_when_given(self):
        command = bench.launcher_command('python', 'm.onnx', (384, 640), 8, 3, 3, 'realtime', 'oldest', None, 'out')
        self.assertEqual(command[command.index('--pace') + 1], 'realtime')
        self.assertEqual(command[command.index('--drop-policy') + 1], 'oldest')
        self.assertEqual(command[command.index('--imgsz') + 1:command.index('--imgsz') + 3], ['384', '640'])
        self.assertEqual(command[command.index('--instances') + 1], '3')
        self.assertIn('--headless', command)
        self.assertNotIn('--max-lag-s', command)
        command = bench.launcher_command('python', 'm.onnx', 640, 1, 1, 3, 'realtime', 'oldest', 0.5, 'out')
        self.assertEqual(command[command.index('--max-lag-s') + 1], '0.5')


class PathTests(unittest.TestCase):
    def test_machine_paths_become_placeholders(self):
        out = bench.REPO / 'results' / 'x'
        replacements = bench.path_replacements(out)
        self.assertEqual(bench.portable(str(out / 'n1' / 'logs'), replacements), str(Path('<out>') / 'n1' / 'logs'))
        self.assertEqual(bench.portable(str(bench.REPO / 'run_demo.py'), replacements),
                         str(Path('<repo>') / 'run_demo.py'))
        self.assertEqual(bench.portable(str(Path.home() / 'f'), replacements), str(Path('<home>') / 'f'))
        self.assertEqual(bench.portable('no paths', replacements), 'no paths')

    def test_scrub_file_in_utf8_and_utf16(self):
        replacements = bench.path_replacements(None)
        repo_file = str(bench.REPO / 'run_demo.py')
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'instance-01.log'
            log.write_bytes(f"$ python {repo_file}\n".encode('utf-8') + f"at {repo_file}\n".encode('utf-16-le'))
            bench.scrub_file(log, replacements)
            data = log.read_bytes()
        expected = str(Path('<repo>') / 'run_demo.py')
        self.assertEqual(data, f"$ python {expected}\n".encode('utf-8') + f"at {expected}\n".encode('utf-16-le'))

    def test_baseline_waits_for_memory_to_settle(self):
        readings = iter([2900.0, 2600.0, 2210.0, 2200.0, 2199.0])
        sleeps = []
        self.assertEqual(bench.settled_memory_mb(lambda: next(readings), sleeps.append), 2200.0)
        self.assertEqual(len(sleeps), 3)
        rising = iter(range(0, 10000, 100))
        self.assertEqual(bench.settled_memory_mb(lambda: next(rising), lambda s: None, timeout_s=5), 500)

    def test_guard_needs_consecutive_samples(self):
        low = sample(0, 1, free_mb=100)
        self.assertIsNone(bench.guard_breach([low, low], 400, 2))
        self.assertIsNone(bench.guard_breach([low, sample(1, 1), low, low], 400, 2))
        self.assertIn('free GPU memory', bench.guard_breach([low, low, low], 400, 2))
        host = sample(0, 1, host_free_gb=1.0)
        self.assertIn('host memory', bench.guard_breach([host] * 3, 400, 2))


if __name__ == '__main__':
    unittest.main()
