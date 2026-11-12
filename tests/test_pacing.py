# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Real-time pacing and the oldest-drop policy of run_one_loop, with a fake clock, reader and model (no GPU)."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import run_demo as demo


class Clock:
    """A fake time module: monotonic() returns the current fake time and sleep() advances it."""

    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Reader:
    """A 25 FPS source of n frames; each frame's pixels hold its 1-based number."""
    backend = 'opencv'; height = 12; width = 16; fps = 25.0; device_verified = False

    def __init__(self, n):
        self.frame_count = n
        self.i = 0

    def read(self, **kw):
        if self.i == self.frame_count:
            return False, None, None
        self.i += 1
        return True, np.full((12, 16, 3), self.i, dtype=np.uint8), self.i / 25


class Model:
    """Tracks instantly or, with `cost`, advances the fake clock by `cost` seconds per frame in each call."""

    def __init__(self, clock, cost=0.0):
        self.clock, self.cost = clock, cost
        self.ids = []

    def track(self, source, **kwargs):
        frames = source if isinstance(source, list) else [source]
        self.clock.now += self.cost * len(frames)
        self.ids.extend(int(f[0, 0, 0]) for f in frames)
        return [SimpleNamespace(boxes=None, names={}) for _ in frames]


class PacingTests(unittest.TestCase):
    def run_loop(self, n, batch=1, cost=0.0, pace='realtime', drop_policy='none', max_lag_s=0.5, stop=None):
        clock = Clock()
        args = SimpleNamespace(batch_size=batch, tracker='bytetrack.yaml', conf=.25, iou=.7, max_frames=0,
                               pace=pace, drop_policy=drop_policy, max_lag_s=max_lag_s)
        model = Model(clock, cost)
        with patch.object(demo, 'time', clock):
            stats = demo.run_one_loop(args, Reader(n), model, 'cpu', 640, 1, None, None, stop, {'actual': None})
        return stats, model, clock

    def test_realtime_reads_each_frame_at_its_due_time(self):
        stats, model, clock = self.run_loop(10)
        self.assertEqual(model.ids, list(range(1, 11)))
        self.assertAlmostEqual(sum(clock.sleeps), 9 / 25)  # frame k waits until k / 25 s after the loop start
        self.assertAlmostEqual(stats['processing_fps'], 10 / (9 / 25))
        self.assertEqual(stats['frames_dropped'], 0)
        self.assertAlmostEqual(stats['due_to_tracked_ms']['p95'], 0.0)

    def test_max_pace_never_sleeps_and_due_is_read_time(self):
        stats, model, clock = self.run_loop(10, pace='max', cost=0.01)
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(model.ids, list(range(1, 11)))
        self.assertEqual(stats['due_to_tracked_ms'], stats['read_to_batch_complete_ms'])

    def test_batch_waits_for_its_frames_to_be_due(self):
        stats, model, _ = self.run_loop(8, batch=4)
        self.assertEqual(model.ids, list(range(1, 9)))
        # In each batch of 4 the frames wait 120, 80, 40 and 0 ms for the fourth to be due.
        self.assertAlmostEqual(stats['due_to_tracked_ms']['p95'], 120.0, delta=1e-6)
        self.assertAlmostEqual(stats['due_to_tracked_ms']['p50'], 60.0, delta=1e-6)

    def test_slow_processing_without_drop_policy_falls_behind(self):
        stats, model, _ = self.run_loop(50, cost=0.06)  # 60 ms per frame against a 40 ms period
        self.assertEqual(stats['frames_dropped'], 0)
        self.assertEqual(model.ids, list(range(1, 51)))
        self.assertGreater(stats['due_to_tracked_ms']['p95'], 900)  # lag grows by 20 ms per frame

    def test_oldest_policy_skips_late_frames_and_bounds_latency(self):
        stats, model, _ = self.run_loop(50, cost=0.06, drop_policy='oldest', max_lag_s=0.2)
        self.assertGreater(stats['frames_dropped'], 0)
        self.assertEqual(stats['frames_dropped'] + stats['frames_processed'], 50)
        self.assertEqual(stats['frames_read'], 50)
        self.assertEqual(model.ids, sorted(model.ids))  # skipped frames leave gaps, order is kept
        self.assertLessEqual(stats['due_to_tracked_ms']['p95'], 200 + 60 + 1e-6)

    def test_stop_during_pacing_wait_ends_the_loop(self):
        class StopWhileWaiting:
            """Not set at first; a stop arrives during the first pacing wait."""
            waits = []

            def is_set(self):
                return False

            def wait(self, seconds):
                self.waits.append(seconds)
                return True

        stop = StopWhileWaiting()
        stats, model, _ = self.run_loop(10, stop=stop)
        self.assertTrue(stats['stopped_by_user'])
        self.assertEqual(model.ids, [1])  # frame 1 is due at once; the stop arrives while waiting for frame 2
        self.assertAlmostEqual(stop.waits[0], 1 / 25)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, 'needs --pace realtime'):
            demo.validate_pacing(SimpleNamespace(pace='max', drop_policy='oldest', max_lag_s=0.5))
        with self.assertRaisesRegex(ValueError, 'max-lag-s'):
            demo.validate_pacing(SimpleNamespace(pace='realtime', drop_policy='oldest', max_lag_s=0))
        demo.validate_pacing(SimpleNamespace(pace='realtime', drop_policy='oldest', max_lag_s=0.5))
        demo.validate_pacing(SimpleNamespace())  # older callers without pacing fields keep max pace

    def test_launcher_forwards_pacing(self):
        import launch_instances
        with patch('sys.argv', ['launch_instances.py', '--pace', 'realtime', '--drop-policy', 'oldest',
                                '--max-lag-s', '0.3']):
            args = launch_instances.parse_args()
        command = launch_instances.child_command(args, 'instance-01', None)
        for flag, value in (('--pace', 'realtime'), ('--drop-policy', 'oldest'), ('--max-lag-s', '0.3')):
            self.assertEqual(command[command.index(flag) + 1], value)


if __name__ == '__main__':
    unittest.main()
