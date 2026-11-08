# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the bundle pool: leasing, reclaiming, expiry and the choice of which frame to display."""
import threading

from pipeline.bundle import COMPLETE, Bundle, BundlePool, Part


class Clock:
    """Manually advanced clock, so expiry and grace periods are tested without sleeping."""

    def __init__(self):
        """Start the clock at a fixed time."""
        self.now = 100.0

    def __call__(self):
        """Return the current fake time."""
        return self.now


def make_pool(size=3, display=True, clock=None):
    """Build a pool of `size` bundles and return it with the list that records every released pointer."""
    released = []
    pool = BundlePool([Bundle(i) for i in range(size)], released.append, display_enabled=display,
                      max_age_s=2.0, clock=clock or Clock())
    return pool, released


def start_frame(pool, frame_id, pointer=None):
    """Acquire a bundle and publish a frame into it; return the bundle as the inference stage receives it."""
    bundle = pool.acquire(threading.Event())
    pool.publish_frame(bundle, pointer if pointer is not None else 1000 + frame_id, frame_id, 0, frame_id)
    return pool.to_infer.get_nowait()


def finish_stages(pool, bundle):
    """Complete the inference stage for a bundle (detections, tracks) and release its lease."""
    pool.mark(bundle, Part.DETECTIONS)
    pool.set_tracks(bundle, f"tracks-{bundle.frame_id}")
    pool.release(bundle)


def test_acquire_returns_a_free_bundle_with_one_lease():
    """Proves a freshly acquired bundle holds one lease and no finished parts."""
    pool, _ = make_pool()
    bundle = pool.acquire(threading.Event())
    assert bundle.leases == 1
    assert bundle.parts == Part.NONE


def test_acquire_returns_none_when_stopped():
    """Proves acquire gives up with None once the stop event is set and no bundle is free."""
    pool, _ = make_pool(size=1)
    pool.acquire(threading.Event())
    stop = threading.Event()
    stop.set()
    assert pool.acquire(stop) is None


def test_acquire_blocks_until_a_bundle_is_reclaimed():
    """Proves acquire waits while every bundle is in use and returns the bundle once it is reclaimed."""
    pool, _ = make_pool(size=1, display=False)
    bundle = start_frame(pool, 0)
    results = []
    worker = threading.Thread(target=lambda: results.append(pool.acquire(threading.Event())))
    worker.start()
    worker.join(0.2)
    assert worker.is_alive()
    finish_stages(pool, bundle)
    worker.join(2)
    assert results and results[0] is bundle


def test_complete_bundle_is_reclaimed_without_display():
    """Proves that with display off a bundle is reclaimed and its buffer released when the stages finish."""
    pool, released = make_pool(display=False)
    bundle = start_frame(pool, 0, pointer=42)
    finish_stages(pool, bundle)
    assert released == [42]
    assert pool.stats()["free"] == 3
    assert bundle.parts == COMPLETE


def test_leased_bundle_is_never_reclaimed_even_when_expired():
    """Proves a bundle that still has a lease is kept even when it is older than the maximum age."""
    clock = Clock()
    pool, released = make_pool(clock=clock)
    bundle = start_frame(pool, 0)
    clock.now += 100
    pool.mark(bundle, Part.DETECTIONS)  # every update runs reclaim
    assert released == []
    assert pool.stats()["active"] == 1


def test_expired_unleased_bundle_is_reclaimed():
    """Proves a finished bundle nobody displayed is reclaimed and counted as expired once it is too old."""
    clock = Clock()
    pool, released = make_pool(size=1, clock=clock)
    bundle = start_frame(pool, 0, pointer=7)
    finish_stages(pool, bundle)
    assert released == []
    clock.now += 3
    pool.acquire(threading.Event())
    assert released == [7]
    assert pool.stats()["expired"] == 1


def test_abandoned_bundle_is_reclaimed():
    """Proves an abandoned bundle is reclaimed at once, its buffer released and the abandonment counted."""
    pool, released = make_pool()
    bundle = start_frame(pool, 0, pointer=9)
    pool.abandon(bundle)
    assert released == [9]
    assert pool.stats()["abandoned"] == 1


def test_display_prefers_newest_complete_and_supersedes_older():
    """Proves display picks the newest complete frame, releases older ones, and frees the shown one on release."""
    pool, released = make_pool()
    first = start_frame(pool, 0, pointer=1)
    second = start_frame(pool, 1, pointer=2)
    finish_stages(pool, first)
    finish_stages(pool, second)
    choice = pool.take_for_display()
    assert choice.bundle is second
    assert choice.complete is True
    assert choice.tracks == "tracks-1"
    assert released == [1]
    pool.release(choice.bundle)
    assert released == [1, 2]


def test_display_falls_back_to_incomplete_after_the_grace_period():
    """Proves that after the grace period display shows an incomplete frame with the previous frame's tracks."""
    clock = Clock()
    pool, _ = make_pool(clock=clock)
    first = start_frame(pool, 0)
    finish_stages(pool, first)
    assert pool.take_for_display().bundle is first
    pool.release(first)
    second = start_frame(pool, 1)
    clock.now += 0.3
    choice = pool.take_for_display()
    assert choice.bundle is second
    assert choice.complete is False
    assert choice.tracks == "tracks-0"
    assert choice.tracks_frame_id == 0


def test_display_waits_for_a_complete_frame_during_the_grace_period():
    """Proves display returns nothing while an incomplete frame is within the grace period, then the complete one."""
    clock = Clock()
    pool, _ = make_pool(clock=clock)
    first = start_frame(pool, 0)
    finish_stages(pool, first)
    pool.release(pool.take_for_display().bundle)
    second = start_frame(pool, 1)
    clock.now += 0.1
    assert pool.take_for_display() is None
    finish_stages(pool, second)
    choice = pool.take_for_display()
    assert choice.bundle is second
    assert choice.complete is True


def test_display_does_not_jump_past_frames_still_in_inference():
    """Proves display never skips over older frames that are still being inferred."""
    pool, _ = make_pool(size=4)
    first = start_frame(pool, 0)
    second = start_frame(pool, 1)
    start_frame(pool, 2)
    start_frame(pool, 3)
    finish_stages(pool, first)
    pool.release(pool.take_for_display().bundle)
    assert pool.take_for_display() is None  # frames 1 to 3 are unfinished, so display must wait
    finish_stages(pool, second)
    choice = pool.take_for_display()
    assert choice.bundle is second
    assert choice.complete is True


def test_publish_records_the_source_due_time():
    """Proves a published frame carries the given due time, or the current time when none is given."""
    clock = Clock()
    pool, _ = make_pool(clock=clock)
    pool.publish_frame(pool.acquire(threading.Event()), 1, 0, 0, 0, due_at=95.0)
    pool.publish_frame(pool.acquire(threading.Event()), 2, 1, 0, 1)
    assert pool.to_infer.get_nowait().due_at == 95.0
    assert pool.to_infer.get_nowait().due_at == clock.now


def test_display_returns_none_without_a_newer_frame():
    """Proves display returns None when the pool holds no frame."""
    pool, _ = make_pool()
    assert pool.take_for_display() is None


def test_buffer_released_once_after_incomplete_display_finishes():
    """Proves a buffer shown before its stages finish is released exactly once, when the last stage finishes."""
    pool, released = make_pool()
    bundle = start_frame(pool, 0, pointer=5)
    choice = pool.take_for_display()
    pool.release(choice.bundle)
    assert released == []
    finish_stages(pool, bundle)
    assert released == [5]


def test_finish_reclaims_complete_undisplayed_bundles():
    """Proves finish() releases the buffers of complete bundles that were never displayed."""
    pool, released = make_pool()
    bundle = start_frame(pool, 0, pointer=3)
    finish_stages(pool, bundle)
    pool.finish()
    assert released == [3]


def test_reused_bundle_is_clean():
    """Proves a reused bundle has a new generation and no tracks, rows or pointer left from its previous frame."""
    pool, _ = make_pool(size=1, display=False)
    bundle = start_frame(pool, 0)
    finish_stages(pool, bundle)
    generation = bundle.generation
    reused = pool.acquire(threading.Event())
    assert reused is bundle
    assert reused.generation == generation + 1
    assert reused.tracks is None
    assert reused.rows is None
    assert reused.nv12_pointer is None
