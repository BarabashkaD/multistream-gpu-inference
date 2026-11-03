# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the tracker core and stage: source coordinates, filtering, clipping, loop resets and parity with the
Ultralytics tracker."""
import threading

import numpy as np
import pytest
import torch

from pipeline.bundle import Bundle, BundlePool, Part
from pipeline.metrics import Metrics
from pipeline.nv12 import Letterbox, nv12_to_input, wrap_nv12
from pipeline.stage import END
from pipeline.tracking import TrackerCore, TrackerStage


def rows_for(box, conf=0.9, cls=2):
    """Builds a detector output of 300 rows with one detection (box, confidence, class) in the first row."""
    rows = np.zeros((300, 6), np.float32)
    rows[0] = [*box, conf, cls]
    return rows


def test_core_tracks_one_object_in_source_coordinates():
    """Proves a detection seen in two frames becomes one track whose box is in source pixels and keeps its class."""
    core = TrackerCore(Letterbox(1920, 1080, 640), (1920, 1080), "bytetrack.yaml", 0.1)
    tracks = core.update(rows_for((100, 240, 200, 300)))
    tracks = core.update(rows_for((102, 240, 202, 300)))
    assert len(tracks) == 1
    assert tracks[0, 2] == pytest.approx(606, abs=8)
    assert tracks[0, 1] == pytest.approx(300, abs=8)
    assert tracks[0, 6] == 2


def test_core_ignores_empty_rows():
    """Proves an all-zero detector output gives no tracks."""
    core = TrackerCore(Letterbox(1920, 1080, 640), (1920, 1080), "bytetrack.yaml", 0.1)
    assert len(core.update(np.zeros((300, 6), np.float32))) == 0


def test_core_clips_tracks_to_the_frame():
    """Proves a track that the tracker predicts outside the frame is clipped to the frame edges."""
    core = TrackerCore(Letterbox(1920, 1080, 640), (1920, 1080), "bytetrack.yaml", 0.1)
    predicted = np.array([[-30.0, -20.0, 1950.0, 1100.0, 1, 0.9, 2, 0]], np.float32)
    core.tracker.update = lambda boxes, image: predicted
    tracks = core.update(rows_for((100, 240, 200, 300)))
    assert tracks[0, :4].tolist() == [0.0, 0.0, 1920.0, 1080.0]


def test_core_drops_rows_below_its_confidence_threshold():
    """Proves detections below the core's confidence threshold never become tracks, however often they repeat."""
    core = TrackerCore(Letterbox(1920, 1080, 640), (1920, 1080), "bytetrack.yaml", 0.5)
    for _ in range(3):
        tracks = core.update(rows_for((100, 240, 200, 300), conf=0.4))
    assert len(tracks) == 0


def test_stage_resets_at_loop_change_and_signals_finished():
    """Proves the tracker stage resets the core when the loop number changes, stores each frame's tracks and
    releases its lease, and sets the finished event at the end marker."""
    calls = []
    released = []

    class FakeCore:
        """Stand-in tracker core that records update and reset calls."""

        def update(self, rows):
            """Records the call and returns no tracks."""
            calls.append("update")
            return np.zeros((0, 8), np.float32)

        def reset(self):
            """Records the call."""
            calls.append("reset")

    stop, failures, finished = threading.Event(), [], threading.Event()
    bundles = [Bundle(i, None, torch.zeros((1, 300, 6))) for i in range(2)]
    pool = BundlePool(bundles, released.append, display_enabled=False)
    for frame_id, loop in ((0, 0), (1, 1)):
        bundle = pool.acquire(stop)
        pool.publish_frame(bundle, 1 + frame_id, object(), frame_id, loop, frame_id)
        taken = pool.to_infer.get_nowait()
        pool.mark(taken, Part.INPUT | Part.DETECTIONS)
        pool.to_track.put(taken)
    pool.to_track.put(END)
    stage = TrackerStage(FakeCore(), pool, finished, Metrics(), stop, failures)
    stage.start()
    stage.join(5)
    assert not failures, failures
    assert calls == ["update", "reset", "update"]
    assert finished.is_set()
    assert released == [1, 2]  # tracks were stored and the lease was released
    assert pool.stats()["free"] == 2


@pytest.mark.gpu
def test_parity_with_model_track(clip_path, model_path):
    """Proves over the first 300 frames the tracker core gives the same track ids and boxes (within half a pixel) as
    Ultralytics' own `track` fed with the same input."""
    import PyNvVideoCodec as nvc
    from run_demo import CudaOrigImagePredictor
    from ultralytics import YOLO

    from pipeline.detector import Detector
    from video_reader import ShapeOnlyImage

    letterbox = Letterbox(1920, 1080, 640)
    detector = Detector(model_path, 0)
    core = TrackerCore(letterbox, (1920, 1080), "bytetrack.yaml", 0.25)
    yolo = YOLO(str(model_path))
    shape = ShapeOnlyImage(1080, 1920)
    demuxer = nvc.CreateDemuxer(str(clip_path))
    decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=True,
                                outputColorType=nvc.OutputColorType.NATIVE)
    tensor = torch.empty((1, 3, 640, 640), dtype=torch.float32, device="cuda:0")
    output = torch.empty((1, 300, 6), dtype=torch.float32, device="cuda:0")
    compared = 0
    while compared < 300:
        for _ in range(decoder.GetNumDecodedFrame(demuxer.Demux())):
            pointer = decoder.GetLockedFrame()
            decoder.SyncOnCUStream()
            nv12_to_input(wrap_nv12(pointer, 1920, 1080), letterbox, 1920, 1080, tensor)
            torch.cuda.current_stream().synchronize()
            detector.run(tensor, output)
            ours = core.update(output[0].cpu().numpy())
            CudaOrigImagePredictor.external_orig_imgs = [shape]
            theirs = yolo.track(tensor, predictor=CudaOrigImagePredictor, persist=True, tracker="bytetrack.yaml",
                                conf=.25, iou=.7, imgsz=640, device=0, verbose=False)[0].boxes
            CudaOrigImagePredictor.external_orig_imgs = None
            decoder.UnlockFrame(pointer)
            their_rows = theirs.data.cpu().numpy() if theirs.is_track else np.zeros((0, 7))
            assert len(ours) == len(their_rows), f"frame {compared}"
            if len(ours):
                ours_sorted = ours[np.argsort(ours[:, 4])]
                theirs_sorted = their_rows[np.argsort(their_rows[:, 4])]
                assert np.array_equal(ours_sorted[:, 4], theirs_sorted[:, 4]), f"frame {compared}"
                assert np.abs(ours_sorted[:, :4] - theirs_sorted[:, :4]).max() < 0.5, f"frame {compared}"
            compared += 1
