#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Run one YOLO26 ONNX detector with ByteTrack on a video held in memory, optionally looping, and print JSON stats.

Decodes with NVDEC into CUDA memory (or OpenCV on the CPU), tracks objects, and can show the annotated frames in a
window and record them to MP4. The model is loaded once and reused across loops, and the tracker is reset between
loops. Prints one JSON summary at the end; --loops 0 repeats until Q, Esc, closing the window or Ctrl-C.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import signal
import numpy as np
import sys
import threading
import time
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / ".ultralytics")

import cv2
import onnxruntime as ort

from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from video_reader import create_reader


class CudaOrigImagePredictor(DetectionPredictor):
    """A detection predictor for CUDA tensor input, which carries no original image of its own.

    The caller sets external_orig_imgs to the frame (or a shape-only stand-in) before each prediction, so boxes are
    scaled back to the original frame size."""

    external_orig_imgs = None

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-processes as the base class does, but with external_orig_imgs in place of the original images.
        Raises RuntimeError if the caller did not set them."""
        if self.external_orig_imgs is None:
            raise RuntimeError("CUDA input predictor missing original-frame metadata")
        return super().postprocess(preds, img, self.external_orig_imgs, **kwargs)


class DisplayMailbox:
    """A one-slot, thread-safe hand-off from the inference thread to the display loop.

    Publishing over an unread item replaces it and counts a drop, so the display always shows the newest frame and
    never slows inference down."""

    def __init__(self) -> None:
        """Creates an empty mailbox with zero published and dropped counts."""
        self._lock = threading.Lock()
        self._item = None
        self._dropped = 0
        self._published = 0

    def has_pending(self) -> bool:
        """True if an item is waiting to be taken."""
        with self._lock:
            return self._item is not None

    def publish(self, item) -> None:
        """Stores an item, replacing and counting as dropped any item that was not taken."""
        with self._lock:
            if self._item is not None:
                self._dropped += 1
            self._item = item
            self._published += 1

    def note_drop(self) -> None:
        """Counts a frame the producer skipped because the display had not taken the previous one."""
        with self._lock:
            self._dropped += 1

    def take(self):
        """Removes and returns the waiting item, or None if there is none."""
        with self._lock:
            item = self._item
            self._item = None
            return item

    @property
    def dropped(self) -> int:
        """The number of frames dropped so far, replaced or skipped."""
        with self._lock:
            return self._dropped

    @property
    def published(self) -> int:
        """The number of items published so far."""
        with self._lock:
            return self._published


def parse_args():
    """Parses the command line; --input and --model are required. Exits with a usage message on bad options."""
    p = argparse.ArgumentParser(description="YOLO26 ONNX + ByteTrack memory-backed loop demo")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="MP4 path; with --loops>1, loop suffixes are added (not valid with --loops 0)")
    p.add_argument("--decoder", choices=("opencv", "nvdec"), default="nvdec")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--iou", type=float, default=.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=1, help="Sequential frames per inference; requires matching static ONNX export")
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--loops", type=int, default=1, help="1=single pass, N=finite passes, 0=repeat until Q/Esc/window close/Ctrl-C")
    p.add_argument("--instance-id", default="single", help="Label used in GUI title and JSON")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="Per-loop test bound; zero means full source")
    return p.parse_args()


def select(path, requested):
    """Chooses the ONNX Runtime backend for the model at `path` and returns (backend name, provider list, all
    available providers). "cuda" must work or RuntimeError is raised; "auto" tries to open the model on CUDA and
    falls back to the CPU, printing the reason to stderr if that fails; "cpu" never touches CUDA."""
    avail = ort.get_available_providers()
    if requested == "cpu":
        return "cpu", ["CPUExecutionProvider"], avail
    if requested == "cuda" and "CUDAExecutionProvider" not in avail:
        raise RuntimeError(f"CUDA requested but unavailable; providers={avail}")
    if "CUDAExecutionProvider" not in avail:
        return "cpu", ["CPUExecutionProvider"], avail
    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = session.get_providers()
        if "CUDAExecutionProvider" not in active:
            if requested == "cuda":
                raise RuntimeError(f"CUDA provider failed; active={active}")
            return "cpu", ["CPUExecutionProvider"], avail
        return "cuda", active, avail
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA provider failed to initialize: {exc}") from exc
        print(f"[backend] explicit CPU fallback: {exc}", file=sys.stderr)
        return "cpu", ["CPUExecutionProvider"], avail


def annotate(frame, result, loop_id, frame_id, fps, copy_frame=True):
    """Draws the boxes with class, score and track id, plus a status line with loop, frame, FPS and box count.
    Returns the annotated image: a copy of `frame`, or `frame` itself, changed in place, if `copy_frame` is false."""
    out = frame.copy() if copy_frame else frame
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xy = boxes.xyxy.cpu().numpy().astype(int)
        cl = boxes.cls.cpu().numpy().astype(int)
        cf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xy)
        names = result.names
        for (x1, y1, x2, y2), c, score, tid in zip(xy, cl, cf, ids):
            color = ((37 * int(c) + 80) % 255, (17 * int(c) + 150) % 255, (97 * int(c) + 40) % 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            name = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
            label = f"{name} {score:.2f}" + (f" id={int(tid)}" if tid is not None else "")
            cv2.putText(out, label, (x1, max(y1 - 7, 18)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"loop={loop_id} frame={frame_id} processing_fps={fps:.1f} boxes={len(boxes) if boxes is not None else 0}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        .7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def is_nvdec(reader) -> bool:
    """True if the reader decodes with NVDEC, in memory or from a file."""
    return str(reader.backend).startswith("nvdec")


def reset_tracking(model) -> None:
    """Clears the tracker state held by the model's predictor so track ids restart on the next pass. Does nothing
    before the first prediction, when there is no predictor yet."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for tracker in getattr(predictor, "trackers", []):
        reset = getattr(tracker, "reset", None)
        if reset is not None:
            reset()
    if hasattr(predictor, "vid_path"):
        predictor.vid_path = [None] * len(predictor.vid_path)


def loop_output_path(base: Path | None, loop_id: int, total_loops: int) -> Path | None:
    """The MP4 path for a loop: None without --output, `base` itself for a single pass, otherwise `base` with
    "-loop-NNN" added before the extension (".mp4" if it has none)."""
    if base is None:
        return None
    if total_loops == 1:
        return base
    suffix = base.suffix or ".mp4"
    return base.with_name(f"{base.stem}-loop-{loop_id:03d}{suffix}")


def validate_model_shape(path, batch_size, imgsz):
    """Reject silent AutoBackend splitting and unsupported dynamic/non-square inputs."""
    import onnx
    graph = onnx.load(str(path), load_external_data=False).graph
    shape = [d.dim_value if d.HasField("dim_value") else None
             for d in graph.input[0].type.tensor_type.shape.dim]
    expected = [batch_size, 3, imgsz, imgsz]
    if shape != expected:
        raise ValueError(f"ONNX input {shape} does not match {expected}; export with --batch-size {batch_size} --imgsz {imgsz}")
    return shape


def run_one_loop(args, reader, model, backend, target_size, loop_id,
                 output_path, mailbox, stop_event, runtime):
    """Batch consecutive frames, tracking each real frame in order; flush at EOF/stop.

    Ultralytics 8.4.190 pads a short static batch and trims raw outputs before
    tracking. Never pad source frames here: padding must not advance ByteTrack.
    """
    batch_size = args.batch_size
    writer = None
    read_count = processed = written = calls = 0
    stopped = False
    processed_times = deque(maxlen=4096)
    fill_wait = deque(maxlen=4096)
    read_to_tracked = deque(maxlen=4096)
    batch_sizes = {}
    loop_start = time.monotonic()
    nvdec = is_nvdec(reader)
    shape_only = nvdec and "bytetrack" in Path(args.tracker).name.lower()
    host_before_track = not nvdec or not shape_only
    pending = []

    def flush():
        nonlocal processed, written, calls
        if not pending:
            return
        ready = time.monotonic()
        k = len(pending)
        fill_wait.extend((ready - item[3]) * 1000 for item in pending)
        kwargs = dict(persist=True, tracker=args.tracker, conf=args.conf,
                      iou=args.iou, imgsz=target_size, batch=batch_size,
                      device=0 if backend == "cuda" else "cpu", verbose=False)
        try:
            if nvdec:
                inputs = pending[0][1] if k == 1 else reader._torch.cat([item[1] for item in pending], dim=0)
                CudaOrigImagePredictor.external_orig_imgs = [
                    item[0] if item[0] is not None else reader.shape_only_image for item in pending]
                results = model.track(inputs, predictor=CudaOrigImagePredictor, **kwargs)
            else:
                inputs = pending[0][0] if batch_size == 1 else [item[0] for item in pending]
                results = model.track(inputs, **kwargs)
        finally:
            CudaOrigImagePredictor.external_orig_imgs = None
        if len(results) != k:
            raise RuntimeError(f"Expected {k} real results, received {len(results)}")
        calls += 1
        batch_sizes[k] = batch_sizes.get(k, 0) + 1
        completed = time.monotonic()
        processed_times.extend([completed] * k)
        read_to_tracked.extend((completed - item[3]) * 1000 for item in pending)
        if runtime["actual"] is None:
            session = getattr(getattr(getattr(model, "predictor", None), "model", None), "session", None)
            if session is not None:
                runtime["actual"] = session.get_providers()
        for (frame, _, device_frame, _, frame_id), result in zip(pending, results):
            processed += 1
            if mailbox is not None and writer is None and mailbox.has_pending():
                mailbox.note_drop()
                continue
            if frame is None and (writer is not None or mailbox is not None):
                # Each entry owns its decoded RGB tensor, not reader.last_device_frame.
                frame = reader.host_frame(device_frame)
            if frame is not None and (writer is not None or mailbox is not None):
                out = annotate(frame, result, loop_id, frame_id,
                               processed / max(time.monotonic() - loop_start, 1e-6), copy_frame=False)
                if writer is not None:
                    writer.write(out)
                    written += 1
                if mailbox is not None:
                    mailbox.publish((loop_id, frame_id, out))
        pending.clear()

    try:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     reader.fps, (reader.width, reader.height))
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not open {output_path}")
        print(f"[loop {loop_id}] start frames={reader.frame_count} batch={batch_size} output={output_path}")
        while True:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if args.max_frames and read_count >= args.max_frames:
                break
            ok, frame, timestamp = reader.read(host_frame=host_before_track)
            if not ok:
                break
            read_at = time.monotonic()
            read_count += 1
            if host_before_track and (frame is None or frame.shape[:2] != (reader.height, reader.width)):
                raise RuntimeError("Unexpected decoder frame shape")
            detector_input = device_frame = None
            if nvdec:
                detector_input, prep = reader.cuda_input(target_size)
                if (detector_input.device.type != "cuda" or detector_input.dtype != reader._torch.float32
                        or tuple(detector_input.shape) != (1, 3, target_size, target_size)):
                    raise RuntimeError("Invalid CUDA detector input")
                if read_count == 1:
                    low, high = float(detector_input.min().item()), float(detector_input.max().item())
                    if low < 0 or high > 1:
                        raise RuntimeError(f"Invalid CUDA input range: {low}, {high}")
                if frame is None and (writer is not None or mailbox is not None):
                    device_frame = reader.last_device_frame
            pending.append((frame, detector_input, device_frame, read_at, read_count))
            if len(pending) == batch_size:
                flush()
        flush()
        if not read_count and not stopped:
            raise RuntimeError(f"Loop {loop_id} decoded no frames")
        if not args.max_frames and not stopped and reader.frame_count and read_count != reader.frame_count:
            raise RuntimeError(f"Loop {loop_id} frame count mismatch: expected {reader.frame_count}, got {read_count}")
        if nvdec and read_count and not reader.device_verified:
            raise RuntimeError("CUDA decoder memory was not verified")
        elapsed = max(time.monotonic() - loop_start, 1e-6)
        def percentiles(values):
            return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else {"p50": 0.0, "p95": 0.0}
        return dict(loop=loop_id, frames_read=read_count, frames_processed=processed,
                    frames_written=written, inference_batches=calls, batch_sizes=batch_sizes,
                    batch_fill_wait_ms=percentiles(fill_wait), read_to_batch_complete_ms=percentiles(read_to_tracked),
                    latency_samples_retained=len(read_to_tracked),
                    output=str(output_path) if output_path else None,
                    processing_seconds=elapsed, processing_fps=processed / elapsed,
                    stopped_by_user=stopped, _processed_times=processed_times,
                    _start=loop_start, _end=time.monotonic())
    finally:
        pending.clear()
        CudaOrigImagePredictor.external_orig_imgs = None
        if writer is not None:
            writer.release()


def process_stream(args, mailbox=None, stop_event=None):
    """Runs the whole job: opens the reader, picks the backend, loads the model once and runs the requested loops.
    Frames go to `mailbox` for display when one is given, and `stop_event` ends the run early. Returns the
    summary stats dict, including private "_" keys the caller must remove before printing. Raises RuntimeError if
    NVDEC is requested without CUDA or no loop completes. Always closes the reader."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    model_shape = validate_model_shape(args.model, args.batch_size, args.imgsz)
    reader = create_reader(args.input, args.decoder, memory_input=True)
    runtime = {"actual": None}
    # Keep JSON history and async timing evidence bounded during --loops 0.
    loops = deque(maxlen=32)
    all_processed_times = deque(maxlen=4096)
    loops_completed = 0
    total_frames = 0
    total_written = 0
    total_start = None
    total_end = None
    stopped_by_user = False
    try:
        print(
            f"[source] decoder={reader.backend} compressed_bytes={getattr(reader, 'encoded_bytes_len', None)} "
            f"frames={reader.frame_count} size={reader.width}x{reader.height} fps={reader.fps:.3f}"
        )
        backend, probe, avail = select(args.model, args.backend)
        print(f"[backend] requested={args.backend} selected={backend}")
        print(f"[backend] ORT available={avail}")
        print(f"[backend] ORT probe active={probe}")
        model = YOLO(str(args.model))
        model_size = getattr(model, "imgsz", args.imgsz)
        target_size = int(model_size[0] if isinstance(model_size, (tuple, list)) else model_size)
        if is_nvdec(reader) and backend != "cuda":
            raise RuntimeError("NVDEC memory mode requires CUDAExecutionProvider; refusing CPU fallback")

        loop_id = 1
        while args.loops == 0 or loop_id <= args.loops:
            if stop_event is not None and stop_event.is_set():
                stopped_by_user = True
                break
            if loop_id > 1:
                reader.reset()
                reset_tracking(model)
                print(f"[loop {loop_id}] decoder seek_to_index(0); tracker reset; model/session retained")
            output_path = loop_output_path(args.output, loop_id, args.loops)
            loop_stats = run_one_loop(
                args, reader, model, backend, target_size, loop_id,
                output_path, mailbox, stop_event, runtime,
            )
            loops.append(loop_stats)
            loops_completed += 1
            total_frames += loop_stats["frames_processed"]
            total_written += loop_stats["frames_written"]
            all_processed_times.extend(loop_stats.pop("_processed_times"))
            loop_start = loop_stats.pop("_start")
            if total_start is None:
                total_start = loop_start
            total_end = loop_stats.pop("_end")
            if loop_stats["stopped_by_user"]:
                stopped_by_user = True
                break
            loop_id += 1

        if total_start is None:
            now = time.monotonic()
            total_start = now
            total_end = now
        total_seconds = max(total_end - total_start, 1e-6)
        if not loops and not stopped_by_user:
            raise RuntimeError("No loops completed")
        return {
            "instance_id": args.instance_id,
            "batch_size": args.batch_size,
            "model_input_shape": model_shape,
            "decoder": reader.backend,
            "memory_bytes": getattr(reader, "encoded_bytes_len", None),
            "memory_input": bool(getattr(reader, "encoded_bytes_len", None) is not None),
            "loops_requested": args.loops,
            "loops_completed": loops_completed,
            "loops": list(loops),
            "loops_history_retained": len(loops),
            "frames_processed": total_frames,
            "frames_written": total_written,
            "processing_seconds": total_seconds,
            "processing_fps": total_frames / total_seconds if total_frames else 0.0,
            "output": str(args.output) if args.output is not None else None,
            "backend": backend,
            "ort_providers": runtime["actual"] or probe,
            "model_loaded_once": True,
            "tracker_reset_between_loops": loops_completed > 1,
            "stopped_by_user": stopped_by_user,
            "_processing_start": total_start,
            "_processing_end": total_end,
            "_processed_times": all_processed_times,
        }
    finally:
        reader.close()


def run_async_gui(args):
    """Runs process_stream on a worker thread and shows its frames in a window on the calling thread, since the
    window must be driven from the main thread. Q, Esc, closing the window or Ctrl-C stops the run. Prints the
    stats as JSON with the display counters added, and re-raises any error from the worker."""
    mailbox = DisplayMailbox()
    stop_event = threading.Event()
    break_previous = signal.signal(signal.SIGBREAK, lambda *_: stop_event.set()) if hasattr(signal, "SIGBREAK") else None
    done_event = threading.Event()
    result_box = {}
    error_box = {}
    title = f"YOLO26 memory loop [{args.instance_id}]"

    def worker():
        """Runs the job and keeps its stats or error for the main thread; always signals completion."""
        try:
            result_box["stats"] = process_stream(args, mailbox=mailbox, stop_event=stop_event)
        except BaseException as exc:
            error_box["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(target=worker, name=f"yolo-inference-{args.instance_id}", daemon=False)
    thread.start()
    displayed = 0
    display_times = deque(maxlen=4096)
    window_created = False
    try:
        while not done_event.is_set() or mailbox.has_pending():
            item = mailbox.take()
            if item is not None:
                loop_id, frame_id, frame = item
                displayed += 1
                now = time.monotonic()
                display_fps = displayed / max(now - display_times[0], 1e-6) if display_times else 0.0
                cv2.putText(
                    frame,
                    f"display_fps={display_fps:.1f} drops={mailbox.dropped} loop={loop_id} frame={frame_id}",
                    (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(title, frame)
                display_times.append(time.monotonic())
                window_created = True

            key = cv2.waitKey(1) & 255
            if key in (27, ord("q")):
                print("[run] stopped by user")
                stop_event.set()
                break
            if window_created:
                try:
                    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        print("[run] window closed by user")
                        stop_event.set()
                        break
                except cv2.error:
                    stop_event.set()
                    break
            if not done_event.is_set() and item is None:
                time.sleep(.001)
    except KeyboardInterrupt:
        print("[run] interrupted by user")
        stop_event.set()
    finally:
        cv2.destroyAllWindows()
        stop_event.set()
        thread.join()
        if break_previous is not None:
            signal.signal(signal.SIGBREAK, break_previous)

    if "error" in error_box:
        raise error_box["error"]
    stats = result_box["stats"]
    processing_start = stats["_processing_start"]
    processing_end = stats["_processing_end"]
    processed_times = stats["_processed_times"]
    common_start = max(processing_start, display_times[0]) if display_times else 0.0
    common_end = min(processing_end, display_times[-1]) if display_times else 0.0
    common_seconds = max(common_end - common_start, 0.0)
    displayed_common = sum(common_start <= t <= common_end for t in display_times) if common_seconds else 0
    processed_common = sum(common_start <= t <= common_end for t in processed_times) if common_seconds else 0
    stats["frames_displayed"] = displayed
    stats["display_published"] = mailbox.published
    stats["display_drops"] = mailbox.dropped
    stats["frames_processed_common"] = processed_common
    stats["frames_displayed_common"] = displayed_common
    stats["common_interval_seconds"] = common_seconds
    stats["common_display_fps"] = displayed_common / common_seconds if common_seconds else 0.0
    stats["display_fps_submit_window"] = (
        displayed / max(display_times[-1] - display_times[0], 1e-6)
        if len(display_times) >= 2 else 0.0
    )
    stats.pop("_processing_start", None)
    stats.pop("_processing_end", None)
    stats.pop("_processed_times", None)
    print(json.dumps(stats, indent=2))


def main():
    """Entry point: checks the arguments and runs headless or with a window, printing the JSON stats. Raises
    ValueError for a negative --loops or --output with --loops 0, and FileNotFoundError for a missing input or
    model."""
    args = parse_args()
    if args.loops < 0:
        raise ValueError("--loops must be >= 0")
    if args.loops == 0 and args.output is not None:
        raise ValueError("--output cannot be combined with --loops 0; use a finite loop count")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.no_display:
        stop_event = threading.Event()
        previous = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        break_previous = signal.signal(signal.SIGBREAK, lambda *_: stop_event.set()) if hasattr(signal, "SIGBREAK") else None
        try:
            stats = process_stream(args, stop_event=stop_event)
        finally:
            signal.signal(signal.SIGINT, previous)
            if break_previous is not None:
                signal.signal(signal.SIGBREAK, break_previous)
        stats.pop("_processing_start", None)
        stats.pop("_processing_end", None)
        stats.pop("_processed_times", None)
        print(json.dumps(stats, indent=2))
    else:
        run_async_gui(args)


if __name__ == "__main__":
    main()
