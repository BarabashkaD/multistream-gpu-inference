# External batched inference design

One inference process serves many pipeline instances. Each instance decodes one stream in its own process and shares decoded frames with the server through CUDA IPC, with no copy. The server prepares and detects whole batches on the GPU and returns detections. Each instance tracks its own stream and shows its own window.

The code is self-contained. The single-process baseline is v2, kept unchanged on the branch [`v2-bundle-pipeline`](https://github.com/BarabashkaD/multistream-gpu-inference/tree/v2-bundle-pipeline). The clip is at `data/pexels-36304379.mp4`.

## Goals

- Sustain 16 streams at 1080p 25 FPS, about 400 inferences per second, with the detector on every decoded frame.
- Run as multiple OS processes that share one GPU, with one copy of the model.
- Keep per-stream track state in each instance.
- Keep one window per instance, with `--no-display` for headless runs.
- Bound memory: review what each process uses, reduce it, and report before and after figures.
- Provide a launcher and a benchmark harness that vary the stream count and produce a scaling table.

## Non-goals

- Containers. The target is native Windows first. CUDA IPC, NVDEC and named pipes are all unproven in Linux containers on WSL2, so this stays a later stage, and this version has no single-command container deployment.
- Emitting tracks to a downstream consumer. It is a project goal, left to a later stage.
- A systems-language component. The tracker is the likely candidate, because it is the one per-frame CPU cost left in each instance.
- `--prepare gl`. It stays in v2, on the branch [`v2-bundle-pipeline`](https://github.com/BarabashkaD/multistream-gpu-inference/tree/v2-bundle-pipeline). Preparation here runs as batched CUDA kernels in the server.
- More than one clip resolution in one server. The server batches frames by stacking them, so every stream must have the size of the first.

## Measured evidence

All figures are from probes on an RTX 2070 SUPER 8 GB, i7-9700K, 32 GB RAM, Windows 11, with 16 separate client processes and no tracker or window. Clients imported neither PyTorch nor ONNX Runtime.

| Experiment (16 clients unless stated) | FPS | GPU | NVDEC | Notes |
| --- | --- | --- | --- | --- |
| Separate v2 processes, 8 instances, for comparison | 157 | 98% | 9% | total caps at about 156; 16 did not fit in host memory: available host memory fell by about 1.2 GB per instance, and a 16-instance run was stopped by the safety limit at 3.0 GB available |
| Decode and IPC with batched prepare, no model, max pace | 797 | 32% | 49% | IPC is not the limit at 400 FPS |
| Same at real time | 408 | 21% | 32% | every checksum matched |
| 640x640, batch 4, max pace | 314 | 90% | 22% | prepare 0.44 ms, infer 2.53 ms per frame |
| 640x640, batch 8 | 356 | 94% | 25% | infer 2.24 ms |
| 640x640, batch 16 | 374 | 95% | 26% | infer 2.19 ms |
| 640x384, batch 8 | 507 | 91% | 35% | infer 1.43 ms, p95 latency 158 ms |
| 640x384, batch 16 | 551 | 94% | 39% | infer 1.33 ms, p95 latency 565 ms, 700 MB more GPU memory |

The 640x640 input holds a 640x360 picture, so 44% of it is padding. A 640x384 model removes the padding and runs inference about 37% faster. This is the change that clears the target.

**Detection differences at 640x384.** Against the 640x640 model on 204 frames of the clip, at confidence 0.5 or more, 97.3% of detections were found by the other model, with mean IoU 0.994 and counts of 625 and 626. At confidence 0.25 or more, 93.4% were found, with mean IoU 0.990 and counts of 1044 and 1051. The models are close but not identical.

**Batched output.** Detections with confidence 0.1 or more matched batch-1 output in 20 of 20 compared batches at batch 8 and 16. Low-confidence rows can change order. Batched preparation matched `nv12_to_input` exactly.

**Instance failure.** One client killed in mid-run left the server and the other 15 clients running to completion.

**NVIDIA's guidance.** CUDA IPC on Windows is documented as supported for compatibility and not recommended because of its performance cost. The cost measured above leaves about twice the required throughput, so this design uses it and records that it depends on a discouraged path.

## Processes

```text
 instance process (one per stream)                      inference server (one)
 +-------------------------------------------+         +-----------------------------------+
 | Reader -> Decoder -> Submit ---- frame ---|-------> | Receive thread -> FairQueue       |
 |             |          |    (pointer plus |  pipe   |                      |            |
 |             v          v     IPC handle)  |         |                      v            |
 |          bundle pool  in-flight table     |         | Batch loop: open handles, stack,  |
 |             ^          ^                  |         | prepare, ONNX, read back          |
 |  Display <--+   Collect <---- rows -------|<------- | reply per frame                   |
 |  (window)       |                         |         +-----------------------------------+
 |              Tracker (ByteTrack, no torch)|
 +-------------------------------------------+
```

The boundaries follow the costs. GPU compute needs one owner to batch it, so the model, preparation and PyTorch live in one process. Tracking is Python work that would fight the server's GIL, so it stays in the instance, which also keeps per-stream state with its stream. Decoding stays in the instance because NVDEC has spare capacity and the frame has to stay available for that instance's window.

`run_fleet.py` starts one server and N instances, watches safety limits and collects results.

## Frame path

1. **Decoder** locks a frame, syncs on the CUDA stream and publishes the bundle, as in v2, but keeps only the pointer. It creates no tensor.
2. **Submit** takes the bundle from `to_infer`, applies the drop rule and records it in the in-flight table. It exports a `cuIpcGetMemHandle` handle once per distinct decoder pointer, and sends the frame id, pointer and any new handle. The bundle's lease moves to the in-flight table.
3. **Server receive thread** routes messages into a `FairQueue` with one FIFO per stream.
4. **Batch loop** takes frames round-robin. It waits up to `linger_ms` for the largest configured batch size, then uses the largest configured size that the waiting frames can fill. (Note, 2026-11-03: the `linger_ms` rule was replaced by the gather-and-pad rule; see Part 7 of [`docs/design/containers-live-stats.md`](containers-live-stats.md).) It opens new handles once per pointer, stacks frames, prepares and synchronises, runs ONNX through IO binding, and copies the result to the host.
5. **Reply:** each frame gets a result holding only rows at or above the server's `min_conf`.
6. **Collect** matches the result to its bundle, stores the rows, marks `Part.DETECTIONS` and passes the bundle to `to_track`.
7. **Tracker** runs on the rows, publishes tracks and reclaim returns the buffer. Display behaves as in v2.

`Part.INPUT` is gone: `COMPLETE` is `FRAME | DETECTIONS | TRACKS`.

Detections travel back over the named pipe, not shared memory as first sketched. The probe carried 7.2 KB per frame this way at over 500 FPS with no effect on throughput, and filtered rows are far smaller.

## Protocol

Messages are `NamedTuple`s pickled over one `multiprocessing.connection` named pipe per instance. Pipes are authenticated with a per-run HMAC key passed through an inherited environment variable, so messages are unpickled only from an authenticated peer.

| Direction | Message | Purpose |
| --- | --- | --- |
| instance to server | `Hello(stream, width, height)` | joins |
| server to instance | `Start(names)` | sent to all once `clients` instances have joined, so all streams start together; `names` maps class ids to labels for the overlay |
| server to instance | `Reject(reason)` | wrong frame size; a stream number that has already joined, or that has left; a frame handle that fails to open; or handles that failed to close |
| instance to server | `Frame(frame_id, pointer, handle)` | one frame; `handle` is bytes when the pointer is new, else `None` |
| server to instance | `Result(frame_id, rows)` | `rows` is float32 bytes, N x 6 |
| instance to server | `Bye(stats)` | no more frames; carries the instance's summary |
| server to instance | `Closed()` | handles are closed, so the decoder may be destroyed; withheld when closing them failed |

The handshake order matters. An instance must not destroy its decoder before `Closed`, because the server still maps that memory. If closing a stream's handles fails, the server keeps the mapping, sends only `Reject`, and never sends `Closed`; the instance then keeps its decoder until the process exits.

## Components

**`pipeline/`, used by instances. It must import neither `torch`, `onnxruntime` nor `ultralytics`. A CPU test enforces this in a subprocess.**

| Module | Purpose |
| --- | --- |
| `bundle.py`, `stage.py`, `metrics.py`, `source.py`, `overlay.py`, `gl_util.py` | Copied from v2. `bundle.py` loses `input_tensor`, `nv12` and `Part.INPUT`, and `detections` becomes a NumPy array |
| `config.py` | Instance configuration |
| `protocol.py` | Message types and pipe address helper |
| `gpu.py` | CUDA context setup and IPC handles through `ctypes`, with no PyTorch |
| `decoder.py` | v2 decoder without `wrap_nv12` |
| `remote.py` | `InFlight`, `SubmitStage`, `CollectStage` |
| `bytetrack.py` | ByteTrack in NumPy and `lap` |
| `tracking.py` | `TrackerCore`, `TrackerStage` over `bytetrack.py` |
| `display.py` | v2 display with `gpu.py` in place of PyTorch, plus window position |
| `app.py` | `run_instance` |

**`server/`, used only by the inference server.**

| Module | Purpose |
| --- | --- |
| `config.py` | Server configuration |
| `letterbox.py` | `Letterbox` with separate target width and height |
| `prepare.py` | `BatchPreparer`: NV12 batch to input tensor |
| `detector.py` | One ONNX session per batch size, with IO binding |
| `imports.py` | Opens and closes IPC handles per stream |
| `queueing.py` | `FairQueue` |
| `service.py` | `InferenceServer`: accept, receive and batch loops |

The server returns rows already in source coordinates, so instances need no letterbox and never import `server/`. The server shares only `pipeline.protocol`, `pipeline.gpu` and `pipeline.metrics`.

**Scripts.** `run_server.py`, `run_instance.py`, `run_fleet.py`, `bench.py`, `tools/export_models.py`.

## Tracker

The instance cannot use Ultralytics' `BYTETracker`, because importing `ultralytics.utils` loads PyTorch, which costs 885 MB of private memory per process. `bytetrack.py` is a port of the Ultralytics ByteTrack tracker to NumPy, with an XYAH Kalman filter, IoU cost with score fusion, two-stage association, and the thresholds from `bytetrack.yaml`: high 0.25, low 0.1, new track 0.25, match 0.8, buffer 30 frames. `lap` provides the assignment solver. A test compares it against Ultralytics, which remains a test-only dependency, and requires identical IDs and boxes within 0.01 pixels.

## Scheduling and back-pressure

- **Fairness.** `FairQueue.take` rotates its start stream on every call, so no stream can occupy every batch.
- **Batch sizes.** `--batch-sizes` defaults to `1,8`. Batch 8 clears the target with lower latency and less GPU memory than 16. The memory review dropped the batch 4 session, which had been the default with `1,4,8`: it saved 8.7% of the server's host private memory and 13.9% of its GPU memory with no loss of throughput, at the cost of a higher p95 latency ([`docs/memory-review.md`](../memory-review.md)).
- **Instance back-pressure.** The bundle pool bounds frames in flight per stream. When it is empty the decoder blocks, then the reader, as in v2.
- **Falling behind.** With the default `--drop-policy oldest` at real-time pace, a frame more than `max_lag_s` past its due time is skipped by Submit and counted as `dropped`. Dropped frames do not appear in the window, as in v2, and the window keeps showing the newest finished frame. With `--pace max` nothing is dropped. A measured run with one deliberately slow tracker shows the effect.
- **Server failure.** If the server's pipe breaks, or a frame waits longer than `server_timeout_s`, the instance records a stage failure and exits non-zero.
- **Instance failure.** The server sees the pipe close, drops that stream's queued frames, closes its handles and carries on.

## Memory

| Item | Before | Target | Evidence |
| --- | --- | --- | --- |
| Instance host private | 1.5 GB (v2) | about 350 MB | no PyTorch: 314 MB in the probe, 43 MB for all imports |
| NumPy thread buffers | +235 MB per process | +9 MB | `OPENBLAS_NUM_THREADS=1`, set before NumPy loads |
| Prepare temporaries | about 100 MB at batch 8 (arithmetic, not measured) | under 60 MB | full-resolution float copies replaced by index-based sampling; whether the target was met was not measured |
| ONNX Runtime GPU arena | 130 MB overhead | removed | `arena_extend_strategy=kSameAsRequested` |
| Server host private | 2994 MB | measured reduction | the memory review: CPU arena, memory patterns, session count |
| Instance GPU memory | 168 MB × 16 | measured | review `maxwidth`, `maxheight`, `latency` decoder options |

The pool holds no GPU tensors in an instance, so pool size costs only decoder surfaces. Default `pool_size` falls from 16 to 8, and the memory review lowered it to 6 with no dropped frames at 16 streams ([`docs/memory-review.md`](../memory-review.md)).

## Safety limits

An earlier crash of the test host has no known cause, and its RAM was not tested. `run_fleet.py` therefore stops every process if free GPU memory falls below 300 MB or available host memory below 3 GB, and reports that as the result.

## Testing

- CPU suite, the default: fake connections and fake converters cover the protocol, `FairQueue`, `InFlight`, Submit and Collect, tracker parity, `BatchPreparer` parity on CPU tensors, and the no-PyTorch import rule.
- GPU suite, marked `gpu`: two instances against a real server with a check that detections arrive, a killed instance, a killed server, a slow instance, frame order through the decoder, and a five-minute soak at 16 streams. Windows are tested for position and shutdown, and checked by eye once.
- Benchmarks: `bench.py` for 1, 2, 4, 8, 12 and 16 streams, headless at real-time pace and max pace, and with windows at real-time pace only.

## Risks

- CUDA IPC on Windows is the path NVIDIA recommends against, and it may change with a driver update.
- 16 windows add GPU work. At the measured 2.08 ms per displayed frame in v2, windows take about 83% of one thread's time, and the windowed case may miss 25 FPS. Benchmarks report both cases.
- The tracker adds CPU work the probes did not include: about 0.9 ms per frame, or 36% of a core across 16 streams.
- All streams use one clip, so real sites with busier scenes may cost more tracker time.
- The 640x384 model is tied to 16:9 sources.
