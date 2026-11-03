# Bundle pipeline design

A new demo built from scratch next to the current one. Every frame travels through independent threads as a *bundle* from a preallocated pool, without being copied after it leaves the decoder.

The current demo, the v1 version on the branch [v1-single-stream](https://github.com/BarabashkaD/multistream-gpu-inference/tree/v1-single-stream) (`run_demo.py`, `video_reader.py`, `launch_instances.py`), stays untouched, and its `run_demo.py` and `video_reader.py` are also at the root of this branch. The new code lives in `pipeline/`, starts from `run_pipeline.py`, and shares `data/` and `yolo26n.onnx` at the repository root.

## Goals

- Move decoded frames through the pipeline with no copy after the decoder's own buffer.
- Run reading, decoding, preparation, inference, tracking and display independently, so a slow stage cannot stall a faster one except through deliberate back-pressure.
- Show frames from GPU memory at a configured rate and size, with annotation drawn by OpenGL.
- Bound all memory with a preallocated pool.
- Report live and final statistics for every stage.

## Non-goals

- Reaching 16 streams or 400 inferences per second in this stage. One process runs one frame at a time, so removing copies and stalls cannot reach that rate. Batched inference measured 405 frames/s at batch 4, for inference alone, so the target needs the next stage.
- Central or batched inference. This stage uses the model at batch size 1. A central inference process is the planned next stage. A second process has read a decoder frame through CUDA IPC, for one frame on native Windows.
- Recording and RTSP input. Containers and downstream output are part of the project's target, left to a later stage.

## Pipeline

```text
 Reader thread      Decoder thread          Prepare thread       Inference thread    Tracker thread
 demux from RAM --> NVDEC, lock frame --> NV12 to input --> ONNX, IO binding --> ByteTrack
 copy packet        acquire bundle          CUDA kernels         detections on GPU   tracks on CPU
      |                  |                            |                      |
      |  packet queue    v                            v                      v
      +----------> +------------------------------------------------------------+
                   |  Bundle pool (preallocated, fixed size, leases, reclaim)   |
                   +------------------------------------------------------------+
                                        ^                         |
                       take newest      |                         | release
                   Display, main thread |                         v
                   NV12 shader + overlay                  decoder unlocks buffer
```

### Bundle contents

| Part | Stored as | Added by |
| --- | --- | --- |
| Original NV12 frame | Device pointer and tensor view, no copy | Decoder |
| Resized and padded RGB input | Preallocated float32 tensor on the GPU | Prepare |
| Detections | Preallocated `[1, 300, 6]` tensor on the GPU | Inference |
| Tracks with boxes and IDs | NumPy array in source coordinates | Tracker |

A bundle is complete when all four parts exist.

### Stage rules

- **Reader:** reads the MP4 once into RAM, demuxes from memory, and copies each packet into memory it owns. It blocks only on the bounded packet queue. At EOF it starts a fresh demuxer for the next loop.
- **Decoder:** the only thread that touches the decoder object. It takes a free bundle when a frame emerges and blocks if none is free, which fills the packet queue and blocks the reader. It unlocks returned buffers itself.
- **Prepare:** takes bundles in order from the decoder's queue (`to_infer`), applies the drop rule, converts NV12 to the detector input through a converter object with `open`, `convert` and `close`, and passes the bundle to inference (`to_detect`). CUDA kernels are the only converter so far. Running it on its own thread lets it overlap inference on the previous frame. An experiment outside the repository measured about 122 FPS for this design against 105 for preparation inside the inference thread.
- **Inference:** takes prepared bundles in order, runs ONNX Runtime with IO binding, and passes the bundle on. It never waits for a host copy, because detections stay on the GPU.
- **Tracker:** copies detections to the host, runs ByteTrack, and publishes tracks. It resets when the loop number changes. Python tracker work runs at about 0.9 ms per update.
- **Display:** runs on a fixed tick on the main thread. It takes the newest complete bundle. If none is newer than the last shown frame, it waits, up to `display_grace_s` (0.2 s) since the last complete frame, so it cannot jump past frames still in inference and leave boxes stale by the whole pool. After that wait it takes the newest incomplete bundle and draws the latest available tracks. If there is no newer frame at all, an empty tick waits up to half a period (frames finish 12 to 32 ms after they are due) before it skips the tick and counts it.

### Pool rules

- A bundle is leased while any stage holds it, and a leased bundle is never reclaimed.
- Reclaim runs on every pool update, and on every wait while a stage blocks for a free bundle.
- An unleased bundle is reclaimed if it is complete and displayed, complete and superseded by a newer displayed frame, abandoned, or older than `max_age_s`.
- Reclaim sends the buffer pointer to the decoder's return queue. The decoder thread unlocks it.

### Timing and drops

- `--pace realtime` releases frames at the source frame rate, so `--display-fps 25` shows the clip at normal speed. This is the default.
- `--pace max` runs unpaced for benchmarks. Display then samples the stream, and the clip plays fast.
- `--drop-policy none` is the default and processes every decoded frame, which is what the throughput target counts. Stages push back through the pool.
- `--drop-policy oldest` lets the prepare stage skip frames more than `max_lag_s` behind their source time, for live sources. Lag counts from the frame's due time, because the pool caps the time since publishing.

## Architecture review

Each finding was measured on the demo clip, an RTX 2070 SUPER, PyNvVideoCodec 2.2.3 and ONNX Runtime 1.31.1 on native Windows.

### Confirmed

| Proposal | Evidence |
| --- | --- |
| Frames travel without a copy and return on release | `GetLockedFrame` returned distinct device pointers. All 8 sampled frames stayed byte-identical while 40 more were decoded, and `UnlockFrame` returned them. |
| Preallocated pool bounds memory | With 16 slots, 611 frames used 20 decoder buffers. |
| Tensors over decoder memory need no copy | A DLPack wrapper gave a tensor at the same address for all 611 frames. |
| Independent reader thread | One decoder consumed packets from a reader thread across 2 loops and returned 1,222 of 1,222 frames. |
| Inference without a host copy | IO binding ran directly on GPU tensors. Inference took 6.45 ms and preparation 0.98 ms, for 123 FPS before tracking. |
| Python threads are viable | During inference, a 0.5 ms sleeper saw a longest gap of 15.9 ms and a p99 of 1.4 ms. |
| OpenGL display from GPU memory | Interop reproduced frames exactly, and display with an overlay took 2.08 ms against 5.49 ms for `annotate` plus `cv2.imshow`. |

### Changed from the proposal

| Proposal | Problem | Decision |
| --- | --- | --- |
| Use `ThreadedDecoder` | It takes only a file path, and its frames are overwritten after two more fetches. It cannot hold frames for a bundle. | Use `CreateDemuxer` and `CreateDecoder` directly. |
| Output surfaces travel in the bundle | NVDEC surfaces are mapped briefly inside the library and `ulNumOutputSurfaces` is fixed at 2. What can travel is the library's locked frame buffer. The library's internal copy out of the surface remains. | Bundles hold locked frame buffers. |
| Reader takes a bundle per packet | Packets are not frames. The first frame needed 5 packets, and 60 of 120 packets were out of display order. | Reader uses a packet queue. The decoder takes a bundle when a frame emerges. Back-pressure still reaches the reader through that queue. |
| Decoder blocks when surfaces run out | `GetLockedFrame` held 128 frames without blocking, so nothing stops it from growing. | The pool is the only limit, and the decoder acquires a bundle before locking. |
| Reader pushes demuxed packets directly | Reading a packet after the next `Demux()` caused an access violation. | Reader copies every packet into owned memory before queueing it. |
| GC releases bundles in any state | Reclaiming a bundle during inference or display would recycle memory still in use. | Leases protect bundles in use. A stage returns its lease before the bundle can go. |
| Display is an independent thread | GLFW requires window creation and event polling on the process's main thread. | Display runs on the main thread, and all other stages run on worker threads. |
| Display uses the resized RGB when downsized | That tensor is float, 640x640, and padded. A second path adds code and no value. | Display always uses NV12 and lets OpenGL scale to the window. |
| Preprocessing is its own module | It takes under 1 ms and must finish before inference anyway. | It is a separate function, run by a converter on the prepare thread. Each frame's GPU work finishes before the bundle moves on, so the order holds. |
| Tracker accepts gaps in frames | ByteTrack counts updates, not time. Dropped frames shorten `track_buffer` in real time and skew motion prediction. | Accepted only under `--drop-policy oldest`, and reported as a statistic. |

### Other decisions

- `GetLockedFrame` returns only a pointer, with no PTS. The decoder keeps a heap of `(loop, pts)` from incoming packets and pops the smallest for each frame. Frames emerged in display order from `Decode()` in testing, with 0 of 116 out of order. `GetLockedFrame` order is checked against `SimpleDecoder` by `test_frames_match_simple_decoder_in_order` in `tests/test_reader_decoder.py`.
- `GetFrame` is not used. It is documented as locking frames, but all 32 frames shared one address.
- Colour conversion uses BT.709 limited range, as the demuxer reports for this clip. Resizing before conversion gave a mean difference of 0.29 out of 255 from the library's RGB path, and the same 7 detections.
- The model output is already final, as `x1, y1, x2, y2, conf, class` in detector pixels, so no NMS step is needed.
- The detection threshold defaults to 0.1, so ByteTrack's low-confidence stage receives data. The current demo's 0.25 starves it.
- Tracks are stored in source coordinates, so the overlay can use a 1920x1080 orthographic projection whatever the window size.

## Measured gain from the prepare thread

Two headless runs of 1833 frames at max pace gave `steady_fps.tracked` of 116.8 and 130.0, against 101.1 before the change. Median prepare time was 1.38 and 1.23 ms, inference 8.13 and 7.07 ms, and tracking 1.42 and 1.27 ms. With display on at 25 FPS, 1206 of 1222 frames were shown with 6 empty ticks, against 1205 and 6 before; an earlier run of the same code showed 1209 with 3 empty ticks.

## Optional OpenGL prepare

`--prepare cuda` is the default. `--prepare gl` swaps the converter for `GlConverter`, which runs the same conversion as a fragment shader on a hidden 1x1 window. `--prepare cuda` does not load GLFW or OpenGL for preparation, and a `--no-display` run loads neither. Converters have five methods: `setup_main` and `teardown_main` run on the main thread, because GLFW windows belong there, and `open`, `convert` and `close` run on the prepare thread, which owns the GL context while it works. `run_pipeline` calls `setup_main` before any stage starts, and after the stages are joined it calls `teardown_main` and then `glfw.terminate()` if GLFW was used. The display no longer terminates GLFW, because the prepare thread can still own a live context when the display closes.

The shader writes three float planes through one framebuffer. The frame reaches OpenGL through a CUDA-registered upload buffer and returns through a registered readback buffer, so `gl` adds two GPU copies per frame. `cuCtxSynchronize()` follows the last copy, so the tensor is complete before the bundle is marked, and the shader clamps while colour clamping is disabled for the float targets. The context cannot be placed on a chosen GPU, so `--prepare gl` accepts only `--gpu-id 0`. The output matches `nv12_to_input` closely on a 0 to 255 scale: measured on one decoded frame of the test clip, the mean absolute difference is 0.0002 (R), 0.0001 (G) and 0.0003 (B) and the maximum is 0.0035 (R), 0.0015 (G) and 0.0041 (B).

Two alternating runs per mode, 1833 frames at max pace, gave `steady_fps.tracked` of 128.0 and 122.3 for `cuda`, and 129.1 and 126.4 for `gl`. Median prepare was 1.30 and 1.35 ms for `cuda`, and 2.02 and 2.76 ms for `gl`. The runs overlap, so there is no clear throughput winner. With the display on at 25 FPS over 2 loops, `cuda` tracked 1222 frames, showed 1206 and had 6 empty ticks, and `gl` tracked 1222 frames, showed 1213 and had 5 empty ticks (one run each). `gl` was tested on an OpenGL 4.6 driver.

## Error handling

- Any exception in a stage is recorded and sets a shared stop event, so all stages exit. The run then raises that error.
- Every blocking wait uses a timeout and checks the stop event.
- A window close, Q or Esc sets the stop event.

## Risks

- `GetLockedFrame`, `UnlockFrame` and `PacketData` construction are lightly documented, and only version 2.2.3 was tested. Pin that version.
- The DLPack wrapper depends on tensors being dropped before their buffer is unlocked. Leases enforce this.
- A window being dragged on Windows can block the event loop. Reclaim inside `acquire` and `max_age_s` keep the pipeline moving.
- Rebinding the ONNX output per slot has not been tested. If it fails, inference binds one output tensor and copies it to the slot on the GPU.
- In this stage each process serves one stream, so several streams mean several processes competing for one GPU. The central inference stage replaces this.

## Success criteria

- `--loops 1 --no-display --pace max` processes 611 of 611 frames at each stage, and every bundle returns to the free list.
- Track IDs and boxes match `model.track` on 300 frames at `conf=0.25`.
- Throughput with tracking beats the current demo's 91 FPS on the same hardware.
- A 20-loop run shows flat GPU memory and process memory.
- With display on at 25 FPS, the window shows annotated video at normal speed with no decoded frames lost.
