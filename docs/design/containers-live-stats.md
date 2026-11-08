# Live statistics table and containers for v3: design

Date: 2026-11-03. Builds on [`docs/design/external-inference.md`](external-inference.md) (v3: one inference
server, N pipeline instances, CUDA IPC frames, named pipes).

## Goal

1. While instances run, show one human-readable table of every instance's statistics, refreshed every N seconds (N
   configurable), natively on Windows and in containers.
2. Run the v3 server and instances headless in Docker, tested on Docker Desktop (WSL2 backend) on Windows 11, in two
   forms: the whole fleet in one container, and one container per process (compose).

## Decisions

| Question | Decision |
| --- | --- |
| Containers | Both: the single-container fleet first (fallback), then one container per process |
| Who collects the statistics | Each process writes its periodic JSON line; a separate monitor only reads the log files. The server does not relay instance statistics: it does not measure them, and relaying is not its job |
| Table rendering | The `rich` library, in the monitor only |
| Base image | `python:3.14-slim`, two-stage builds, as small as possible. Alpine is impossible (musl; PyTorch, ONNX Runtime GPU and PyNvVideoCodec ship glibc wheels only) |
| Test platform | Docker Desktop with the WSL2 backend on Windows 11; the live table natively first |

Rejected collection options: the server relaying stats (wrong responsibility, protocol change); UDP push (one more
port, nothing kept); Prometheus + Grafana (web dashboard, two more containers); Docker's log API (`docker.sock` in a
container gives full control of the host's Docker); `watch` + `jq` (Linux only, no totals, hard to test).

## Probe results (run 2026-11-03, before this revision)

Throwaway probe scripts, not committed. Docker Desktop 29.8.2 with the WSL2 backend, RTX 2070 SUPER, driver
616.56.

| Probe | Result |
| --- | --- |
| P0: GPU in a container | `docker run --rm --gpus all python:3.14-slim nvidia-smi` lists the GPU. `libcuda.so.1`, `libnvcuvid.so.1` and `libnvidia-ml.so.1` are injected. The fleet's NVML calls (memory, utilisation, decoder utilisation) work |
| P1: NVDEC in the slim image | PyNvVideoCodec 2.2.3 decodes the clip: 611 frames, frame 100 byte-identical to the native decode (same MD5). One decoder at max pace: 606 FPS in the container, 726 FPS natively. No CUDA runtime library is needed in the image |
| P2: CUDA IPC, two processes in one container | Fails by default with `CUDA_ERROR_INVALID_HANDLE` (400), also for a plain `cuMemAlloc` buffer, also with `--ipc=host` or `--shm-size=2g`. Works with `--cap-add SYS_PTRACE` (an NVDEC frame read in the other process matches the native MD5). `--security-opt seccomp=unconfined` alone does not help |
| P3: CUDA IPC across two containers | Separate PID namespaces: fails with error 201. Works when the importer joins the exporter's PID namespace (`--pid=container:<exporter>`) or both use `--pid=host`. Only the importing side (in v3 the server) needs `SYS_PTRACE`; the exporting side (the instances) needs no added capability. A shared IPC namespace is not needed |
| Unix socket buffer | 93 KB can be written before a send blocks (`SO_SNDBUF` 212992). The pool cap's assumption (results in flight fit 64 KiB: 8 x 7.3 KB) holds |
| Docker VM resources | 4 CPUs and 7.7 GB of memory. Natively 16 streams used about 8 GB of private memory and about 4 cores, so 16 streams will not fit as configured |

Probe image with PyNvVideoCodec, NumPy and psutil: 242 MB.

## Facts from the current code that this design must fix

- `pipeline/app.py:35-42`: the instance's periodic line has rates, pool state and memory, but no stream number and
  no stage times. Stage medians and p95 exist only in the final summary line.
- `pipeline/metrics.py`: medians and p95 cover the last 1024 samples (about 41 s at 25 FPS). `current_fps` uses a
  fixed 1 s window.
- `pipeline/metrics.py:61-63`: `_percentile` returns index `int(n * 0.95) - 1`, which is below the nearest rank, and
  for fewer than 21 samples returns the last element.
- `memory_info().private` is Windows-only in psutil: `pipeline/app.py:42`, `server/service.py:220`,
  `pipeline/fleet.py:260,262`. On Linux the first two raise in their report threads; in the fleet the
  `AttributeError` escapes the `(psutil.Error, OSError)` handler and ends the run as a "monitor error".
- `pipeline/fleet.py:146,149`: `summarise` treats an instance line with `counts` as finished and a server line with
  `batches` as alive. Periodic lines that carry those keys would make it raise on a killed process's last line.
- Reporter threads are never joined (`pipeline/app.py:93-106`, `server/service.py:215-220`), so a periodic line can
  be printed after the final line.
- `server/service.py:183`: the `batches` Counter is updated without the lock.
- `pipeline/protocol.py:48`: `default_address()` returns a Windows pipe name.
- `pipeline/fleet.py:81`: per-process GPU memory comes from a Windows performance counter.
- `run_fleet.py` has no `--input` option; `report_interval_s` (default 1.0 in `InstanceConfig` and `ServerConfig`)
  has no command-line option.
- `pipeline/fleet.py:194` sets `TRAFFIC_INFERENCE_KEY` itself for each run.

## Part 1: periodic statistics line

New option on `run_instance.py` and `run_server.py`: `--report-interval-s` (float, default 1.0, must be > 0), mapped
to the existing `report_interval_s`. New option on both: `--stats-file PATH`: when set, every periodic line and the
final line are also written to `PATH`, opened once in mode `"w"` (truncated, so a reused volume shows only this
run), UTF-8, each line flushed. It is kept rather than shell redirection in compose, because it keeps `docker logs`,
the exit code and no shell in the container. Standard output and the stats file are written under one lock.

Line kinds: each periodic line has `"kind": "period"`; the final summary line keeps its content and gains
`"kind": "final"`. The reporter thread is stopped and joined before the final line is written, so the final line is
always the last line.

Readers of "the last line" change: `fleet._last_json` returns the last line with `kind == "final"`, otherwise `None`.
Logs written before this change are not supported (they live only in the git-ignored `results/`).
`summarise`, `bench.py` and their tests use it; new tests cover a log whose last line is a period line.

Instance periodic line, exact keys:

```json
{"kind": "period", "role": "instance", "stream": 3, "time": 95.0, "period_s": 1.0,
 "current": {"decoded": 25.0, "tracked": 25.0, "dropped": 0.0},
 "fps": {"tracked": 25.0},
 "counts": {"decoded": 2375, "tracked": 2375, "dropped": 0},
 "period_ms": {"decode": {"p50": 0.9, "p95": 1.3}, "remote": {"p50": 38.0, "p95": 52.0},
               "track": {"p50": 0.85, "p95": 1.1}, "latency": {"p50": 44.0, "p95": 48.0}},
 "pool": {"...": "unchanged from today"}, "memory_mb": 340}
```

- `time`: seconds since the process started (psutil's process creation time), rounded to 0.1.
- `current`: events per second over the last `period_s` (`current_fps(window_s=period_s)`), for every counter the
  process has, as today; the table uses `tracked` and `dropped`.
- `counts`: cumulative counts, every counter.
- `period_ms`: median (`p50`) and `p95` of each timing name's samples recorded since the previous periodic line;
  names with no new samples are omitted. `display` appears only when a window is open.
- Values are rounded to 2 decimals.

Server periodic line: `"kind": "period"`, `"role": "server"`, `time`, `period_s`, `current`, `queued`, `memory_mb`,
`batches` (cumulative counts by size, copied under the server's lock), `available` (cumulative counts of batches by
the number of frames waiting when each was formed, the largest size's bucket labelled `N+`; see Part 7) and `period_ms` for `prepare`, `infer`,
`reply` and `batch_wait`.

`Metrics.take_period() -> dict[str, list[float]]`: `record_ms` also increments a per-name total count;
`take_period` returns, per name, the samples recorded since its previous call, taken from the tail of the existing
1024-sample deque (at most 1024), and remembers the totals. No second store; `snapshot()` is unchanged. Called only by the
reporter thread.

`_percentile` becomes nearest rank: index `ceil(n * fraction) - 1`. For the final summaries (1024 samples) this moves
p95 by one sample; the change is noted in [performance-and-design.md](../performance-and-design.md).

Memory: `pipeline/memory.py: private_mb(process) -> float` returns `memory_info().private` on Windows and
`memory_full_info().uss` on Linux, falling back to `memory_info().rss` if `uss` is denied. Used by the instance, the
server and the fleet. Windows numbers are unchanged.

## Part 2: monitor

`monitor.py --logs DIR [--period-s 2] [--once] [--streams N]`, module `pipeline/monitor.py`.

- Every period it reads the tail of each `DIR/*.log` (last 64 KiB, doubled up to 1 MiB until it holds a complete
  line). It takes the last complete JSON line with a `kind` and a `role`. A partial last line is ignored;
  malformed lines are skipped and counted in the footer.
- Rows are built from the line's `role` (`instance` rows, one `server` header). Files with neither are ignored.
- On a terminal it redraws one table with `rich.live.Live.update`; when the output is not a terminal it prints
  `console.print(table)` per period, headed by the wall-clock time (rich strips control codes itself).
- Header: run time, server inferred FPS, batches by size, prepare and infer p50, queue length, server memory, or
  "server: no data".
- One row per instance, sorted by stream: stream, FPS last period (`current.tracked`), FPS average (`fps.tracked`),
  decode p50, round trip (`remote`) p50 and p95, track p50, latency p95, dropped (`counts.dropped`), pool in use,
  memory MB, state.
- States: `running`; `done` when the last line is final; `failed` when a Python traceback follows the last JSON line
  (natively the fleet merges stderr into the log, so a traceback lands there; other text, such as warnings, does not
  count); `stale` when the line's `time` has not
  advanced for 3 x the line's own `period_s`, judged on the monitor's clock from the moment it last saw `time`
  change (not from file mtime, which is unreliable on bind mounts).
- Totals row: sum of FPS, worst latency p95, sum of dropped.
- `--once` prints one table and exits 0. A missing `DIR` is an error. With `--streams N` the monitor exits 0 once the
  server's line is final and N instance rows are not `running` (the server writes its final line only after every
  instance has left, so a killed instance with no final line also counts as ended); used by compose. Without it, it
  runs until stopped.
- `rich` is imported only by `pipeline/monitor.py`. Import checks (`tests/test_no_torch.py` style) assert that the
  instance and server modules do not import `rich`.

Fleet: `run_fleet.py --monitor-period-s S` (default 2.0, 0 = off), `FleetConfig.monitor_period_s`. When on and
standard error is a terminal, the fleet imports `pipeline.monitor` lazily inside `run_fleet` and runs it in a thread
over the run's log folder, writing to **standard error**, so the final JSON on standard output stays parseable
(`run_fleet.py > out.json` works). A monitor exception is printed once to standard error and never ends the run. The
monitor is stopped in the `finally` block before the processes are killed. `bench.py` passes 0.

Also new: `run_fleet.py --input PATH` (default unchanged), passed to every instance.

## Part 3: Linux port

- `default_address()`: on Linux `/tmp/traffic-inference-<pid>.sock`; Windows unchanged. Every `--address` accepts
  either form. Before loading models the server removes a socket file left at that path, and it removes it on exit.
  Socket paths must stay under 108 bytes; tests use a short temporary directory.
- `private_mb` as above. Fleet GPU memory per process is `None` on Linux.
- Tests that build pipe names (`tests/conftest.py:55`, `tests/test_service.py`, `tests/test_protocol.py`) use one
  helper returning a pipe name on Windows and a short socket path on Linux.
- Display code is not ported: containers always run `--no-display`.
- Tests that need packages absent from the image (`ultralytics`, `cv2`, `glfw`, `OpenGL`) skip with
  `pytest.importorskip` at module top, so collection does not fail. On Windows nothing is skipped.

## Part 4: images

One `Dockerfile` with a build stage and two final targets, all on `python:3.14-slim` pinned by digest (the same
digest in every stage). The build stage creates `/opt/venv`; the final stages copy `/opt/venv` to the same path.
No compilers, no pip cache in the final images. Code lives at `/app` (the repository's content), so `V3_ROOT` is
`/app`, models are `/app/models` and results `/app/results`.

| Target | Requirements | Used for | Expected size |
| --- | --- | --- | --- |
| `instance` | `requirements-instance.txt`: PyNvVideoCodec, NumPy, `lap`, psutil, `rich` | instances and the monitor | about 250 to 300 MB (probe image without `lap` and `rich`: 242 MB) |
| `server` | `requirements-server.txt`: the instance list plus PyTorch (CUDA 13.0 Linux wheel), ONNX Runtime GPU, nvidia-ml-py, pytest | the server, the single-container fleet, the test run | 5 to 6 GB, almost all PyTorch's CUDA libraries |

- cuda-bindings and cuda-pathfinder are left out of the instance image: only the display imports `cuda.bindings`, and P1 decoded
  without them. They are in the server image because PyTorch depends on them.
- Pins are copied exactly from `requirements.txt`; `rich==15.0.0` and its dependencies are pinned in all three
  files. `requirements-server.txt` keeps `--extra-index-url https://download.pytorch.org/whl/cu130`.
- Models, the clip and results are mounted at run time, never copied into an image. `.dockerignore` excludes
  `models/`, `results/`, `.venv`, `.git`, caches and `hang.log`.
- Containers run with `--gpus all` and `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`.
- Gate P1b after building `server`: `run_server.py` starts in the container and ONNX Runtime reports the CUDA
  execution provider active (`server/detector.py` already raises otherwise). If it fails, stop and report.
- Real sizes are measured after the build and reported.

## Part 5: single-container fleet

From the repository root:

```powershell
docker run --rm -it --gpus all --cap-add SYS_PTRACE -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility `
  -v "${PWD}/models:/app/models:ro" -v "${PWD}/data:/data:ro" -v "${PWD}/results:/app/results" `
  traffic-v3:server python run_fleet.py --streams 2 --no-display --input /data/pexels-36304379.mp4
```

The fleet makes its own key, as natively. `SYS_PTRACE` is required for CUDA IPC (P2). The live table goes to the
terminal (standard error); the final JSON follows on standard output. The fleet's safety limits apply to the VM's
memory.

## Part 6: compose, one container per process

- `tools/make_compose.py --streams N [--output PATH]` writes `compose.yaml`: one `server` service, N `instance<k>`
  services (`--stream k`) and one `monitor` service. The committed file is generated for N = 16; a test checks that
  regenerating it gives the same text.
- `server`: target `server`, `cap_add: [SYS_PTRACE]`, `--address /run/traffic/server.sock --clients N`.
- `instance<k>`: target `instance`, `pid: "service:server"` (shares the server's PID namespace, required by P3), no
  added capability, `depends_on: [server]`. No healthcheck: instances already retry the connection for 120 s
  (`connect_timeout_s`), a stale socket file refuses connections and is retried, and the server removes it before
  loading models.
- The socket lives on a named volume `traffic-run` at `/run/traffic` (never a bind mount).
- Every process writes `--stats-file /logs/<name>.log` on a named volume `traffic-logs`; the `monitor` service runs
  `monitor.py --logs /logs --streams N`, `tty: false`, and is watched with `docker compose logs -f monitor`.
- Key: `TRAFFIC_INFERENCE_KEY: ${TRAFFIC_INFERENCE_KEY:?set it}` on server and instances. No key in a committed
  file. [performance-and-design.md](../performance-and-design.md#security-notes) notes that the value is visible to anyone who can run `docker inspect` or `docker compose
  config`, who can also exec into the containers.
- `restart: "no"` everywhere: a failed instance stays failed and visible.
- Only `instance0` carries the `instance` build section; the other instances and the monitor use the same image with
  `pull_policy: never`. Between runs `docker compose down -v` clears the log volume, so an earlier, larger run's
  files do not show up as extra rows.
- Containers run as root: the added capability does not reach a non-root user.

## Part 7: batching and the frames-available histogram

Added after the first implementation (2026-11-03); it replaces the `linger_ms` rule of the external-inference design.

- Gather rule: each round the server gathers frames for up to `--gather-ms` (default 10 ms). It stops early once every
  active stream has a frame waiting, or a full batch of the largest loaded size is waiting. It then takes every waiting
  frame, up to the largest size, in round-robin order.
- Batch size: the smallest loaded size that holds the taken frames. The rest of the batch is padded with copies of a
  real frame; the rows for the copies are never sent. `counts.padded` counts the copies.
- Timings: `prepare`, `infer` and `reply` are divided by the number of real frames, so padded copies do not lower
  or raise a per-frame figure.
- `available`: for each batch that ran, the number of frames waiting when it was formed (read from the queue right after
  the take). Buckets are the numbers 1 up to one below the largest size, and `N+` for the largest size or more
  (`8+` with sizes `1,8`). A batch that never ran (every handle failed) is not counted. It appears in the server's
  periodic line and final summary (`available`), in the fleet summary and `bench.py` rows, and as the monitor's caption
  `frames available per batch: 1×34 2×5 8+×615`. Mostly `N+` means frames queue and the GPU is the limit; mostly 1 or 2
  means spare capacity.
- Experiment and re-measured figures: the design-phase probe comparing the earlier rule with this one, and the
  re-measurement with gathered, padded batches, are recorded in
  [Earlier measurements (2026-11-05 and before)](#earlier-measurements-2026-11-05-and-before) at the end of this
  document; the figures are not repeated in this part.

## Before 16 streams in containers

Docker's VM has 4 CPUs and 7.7 GB. Before any 16-stream container run, re-read the VM's limits. If they are still
below about 8 CPUs and 12 GB, raise them first with the WSL settings `memory` and `processors` in the `[wsl2]`
section of `.wslconfig`; the project never changes that file itself. Runs of 2 to 4 streams fit as configured.

## Order of work

1. Parts 1 and 2 natively (no Docker).
2. Part 3 and Part 4, gate P1b, then Part 5 with 2 streams.
3. Part 6 with 2 streams, then 16 if the VM allows.

## Testing

- Monitor: canned log folders: rows and sorting, totals, `running`/`done`/`failed`/`stale`, missing server, malformed
  and partial lines, a tail with no newline, non-terminal output without control codes, `--streams` exit.
- Statistics line: exact keys for instance and server lines; `take_period` returns each sample once; the final line
  is last (reporter joined); the stats file is truncated on open.
- `_last_json` and `summarise` with a period line last and a final line last.
- `private_mb` on Windows equals `memory_info().private`.
- Fleet: the monitor thread starts and stops; 0 starts none; a monitor exception does not end the run; standard
  output holds only the final JSON.
- `make_compose.py`: regenerated file equals the committed one; N = 2 gives 2 instance services.
- In containers: the CPU suite inside the `server` image; one GPU run per form (2 streams).
- GPU runs one at a time, never overlapping native and container runs, with the fleet's safety limits.
- Not tested: a real Linux host with the NVIDIA Container Toolkit. The README and [performance-and-design.md](../performance-and-design.md) say so.

## Risks

- `SYS_PTRACE` and a shared PID namespace are required for CUDA IPC under WSL2 (P2, P3). Both are narrower than
  `--privileged`, which also works. On a real Linux host the requirement may differ.
- Decoding in the container was 17% slower than natively for one decoder at max pace (P1); 400 FPS in containers is
  measured, not assumed.
- Docker's VM is too small for 16 streams as configured.
- The server image is large (PyTorch's CUDA libraries). Removing PyTorch from the server is a separate change.

## Out of scope

Displaying frames in containers; Kubernetes; a real Linux host; Prometheus or other monitoring stacks; removing
PyTorch from the server; the downstream output of tracks.

## Part 8: self-contained images and one build script (added 2026-11-03)

Goal: a demo on an unknown machine needs only Docker and an NVIDIA driver. Decision: the models are copied
into the server image from `models/` at build time (option 1), not exported inside the Docker build; images are
built on a prepared machine and carried with `docker save` / `docker load`.

- Both images contain the clip at `/data/pexels-36304379.mp4`, passed to the instances with `--input` (compose does
  this; the code's default is `data/` next to the code, which is `/app/data` in the images). It
  comes from a named build context `clip` pointing at the repository's `data/` folder
  (`docker build --build-context clip=./data`; compose `additional_contexts`).
- The server image contains `/app/models/b*.onnx` (the default `--models`), copied from the build context:
  `.dockerignore` ignores `models/*` except `models/b*.onnx`.
- Overrides: mounting a host folder over `/data` or `/app/models` replaces the image's files (no code change);
  `--input /data/<file>` selects another clip.
- Compose: `compose.yaml` mounts neither models nor data (the images are self-contained); a generated
  `compose.mounts.yaml` adds the two mounts (`./models:/app/models:ro` on the server, `./data:/data:ro` on every
  instance) when passed as a second `-f` file. `tools/make_compose.py` writes both; the committed pair is for N = 16
  and a test checks both.
- `tools/build_images.ps1` (Windows, PowerShell 7.3+) and `tools/build_images.sh` (Ubuntu 24.04, bash), with
  the same steps and messages and no machine-specific values, prepare and build everything, each step only when
  needed: check Docker (on Ubuntu also `docker buildx` and GPU access in a container, printing the install steps
  for the buildx plugin or the NVIDIA Container Toolkit and exiting if one is missing); download the clip to a
  temporary file if it is missing or its SHA-256 differs, verify it and move it into place (fail and delete it on a
  mismatch); export the missing models among `-Sizes`/`--sizes` (default `1,8,16`) in one throwaway container
  (the Dockerfile's pinned `python:3.14-slim`, `requirements-export.txt` with CPU-only PyTorch, Ultralytics and
  ONNX, OpenCV's system libraries from apt), so building needs only Docker, not a host venv; on Ubuntu the same
  container run gives the files back to the host user; build both images with the clip context; smoke-test them
  unless `-SkipTests`/`--skip-tests` (the clip's size in the instance image, and the server's detector loads the
  models with the CUDA provider); with `-SaveTo`/`--save-to PATH` write both images with `docker save` and print
  the `docker load` command.

## Earlier measurements (2026-11-05 and before)

These figures were measured before 2026-11-06 with the code of this branch's milestones and were not repeated on 2026-11-06; [performance-and-design.md](../performance-and-design.md) holds the 2026-11-06 figures. Conditions as there: Windows 11, NVIDIA GeForce RTX 2070 SUPER (8 GB, driver 616.56), Intel Core i7-9700K, 32 GB RAM, one test clip (1920x1080, 25 FPS, 611 frames), 2 loops per stream unless stated; Docker Desktop with the WSL2 back end.

### Compose, one container per process

Docker Desktop VM with 4 CPUs and 7.7 GiB, measured on 2026-11-05; every container of every run exited 0:

| Streams | 1 | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| Total FPS | 25.2 | 50.2 | 100.6 | 201.9 | 302.3 | 404.4 |
| Dropped | 0 | 0 | 0 | 0 | 0 | 12 |
| Worst p95 latency ms | 33 | 49 | 52 | 53 | 54 | 428 |
| GPU % | 31 | 29 | 52 | 53 | 63 | 79 |
| NVDEC % | 5 | 4 | 6 | 12 | 19 | 26 |

With `1,8`, compose's capacity at 16 streams is 404.7 FPS against the 400 needed, so a queue that forms cannot drain, which is the 428 ms p95. With `1,16` (two runs): 404.8 and 404.4 FPS, 9 and 28 of 19,552 frames dropped, worst p95 196 and 83 ms, at the cost of about 13 ms more median latency and about 290 MB more GPU memory.

At 16 streams compose has 1% of capacity spare with the default batch sizes `1,8`, and 10% with `1,16`.

### Capacity without the 25 FPS cap (`--pace max`), native against compose

Total FPS:

| Streams | 1 | 2 | 4 | 8 | 12 | 16 | 16 with `1,16` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native Windows | 193.2 | 355.8 | 467.2 | 454.1 | 468.0 | 460.8 (466.1 in a re-run) | 492.6 |
| Native GPU % | 80 | 84 | 90 | 87 | 90 | 88 | |
| Compose | 163.0 | 322.4 | 417.8 | 412.9 | 404.4 | 404.7 | 440.8 |

From 4 streams up the GPU is the limit natively, at 454 to 468 FPS and 87 to 90% busy; at 16 streams 460.8 FPS is 15% above the 400 FPS those streams need. Latency grows as the streams queue for the GPU: p95 was 53.6 ms at 4 streams and 252.9 ms at 16. Native Windows has 10 to 19% more capacity than compose at every stream count; from 4 streams on, inference per frame settles at 1.50 to 1.55 ms natively and 1.77 to 1.79 ms in compose, with the same ONNX Runtime, PyTorch, CUDA and cuDNN versions. NVDEC was slower in the container too: 606 against 726 FPS for one decoder. The cause was not isolated further. At 1 and 2 streams the server is not the limit: a stream holds at most 6 frames, so its rate is at most 6 divided by the time a frame is held.

### Batch sizes `1,8,16`

At 16 streams and real time, a `1,8,16` run gave 401.8 FPS against 402.4 for `1,8`, with 1753 against 1051 MB of server GPU memory. (Batch sizes `1,16` as the default, against `1,4,8`: server private 3338 MB (+8.0%) and GPU 1853 MB (+19%), no faster.)

### Soak (earlier batching rule, not re-run)

16 streams for 12 loops (117312 frames) ran at 400.3 FPS with no frame dropped. Over 291 one-second samples the server's private memory stayed at 2645 to 2646 MB, and the last instance's stayed between 333 and 345 MB after the first quarter of the run, with no upward trend. No compose soak was run.

### A consumer that falls behind

Natively (earlier batching rule, not re-run), with stream 0's tracker delayed by 120 ms a frame (`run_fleet.py --slow-stream 0 --slow-ms 120`), stream 0 tracked 392 frames at 7.8 FPS and dropped 830, while the other 15 streams ran at 25.06 to 25.25 FPS with no frame dropped and a worst p95 of 100 ms. In compose with `1,16`, stream 0 tracked 8.2 FPS and dropped 815 of 1,222 frames; the other 15 held 25.17 to 25.32 FPS, dropped 9 frames between them, and their worst p95 was 84 ms, against 83 ms in a run with no slow stream.

### Gather rule: design-phase probe

The earlier rule waited up to 4 ms for 8 frames and otherwise ran a batch of 1, so batches rarely filled; at max pace one stream reached only 60.8 FPS, because its pool of 6 frames could never supply 8. With the gather rule of Part 7, one stream at max pace reaches 193.2 FPS. In the probe at 4 streams, the earlier rule ran 1004 batches of 1 and 180 of 8, with latency p50 66 ms and p95 112 ms; the new rule ran 34 of 1 and 615 of 8, p50 26 ms and p95 50 ms. At 16 streams in that probe p95 went from 277 to 212 ms, with the same FPS and no drops. A gather rule that always waited for every stream left a standing backlog at 16 streams, p50 latency 229 ms; waiting therefore also stops at a full batch.

### 16 windows (earlier batching rule, not re-run)

With one window per stream, 16 streams fall behind: 376.2 FPS in total, the slowest stream at 23.1 FPS, 1037 of 19552 frames dropped, 65% of the tracked frames shown, p95 latency 730 ms and GPU 92%. The windows' OpenGL drawing runs on the same GPU as the server, which fits these numbers, but no run isolated the cause. From 4 windows up many display ticks find no new frame (4 windows showed 73% of the tracked frames with nothing dropped); the cause was not traced.
