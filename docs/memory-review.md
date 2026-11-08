# Memory review

Every experiment is one `run_fleet.py --streams 16 --no-display --loops 2` run with one change. Figures come from
`results/latest/summary.json`. Memory is in MB. "Server" and "instance" are the real worker process, not the
virtual-environment launcher. GPU memory is the per-process Windows `Dedicated Usage` counter, sampled once after about
20 s of running. The GPU peak is the whole card as NVML reports it, so it includes about 2320 MB used by the desktop and
other programs outside this project; that figure is NVML's used memory with nothing of this project running, read in
the design-phase probes, not in these runs. The lowest available host memory comes from the fleet's once-a-second
samples. Latency figures (medians and p95) cover each process's last 1024 timing samples (`pipeline/metrics.py`,
`_TIMING_SAMPLES`), so they leave out the start-up transient.

Rule: keep a change only if it saves at least 5% of the figure it targets and costs under 2% of `total_fps`. Servers
and instances are measured at 16 streams, 24.5 FPS target. All runs had `failed` 0, no stop reason, and 19552 frames.

Baseline, run twice: total 402.1 and 402.4 FPS, slowest stream 25.07 and 25.11, p95 latency 81 and 86 ms, dropped 0,
GPU 82 to 83%, GPU peak 6383 and 6391, server 1553 and 1561 GPU and 3090 and 3089 private, instance 180.2 GPU and
356 to 357 private.

The server-side candidates 1, 2 and 5 to 7 were compared against the baseline, or, for 5 to 7, against the
baseline plus the changes already kept. Candidates 3, 4 and 8 were run last, against the run with every kept change but
batch sizes (server 1187 GPU, 2832 private), because the server was still above 2048 MB private afterwards, which is
the review's condition for running them. They were therefore not skipped.

## Server process

| # | Item | Before | After | Verdict | Measured |
| --- | --- | --- | --- | --- | --- |
| 1 | Batch sizes | `1,4,8` | `1,8` | Kept | private 3090 to 2818 and 2823 (-8.7%), GPU 1553 to 1337 and 1349 (-13.9%), GPU peak -200; total FPS 401.9 and 402.3, slowest 25.06 and 25.12. p95 latency rose from 81 and 86 ms to 107 and 94 ms |
| 1 | Batch sizes | `1,4,8` | `1,16` | Rejected | private 3338 (+8.0%), GPU 1853 (+19%), peak 6683; total 401.3, p95 101 ms. The batch 16 session costs more than it saves, and the run was not faster. `b16.onnx` was present |
| 2 | ONNX Runtime CPU memory arena and memory pattern | on | `enable_cpu_mem_arena = False`, `enable_mem_pattern = False` | Kept | private 3090 to 2836 (-8.2%), GPU 1553 to 1299 (-16.4%), peak 6383 to 6170; total 402.3, p95 92 ms |
| 3 | ONNX Runtime threading | default | sequential, 1 intra-op thread, 1 inter-op thread | Rejected | private 2832 to 2821 (-0.4%); total 402.0, p95 100 ms |
| 4 | `torch.cuda.empty_cache()` after warm-up | none | once, at the end of `BatchDetector.__init__` | Rejected | private 2832 to 2858 (+0.9%), GPU 1187 to 1211 (+2%); total 402.1 |
| 8 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the server | not set | set | Rejected | accepted without any error, but PyTorch documents the option for Linux and I did not confirm that Windows honours it. Private 2832 to 2792 (-1.4%), GPU 1187 to 1145 (-3.5%), below the threshold; total 402.3 |
| 7 | Frame pool, as seen by the server | 8 per instance | 6 per instance | Kept, see the instance table | server GPU 1315 and 1311 to 1187 and 1187 (-9.5%) |

With every kept change (the final configuration): private 2646 (-14.4% from 3090), GPU 1051 (-32% from 1553).

## Instance process

| # | Item | Before | After | Verdict | Measured |
| --- | --- | --- | --- | --- | --- |
| 5 | `maxwidth` and `maxheight` in `CreateDecoder` | not set | the source's width and height | Rejected | GPU 180.2 to 172.2 (-4.4%), below the threshold; private unchanged at 357; total 402.1. Over 16 instances this is about 128 MB, but the rule is applied per process |
| 6 | `latency=DisplayDecodeLatencyType.LOW` in `CreateDecoder` | default | LOW | Kept | GPU 180.2 to 164.2 (-8.9%), private 357 to 341.5 and 339.1 (-4.3% and -5.0%); total 402.1 and 401.7; frame count 19552 as before, and the GPU suite passes |
| 7 | `InstanceConfig.pool_size` | 8 | 6 | Kept | GPU 164.2 to 156.2 (-4.9%), plus the server saving above (-9.5%); `dropped` 0 in both runs; total 402.0 and 401.9; p95 93 ms in both; peak 5886 to 5762. Private memory unchanged. `tests/test_config.py` now expects 6 |

Final instance: GPU 156.2 (-13.3% from 180.2), private 340 (-4.7% from 357).

## Totals for 16 streams

| | Baseline | Final | Limit |
| --- | --- | --- | --- |
| GPU, server plus 16 instances (per-process counters) | 1553 + 16 x 180.2 = 4436 | 1051 + 16 x 156.2 = 3550 | 8192 |
| GPU, whole card at peak (NVML, includes about 2320 desktop) | 6383 and 6391 | 5630 and 5625 | 8192 |
| Host, server plus 16 instances (private) | 3090 + 16 x 357 = 8802 | 2646 + 16 x 340 = 8086 | 32768 |
| Total FPS, slowest stream, p95 latency, dropped | 402.1 to 402.4, 25.07 to 25.11, 81 to 86 ms, 0 | 402.2 and 402.3, 25.04 and 25.10, 97 and 101 ms, 0 | |

The per-process counters sum to more than the NVML peak minus the desktop (3550 against about 3310). Two causes are
likely but were not measured: each process may count its own mapping of memory it shares through CUDA IPC, and the two
figures are sampled at different moments. Treat the NVML peak as the honest limit check: 16 streams leave about 2.5 GB of the 8 GB card free. Host
memory use is about 8 GB of 32 GB, and the lowest available memory seen during any run was 12.0 GB, which includes
everything else running on the machine. No memory key came back `None`.

The cost is latency: p95 is 11 to 20 ms worse than the baseline in the final runs (81 and 86 ms before, 97 and 101 ms
after). That follows the removal of the
batch 4 session; I did not trace why. It is far below the 2 s age limit and `min_fps` did not move.

## Why the server holds about 2.6 GB of host memory

Measured with the final options, 16 instances connected, all with `--no-display --loops 2`:

| Server batch sizes | `server_private_mb` | Slowest stream | Note |
| --- | --- | --- | --- |
| none (the earlier probe, no model loaded) | 1583 | | Python, PyTorch, the CUDA context |
| `1` | 2182 | 6.6 FPS, 14273 dropped | one session; this server cannot serve 16 streams |
| `1,8` | 2642 | 25.0 FPS | two sessions |
| `1,4,8` | 2832 | 25.1 FPS | three sessions |

So the first session adds about 600 MB, the batch 8 session about 460 MB, and the batch 4 session about 190 MB. The
last two steps compare v3 runs. The first step (1583 to 2182 MB) compares the design-phase probe server, which loaded
no model, with a v3 server, so it also holds whatever else differs between those two programs. Why the first is the
largest is inference, not measured: probably loading the CUDA execution
provider brings in the CUDA, cuBLAS and cuDNN libraries, whose host-side commit is paid once, together with ONNX
Runtime's own initialisation. Each later session probably adds host-side graph, weights, workspace and CUDA handles;
the model file itself is only 9.8 MB.

What I measured and what I did not: the 1583 MB with no model (in the design-phase probe) and the v3 servers with one,
two and three sessions are measured. The 16 connections,
named pipes, per-client threads, frame imports and preparer buffers are inside the 2642 MB for `1,8` but their share
was not isolated, because that needs a separate server run with no model and 16 clients, which the server does not
support. Turning off the CPU arena and memory pattern (candidate 2) took about 250 MB off a three-session server. The likely
reason, not measured, is that the CPU arena held memory for sessions whose work runs on the GPU.

## Kept

Batch sizes `1,8` (server, `server/config.py`, `run_fleet.py`, `bench.py`, `pipeline/fleet.py`); no CPU arena and no memory
pattern (`server/detector.py`); LOW display latency (`pipeline/decoder.py`); frame pool of 6 (`pipeline/config.py`).
Pinned by `tests/test_memory_options.py`.
