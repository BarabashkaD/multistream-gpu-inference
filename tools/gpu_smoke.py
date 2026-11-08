#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Test immutable candidate images on an explicitly authorized Linux NVIDIA host.

Never builds, tags, pushes, installs software or logs in. Requires Docker Compose.
The result covers model warmup/inference, CUDA IPC and two cross-container streams,
not display, performance guarantees or redistribution rights.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile

IMAGE = re.compile(r"ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")


def validate_images(instance, server, revision):
    if not SHA.fullmatch(revision):
        raise ValueError("Expected a full lowercase commit SHA")
    if not all(IMAGE.fullmatch(ref) for ref in (instance, server)):
        raise ValueError("Use immutable ghcr.io/...@sha256:<64 hex> references")
    if instance == server:
        raise ValueError("Instance and server must be distinct candidate images")


def compose_spec(instance, server):
    gpu = {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
    common = {"restart": "no", "network_mode": "none", "pull_policy": "never",
              "deploy": {"resources": {"reservations": {"devices": [gpu]}}},
              "environment": {"TRAFFIC_INFERENCE_KEY": "${TRAFFIC_INFERENCE_KEY:?required}"},
              "volumes": ["ipc:/run/traffic"]}
    services = {"server": {**common, "image": server, "cap_add": ["SYS_PTRACE"],
        "command": ["python", "run_server.py", "--address", "/run/traffic/server.sock", "--clients", "2"]}}
    for i in range(2):
        services[f"instance{i}"] = {**common, "image": instance, "pid": "service:server",
            "depends_on": ["server"], "command": ["python", "run_instance.py", "--address",
            "/run/traffic/server.sock", "--stream", str(i), "--input", "/data/pexels-36304379.mp4",
            "--no-display", "--pace", "max", "--drop-policy", "none", "--loops", "1"]}
    return {"services": services, "volumes": {"ipc": {}}}


def check_summaries(results):
    for name in ("instance0", "instance1"):
        result = results[name]
        if (result["counts"]["tracked"] != 611 or result["counts"]["rows"] <= 1833
                or result["pool"]["free"] != result["pool"]["size"]):
            raise ValueError(f"Cross-container frame/detection/pool check failed: {name}")
    server = results["server"]
    if server["counts"]["inferred"] != 1222 or server["dead"] or len(server["streams"]) != 2:
        raise ValueError("Cross-container server summary failed")


def run(args, *, env=None, timeout=600):
    return subprocess.check_output(args, text=True, env=env, timeout=timeout).strip()


def smoke(instance, server, revision, output):
    validate_images(instance, server, revision)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "status": "failed", "revision": revision,
              "images": {"instance": instance, "server": server},
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    project = "mgi-smoke-" + secrets.token_hex(6)
    probe_name = project + "-probe"
    try:
        report["docker"] = run(["docker", "version", "--format", "{{.Server.Version}}"])
        report["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        report["image_ids"] = {}
        for target, ref in report["images"].items():
            run(["docker", "pull", ref])
            info = json.loads(run(["docker", "image", "inspect", ref]))[0]
            if (info["Config"]["Labels"] or {}).get("org.opencontainers.image.revision") != revision:
                raise ValueError(f"{target}: source revision mismatch")
            report["image_ids"][target] = info["Id"]
        # No bind-mounted application code: tests and models come from the candidate.
        probe = run(["docker", "run", "--rm", "--name", probe_name, "--network", "none", "--gpus", "all",
                     server, "python", "tools/run_gpu_tests.py"], timeout=1200)
        (output / "gpu-tests.log").write_text(probe + "\n")
        with tempfile.TemporaryDirectory(prefix="mgi-smoke-") as folder:
            spec = Path(folder) / "compose.json"
            spec.write_text(json.dumps(compose_spec(instance, server)))
            cmd = ["docker", "compose", "-p", project, "-f", str(spec)]
            env = {**os.environ, "TRAFFIC_INFERENCE_KEY": secrets.token_hex(32)}
            try:
                run([*cmd, "up", "-d", "--no-build"], env=env)
                summaries = {}
                for service in ("instance0", "instance1", "server"):
                    cid = run([*cmd, "ps", "-a", "-q", service], env=env)
                    if not cid or "\n" in cid:
                        raise ValueError(f"Expected one container: {service}")
                    code = run(["docker", "wait", cid], timeout=600)
                    log = run(["docker", "logs", cid])
                    (output / f"{service}.log").write_text(log + "\n")
                    if code != "0":
                        raise RuntimeError(f"{service} exited {code}")
                    summaries[service] = json.loads(log.splitlines()[-1])
                check_summaries(summaries)
                report["summaries"] = summaries
            finally:
                subprocess.run([*cmd, "down", "--volumes", "--remove-orphans"], env=env, timeout=120, check=True)
        report["status"] = "passed"
        report["scope"] = ["all model warmups", "CUDA IPC", "two cross-container streams"]
    except Exception as exc:
        report["error"] = str(exc)
        if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)) and exc.output:
            captured = exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else exc.output
            (output / "failed-command.log").write_text(captured)
        raise
    finally:
        cleanup_failure = None
        try:
            subprocess.run(["docker", "rm", "-f", probe_name], capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if report["status"] == "passed":
                cleanup_failure = exc
            report["cleanup_error"] = str(exc)
            report["status"] = "failed"
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "gpu-report.json").write_text(json.dumps(report, indent=2) + "\n")
        if cleanup_failure is not None:
            raise cleanup_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=Path("gpu-report"))
    args = parser.parse_args()
    smoke(args.instance, args.server, args.revision, args.output)


if __name__ == "__main__":
    main()
