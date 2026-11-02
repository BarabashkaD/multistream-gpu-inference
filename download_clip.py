# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Download the sample traffic clip to data/pexels-36304379.mp4 unless it is already there.

Takes no arguments and resolves the output folder against the current directory, creating it if needed. Streams the
file in 1 MiB chunks and prints its path and size. A failed download raises the urllib error and may leave a partial
file, which a later run would then treat as complete.
"""
from pathlib import Path
from urllib.request import Request,urlopen
URL="https://videos.pexels.com/video-files/36304379/15396218_1920_1080_25fps.mp4"; OUT=Path("data/pexels-36304379.mp4")
OUT.parent.mkdir(parents=True,exist_ok=True)
if not OUT.exists():
 with urlopen(Request(URL,headers={"User-Agent":"multistream-gpu-inference/1.0"})) as r, OUT.open("wb") as f:
  while chunk:=r.read(1024*1024): f.write(chunk)
print(f"{OUT} {OUT.stat().st_size} bytes")
