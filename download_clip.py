# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Downloads the sample traffic clip to data/pexels-36304379.mp4 under the current folder.

Does nothing when the file already exists. Prints the path and size; a network failure raises an exception and
leaves a partial file, which must be deleted before running again.
"""
from pathlib import Path
from urllib.request import Request,urlopen
URL="https://videos.pexels.com/video-files/36304379/15396218_1920_1080_25fps.mp4"; OUT=Path("data/pexels-36304379.mp4")
OUT.parent.mkdir(parents=True,exist_ok=True)
if not OUT.exists():
 with urlopen(Request(URL,headers={"User-Agent":"multistream-gpu-inference/1.0"})) as r, OUT.open("wb") as f:
  while chunk:=r.read(1024*1024): f.write(chunk)
print(f"{OUT} {OUT.stat().st_size} bytes")
