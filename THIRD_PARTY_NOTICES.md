# Third-party notices

This project (v2: bundle pipeline) is licensed under AGPL-3.0-or-later; see LICENSE and NOTICE. It depends on the
software below, which users install themselves with pip. None of it is redistributed in this repository.

## Model and tracker

- **Ultralytics** (AGPL-3.0), https://github.com/ultralytics/ultralytics: detection model loading, the
  ByteTrack tracker and the model export.
- **YOLO26n weights** (AGPL-3.0, Ultralytics), downloaded by the export script (prepare_model.py) and not stored in
  the repository.
- **ByteTrack** (MIT, https://github.com/ifzhang/ByteTrack): the tracking algorithm.

## NVIDIA software

The NVIDIA libraries below are covered by the additional permission in NOTICE, section 3. Users install them
from PyPI, directly, as dependencies of PyTorch, or inside the PyTorch, PyNvVideoCodec and Triton wheels, and
accept their licenses themselves. None of them is redistributed in this repository.

- **CUDA Toolkit libraries** (nvidia-cublas, nvidia-cuda-cupti, nvidia-cuda-nvrtc, nvidia-cuda-runtime,
  nvidia-cufft, nvidia-cufile, nvidia-curand, nvidia-cusolver, nvidia-cusparse, nvidia-nvjitlink,
  nvidia-nvshmem-cu13; installed with PyTorch on Linux, and most of them bundled in the PyTorch CUDA build for Windows; cuFile and
  NVSHMEM are Linux-only): NVIDIA CUDA Toolkit End User License Agreement,
  https://docs.nvidia.com/cuda/eula/index.html. It allows installing and using the libraries, and
  redistributing the runtime libraries of its Attachment A inside an application. It also says "You may not
  use the SDK in any manner that would cause it to become subject to an open source software license"
  (section 1.2, item 5); this project distributes none of them, and NOTICE, section 3 keeps them outside the
  AGPL-3.0 terms of the author's code. The libraries are licensed for systems with NVIDIA GPUs, and
  life-critical uses need an agreement with NVIDIA.
- **cuDNN** (nvidia-cudnn-cu13): License Agreement for NVIDIA Software Development Kits with the cuDNN
  supplement, https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html. Equivalent terms to the CUDA
  Toolkit libraries.
- **cuSPARSELt** (nvidia-cusparselt-cu13, Linux): License Agreement for NVIDIA Software Development Kits with
  the cuSPARSELt supplement, in the wheel (nvidia/cusparselt/LICENSE.txt),
  https://pypi.org/project/nvidia-cusparselt-cu13/. Equivalent terms to cuDNN.
- **NCCL** (nvidia-nccl-cu13, Linux): BSD-3-Clause, the license text in the wheel (its metadata says
  LicenseRef-NVIDIA-Proprietary), https://pypi.org/project/nvidia-nccl-cu13/.
- **NVTX** (nvidia-nvtx, Linux): Apache-2.0 with LLVM exceptions,
  https://github.com/NVIDIA/NVTX/blob/release-v3/LICENSE.txt.
- **cuda-toolkit** (Linux): a metapackage without files that selects the CUDA libraries above; it states no
  license of its own, https://pypi.org/project/cuda-toolkit/.
- **CUDA Python** (cuda-bindings, cuda-pathfinder): Apache-2.0, https://github.com/NVIDIA/cuda-python.
- **PyNvVideoCodec** (pynvvideocodec): MIT, https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html.
  Its wheels also contain parts of FFmpeg (libavformat, libavutil) under LGPL-3.0, as NOTICES.txt in the
  wheel states, and the CUDA 12 runtime under the CUDA EULA; LGPL-3.0 allows use with AGPL-3.0 code.
- **NVML Python bindings** (nvidia-ml-py): BSD-3-Clause, the license in pynvml.py,
  https://pypi.org/project/nvidia-ml-py/.
- **NVIDIA driver** (the CUDA driver API, NVML and NVCUVID for NVDEC decoding): installed by the user with the
  GPU driver under NVIDIA's driver license; not a Python package.
- **Triton** (triton, Linux, a dependency of PyTorch): MIT, https://github.com/triton-lang/triton. Its wheel
  also contains NVIDIA CUDA tools (ptxas, cuobjdump, nvdisasm) and CUPTI under NVIDIA's CUDA license. This
  project does not call Triton directly.
- **ONNX Runtime GPU** (onnxruntime-gpu): MIT, Microsoft, https://github.com/microsoft/onnxruntime. The
  ThirdPartyNotices.txt in the package lists its bundled components. It loads the CUDA and cuDNN libraries
  above at run time.

## Test video

"Busy urban intersection from above" by Scott Precious, Pexels video 36304379,
https://www.pexels.com/video/busy-urban-intersection-from-above-36304379/, used under the Pexels license.
The clip is downloaded by `download_clip.py` and not stored in the repository.
Pexels license: https://www.pexels.com/license/

## Python packages

Every package that pip resolves for this version's requirement files on Windows and on Linux (Python 3.14),
with the license its metadata states, or its license file where the metadata states none or differs.
A package resolved to several builds lists each version.

| Package | Version | License | Source |
|---|---|---|---|
| anyio | 4.16.0 | MIT | https://anyio.readthedocs.io/en/latest/ |
| certifi | 2026.10.23 | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | MIT | https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md |
| colorama | 0.4.6 | BSD License | https://github.com/tartley/colorama |
| contourpy | 1.4.0 | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cuda-bindings | 13.4.4 | Apache-2.0 | https://github.com/NVIDIA/cuda-python |
| cuda-pathfinder | 1.8.5 | Apache-2.0 | https://github.com/NVIDIA/cuda-python |
| cuda-toolkit | 13.0.3.0 | none stated (metapackage without files) | https://developer.nvidia.com/cuda-toolkit |
| cycler | 0.12.1 | BSD License | https://matplotlib.org/cycler/ |
| filelock | 4.2.9 | MIT | https://py-filelock.readthedocs.io/en/latest/changelog.html |
| flatbuffers | 25.12.19 | Apache 2.0 | https://google.github.io/flatbuffers/ |
| fonttools | 4.68.0 | MIT | http://github.com/fonttools/fonttools |
| fsspec | 2026.9.0 | BSD-3-Clause | https://filesystem-spec.readthedocs.io/en/latest/changelog.html |
| glfw | 2.11.0 | MIT | https://github.com/FlorianRhiem/pyGLFW |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| httpcore | 1.0.9 | BSD-3-Clause | https://www.encode.io/httpcore |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx/blob/master/CHANGELOG.md |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna/blob/master/HISTORY.md |
| iniconfig | 2.3.1 | MIT | https://github.com/pytest-dev/iniconfig |
| Jinja2 | 3.1.6 | BSD License | https://jinja.palletsprojects.com/changes/ |
| kiwisolver | 1.5.1 | BSD License | https://github.com/nucleic/kiwi |
| lap | 0.5.13 | BSD-2-Clause | https://github.com/gatagat/lap |
| MarkupSafe | 3.0.4 | BSD-3-Clause | https://palletsprojects.com/donate |
| matplotlib | 3.11.2 | Python Software Foundation License | https://matplotlib.org |
| ml_dtypes | 0.6.0 | Apache-2.0 | https://github.com/jax-ml/ml_dtypes |
| mpmath | 1.3.0 | BSD | http://mpmath.org/ |
| networkx | 3.7 | BSD-3-Clause | https://networkx.org/ |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cuda-runtime | 13.0.96 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cudnn-cu13 | 9.24.0.43 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software | https://developer.nvidia.com/cusparselt |
| nvidia-ml-py | 13.615.71 | BSD | https://forums.developer.nvidia.com |
| nvidia-nccl-cu13 | 2.30.7 | BSD-3-Clause (license file; metadata: LicenseRef-NVIDIA-Proprietary) | https://developer.nvidia.com/cuda-zone |
| nvidia-nvjitlink | 13.5.37 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary | https://developer.nvidia.com/cuda-zone |
| nvidia-nvtx | 13.0.85 | Apache 2.0 | https://developer.nvidia.com/cuda-zone |
| onnx | 1.23.2 | Apache-2.0 | https://onnx.ai/ |
| onnxruntime-gpu | 1.31.1 | MIT License | https://onnxruntime.ai |
| opencv-python | 5.0.0.93 | Apache 2.0 | https://github.com/opencv/opencv-python |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://packaging.pypa.io/ |
| pillow | 13.0.0 | MIT-CMU | https://pillow.readthedocs.io/en/stable/releasenotes/index.html |
| pluggy | 1.7.0 | MIT | https://github.com/pytest-dev/pluggy |
| polars | 2.1.0 | MIT License | https://www.pola.rs/ |
| polars-runtime-32 | 2.1.0 | MIT License | https://github.com/pola-rs/polars/releases |
| protobuf | 7.36.2 | 3-Clause BSD License | https://developers.google.com/protocol-buffers/ |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| Pygments | 2.21.0 | BSD-2-Clause | https://pygments.org |
| pynvvideocodec | 2.2.3 | MIT | https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html |
| PyOpenGL | 3.1.10 | BSD License | https://mcfletch.github.io/pyopengl/ |
| pyparsing | 3.3.3 | MIT | https://pyparsing-docs.readthedocs.io/en/latest/ |
| pytest | 9.2.0 | MIT | https://docs.pytest.org/en/stable/changelog.html |
| python-dateutil | 4.0.0 | BSD-3-Clause | https://github.com/dateutil/dateutil |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/ |
| requests | 2.34.2 | Apache-2.0 | https://requests.readthedocs.io |
| setuptools | 84.2.1 | MIT | https://github.com/pypa/setuptools |
| sympy | 1.14.0 | BSD | https://sympy.org |
| torch | 2.14.1+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | https://pytorch.org |
| torchvision | 0.29.1+cu130 | BSD | https://github.com/pytorch/vision |
| triton | 3.8.0 | MIT | https://github.com/triton-lang/triton/ |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions/issues |
| tzdata | 2026.6 | Apache-2.0 | https://github.com/python/tzdata |
| ultralytics | 8.4.190 | AGPL-3.0 | https://ultralytics.com |
| ultralytics-platform | 0.1.104 | AGPL-3.0-only | https://github.com/ultralytics/sdk |
| ultralytics-thop | 2.2.5 | AGPL-3.0 | https://ultralytics.com |
| urllib3 | 2.8.0 | MIT | https://github.com/urllib3/urllib3/blob/main/CHANGES.rst |
