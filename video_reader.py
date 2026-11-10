# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Video readers for the demo: an OpenCV CPU reader and NVDEC readers that decode into CUDA memory.

create_reader picks one by name. All readers expose the same members (width, height, fps, frame_count, read, reset,
close), so the run loop does not depend on which one it was given.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ShapeOnlyImage:
    """A stand-in for a frame that carries only its (height, width, 3) shape, for code that reads just the shape."""

    __slots__ = ("shape",)

    def __init__(self, height: int, width: int):
        """Stores the shape for a three-channel image of the given size."""
        self.shape = (int(height), int(width), 3)


class OpenCVReader:
    """CPU OpenCV/FFmpeg path retained as an explicit fallback."""

    backend = "opencv"

    def __init__(self, path: Path):
        """Opens the file and reads its size, frame rate (25 if the file reports none) and frame count.
        Raises RuntimeError if OpenCV cannot open it."""
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open input: {self.path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = 0
        self.device_verified = False

    def read(self, host_frame: bool = True) -> tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Returns (True, BGR frame, timestamp in seconds) for the next frame, or (False, None, None) at the end.
        `host_frame` is ignored: frames always live in host memory."""
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        timestamp = self.index / self.fps if self.fps > 0 else None
        self.index += 1
        return True, frame, timestamp

    def reset(self) -> None:
        """Seeks back to the first frame so the clip can be read again."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.index = 0

    def close(self) -> None:
        """Releases the OpenCV capture."""
        self.cap.release()


class NvdecReader:
    """NVDEC reader with CUDA detector preprocessing and optional host bridge."""

    backend = "nvdec"

    def __init__(self, path: Path, gpu_id: int = 0, source=None):
        """Opens an NVDEC decoder on GPU `gpu_id` that outputs RGB frames in device memory. `source` is what the
        decoder reads (a file path by default; InMemoryNvdecReader passes the encoded bytes). Raises RuntimeError
        if PyNvVideoCodec or torch cannot be imported, the decoder cannot start, or the stream metadata is invalid."""
        self.path = Path(path)
        self.source = str(self.path) if source is None else source
        try:
            import PyNvVideoCodec as nvc
            from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder
            import torch
        except Exception as exc:
            raise RuntimeError("NVDEC requires NVIDIA PyNvVideoCodec and its CUDA runtime") from exc

        self._nvc = nvc
        self._torch = torch
        try:
            self.decoder = SimpleDecoder(
                self.source,
                gpu_id=gpu_id,
                use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
                bWaitForSessionWarmUp=True,
            )
        except Exception as exc:
            raise RuntimeError(f"NVDEC initialization failed for {self.path}") from exc

        metadata = self.decoder.get_stream_metadata()
        self.width = int(metadata.width)
        self.height = int(metadata.height)
        self.fps = float(metadata.average_fps)
        self.frame_count = int(len(self.decoder))
        self.index = 0
        self.device_verified = False
        self.last_device_frame = None
        self.shape_only_image = ShapeOnlyImage(self.height, self.width)
        if self.width <= 0 or self.height <= 0 or self.fps <= 0 or self.frame_count <= 0:
            self.close()
            raise RuntimeError(
                f"NVDEC returned invalid stream metadata: {self.width}x{self.height} "
                f"{self.fps} FPS, {self.frame_count} frames"
            )

    def read(self, host_frame: bool = True) -> tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Decodes the next frame into a CUDA tensor kept as last_device_frame and returns (True, frame, timestamp
        in seconds), or (False, None, None) at the end. The frame is a host BGR array only if `host_frame` is true,
        otherwise None. Waits for the GPU copy to finish. Closes the reader and raises RuntimeError if the decoder
        returns a frame that is not a CUDA tensor of the expected shape."""
        if self.index >= self.frame_count:
            return False, None, None

        decoded = self.decoder[self.index]
        if not hasattr(decoded, "__dlpack__"):
            self.close()
            raise RuntimeError("NVDEC returned a frame without DLPack device-memory support")

        tensor = self._torch.utils.dlpack.from_dlpack(decoded)
        if not tensor.is_cuda:
            self.close()
            raise RuntimeError(f"NVDEC returned non-CUDA frame: {tensor.device}")
        if tensor.ndim != 3 or tuple(tensor.shape) != (self.height, self.width, 3):
            self.close()
            raise RuntimeError(f"Unexpected NVDEC RGB frame shape: {tuple(tensor.shape)}")

        self.device_verified = True
        device_rgb = tensor.clone()
        self._torch.cuda.current_stream(device_rgb.device).synchronize()
        self.last_device_frame = device_rgb

        timestamp = self.index / self.fps
        self.index += 1
        if not host_frame:
            return True, None, timestamp
        return True, self.host_frame(), timestamp

    def host_frame(self, device_frame=None) -> np.ndarray:
        """Copies the last decoded frame to host memory as a contiguous BGR array. Raises RuntimeError if no frame
        has been read yet."""
        source = self.last_device_frame if device_frame is None else device_frame
        if source is None:
            raise RuntimeError("host_frame() called before read()")
        bgr = source.flip(-1).cpu().numpy()
        return bgr if bgr.flags.c_contiguous else np.ascontiguousarray(bgr)

    def reset(self) -> None:
        """Seeks the decoder back to the first frame and drops the last frame. Raises RuntimeError if the reader
        is closed."""
        if self.decoder is None:
            raise RuntimeError("Cannot reset a closed NVDEC decoder")
        self.decoder.seek_to_index(0)
        self.index = 0
        self.last_device_frame = None

    def cuda_input(self, target_size: int) -> tuple["object", dict[str, int | float]]:
        """Builds the detector input from the last decoded frame on the GPU: a float32 RGB tensor of shape
        (1, 3, target_size, target_size) scaled to 0..1, letterboxed with grey (114/255) padding. Returns it with
        a dict of the original and resized sizes, padding and ratio. Raises RuntimeError if no frame has been read
        or the frame is not a uint8 CUDA tensor."""
        if self.last_device_frame is None:
            raise RuntimeError("cuda_input() called before read()")
        if not self.last_device_frame.is_cuda:
            raise RuntimeError("NVDEC detector input is not CUDA-resident")
        if self.last_device_frame.dtype != self._torch.uint8:
            raise RuntimeError(f"Unexpected NVDEC dtype: {self.last_device_frame.dtype}")

        h, w = self.last_device_frame.shape[:2]
        target_h = target_w = int(target_size)
        ratio = min(target_h / h, target_w / w)
        new_w = max(round(w * ratio), 1)
        new_h = max(round(h * ratio), 1)
        dw = target_w - new_w
        dh = target_h - new_h
        left = round(dw / 2 - 0.1)
        top = round(dh / 2 - 0.1)

        chw = self.last_device_frame.permute(2, 0, 1).unsqueeze(0).contiguous()
        chw = chw.to(dtype=self._torch.float32).div_(255.0)
        resized = self._torch.nn.functional.interpolate(
            chw, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        output = self._torch.full(
            (1, 3, target_h, target_w),
            114.0 / 255.0,
            dtype=self._torch.float32,
            device=self.last_device_frame.device,
        )
        output[..., top : top + new_h, left : left + new_w] = resized
        return output.contiguous(), {
            "orig_h": int(h),
            "orig_w": int(w),
            "new_h": int(new_h),
            "new_w": int(new_w),
            "pad_left": int(left),
            "pad_top": int(top),
            "target": int(target_size),
            "ratio": float(ratio),
        }

    def close(self) -> None:
        """Releases the decoder and the last frame and collects garbage so GPU memory is freed. Safe to call twice
        and on a reader whose constructor failed part way."""
        decoder = getattr(self, "decoder", None)
        self.decoder = None
        self.last_device_frame = None
        if decoder is not None:
            del decoder
            gc.collect()


class InMemoryNvdecReader(NvdecReader):
    """NVDEC reader backed by one resident compressed MP4 byte buffer."""

    backend = "nvdec-memory"

    def __init__(self, path: Path, gpu_id: int = 0):
        """Reads the whole file into memory once and opens the decoder on those bytes, so looping never touches
        the disk again."""
        self.encoded_bytes = Path(path).read_bytes()
        self.encoded_bytes_len = len(self.encoded_bytes)
        super().__init__(Path(path), gpu_id=gpu_id, source=self.encoded_bytes)

    def reset(self) -> None:
        """Seeks back to the first frame; the bytes stay resident, so nothing is re-read."""
        super().reset()


def create_reader(path: Path, decoder: str, memory_input: bool = True):
    """Returns an OpenCVReader for "opencv" or an NVDEC reader for "nvdec", the in-memory one if `memory_input` is
    true. Raises ValueError for any other decoder name."""
    if decoder == "opencv":
        return OpenCVReader(path)
    if decoder == "nvdec":
        return InMemoryNvdecReader(path) if memory_input else NvdecReader(path)
    raise ValueError(f"Unsupported decoder: {decoder}")
