# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""The display stage: shows decoded frames from GPU memory in an OpenGL window with tracks and a status line drawn over them."""
from __future__ import annotations

import ctypes
import threading
import time

import cuda.bindings.driver as drv
import glfw
import numpy as np
from OpenGL import GL

from pipeline.bundle import BundlePool
from pipeline.config import InstanceConfig
from pipeline.gl_util import compile_shader, cuda_ok, make_texture
from pipeline.gpu import use_gpu
from pipeline.metrics import Metrics
from pipeline.overlay import FontAtlas, box_geometry, track_label

# Frame quad as vertex arrays (no glBegin/glEnd, see gl_prepare). The top of the image is v = 0.
_FRAME_VERTICES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
_FRAME_UVS = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
_STATUS_POSITION = (12.0, 28.0)
_STATUS_COLOUR = (255, 255, 0)
_STATUS_INTERVAL_S = 1.0
_LABEL_OFFSET = 7.0
_LABEL_MIN_Y = 18.0

_VERTEX_SHADER = """
#version 120
varying vec2 uv;
void main() {
    uv = gl_MultiTexCoord0.st;
    gl_Position = gl_Vertex;
}
"""

# BT.709 limited range, matching what the demuxer reports for this clip.
_FRAGMENT_SHADER = """
#version 120
uniform sampler2D luma;
uniform sampler2D chroma;
varying vec2 uv;
void main() {
    float y = (texture2D(luma, uv).r - 16.0 / 255.0) * (255.0 / 219.0);
    vec2 c = (texture2D(chroma, uv).rg - 128.0 / 255.0) * (255.0 / 224.0);
    gl_FragColor = vec4(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x, 1.0);
}
"""


_COMPACT_BELOW_WIDTH = 960


def status_text(current: dict, average: dict, overlay_lag: int, compact: bool = False) -> str:
    """Format the status line from the current and average frame rates and the overlay lag in frames.

    Missing rates read as zero. The compact form is shorter, for narrow windows.
    """
    if compact:
        return (f"{current.get('displayed', 0.0):.1f} fps"
                f" | trk {current.get('tracked', 0.0):.1f} ({average.get('tracked', 0.0):.1f})"
                f" | lag {overlay_lag}")
    return (f"display {current.get('displayed', 0.0):.1f} fps"
            f" | tracked {current.get('tracked', 0.0):.1f} fps (avg {average.get('tracked', 0.0):.1f})"
            f" | overlay_lag={overlay_lag}")


class DisplayStage:
    """Shows the pool's frames at the display rate on the calling thread, which owns the window and GL context.

    run() opens the window, shows one frame per tick, stops on a window close or the Q or Escape key (setting the
    shared stop event), and closes the window when done.
    """

    def __init__(self, config: InstanceConfig, pool: BundlePool, source_size: tuple[int, int],
                 names: dict[int, str], metrics: Metrics, stop: threading.Event, finished: threading.Event):
        """Record the collaborators and sizes; no window or GPU resource is created until open()."""
        self.config = config
        self.pool = pool
        self.source_width, self.source_height = source_size
        self.names = names
        self.metrics = metrics
        self.stop = stop
        self.finished = finished
        self.window = None
        self.resource = None
        self.title = ""
        self.text_scale = 1.0
        self._shown_any = False
        self._status = ""
        self._status_built = -_STATUS_INTERVAL_S
        self.frame_bytes = self.source_width * self.source_height * 3 // 2

    def open(self) -> None:
        """Create the window, shaders, textures, the pixel buffer shared with CUDA and the font atlas.

        Raises RuntimeError if GLFW cannot start, the window cannot be created or a shader fails to compile.
        """
        use_gpu(self.config.gpu_id)  # make the CUDA context current here
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        width, height = self.config.display_size
        self.title = f"Stream {self.config.stream}"
        self.text_scale = self.source_height / height
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, self.title, None, None)
        glfw.default_window_hints()
        if not self.window:
            raise RuntimeError("Could not create the display window")
        if self.config.window_position is not None:
            glfw.set_window_pos(self.window, *self.config.window_position)
        glfw.show_window(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)
        GL.glViewport(0, 0, self.fb_width, self.fb_height)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glLineWidth(2.0)

        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, compile_shader(GL.GL_VERTEX_SHADER, _VERTEX_SHADER))
        GL.glAttachShader(self.program, compile_shader(GL.GL_FRAGMENT_SHADER, _FRAGMENT_SHADER))
        GL.glLinkProgram(self.program)
        GL.glUseProgram(self.program)
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "luma"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "chroma"), 1)
        GL.glUseProgram(0)

        self.pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.frame_bytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        self.luma_texture = make_texture(GL.GL_R8, self.source_width, self.source_height, GL.GL_RED)
        self.chroma_texture = make_texture(GL.GL_RG8, self.source_width // 2, self.source_height // 2, GL.GL_RG)
        self.atlas = FontAtlas()
        self.atlas_texture = make_texture(GL.GL_ALPHA8, self.atlas.width, self.atlas.height, GL.GL_ALPHA,
                                          self.atlas.pixels.tobytes())
        self.resource = cuda_ok(drv.cuGraphicsGLRegisterBuffer(
            int(self.pbo), drv.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))

    def close(self) -> None:
        """Unregister the CUDA resource and destroy the window; does nothing if no window was opened."""
        if self.window is None:
            return
        if self.resource is not None:
            cuda_ok(drv.cuGraphicsUnregisterResource(self.resource))
            self.resource = None
        glfw.destroy_window(self.window)
        self.window = None

    def _upload(self, pointer: int) -> None:
        """Copy the NV12 frame at the device pointer into the pixel buffer and from it into the luma and chroma
        textures, without leaving the GPU."""
        cuda_ok(drv.cuGraphicsMapResources(1, self.resource, None))
        mapped, _size = cuda_ok(drv.cuGraphicsResourceGetMappedPointer(self.resource))
        cuda_ok(drv.cuMemcpyDtoD(mapped, pointer, self.frame_bytes))
        cuda_ok(drv.cuGraphicsUnmapResources(1, self.resource, None))
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.luma_texture)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.source_width, self.source_height,
                           GL.GL_RED, GL.GL_UNSIGNED_BYTE, None)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.chroma_texture)
        offset = ctypes.c_void_p(self.source_width * self.source_height)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.source_width // 2, self.source_height // 2,
                           GL.GL_RG, GL.GL_UNSIGNED_BYTE, offset)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _draw_frame(self) -> None:
        """Draw the video frame as a full-window quad, converting NV12 to RGB in the fragment shader."""
        GL.glUseProgram(self.program)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
        GL.glVertexPointer(2, GL.GL_FLOAT, 0, _FRAME_VERTICES)
        GL.glTexCoordPointer(2, GL.GL_FLOAT, 0, _FRAME_UVS)
        GL.glDrawArrays(GL.GL_QUADS, 0, 4)
        GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glUseProgram(0)

    def _draw_overlay(self, tracks: np.ndarray, status: str) -> None:
        """Draw track boxes in source pixel coordinates, then their labels and the status line from the font atlas.

        Labels are placed above each box but kept inside the image; text scales with the window relative to the source.
        """
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, self.source_width, self.source_height, 0, -1, 1)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)

        vertices, colours = box_geometry(tracks)
        if len(vertices):
            GL.glVertexPointer(2, GL.GL_FLOAT, 0, vertices)
            GL.glColorPointer(3, GL.GL_UNSIGNED_BYTE, 0, colours)
            GL.glDrawArrays(GL.GL_LINES, 0, len(vertices))

        scale = self.text_scale
        labels = []
        box_colours = colours[::8]
        for row, colour in zip(tracks, box_colours):
            y = max(float(row[1]) - _LABEL_OFFSET * scale, _LABEL_MIN_Y * scale)
            labels.append((track_label(row, self.names), float(row[0]), y, tuple(int(c) for c in colour)))
        if status:
            labels.append((status, _STATUS_POSITION[0] * scale, _STATUS_POSITION[1] * scale, _STATUS_COLOUR))
        quad_vertices, quad_uvs, quad_colours = self.atlas.text_geometry(labels, scale=scale)
        if len(quad_vertices):
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.atlas_texture)
            GL.glEnable(GL.GL_BLEND)
            GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glVertexPointer(2, GL.GL_FLOAT, 0, quad_vertices)
            GL.glTexCoordPointer(2, GL.GL_FLOAT, 0, quad_uvs)
            GL.glColorPointer(3, GL.GL_UNSIGNED_BYTE, 0, quad_colours)
            GL.glDrawArrays(GL.GL_QUADS, 0, len(quad_vertices))
            GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def show(self, pointer: int, tracks: np.ndarray, status: str) -> None:
        """Render one frame into the back buffer: the video at `pointer`, then the tracks (none if None) and status."""
        self._upload(pointer)
        self._draw_frame()
        self._draw_overlay(tracks if tracks is not None else np.zeros((0, 8), np.float32), status)

    def read_pixels(self) -> np.ndarray:
        """RGB pixels of the back buffer, top row first."""
        GL.glFinish()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        raw = GL.glReadPixels(0, 0, self.fb_width, self.fb_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(raw, np.uint8).reshape(self.fb_height, self.fb_width, 3)[::-1]

    def _next_choice(self):
        """Wait up to half a tick. Frames finish a few ms after they are due, so a fixed tick can land just before one."""
        deadline = time.monotonic() + 0.5 / self.config.display_fps
        choice = self.pool.take_for_display()
        while choice is None and time.monotonic() < deadline and not self.stop.is_set():
            self.stop.wait(0.001)
            choice = self.pool.take_for_display()
        return choice

    def _status_for(self, overlay_lag: int) -> str:
        """Rebuilt at most once a second, so the label cache is not filled with one-off strings."""
        now = time.monotonic()
        if now - self._status_built >= _STATUS_INTERVAL_S:
            snapshot = self.metrics.snapshot()
            compact = self.config.display_size[0] < _COMPACT_BELOW_WIDTH
            self._status = status_text(self.metrics.current_fps(), snapshot["fps"], overlay_lag, compact)
            self._status_built = now
        return self._status

    def _tick(self) -> bool:
        """Show the next frame from the pool and swap buffers; return False if no frame was available.

        The bundle's lease is released even if drawing fails. Counts shown frames, complete or not, and ticks that
        found nothing to show, separately before and after the first frame.
        """
        choice = self._next_choice()
        if choice is None:
            self.metrics.count("ticks_without_frame" if self._shown_any else "ticks_before_first_frame")
            return False
        try:
            started = time.perf_counter()
            bundle = choice.bundle
            lag = bundle.frame_id - choice.tracks_frame_id
            self.show(bundle.nv12_pointer, choice.tracks, self._status_for(lag))
        finally:
            self.pool.release(choice.bundle)
        glfw.swap_buffers(self.window)
        self.metrics.record_ms("display", (time.perf_counter() - started) * 1e3)
        self._shown_any = True
        self.metrics.count("displayed")
        self.metrics.count("displayed_complete" if choice.complete else "displayed_incomplete")
        return True

    def _stop_requested(self) -> bool:
        """Whether the user closed the window or pressed Q or Escape."""
        return (glfw.window_should_close(self.window)
                or glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS
                or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS)

    def run(self) -> None:
        """Open the window and show frames at the display rate until stopped or the stream is finished and drained.

        A tick that overruns its period is counted as late and the schedule restarts from now. The window is closed on
        every exit path.
        """
        period = 1.0 / self.config.display_fps
        next_tick = time.monotonic()
        try:
            self.open()
            while not self.stop.is_set():
                glfw.poll_events()
                if self._stop_requested():
                    self.stop.set()
                    break
                shown = self._tick()
                if self.finished.is_set() and not shown:
                    break
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self.stop.wait(delay)
                else:
                    self.metrics.count("ticks_late")
                    next_tick = time.monotonic()
        finally:
            self.close()
