# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""OpenGL frame preparation: NV12 in decoder memory to the planar float32 detector input.

Optional: pipeline.app imports this module only for --prepare gl. The conversion is a fragment shader that
resizes with bilinear texture filtering, converts BT.709 limited-range YUV to RGB and clamps to [0, 1].
It writes three float planes through one framebuffer; the planes travel through CUDA-GL interop buffers.
Per frame this costs two GPU copies on top of the draw: decoder memory to the upload buffer and the
readback buffer to the input tensor.
"""
from __future__ import annotations

import ctypes
import sys
from functools import partial

import cuda.bindings.driver as drv
import glfw
import numpy as np
import torch
from OpenGL import GL

from pipeline.gl_util import compile_shader, cuda_ok, make_texture
from pipeline.nv12 import PAD_VALUE, Letterbox

_PLANES = 3

_VERTEX = """
#version 120
varying vec2 uv;
void main() {
    uv = gl_MultiTexCoord0.st;
    gl_Position = gl_Vertex;
}
"""

# Same maths as nv12_to_input: BT.709 limited range, result clamped to [0, 1].
_FRAGMENT = """
#version 120
uniform sampler2D luma;
uniform sampler2D chroma;
varying vec2 uv;
void main() {
    float y = (texture2D(luma, uv).r * 255.0 - 16.0) * (255.0 / 219.0);
    vec2 c = (texture2D(chroma, uv).rg * 255.0 - 128.0) * (255.0 / 224.0);
    vec3 rgb = vec3(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x) / 255.0;
    rgb = clamp(rgb, 0.0, 1.0);
    gl_FragData[0] = vec4(rgb.r);
    gl_FragData[1] = vec4(rgb.g);
    gl_FragData[2] = vec4(rgb.b);
}
"""


# Drawn with vertex arrays, and so is the display (a test enforces it): PyOpenGL suspends error checking for
# every thread between glBegin and glEnd with a process-wide flag, which would hide errors raised on this thread.
# Framebuffer row 0 is the bottom, so the bottom vertex samples the top of the image (v = 0).
_QUAD_VERTICES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
_QUAD_UVS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def _run_all(steps) -> None:
    """Run every step even if one fails. The first error is raised afterwards, unless an exception is already
    propagating (the caller is in a finally or except block): that one must not be replaced."""
    propagating = sys.exc_info()[0] is not None
    errors: list[Exception] = []
    for step in steps:
        try:
            step()
        except Exception as error:
            errors.append(error)
    if errors and not propagating:
        raise errors[0]


class GlConverter:
    """NV12 to detector input with an OpenGL fragment shader.

    setup_main and teardown_main run on the main thread (GLFW windows belong there). open, convert and close
    run on the prepare thread, which owns the GL context and torch's CUDA context while it works.
    """

    def __init__(self, letterbox: Letterbox, source_size: tuple[int, int], gpu_id: int = 0):
        """Record the geometry and buffer sizes; no window or GL object exists until setup_main and open."""
        self.letterbox = letterbox
        self.width, self.height = source_size
        self.gpu_id = gpu_id
        self.target_height, self.target_width = letterbox.target_height, letterbox.target_width
        self.frame_bytes = self.width * self.height * 3 // 2
        self.plane_bytes = self.target_height * self.target_width * 4
        self.window = None
        self._reset_gl_state()

    def _reset_gl_state(self) -> None:
        """Forget every GL and CUDA handle, so a closed converter looks like a fresh one."""
        self.context_current = False
        self.program = self.up_buffer = self.down_buffer = self.framebuffer = None
        self.luma = self.chroma = None
        self.targets: list[int] = []
        self.up_resource = self.down_resource = None

    def setup_main(self) -> None:
        """Start GLFW and create the hidden 1x1 window that owns the preparation GL context. Main thread only.

        Raises RuntimeError if GLFW cannot start or the window cannot be created.
        """
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        try:
            self.window = glfw.create_window(1, 1, "prepare", None, None)
        finally:
            glfw.default_window_hints()  # the display window created later must be visible
        if not self.window:
            raise RuntimeError("Could not create the OpenGL window for frame preparation")

    def open(self) -> None:
        """Make the GL and CUDA contexts current on this thread and build the shader and buffers.

        Raises RuntimeError if setup_main has not run. A failure part-way through closes what was created and
        re-raises.
        """
        if not self.window:
            raise RuntimeError("setup_main must run before open")
        try:
            glfw.make_context_current(self.window)
            self.context_current = True
            torch.zeros(1, device=f"cuda:{self.gpu_id}")  # makes torch's CUDA context current on this thread
            self._build()
        except BaseException:
            self.close()
            raise

    def _build(self) -> None:
        """Create the shader program, upload and readback buffers, textures and framebuffer, and register the buffers with CUDA.

        Raises RuntimeError if the program does not link or the framebuffer is incomplete.
        """
        self.program = GL.glCreateProgram()
        shaders: list[int] = []
        try:
            for kind, source in ((GL.GL_VERTEX_SHADER, _VERTEX), (GL.GL_FRAGMENT_SHADER, _FRAGMENT)):
                shaders.append(compile_shader(kind, source))
                GL.glAttachShader(self.program, shaders[-1])
            GL.glLinkProgram(self.program)
            if not GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS):
                raise RuntimeError(GL.glGetProgramInfoLog(self.program))
        finally:
            _run_all([partial(GL.glDeleteShader, shader) for shader in shaders])  # the program keeps what it linked
        GL.glUseProgram(self.program)
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "luma"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "chroma"), 1)
        GL.glUseProgram(0)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        self.up_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.up_buffer)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.frame_bytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        self.luma = make_texture(GL.GL_R8, self.width, self.height, GL.GL_RED)
        self.chroma = make_texture(GL.GL_RG8, self.width // 2, self.height // 2, GL.GL_RG)

        self.framebuffer = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
        for plane in range(_PLANES):
            target = make_texture(GL.GL_R32F, self.target_width, self.target_height, GL.GL_RED, data_type=GL.GL_FLOAT, linear=False)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + plane, GL.GL_TEXTURE_2D, target, 0)
            self.targets.append(target)
        GL.glDrawBuffers(_PLANES, [GL.GL_COLOR_ATTACHMENT0 + plane for plane in range(_PLANES)])
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"framebuffer incomplete: {status}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glClampColor(GL.GL_CLAMP_FRAGMENT_COLOR, GL.GL_FALSE)  # the shader clamps; float targets keep the result

        self.down_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self.down_buffer)
        GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, _PLANES * self.plane_bytes, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

        self.up_resource = cuda_ok(drv.cuGraphicsGLRegisterBuffer(
            int(self.up_buffer), drv.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))
        self.down_resource = cuda_ok(drv.cuGraphicsGLRegisterBuffer(
            int(self.down_buffer), drv.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY))

    def convert(self, bundle) -> None:
        """Fill the bundle's input tensor from its NV12 frame through the shader, and wait until it is complete.

        The steps below run in order: copy in, upload to textures, draw the letterboxed picture, read the planes
        back, copy out to the tensor.
        """
        # 1. decoder memory to the upload buffer
        cuda_ok(drv.cuGraphicsMapResources(1, self.up_resource, None))
        mapped, _size = cuda_ok(drv.cuGraphicsResourceGetMappedPointer(self.up_resource))
        cuda_ok(drv.cuMemcpyDtoD(mapped, bundle.nv12_pointer, self.frame_bytes))
        cuda_ok(drv.cuGraphicsUnmapResources(1, self.up_resource, None))
        # 2. upload buffer to the luma and chroma textures
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.up_buffer)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.luma)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL.GL_RED, GL.GL_UNSIGNED_BYTE,
                           ctypes.c_void_p(0))
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.chroma)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.width // 2, self.height // 2, GL.GL_RG,
                           GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(self.width * self.height))
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        # 3. draw the letterboxed picture over a padding-coloured clear
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
        GL.glViewport(0, 0, self.target_width, self.target_height)
        GL.glClearColor(PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glViewport(self.letterbox.left, self.letterbox.top, self.letterbox.new_width, self.letterbox.new_height)
        GL.glUseProgram(self.program)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
        GL.glVertexPointer(2, GL.GL_FLOAT, 0, _QUAD_VERTICES)
        GL.glTexCoordPointer(2, GL.GL_FLOAT, 0, _QUAD_UVS)
        GL.glDrawArrays(GL.GL_QUADS, 0, 4)
        GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glUseProgram(0)
        # 4. read the three planes into one buffer
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self.down_buffer)
        for plane in range(_PLANES):
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0 + plane)
            GL.glReadPixels(0, 0, self.target_width, self.target_height, GL.GL_RED, GL.GL_FLOAT,
                            ctypes.c_void_p(plane * self.plane_bytes))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glFinish()  # blocks until every issued command, including the buffer writes, is complete
        # 5. readback buffer to the input tensor
        cuda_ok(drv.cuGraphicsMapResources(1, self.down_resource, None))
        mapped, _size = cuda_ok(drv.cuGraphicsResourceGetMappedPointer(self.down_resource))
        cuda_ok(drv.cuMemcpyDtoD(bundle.input_tensor.data_ptr(), mapped, _PLANES * self.plane_bytes))
        cuda_ok(drv.cuCtxSynchronize())  # the tensor must be complete before the bundle is marked
        cuda_ok(drv.cuGraphicsUnmapResources(1, self.down_resource, None))

    def close(self) -> None:
        """Release CUDA and GL objects. Safe after a partial open.

        Every step runs even if an earlier one fails, and the context is always released. A cleanup error is
        raised at the end, unless close runs while another error propagates: that one must not be replaced.
        """
        steps = [partial(lambda r: cuda_ok(drv.cuGraphicsUnregisterResource(r)), resource)
                 for resource in (self.up_resource, self.down_resource) if resource is not None]
        if self.context_current:
            steps += self._delete_steps()
            steps.append(partial(glfw.make_context_current, None))
        try:
            _run_all(steps)
        finally:
            self._reset_gl_state()

    def _delete_steps(self) -> list:
        """One step per kind of GL object, so that a failing delete does not skip the others."""
        steps = []
        if self.program is not None:
            steps.append(partial(GL.glDeleteProgram, self.program))
        if self.framebuffer is not None:
            steps.append(partial(GL.glDeleteFramebuffers, 1, [self.framebuffer]))
        textures = [t for t in (self.luma, self.chroma, *self.targets) if t is not None]
        if textures:
            steps.append(partial(GL.glDeleteTextures, textures))
        buffers = [b for b in (self.up_buffer, self.down_buffer) if b is not None]
        if buffers:
            steps.append(partial(GL.glDeleteBuffers, len(buffers), buffers))
        return steps


    def teardown_main(self) -> None:
        """Destroy the hidden window. Main thread only; does nothing if it was never created."""
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
