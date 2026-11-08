# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""OpenGL and CUDA helpers for the display.

Imports only cuda.bindings.driver and OpenGL.GL. It must not import glfw, and pipeline.app must not import it at
module level, so a --no-display run loads none of it.
"""
from __future__ import annotations

import cuda.bindings.driver as drv
from OpenGL import GL


def cuda_ok(result):
    """Unwrap a cuda-python result tuple, raising when the status is not success."""
    status, *values = result if isinstance(result, tuple) else (result,)
    if status != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA call failed: {status}")
    return values[0] if len(values) == 1 else values


def compile_shader(kind, source: str) -> int:
    """Compile a shader of the given kind from source and return its id; raise RuntimeError with the compiler log on failure."""
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(shader))
    return shader


def make_texture(internal_format, width: int, height: int, data_format, pixels=None,
                 data_type=GL.GL_UNSIGNED_BYTE, linear: bool = True) -> int:
    """Create a 2D texture with edge clamping and linear or nearest filtering, and return its id.

    pixels may be None to allocate storage only; the texture is left bound.
    """
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    interpolation = GL.GL_LINEAR if linear else GL.GL_NEAREST
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, interpolation)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, interpolation)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, pixels)
    return texture
