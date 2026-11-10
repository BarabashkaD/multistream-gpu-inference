# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Cleanup and shutdown paths of the OpenGL prepare and of run_pipeline, with fakes and no window or GPU."""
import itertools
import queue
import sys
import threading
import types
import zlib

import cuda.bindings.driver as real_drv
import pytest

from pipeline import app, gl_prepare, gl_util
from pipeline.config import PipelineConfig
from pipeline.nv12 import Letterbox

SUCCESS = real_drv.CUresult.CUDA_SUCCESS


class FakeGL:
    """Records every call; names in `fail` raise. Constants get stable integers."""

    def __init__(self):
        """Starts with no recorded calls and no names set to fail."""
        self.calls = []
        self.fail = set()
        self._ids = itertools.count(1)

    def __getattr__(self, name):
        """Answers any OpenGL name: a GL_ constant becomes a stable integer, anything else a recording function."""
        if name.startswith("GL_"):
            return zlib.crc32(name.encode()) % 100000 + 1

        def call(*args):
            """Records the call, raises if the name is in `fail`, and returns what a successful call would: 1 for
            compile and link status, a complete framebuffer status, a new id for glGen and glCreate calls."""
            self.calls.append(name)
            if name in self.fail:
                raise RuntimeError(f"{name} failed")
            if name in ("glGetShaderiv", "glGetProgramiv"):
                return 1
            if name == "glCheckFramebufferStatus":
                return self.GL_FRAMEBUFFER_COMPLETE
            if name.startswith(("glGen", "glCreate")):
                return next(self._ids)
            return None

        return call


class FakeGlfw:
    """Stand-in for the glfw module that records calls and hands out a fixed window, or None to fail creation."""
    VISIBLE, FALSE = 1, 0

    def __init__(self, window=object()):
        """Takes the window object that create_window will return."""
        self.calls = []
        self.window = window

    def init(self):
        """Records the call and reports success."""
        self.calls.append("init")
        return True

    def window_hint(self, *args):
        """Records the call."""
        self.calls.append("window_hint")

    def default_window_hints(self):
        """Records the call."""
        self.calls.append("default_window_hints")

    def create_window(self, *args):
        """Records the call and returns the configured window."""
        self.calls.append("create_window")
        return self.window

    def make_context_current(self, window):
        """Records the call with its window; None means the context was released."""
        self.calls.append(("make_context_current", window))

    def destroy_window(self, window):
        """Records the call."""
        self.calls.append("destroy_window")

    def terminate(self):
        """Records the call."""
        self.calls.append("terminate")


class FakeDrv:
    """Stand-in for the CUDA driver bindings that hands out resource handles and can fail the unregister call."""
    CUresult = real_drv.CUresult
    CUgraphicsRegisterFlags = real_drv.CUgraphicsRegisterFlags

    def __init__(self):
        """Starts with no recorded calls and no names set to fail."""
        self.calls = []
        self.fail = set()
        self._handles = itertools.count(100)

    def cuGraphicsGLRegisterBuffer(self, buffer, flags):
        """Records the call and returns success with a new handle."""
        self.calls.append("register")
        return (SUCCESS, next(self._handles))

    def cuGraphicsUnregisterResource(self, resource):
        """Records the call and returns success, or an error status when "unregister" is in `fail`."""
        self.calls.append("unregister")
        if "unregister" in self.fail:
            return (real_drv.CUresult.CUDA_ERROR_UNKNOWN,)
        return (SUCCESS,)


@pytest.fixture
def fakes(monkeypatch):
    """Replaces OpenGL, glfw, the CUDA driver bindings and torch in the converter modules; returns the fakes."""
    gl, glfw, drv = FakeGL(), FakeGlfw(), FakeDrv()
    for module in (gl_prepare, gl_util):
        monkeypatch.setattr(module, "GL", gl)
    monkeypatch.setattr(gl_prepare, "glfw", glfw)
    monkeypatch.setattr(gl_prepare, "drv", drv)
    monkeypatch.setattr(gl_prepare, "torch", types.SimpleNamespace(zeros=lambda *a, **k: None))
    return types.SimpleNamespace(gl=gl, glfw=glfw, drv=drv)


def _converter(window=object()):
    """Builds a GlConverter for a 1920x1080 source and 640 px input, with its window already set as setup_main would."""
    converter = gl_prepare.GlConverter(Letterbox(1920, 1080, 640), (1920, 1080))
    converter.window = window
    return converter


RELEASED = ("make_context_current", None)


def test_close_without_open_does_nothing(fakes):
    """Proves closing a converter that was never opened makes no OpenGL, glfw or CUDA call."""
    _converter().close()
    assert fakes.gl.calls == [] and fakes.glfw.calls == [] and fakes.drv.calls == []


def test_open_failing_partway_releases_the_context_and_resets_state(fakes):
    """Proves an open that fails midway releases the OpenGL context and leaves no program or targets behind."""
    fakes.gl.fail.add("glGenFramebuffers")
    converter = _converter()
    with pytest.raises(RuntimeError, match="glGenFramebuffers failed"):
        converter.open()
    assert RELEASED in fakes.glfw.calls
    assert converter.context_current is False
    assert converter.program is None and converter.targets == []


def test_open_failing_after_the_first_registration_unregisters_it(fakes):
    """Proves a failure while registering the second buffer with CUDA unregisters the first and releases the
    context."""
    converter = _converter()
    registered = fakes.drv.cuGraphicsGLRegisterBuffer

    def second_fails(buffer, flags):
        """Registers the first buffer normally and raises on any later one."""
        if "register" in fakes.drv.calls:
            raise RuntimeError("second register failed")
        return registered(buffer, flags)

    fakes.drv.cuGraphicsGLRegisterBuffer = second_fails
    with pytest.raises(RuntimeError, match="second register failed"):
        converter.open()
    assert fakes.drv.calls.count("unregister") == 1
    assert RELEASED in fakes.glfw.calls


def test_cleanup_error_on_the_clean_path_raises_and_releases_the_context(fakes):
    """Proves an OpenGL error during a normal close is raised, and the context is still released."""
    converter = _converter()
    converter.open()
    fakes.gl.fail.add("glDeleteProgram")
    with pytest.raises(RuntimeError, match="glDeleteProgram failed"):
        converter.close()
    assert RELEASED in fakes.glfw.calls
    assert converter.context_current is False


def test_cuda_cleanup_error_on_the_clean_path_raises(fakes):
    """Proves a CUDA error while unregistering a buffer during a normal close is raised, and the context is still
    released."""
    converter = _converter()
    converter.open()
    fakes.drv.fail.add("unregister")
    with pytest.raises(RuntimeError, match="CUDA call failed"):
        converter.close()
    assert RELEASED in fakes.glfw.calls


def test_cleanup_error_does_not_replace_a_propagating_exception(fakes):
    """Proves a cleanup error in close does not mask an exception already propagating through the caller's finally
    block."""
    converter = _converter()
    converter.open()
    fakes.gl.fail.add("glDeleteProgram")
    with pytest.raises(ValueError, match="original"):
        try:
            raise ValueError("original")
        finally:
            converter.close()
    assert RELEASED in fakes.glfw.calls


def test_a_failing_delete_does_not_stop_later_deletes(fakes):
    """Proves when deleting the program fails, close still deletes the framebuffers, textures and buffers."""
    converter = _converter()
    converter.open()
    fakes.gl.fail.add("glDeleteProgram")
    with pytest.raises(RuntimeError):
        converter.close()
    for later in ("glDeleteFramebuffers", "glDeleteTextures", "glDeleteBuffers"):
        assert later in fakes.gl.calls, later


def test_shaders_are_deleted_after_linking(fakes):
    """Proves both shader objects are deleted once the program is linked, as they are no longer needed."""
    converter = _converter()
    converter.open()
    assert fakes.gl.calls.count("glDeleteShader") == 2
    assert fakes.gl.calls.index("glDeleteShader") > fakes.gl.calls.index("glLinkProgram")


def test_shaders_are_deleted_when_linking_fails(fakes):
    """Proves both shader objects are deleted even when linking the program fails."""
    fakes.gl.fail.add("glLinkProgram")
    converter = _converter()
    with pytest.raises(RuntimeError, match="glLinkProgram failed"):
        converter.open()
    assert fakes.gl.calls.count("glDeleteShader") == 2


def test_teardown_main_before_setup_main_is_safe(fakes):
    """Proves teardown_main on a converter whose setup_main never ran makes no glfw call."""
    converter = gl_prepare.GlConverter(Letterbox(1920, 1080, 640), (1920, 1080))
    converter.teardown_main()
    assert fakes.glfw.calls == []


def test_teardown_main_after_a_failed_create_window_is_safe(fakes):
    """Proves teardown_main after a failed window creation does not try to destroy a window."""
    fakes.glfw.window = None
    converter = gl_prepare.GlConverter(Letterbox(1920, 1080, 640), (1920, 1080))
    with pytest.raises(RuntimeError, match="Could not create"):
        converter.setup_main()
    converter.teardown_main()
    assert "destroy_window" not in fakes.glfw.calls


class _Converter:
    """Stand-in converter that records setup_main and teardown_main and can fail either."""

    def __init__(self, setup_error=None, teardown_error=None):
        """Takes the exception, if any, that each step raises."""
        self.setup_error, self.teardown_error = setup_error, teardown_error
        self.events = []

    def setup_main(self):
        """Records the step, then raises the configured error if there is one."""
        self.events.append("setup")
        if self.setup_error:
            raise self.setup_error

    def teardown_main(self):
        """Records the step, then raises the configured error if there is one."""
        self.events.append("teardown")
        if self.teardown_error:
            raise self.teardown_error


def test_release_graphics_terminates_glfw_even_if_teardown_raises(monkeypatch):
    """Proves glfw is terminated even when the converter's teardown raises, and that the error still comes out."""
    glfw = FakeGlfw()
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    config = PipelineConfig(display=False, prepare="gl")
    with pytest.raises(RuntimeError, match="teardown boom"):
        app._release_graphics(config, _Converter(teardown_error=RuntimeError("teardown boom")))
    assert "terminate" in glfw.calls


class _FakeStage:
    """Stand-in stage thread that finishes at once and, if given a failure, reports it when started."""

    def __init__(self, name, failure=None, failures=None):
        """Takes the stage name, the failure to report and the shared list the failure is appended to."""
        self.name = name
        self.failure, self.failures = failure, failures
        self.production_done = threading.Event()  # the decoder's handshake; a fake finishes at once
        self.production_done.set()

    def start(self):
        """Appends the stage's failure, if any, to the shared failure list."""
        if self.failure:
            self.failures.append((self.name, self.failure))

    def join(self, timeout=None):
        """Returns at once."""
        pass

    def is_alive(self):
        """Reports the thread as finished."""
        return False


class _FakePool:
    """Stand-in bundle pool with just the calls run_pipeline makes at the end of a run."""

    def __init__(self):
        """Empty stage queues, drained by run_pipeline at shutdown."""
        self.to_infer, self.to_detect, self.to_track = queue.Queue(), queue.Queue(), queue.Queue()

    def finish(self):
        """Does nothing."""
        pass

    def stats(self):
        """Returns an empty pool's statistics."""
        return {"free": 0, "size": 0, "leased": 0}


@pytest.fixture
def pipeline(monkeypatch):
    """run_pipeline with every heavy part replaced; returns the converter and a way to add a stage failure."""
    state = types.SimpleNamespace(converter=_Converter(), stage_failure=None)

    class Source:
        """Stand-in video source with a fixed 1920x1080 size."""
        width, height = 1920, 1080

        def __init__(self, path):
            """Ignores the path."""
            pass

    detector = types.SimpleNamespace(output_shape=(1, 1), names={})
    monkeypatch.setattr(app, "MemorySource", Source)
    monkeypatch.setattr(app, "Detector", lambda model, gpu, batch_size, size: detector)
    monkeypatch.setattr(app, "_build_bundles", lambda config, det: [])
    monkeypatch.setattr(app, "BundlePool", lambda *a, **k: _FakePool())
    monkeypatch.setattr(app, "TrackerCore", lambda *a, **k: None)
    monkeypatch.setattr(app, "_make_converter", lambda *a: state.converter)
    monkeypatch.setattr(app, "_wait", lambda *a: None)

    def stage(name):
        """Returns a factory for a fake stage; only the reader gets the failure set on `state`."""
        def make(*args):
            """Builds the fake stage, sharing the last constructor argument as its failure list."""
            return _FakeStage(name, state.stage_failure if name == "reader" else None, args[-1])
        return make

    for cls, name in (("ReaderStage", "reader"), ("DecoderStage", "decoder"), ("PrepareStage", "prepare"),
                      ("InferenceStage", "inference"), ("TrackerStage", "tracker")):
        monkeypatch.setattr(app, cls, stage(name))
    return state


def _config():
    """A configuration with the display off and a report interval too long to fire during a test."""
    return PipelineConfig(display=False, report_interval_s=60.0)


def test_failed_setup_main_still_releases_graphics(pipeline):
    """Proves when setting up graphics fails, run_pipeline still tears them down and raises the setup error."""
    pipeline.converter = _Converter(setup_error=RuntimeError("no window"))
    with pytest.raises(RuntimeError, match="no window"):
        app.run_pipeline(_config())
    assert pipeline.converter.events == ["setup", "teardown"]


def test_graphics_failure_does_not_replace_a_stage_failure(pipeline):
    """Proves a graphics teardown error does not hide a stage failure, which stays the raised error."""
    pipeline.converter = _Converter(teardown_error=RuntimeError("teardown boom"))
    pipeline.stage_failure = ValueError("reader boom")
    with pytest.raises(RuntimeError, match="stage reader failed: reader boom") as caught:
        app.run_pipeline(_config())
    assert isinstance(caught.value.__cause__, ValueError)


def test_graphics_failure_alone_is_raised(pipeline):
    """Proves a graphics teardown error with no other failure is raised as a failure of the graphics stage."""
    pipeline.converter = _Converter(teardown_error=RuntimeError("teardown boom"))
    with pytest.raises(RuntimeError, match="stage graphics failed: teardown boom"):
        app.run_pipeline(_config())


def test_graphics_failure_does_not_replace_an_exception_already_propagating(pipeline, monkeypatch):
    """Proves a graphics teardown error does not mask an exception already propagating out of run_pipeline."""
    pipeline.converter = _Converter(teardown_error=RuntimeError("teardown boom"))

    def interrupted(*args):
        """Stands in for the wait and raises, as an interrupt would."""
        raise MemoryError("original")

    monkeypatch.setattr(app, "_wait", interrupted)
    with pytest.raises(MemoryError, match="original"):
        app.run_pipeline(_config())
