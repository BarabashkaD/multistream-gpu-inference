# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dmitry Veremeev
# Additional terms under AGPL-3.0 section 7 apply: see NOTICE.
"""Tests for the NV12 module: letterbox geometry and box mapping, NV12 to network input conversion, and wrapping."""
import numpy as np
import pytest
import torch

from pipeline.nv12 import PAD_VALUE, Letterbox, nv12_to_input, wrap_nv12


def test_letterbox_for_1080p():
    """Proves a 1920x1080 source scaled into 640 gets a height of 360, 140 rows of padding above, and a gain of 1/3."""
    box = Letterbox(1920, 1080, 640)
    assert box.new_height == 360
    assert box.top == 140
    assert box.gain == pytest.approx(1 / 3)


def test_to_source_removes_padding_and_scales():
    """Proves boxes in network coordinates map back to source pixels by removing the padding and undoing the scale."""
    box = Letterbox(1920, 1080, 640)
    result = box.to_source(np.array([[100.0, 140.0, 200.0, 500.0]], np.float32))
    assert result[0].tolist() == pytest.approx([300.0, 0.0, 600.0, 1080.0])


def test_to_source_clips_boxes_to_the_frame():
    """Proves boxes reaching into the padding or beyond the picture are clipped to the source frame."""
    box = Letterbox(1920, 1080, 640)
    result = box.to_source(np.array([[-10.0, 100.0, 700.0, 520.0]], np.float32))
    assert result[0].tolist() == pytest.approx([0.0, 0.0, 1920.0, 1080.0])


def test_letterbox_pads_the_sides_of_a_portrait_source():
    """Proves a portrait source is padded left and right instead of top and bottom, and boxes still map back."""
    box = Letterbox(1080, 1920, 640)
    assert (box.new_width, box.left, box.top) == (360, 140, 0)
    result = box.to_source(np.array([[140.0, 0.0, 500.0, 640.0]], np.float32))
    assert result[0].tolist() == pytest.approx([0.0, 0.0, 1080.0, 1920.0])


def grey_nv12():
    """Builds a 1920x1080 NV12 frame of flat luma 126 with neutral chroma."""
    nv12 = torch.full((1620, 1920), 128, dtype=torch.uint8)  # neutral chroma
    nv12[:1080] = 126
    return nv12


def test_input_places_the_picture_between_padding_rows():
    """Proves the converted image has padding values above and below the picture and the converted grey inside."""
    output = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    nv12_to_input(grey_nv12(), Letterbox(1920, 1080, 640), 1920, 1080, output)
    grey = 110.0 / 219.0
    for row, value in ((139, PAD_VALUE), (140, grey), (300, grey), (499, grey), (500, PAD_VALUE)):
        assert output[0, :, row, 300].tolist() == pytest.approx([value] * 3, abs=1e-4)


def test_input_converts_chroma_with_bt709_limited_range():
    """Proves a known chroma value converts to the expected RGB using the BT.709 limited-range matrix."""
    nv12 = grey_nv12()
    nv12[1080:, 1::2] = 160  # V is the second byte of each chroma pair
    output = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    nv12_to_input(nv12, Letterbox(1920, 1080, 640), 1920, 1080, output)
    assert output[0, :, 300, 300].tolist() == pytest.approx([0.72725, 0.43541, 0.50228], abs=1e-3)


@pytest.mark.gpu
def test_wrap_is_a_view_of_the_same_memory():
    """Proves wrap_nv12 returns a tensor over the same GPU memory, with the NV12 shape, not a copy."""
    source = torch.zeros((1620, 1920), dtype=torch.uint8, device="cuda:0")
    view = wrap_nv12(source.data_ptr(), 1920, 1080)
    assert view.data_ptr() == source.data_ptr()
    assert view.shape == (1620, 1920)
    source[0, 0] = 77
    assert int(view[0, 0]) == 77


@pytest.mark.gpu
def test_input_matches_the_library_rgb_path(clip_path):
    """Proves converting a decoded frame with nv12_to_input matches the library reader's RGB input to within one
    level on average."""
    import PyNvVideoCodec as nvc

    from video_reader import create_reader

    reader = create_reader(clip_path, "nvdec", memory_input=True)
    reader.read(host_frame=False)
    reference, _ = reader.cuda_input(640)
    reader.close()

    demuxer = nvc.CreateDemuxer(str(clip_path))
    decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=True,
                                outputColorType=nvc.OutputColorType.NATIVE)
    pointer = None
    while pointer is None:
        for _ in range(decoder.GetNumDecodedFrame(demuxer.Demux())):
            pointer = pointer or decoder.GetLockedFrame()
    decoder.SyncOnCUStream()

    output = torch.empty((1, 3, 640, 640), dtype=torch.float32, device="cuda:0")
    nv12_to_input(wrap_nv12(pointer, 1920, 1080), Letterbox(1920, 1080, 640), 1920, 1080, output)
    assert float((output - reference).abs().mean()) * 255 < 1.0
