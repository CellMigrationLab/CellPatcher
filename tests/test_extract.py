import numpy as np
import pytest

from cellpatcher.extract import extract_patch
from cellpatcher.types import PatchRegion


def test_extract_patch_yx_in_bounds():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="p", y0=1, y1=4, x0=2, x1=5)
    out = extract_patch(image, region, axes="YX")
    assert np.array_equal(out, image[1:4, 2:5])


def test_extract_patch_cyx_preserves_channels():
    image = np.arange(2 * 5 * 5).reshape(2, 5, 5)
    region = PatchRegion(patch_id="p", y0=1, y1=4, x0=1, x1=4)
    out = extract_patch(image, region, axes="CYX")
    assert out.shape == (2, 3, 3)
    assert np.array_equal(out, image[:, 1:4, 1:4])


def test_extract_patch_yxc_preserves_channels_last():
    image = np.arange(5 * 5 * 3).reshape(5, 5, 3)
    region = PatchRegion(patch_id="p", y0=0, y1=3, x0=1, x1=5)
    out = extract_patch(image, region, axes="YXC")
    assert out.shape == (3, 4, 3)
    assert np.array_equal(out, image[0:3, 1:5, :])


def test_extract_patch_zyx_preserves_z_axis():
    image = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    region = PatchRegion(patch_id="p", y0=1, y1=4, x0=2, x1=6)
    out = extract_patch(image, region, axes="ZYX")
    assert out.shape == (4, 3, 4)
    assert np.array_equal(out, image[:, 1:4, 2:6])


def test_extract_patch_tczyx_preserves_non_spatial_axes_and_constant_padding():
    image = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    region = PatchRegion(patch_id="p", y0=-1, y1=3, x0=4, x1=8)
    out = extract_patch(image, region, axes="TCZYX", pad_mode="constant", constant_values=-9)
    assert out.shape == (2, 3, 4, 4, 4)
    assert np.all(out[..., 0, :] == -9)


def test_extract_patch_clip_out_of_bounds():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="p", y0=-2, y1=3, x0=3, x1=7)
    out = extract_patch(image, region, axes="YX", pad_mode="clip")
    assert np.array_equal(out, image[0:3, 3:5])


def test_extract_patch_constant_padding_preserves_requested_shape():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="p", y0=-1, y1=3, x0=3, x1=7)
    out = extract_patch(image, region, axes="YX", pad_mode="constant", constant_values=-1)
    assert out.shape == (4, 4)
    assert out[0, 0] == -1


def test_extract_patch_clip_fully_out_of_bounds_raises():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="p", y0=6, y1=8, x0=6, x1=8)
    with pytest.raises(ValueError):
        extract_patch(image, region, axes="YX", pad_mode="clip")


def test_extract_patch_constant_fully_out_of_bounds_yx():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="oob", y0=10, y1=12, x0=10, x1=12)
    patch = extract_patch(image, region, axes="YX", pad_mode="constant", constant_values=7)
    assert patch.shape == (2, 2)
    assert np.all(patch == 7)


def test_extract_patch_constant_fully_out_of_bounds_cyx():
    image = np.arange(3 * 5 * 5).reshape(3, 5, 5)
    region = PatchRegion(patch_id="oob", y0=10, y1=12, x0=10, x1=12)
    patch = extract_patch(image, region, axes="CYX", pad_mode="constant", constant_values=0)
    assert patch.shape == (3, 2, 2)
    assert np.all(patch == 0)


def test_extract_patch_constant_fully_out_of_bounds_yxc():
    image = np.zeros((5, 5, 3), dtype=np.float32)
    region = PatchRegion(patch_id="oob", y0=10, y1=12, x0=10, x1=12)
    patch = extract_patch(image, region, axes="YXC", pad_mode="constant", constant_values=9)
    assert patch.shape == (2, 2, 3)
    assert np.all(patch == 9)


def test_extract_patch_invalid_axes_raises():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="p", y0=1, y1=2, x0=1, x1=2)
    with pytest.raises(ValueError):
        extract_patch(image, region, axes="TC")


def test_extract_patch_invalid_pad_mode_raises():
    image = np.arange(25).reshape(5, 5)
    region = PatchRegion(patch_id="p", y0=1, y1=2, x0=1, x1=2)
    with pytest.raises(ValueError):
        extract_patch(image, region, axes="YX", pad_mode="mirror")
