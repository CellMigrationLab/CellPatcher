import numpy as np
import pytest

from cellpatcher.resize import normalize_output_size, resize_patch, resize_patches


def test_normalize_output_size_none():
    assert normalize_output_size(None) is None


def test_normalize_output_size_int():
    assert normalize_output_size(224) == (224, 224)


def test_normalize_output_size_tuple():
    assert normalize_output_size((32, 64)) == (32, 64)


def test_normalize_output_size_invalid_raises():
    with pytest.raises(ValueError):
        normalize_output_size((0, 4))
    with pytest.raises(ValueError):
        normalize_output_size((1, 2, 3))


def test_resize_patch_none_output_size_unchanged_shape():
    arr = np.zeros((5, 6))
    out = resize_patch(arr, axes="YX", output_size=None)
    assert out.shape == (5, 6)


def test_resize_patch_resizes_yx():
    arr = np.arange(25).reshape(5, 5)
    out = resize_patch(arr, axes="YX", output_size=(10, 8), order=0)
    assert out.shape == (10, 8)


def test_resize_patch_resizes_cyx_preserves_c():
    arr = np.arange(2 * 5 * 5).reshape(2, 5, 5)
    out = resize_patch(arr, axes="CYX", output_size=8, order=0)
    assert out.shape == (2, 8, 8)


def test_resize_patch_resizes_yxc_preserves_c_last():
    arr = np.arange(5 * 5 * 3).reshape(5, 5, 3)
    out = resize_patch(arr, axes="YXC", output_size=7, order=0)
    assert out.shape == (7, 7, 3)


def test_resize_patch_resizes_zyx_preserves_z():
    arr = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    out = resize_patch(arr, axes="ZYX", output_size=(9, 7), order=0)
    assert out.shape == (4, 9, 7)


def test_resize_patches_returns_list():
    arrs = [np.zeros((4, 4)), np.zeros((4, 4))]
    out = resize_patches(arrs, axes="YX", output_size=2, order=0)
    assert isinstance(out, list)
    assert [a.shape for a in out] == [(2, 2), (2, 2)]


def test_resize_patch_invalid_axes_raises():
    arr = np.zeros((5, 5))
    with pytest.raises(ValueError):
        resize_patch(arr, axes="TC", output_size=3)
