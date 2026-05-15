import numpy as np
import pytest

from cellpatcher.masking import apply_label_mask_to_patch


def test_apply_label_mask_yx():
    image = np.arange(16).reshape(4, 4)
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 7
    out = apply_label_mask_to_patch(image, labels, label_id=7, image_axes="YX", label_axes="YX", fill_value=0)
    assert np.all(out[labels != 7] == 0)
    assert np.array_equal(out[labels == 7], image[labels == 7])


def test_apply_label_mask_cyx_broadcasts_channels():
    image = np.ones((3, 4, 4), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[2, 2] = 1
    out = apply_label_mask_to_patch(image, labels, label_id=1, image_axes="CYX", label_axes="YX", fill_value=0)
    assert np.all(out[:, 2, 2] == 1)
    assert np.sum(out) == pytest.approx(3.0)


def test_apply_label_mask_yxc_broadcasts_channels():
    image = np.ones((4, 4, 2), dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 5
    out = apply_label_mask_to_patch(image, labels, label_id=5, image_axes="YXC", label_axes="YX", fill_value=0)
    assert np.all(out[0, 0, :] == 1)
    assert np.sum(out) == pytest.approx(2.0)


def test_apply_label_mask_fill_value_respected():
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1, 1] = 1
    out = apply_label_mask_to_patch(image, labels, label_id=1, fill_value=-3.5)
    assert out[0, 0] == pytest.approx(-3.5)


def test_apply_label_mask_label_id_none_raises():
    with pytest.raises(ValueError):
        apply_label_mask_to_patch(np.ones((2, 2)), np.ones((2, 2)), label_id=None)


def test_apply_label_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_label_mask_to_patch(np.ones((3, 3)), np.ones((2, 2)), label_id=1)
