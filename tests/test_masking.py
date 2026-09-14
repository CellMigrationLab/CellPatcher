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


# ---------------------------------------------------------------------------
# Additional masking edge cases
# ---------------------------------------------------------------------------

from cellpatcher.masking import apply_label_masks_to_patches, _label_to_2d


def test_label_to_2d_rejects_extra_non_singleton_dims():
    """label_patch with extra non-singleton dims after moving YX to (0,1) should raise."""
    label_arr = np.ones((4, 4, 3), dtype=np.int32)  # YXC with C=3 (non-singleton)
    with pytest.raises(ValueError, match="extra shape"):
        _label_to_2d(label_arr, label_axes="YXC")


def test_apply_label_mask_to_patch_rejects_1d_image():
    img = np.array([1, 2, 3])
    labels = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="at least 2D"):
        apply_label_mask_to_patch(img, labels, label_id=1)


def test_apply_label_masks_to_patches_length_mismatch_raises():
    image_patches = [np.ones((3, 3))]
    label_patches = [np.ones((3, 3)), np.ones((3, 3))]
    from cellpatcher.types import PatchRegion
    regions = [PatchRegion(patch_id="r", y0=0, y1=3, x0=0, x1=3, label_id=1)]
    with pytest.raises(ValueError):
        apply_label_masks_to_patches(image_patches, label_patches, regions)


def test_apply_label_masks_to_patches_none_label_id_raises():
    image_patches = [np.ones((3, 3))]
    label_patches = [np.ones((3, 3))]
    from cellpatcher.types import PatchRegion
    regions = [PatchRegion(patch_id="r", y0=0, y1=3, x0=0, x1=3)]  # label_id=None
    with pytest.raises(ValueError, match="label_id"):
        apply_label_masks_to_patches(image_patches, label_patches, regions)


def test_apply_label_masks_to_patches_applies_mask_correctly():
    """apply_label_masks_to_patches zeros out non-label pixels for each patch."""
    image = np.ones((4, 4), dtype=np.float32) * 9
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 7
    from cellpatcher.types import PatchRegion
    regions = [PatchRegion(patch_id="r", y0=0, y1=4, x0=0, x1=4, label_id=7)]
    result = apply_label_masks_to_patches([image], [labels], regions)
    assert result[0][0, 0] == 0
    assert result[0][1, 1] == 9


# ---------------------------------------------------------------------------
# Masking singleton-extra-dim path (covered for completeness)
# ---------------------------------------------------------------------------

def test_label_to_2d_accepts_singleton_extra_dim():
    """label_patch with one singleton non-spatial dim should be reshaped to 2D."""
    label_arr = np.ones((4, 4, 1), dtype=np.int32)  # YXC with C=1 (singleton)
    result = _label_to_2d(label_arr, label_axes="YXC")
    assert result.shape == (4, 4)
