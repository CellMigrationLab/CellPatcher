import numpy as np
import pytest

from cellpatcher.config import CellConfig
from cellpatcher.strategies.cell import CellSegmentationStrategy


def test_pre_labeled_mask_returns_two_patches_sorted_and_ids():
    labels = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
    ])
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    patches = s.generate_regions(image_shape=(5, 5), labels=labels, image_id="img")
    assert len(patches) == 2
    assert [p.label_id for p in patches] == [1, 2]
    assert [p.patch_id for p in patches] == ["img_cell_00001", "img_cell_00002"]
    assert patches[0].area_px == 4
    assert abs(patches[0].center_y - 2.0) < 1e-6


def test_binary_connected_components_returns_two():
    labels = np.zeros((6, 6), dtype=int)
    labels[1:3, 1:3] = 1
    labels[4:5, 4:6] = 1
    s = CellSegmentationStrategy(CellConfig(binary_mask_policy="connected_components", exclude_border=False))
    patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert len(patches) == 2


def test_binary_single_object_returns_one():
    labels = np.zeros((6, 6), dtype=int)
    labels[1:3, 1:3] = 1
    labels[4:5, 4:6] = 1
    s = CellSegmentationStrategy(CellConfig(binary_mask_policy="single_object", exclude_border=False))
    patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert len(patches) == 1


def test_border_object_excluded_when_exclude_border_true():
    labels = np.zeros((5, 5), dtype=int)
    labels[0:2, 0:2] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=True))
    assert s.generate_regions(image_shape=(5, 5), labels=labels) == []


def test_border_object_included_and_clipped_when_exclude_border_false():
    labels = np.zeros((5, 5), dtype=int)
    labels[0:2, 0:2] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=4))
    patches = s.generate_regions(image_shape=(5, 5), labels=labels)
    assert len(patches) == 1
    assert patches[0].touches_border is True
    assert patches[0].y0 == 0


def test_patch_size_auto_uses_object_bbox_size():
    labels = np.zeros((8, 8), dtype=int)
    labels[2:5, 3:7] = 1  # bbox 3x4 => auto patch 4
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size="auto"))
    patch = s.generate_regions(image_shape=(8, 8), labels=labels)[0]
    assert patch.height == 4 and patch.width == 4


def test_fixed_patch_size_produces_square_when_possible():
    labels = np.zeros((10, 10), dtype=int)
    labels[4:6, 4:6] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=6))
    patch = s.generate_regions(image_shape=(10, 10), labels=labels)[0]
    assert patch.height == 6 and patch.width == 6



def test_default_center_on_matches_notebook_bbox_center_behavior():
    labels = np.zeros((10, 10), dtype=int)
    labels[2:5, 3:7] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=4))
    patch = s.generate_regions(image_shape=(10, 10), labels=labels)[0]
    # bbox y=[2..4], x=[3..6] -> notebook center uses ((min+max+1)/2) = (3.5, 5.0)
    assert patch.center_y == 3.5
    assert patch.center_x == 5.0

def test_invalid_patch_size_raises():
    labels = np.zeros((5, 5), dtype=int)
    labels[2:3, 2:3] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size="bad"))
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(5, 5), labels=labels)


def test_min_area_and_max_area_filters():
    labels = np.zeros((8, 8), dtype=int)
    labels[1:2, 1:2] = 1  # area 1
    labels[3:7, 3:7] = 2  # area 16
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, min_area=2, max_area=10))
    patches = s.generate_regions(image_shape=(8, 8), labels=labels)
    assert len(patches) == 0


def test_missing_labels_raises():
    s = CellSegmentationStrategy()
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(5, 5), labels=None)


def test_labels_shape_mismatch_raises():
    s = CellSegmentationStrategy()
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(5, 5), labels=np.zeros((4, 5)))


def test_3d_labels_without_mask_channel_raises():
    s = CellSegmentationStrategy(CellConfig(mask_channel=None), mask_axes="CYX")
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(5, 5), labels=np.zeros((2, 5, 5)))


def test_3d_labels_with_valid_mask_channel_works():
    labels = np.zeros((2, 5, 5), dtype=int)
    labels[1, 2:4, 2:4] = 1
    s = CellSegmentationStrategy(CellConfig(mask_channel=1, exclude_border=False), mask_axes="CYX")
    patches = s.generate_regions(image_shape=(5, 5), labels=labels)
    assert len(patches) == 1


def test_invalid_mask_channel_raises():
    labels = np.zeros((2, 5, 5), dtype=int)
    s = CellSegmentationStrategy(CellConfig(mask_channel=3), mask_axes="CYX")
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(5, 5), labels=labels)


# ---------------------------------------------------------------------------
# mask_axes-aware channel selection tests
# ---------------------------------------------------------------------------

def test_cyx_mask_selects_correct_channel():
    """CYX layout: channel axis is 0. mask_channel=1 picks labels[1]."""
    labels = np.zeros((3, 6, 6), dtype=int)
    labels[1, 2:4, 2:4] = 7  # only channel 1 has an object
    s = CellSegmentationStrategy(CellConfig(mask_channel=1, exclude_border=False), mask_axes="CYX")
    patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert len(patches) == 1
    assert patches[0].label_id == 7


def test_yxc_mask_selects_correct_channel():
    """YXC layout: channel axis is 2. mask_channel=2 picks labels[:,:,2].
    Under the old channel-first assumption this would wrongly look at axis 0.
    """
    labels = np.zeros((6, 6, 3), dtype=int)
    labels[2:4, 2:4, 2] = 5  # only channel index 2 (last axis) has an object
    s = CellSegmentationStrategy(CellConfig(mask_channel=2, exclude_border=False), mask_axes="YXC")
    patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert len(patches) == 1
    assert patches[0].label_id == 5


def test_multidim_labels_without_c_axis_raises():
    """mask_axes='YXZ' has no C axis — must raise ValueError."""
    labels = np.zeros((6, 6, 2), dtype=int)
    s = CellSegmentationStrategy(CellConfig(mask_channel=0), mask_axes="YXZ")
    with pytest.raises(ValueError, match="C"):
        s.generate_regions(image_shape=(6, 6), labels=labels)


def test_multidim_labels_with_mask_channel_none_raises():
    """3D labels with mask_axes='CYX' but mask_channel=None must raise ValueError."""
    labels = np.zeros((2, 6, 6), dtype=int)
    s = CellSegmentationStrategy(CellConfig(mask_channel=None), mask_axes="CYX")
    with pytest.raises(ValueError, match="mask_channel"):
        s.generate_regions(image_shape=(6, 6), labels=labels)


def test_out_of_range_mask_channel_raises():
    """mask_channel=5 for a C axis of size 2 must raise ValueError."""
    labels = np.zeros((2, 6, 6), dtype=int)
    s = CellSegmentationStrategy(CellConfig(mask_channel=5), mask_axes="CYX")
    with pytest.raises(ValueError, match="out of bounds"):
        s.generate_regions(image_shape=(6, 6), labels=labels)


# ---------------------------------------------------------------------------
# Float and negative label validation tests
# ---------------------------------------------------------------------------

def test_float_binary_mask_works():
    """Float mask with only 0.0 and 1.0 should be treated as binary."""
    labels = np.zeros((6, 6), dtype=float)
    labels[2:4, 2:4] = 1.0
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, binary_mask_policy="connected_components"))
    patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert len(patches) == 1


def test_non_binary_float_labels_raise():
    """Float mask with values like 1.7, 2.3 must raise ValueError."""
    labels = np.array([[0.0, 1.7], [2.3, 0.0]], dtype=float)
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError, match="Non-binary float"):
        s.generate_regions(image_shape=(2, 2), labels=labels)


def test_negative_integer_labels_raise():
    """Labels containing negative values must raise ValueError."""
    labels = np.array([[0, -1], [1, 0]], dtype=int)
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError, match="non-negative"):
        s.generate_regions(image_shape=(2, 2), labels=labels)


def test_non_contiguous_label_ids_work():
    """Labeled mask with IDs 1, 5, 42 should produce patches for all three."""
    labels = np.zeros((10, 10), dtype=int)
    labels[0:2, 0:2] = 1
    labels[4:6, 4:6] = 5
    labels[7:9, 7:9] = 42
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    patches = s.generate_regions(image_shape=(10, 10), labels=labels)
    assert [p.label_id for p in patches] == [1, 5, 42]
