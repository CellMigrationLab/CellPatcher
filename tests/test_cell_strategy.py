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
    # The strategy gives the last level of patch_id; run_patching prefixes the
    # position, object set and planes (cellpatcher.ids).
    assert [p.patch_id for p in patches] == ["label1", "label2"]
    assert patches[0].area_px == 4
    assert abs(patches[0].center_y - 2.0) < 1e-6


def test_binary_connected_components_returns_two():
    labels = np.zeros((6, 6), dtype=bool)
    labels[1:3, 1:3] = True
    labels[4:5, 4:6] = True
    s = CellSegmentationStrategy(CellConfig(binary_mask_policy="connected_components", exclude_border=False))
    with pytest.warns(UserWarning, match="connected components"):
        patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert len(patches) == 2


def test_bool_mask_is_refused_by_default():
    labels = np.zeros((6, 6), dtype=bool)
    labels[1:3, 1:3] = True
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError, match="binary_mask_policy='connected_components'"):
        s.generate_regions(image_shape=(6, 6), labels=labels)


@pytest.mark.parametrize("policy", ["single_object", "connected_components"])
def test_integer_mask_holding_one_value_is_kept_as_given(policy):
    # Decision D47: an integer mask is instance labels whatever the policy;
    # 0/255 is one object labelled 255, two separate blobs of 1 one object.
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1:3, 1:3] = 255
    labels[4:5, 4:6] = 255
    s = CellSegmentationStrategy(CellConfig(binary_mask_policy=policy, exclude_border=False))
    patches = s.generate_regions(image_shape=(6, 6), labels=labels)
    assert [p.label_id for p in patches] == [255]


def test_default_policy_is_single_object():
    assert CellConfig().binary_mask_policy == "single_object"


def test_labels_above_int32_max_are_refused_not_wrapped():
    labels = np.zeros((4, 4), dtype=np.uint32)
    labels[1:3, 1:3] = 2**31
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError, match="int32 maximum"):
        s.generate_regions(image_shape=(4, 4), labels=labels)


def test_largest_int32_label_is_kept():
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[1:3, 1:3] = 2**31 - 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    assert [p.label_id for p in s.generate_regions(image_shape=(4, 4), labels=labels)] == [2**31 - 1]


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


def test_fixed_patch_size_near_border_can_shrink_due_to_clipping():
    labels = np.zeros((10, 10), dtype=int)
    labels[0:2, 0:2] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=6))
    patches = s.generate_regions(image_shape=(10, 10), labels=labels)
    assert len(patches) == 1
    patch = patches[0]
    assert patch.touches_border is True
    assert patch.height < 6 or patch.width < 6


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


def test_fixed_patch_size_larger_than_image_is_clipped_to_image_bounds():
    labels = np.zeros((10, 10), dtype=int)
    labels[4:6, 4:6] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=20))
    patch = s.generate_regions(image_shape=(10, 10), labels=labels)[0]
    assert (patch.y0, patch.y1, patch.x0, patch.x1) == (0, 10, 0, 10)



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
    with pytest.warns(UserWarning, match="connected components"):
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


# ---------------------------------------------------------------------------
# centroid center_on option
# ---------------------------------------------------------------------------

def test_centroid_center_on_uses_mean_of_pixel_coordinates():
    """center_on='centroid' should place the patch at the mean of foreground pixels."""
    labels = np.zeros((10, 10), dtype=int)
    labels[2:5, 3:7] = 1  # ys in [2,3,4], xs in [3,4,5,6]
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=4, center_on="centroid"))
    patch = s.generate_regions(image_shape=(10, 10), labels=labels)[0]
    ys, xs = np.where(labels == 1)
    assert abs(patch.center_y - float(np.mean(ys))) < 1e-6
    assert abs(patch.center_x - float(np.mean(xs))) < 1e-6


def test_centroid_differs_from_bbox_center_for_asymmetric_object():
    """For a non-symmetric object, centroid and bbox_center should differ."""
    labels = np.zeros((10, 10), dtype=int)
    # Place pixels at corners of a 3x6 region but only partially filled
    labels[2, 3] = 1
    labels[2, 8] = 1
    labels[4, 3] = 1
    s_centroid = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=3, center_on="centroid"))
    s_bbox = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=3, center_on="bbox_center"))
    p_centroid = s_centroid.generate_regions(image_shape=(10, 10), labels=labels)[0]
    p_bbox = s_bbox.generate_regions(image_shape=(10, 10), labels=labels)[0]
    assert (p_centroid.center_y, p_centroid.center_x) != (p_bbox.center_y, p_bbox.center_x)


# ---------------------------------------------------------------------------
# max_area filtering
# ---------------------------------------------------------------------------

def test_max_area_excludes_large_cells():
    labels = np.zeros((10, 10), dtype=int)
    labels[1:2, 1:2] = 1   # area 1
    labels[3:8, 3:8] = 2   # area 25
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, max_area=10))
    patches = s.generate_regions(image_shape=(10, 10), labels=labels)
    assert len(patches) == 1
    assert patches[0].label_id == 1


# ---------------------------------------------------------------------------
# 1D labels should raise
# ---------------------------------------------------------------------------

def test_1d_labels_raises():
    import numpy as np
    labels = np.array([1, 2, 3])
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(5, 5), labels=labels)


# ---------------------------------------------------------------------------
# Further cell strategy edge cases
# ---------------------------------------------------------------------------

def test_cell_strategy_patch_size_zero_raises():
    """patch_size=0 passed directly to strategy should raise (not caught by CellConfig)."""
    labels = np.zeros((5, 5), dtype=int)
    labels[2, 2] = 1
    s = CellSegmentationStrategy(CellConfig(exclude_border=False, patch_size=0))
    with pytest.raises(ValueError, match="patch_size"):
        s.generate_regions(image_shape=(5, 5), labels=labels)


def test_cell_strategy_empty_labels_raises():
    """Empty (zero-size) 2D labels array should raise."""
    labels = np.zeros((0, 5), dtype=int)
    s = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(0, 5), labels=labels)


def test_cell_strategy_mask_axes_length_mismatch_raises():
    """mask_axes length must match labels.ndim for multidimensional labels."""
    labels = np.zeros((2, 2, 5, 5), dtype=int)  # ndim=4
    s = CellSegmentationStrategy(CellConfig(mask_channel=0), mask_axes="CYX")  # len=3
    with pytest.raises(ValueError, match="mask_axes"):
        s.generate_regions(image_shape=(5, 5), labels=labels)


def test_cell_strategy_4d_labels_produces_non_2d_selected_raises():
    """4D labels with TCYX axes → channel selection returns 3D → raises."""
    labels = np.zeros((2, 2, 5, 5), dtype=int)  # TCYX: T=2, C=2, Y=5, X=5
    s = CellSegmentationStrategy(CellConfig(mask_channel=0), mask_axes="TCYX")
    with pytest.raises(ValueError, match="3D"):
        s.generate_regions(image_shape=(5, 5), labels=labels)
