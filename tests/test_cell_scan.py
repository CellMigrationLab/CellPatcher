import numpy as np
import pytest

from cellpatcher.cell_scan import scan_cell_objects, scan_cell_patch_size
from cellpatcher.config import CellConfig


def test_scan_cell_objects_labeled_mask():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:3, 1:4] = 1
    labels[4:7, 5:7] = 2

    records = scan_cell_objects(image_shape=(8, 8), labels=labels, config=CellConfig(exclude_border=False))

    assert [r.label_id for r in records] == [1, 2]
    assert records[0].area_px == 6
    assert records[0].bbox_height == 2
    assert records[0].bbox_width == 3
    assert records[0].square_size == 3
    assert records[0].included is True


def test_border_exclusion():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[0:2, 1:3] = 1

    records = scan_cell_objects(image_shape=(5, 5), labels=labels, config=CellConfig(exclude_border=True))
    assert len(records) == 1
    assert records[0].included is False
    assert records[0].reason == "border"


def test_area_filtering():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2, 2] = 1

    records = scan_cell_objects(image_shape=(6, 6), labels=labels, config=CellConfig(exclude_border=False, min_area=2))
    assert records[0].included is False
    assert records[0].reason == "area"


def test_binary_connected_components():
    labels = np.zeros((6, 6), dtype=bool)
    labels[1, 1] = True
    labels[4, 4] = True

    with pytest.warns(UserWarning, match="connected components"):
        records = scan_cell_objects(
            image_shape=(6, 6),
            labels=labels,
            config=CellConfig(exclude_border=False, binary_mask_policy="connected_components"),
        )
    assert len(records) == 2


def test_integer_one_valued_mask_is_one_object_as_in_patching():
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1, 1] = 1
    labels[4, 4] = 1

    records = scan_cell_objects(image_shape=(6, 6), labels=labels, config=CellConfig(exclude_border=False))
    assert [r.label_id for r in records] == [1]


def test_non_contiguous_labels_preserved():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 5
    labels[5:7, 0:2] = 42

    records = scan_cell_objects(image_shape=(7, 7), labels=labels, config=CellConfig(exclude_border=False))
    assert [r.label_id for r in records] == [1, 5, 42]


def test_float_binary_labels_are_refused():
    # Owner decision D-b: float masks are refused even of 0.0/1.0, whatever binary_mask_policy.
    labels = np.zeros((5, 5), dtype=np.float32)
    labels[2:4, 2:4] = 1.0
    for policy in ("single_object", "connected_components"):
        with pytest.raises(ValueError, match="save the mask as integers"):
            scan_cell_objects(image_shape=(5, 5), labels=labels,
                              config=CellConfig(exclude_border=False, binary_mask_policy=policy))


def test_boolean_labels_need_the_connected_components_opt_in():
    labels = np.zeros((5, 5), dtype=bool)
    labels[2:4, 2:4] = True

    with pytest.raises(ValueError, match="binary"):
        scan_cell_objects(image_shape=(5, 5), labels=labels, config=CellConfig(exclude_border=False))
    with pytest.warns(UserWarning, match="connected components"):
        records = scan_cell_objects(
            image_shape=(5, 5),
            labels=labels,
            config=CellConfig(exclude_border=False, binary_mask_policy="connected_components"),
        )
    assert len(records) == 1
    assert records[0].included is True


def test_non_binary_float_labels_rejected():
    labels = np.array([[0.0, 1.7], [2.3, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        scan_cell_objects(image_shape=(2, 2), labels=labels)


def test_negative_labels_rejected():
    labels = np.array([[0, -1], [0, 1]], dtype=np.int32)
    with pytest.raises(ValueError):
        scan_cell_objects(image_shape=(2, 2), labels=labels)


def test_scan_cell_patch_size_multiple_images():
    labels1 = np.zeros((8, 8), dtype=np.int32)
    labels1[1:3, 1:4] = 1  # square 3
    labels1[4:7, 5:7] = 2  # square 3

    labels2 = np.zeros((8, 8), dtype=np.int32)
    labels2[2:6, 2:7] = 3  # square 5

    summary = scan_cell_patch_size(
        image_shapes=[(8, 8), (8, 8)],
        labels_list=[labels1, labels2],
        config=CellConfig(exclude_border=False),
    )

    assert summary.n_images == 2
    assert summary.n_objects_total == 3
    assert summary.n_included == 3
    assert summary.n_skipped_border == 0
    assert summary.n_skipped_area == 0
    assert summary.max_bbox_height == 4
    assert summary.max_bbox_width == 5
    assert summary.max_square_size == 5
    assert summary.mean_square_size == pytest.approx(11 / 3)
    assert summary.median_square_size == pytest.approx(3.0)
    assert summary.percentile_90_square_size == pytest.approx(4.6)
    assert summary.percentile_95_square_size == pytest.approx(4.8)


def test_no_included_objects():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[0, 0] = 1
    summary = scan_cell_patch_size(
        image_shapes=(5, 5),
        labels_list=labels,
        config=CellConfig(exclude_border=True),
    )
    assert summary.n_included == 0
    assert summary.max_square_size == 0
    assert summary.mean_square_size == 0.0
    assert summary.recommended_size("max") == 0
    assert summary.recommended_size("median") == 0


def test_recommended_size_methods_and_unknown():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:3] = 1  # 2
    labels[4:9, 4:8] = 2  # 5
    summary = scan_cell_patch_size(
        image_shapes=(10, 10),
        labels_list=labels,
        config=CellConfig(exclude_border=False),
    )
    assert summary.recommended_size("max") == 5
    assert summary.recommended_size("median") == 4
    assert summary.recommended_size("mean") == 4
    assert summary.recommended_size("p90") == 5
    assert summary.recommended_size("p95") == 5
    with pytest.raises(ValueError):
        summary.recommended_size("unknown")


def test_yxc_image_axes():
    image_shape = (6, 7, 3)
    labels = np.zeros((6, 7), dtype=np.int32)
    labels[1:4, 2:5] = 1
    records = scan_cell_objects(
        image_shape=image_shape,
        labels=labels,
        image_axes="YXC",
        mask_axes="YX",
        config=CellConfig(exclude_border=False),
    )
    assert len(records) == 1
    assert records[0].bbox_height == 3
    assert records[0].bbox_width == 3


def test_mismatched_lengths_raises():
    labels = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        scan_cell_patch_size(image_shapes=[(5, 5), (5, 5)], labels_list=[labels])


# ---------------------------------------------------------------------------
# Additional cell scan edge cases
# ---------------------------------------------------------------------------

from cellpatcher.cell_scan import _normalize_shape_inputs


def test_normalize_shape_inputs_none_labels_raises():
    with pytest.raises(ValueError, match="labels_list"):
        _normalize_shape_inputs(image_shapes=(5, 5), labels_list=None)


def test_normalize_shape_inputs_bad_image_shapes_type_raises():
    labels = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="image_shapes"):
        _normalize_shape_inputs(image_shapes=5, labels_list=[labels])


def test_scan_cell_objects_shape_mismatch_raises():
    """labels shape does not match image spatial shape → ValueError."""
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[1:4, 1:4] = 1
    with pytest.raises(ValueError):
        scan_cell_objects(image_shape=(5, 5), labels=labels, config=CellConfig(exclude_border=False))


def test_scan_cell_objects_max_area_filter():
    """Objects with area > max_area should be excluded."""
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:2, 1:2] = 1   # area 1
    labels[3:8, 3:8] = 2   # area 25
    records = scan_cell_objects(image_shape=(10, 10), labels=labels, config=CellConfig(exclude_border=False, max_area=10))
    included = [r for r in records if r.included]
    assert [r.label_id for r in included] == [1]
    excluded = [r for r in records if not r.included]
    assert excluded[0].reason == "area"


def test_scan_cell_patch_size_with_area_skipped():
    """Objects skipped by area filter are counted in n_skipped_area."""
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:2, 1:2] = 1   # area 1 → below min_area=5
    labels[3:7, 3:7] = 2   # area 16 → included
    summary = scan_cell_patch_size(
        image_shapes=(10, 10),
        labels_list=labels,
        config=CellConfig(exclude_border=False, min_area=5),
    )
    assert summary.n_skipped_area == 1
    assert summary.n_included == 1
    assert summary.min_square_size > 0


# ---------------------------------------------------------------------------
# Additional cell scan edge cases
# ---------------------------------------------------------------------------

def test_scan_cell_objects_zero_size_spatial_dim_raises():
    """image_shape with zero spatial dimension should raise."""
    labels = np.zeros((0, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="[Ss]patial"):
        scan_cell_objects(image_shape=(0, 5), labels=labels, config=CellConfig(exclude_border=False))
