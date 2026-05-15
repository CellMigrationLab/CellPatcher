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
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1, 1] = 1
    labels[4, 4] = 1

    records = scan_cell_objects(
        image_shape=(6, 6), labels=labels, config=CellConfig(exclude_border=False, binary_mask_policy="connected_components")
    )
    assert len(records) == 2


def test_non_contiguous_labels_preserved():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 5
    labels[5:7, 0:2] = 42

    records = scan_cell_objects(image_shape=(7, 7), labels=labels, config=CellConfig(exclude_border=False))
    assert [r.label_id for r in records] == [1, 5, 42]


def test_float_binary_labels_accepted():
    labels = np.zeros((5, 5), dtype=np.float32)
    labels[2:4, 2:4] = 1.0

    records = scan_cell_objects(image_shape=(5, 5), labels=labels, config=CellConfig(exclude_border=False))
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
