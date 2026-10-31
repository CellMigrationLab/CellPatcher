"""object_sides, border_labels and object_boxes (D86 API C1) and scan_cell_objects on them.

The references below are the implementations the new functions replace: the
per-object loop of scan_cell_objects before D86, and MorphoFeature's
object_extent.object_sides/assembly.border_labels (copied byte-for-byte in
MorphoEmbed's provider_crop_scale). The outputs must be identical.
"""

import numpy as np
import pytest

from cellpatcher import (
    CellConfig,
    border_labels,
    object_boxes,
    object_sides,
    scan_cell_objects,
)


def _old_scan(labeled, config):
    """scan_cell_objects' loop before D86: one pass over the mask per object."""
    image_h, image_w = labeled.shape
    records = []
    for label_id in sorted(int(v) for v in np.unique(labeled) if int(v) != 0):
        ys, xs = np.where(labeled == label_id)
        area_px = int(ys.size)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        bbox_height, bbox_width = y_max - y_min + 1, x_max - x_min + 1
        touches_border = y_min == 0 or x_min == 0 or y_max == image_h - 1 or x_max == image_w - 1
        included, reason = True, None
        if config.exclude_border and touches_border:
            included, reason = False, "border"
        elif area_px < config.min_area or (config.max_area is not None and area_px > config.max_area):
            included, reason = False, "area"
        records.append((label_id, area_px, bbox_height, bbox_width, max(bbox_height, bbox_width),
                        touches_border, included, reason))
    return records


def _reference_border_labels(mask):
    """MorphoFeature assembly.border_labels (06fb395)."""
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return frozenset(int(label) for label in np.unique(edges) if label != 0)


def _reference_object_sides(mask):
    """MorphoFeature object_extent.object_sides (06fb395)."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return {}
    labels = mask[rows, cols]
    order = np.argsort(labels, kind="stable")
    labels, rows, cols = labels[order], rows[order], cols[order]
    unique, starts = np.unique(labels, return_index=True)
    height = np.maximum.reduceat(rows, starts) - np.minimum.reduceat(rows, starts) + 1
    width = np.maximum.reduceat(cols, starts) - np.minimum.reduceat(cols, starts) + 1
    border = _reference_border_labels(mask)
    sides = np.maximum(height, width)
    return {int(label): int(side) for label, side in zip(unique, sides, strict=True) if int(label) not in border}


def _random_masks(count=40):
    """Random instance masks: scattered, touching, gapped and border objects, several dtypes."""
    rng = np.random.default_rng(86)
    for index in range(count):
        height, width = rng.integers(1, 40, size=2)
        dtype = (np.uint8, np.uint16, np.int32, np.uint32, np.int64)[index % 5]
        largest = 250 if dtype == np.uint8 else 70_000
        mask = np.zeros((height, width), dtype=dtype)
        for _ in range(rng.integers(0, 12)):
            y0, x0 = rng.integers(0, height), rng.integers(0, width)
            y1, x1 = y0 + rng.integers(1, 12), x0 + rng.integers(1, 12)
            mask[y0:y1, x0:x1] = rng.integers(1, largest)
        # Speckle: objects split in several pieces and single-pixel objects.
        speckle = rng.random(mask.shape) < 0.05
        mask[speckle] = rng.integers(1, 6, size=int(speckle.sum()))
        yield mask


@pytest.mark.parametrize("config", [
    CellConfig(),
    CellConfig(exclude_border=False, min_area=1),
    CellConfig(exclude_border=True, min_area=3, max_area=40),
])
def test_scan_cell_objects_is_unchanged(config):
    seen_border = 0
    for mask in _random_masks():
        new = [
            (r.label_id, r.area_px, r.bbox_height, r.bbox_width, r.square_size, r.touches_border, r.included, r.reason)
            for r in scan_cell_objects(image_shape=mask.shape, labels=mask, config=config)
        ]
        assert new == _old_scan(mask.astype(np.int32), config)
        assert all(type(value) in (int, bool, str, type(None)) for record in new for value in record)
        seen_border += sum(record[5] for record in new)
    assert seen_border > 0


def test_object_sides_and_border_labels_match_the_copies_in_other_bricks():
    for mask in _random_masks():
        assert border_labels(mask) == _reference_border_labels(mask)
        assert object_sides(mask) == _reference_object_sides(mask)


def test_object_boxes_of_a_known_mask():
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[0, 2:4] = 3          # on the top edge
    mask[2:5, 1:3] = 7        # interior, 3 x 2
    mask[2, 5] = 9            # one pixel
    mask[4, 5] = 9            # same object, second piece
    boxes = object_boxes(mask)
    assert boxes.labels.tolist() == [3, 7, 9]
    assert boxes.area.tolist() == [2, 6, 2]
    assert boxes.height.tolist() == [1, 3, 3]
    assert boxes.width.tolist() == [2, 2, 1]
    assert boxes.touches_border.tolist() == [True, False, False]
    assert object_sides(mask) == {7: 3, 9: 3}
    assert border_labels(mask) == frozenset({3})


def test_an_empty_mask_has_no_objects():
    assert object_sides(np.zeros((4, 4), dtype=np.uint8)) == {}
    assert border_labels(np.zeros((4, 4), dtype=np.uint8)) == frozenset()


@pytest.mark.parametrize("mask, match", [
    (np.zeros((2, 3, 3), dtype=np.uint8), "2D instance mask"),
    (np.zeros((0, 3), dtype=np.uint8), "empty"),
    (np.array([[0, 1.0]]), "integer"),
    (np.array([[0, -1]], dtype=np.int16), "negative|-1"),
    (np.array([[0, 3_000_000_000]], dtype=np.uint32), "2,147,483,647|2147483647"),
])
def test_masks_that_are_not_instance_labels_are_refused(mask, match):
    for function in (object_sides, border_labels, object_boxes):
        with pytest.raises(ValueError, match=match):
            function(mask)
