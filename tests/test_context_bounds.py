"""context_bounds: the execution window of a patch (MorphoBricks-Spec ADR 0004, EW4/EW5).

The first four tests mirror ADR 0004's reference cases (``tools/test_execution_frames.py``,
class ``WindowRecord``) at MorphoBricks-Spec ``main`` commit ``b7b74f4`` (the tool files last
changed in ``3a55b73``): an interior patch keeps its whole margin, the plane edge clips and
records, a zero margin is the patch, and every refusal. The Spec tool is not copied: CellPatcher
has its own implementation, checked here against the same numbers.
"""

import random

import numpy as np
import pytest

from cellpatcher import PatchBoundsError, WindowRecord, context_bounds


def test_an_interior_patch_keeps_its_whole_margin():
    record = context_bounds((256, 512, 256, 512), (67, 67), (1024, 1024))
    assert record.context_bounds == (189, 579, 189, 579)
    assert record.origin == (189, 189)
    assert record.margin_requested == (67, 67, 67, 67)
    assert record.margin_clipped == (67, 67, 67, 67)
    assert record.local_interior == (67, 323, 67, 323)
    assert record.plane_shape == (1024, 1024)


def test_the_plane_edge_clips_and_records():
    record = context_bounds((0, 256, 900, 1000), (67, 30), (300, 1000))
    assert record.context_bounds == (0, 300, 870, 1000)
    assert record.origin == (0, 870)
    assert record.margin_requested == (67, 67, 30, 30)
    assert record.margin_clipped == (0, 44, 30, 0)
    assert record.local_interior == (0, 256, 30, 130)


def test_zero_margin_is_the_patch():
    record = context_bounds((10, 20, 30, 40), (0, 0), (100, 100))
    assert record.context_bounds == (10, 20, 30, 40)
    assert record.origin == (10, 30)
    assert record.margin_clipped == (0, 0, 0, 0)
    assert record.local_interior == (0, 10, 0, 10)


@pytest.mark.parametrize("bounds, margin, shape, error", [
    # The reference refusals.
    ((0, 10, 0, 10), (-1, 0), (100, 100), ValueError),
    ((0, 10, 0, 10), (1.5, 0), (100, 100), TypeError),
    ((0, 10, 0, 10), (True, 0), (100, 100), TypeError),
    ((0, 0, 0, 10), (1, 1), (100, 100), ValueError),
    ((0, 10, 0, 110), (1, 1), (100, 100), ValueError),
    # The same rules on the other arguments and axes.
    ((0, 10, 0, 10), (0, -3), (100, 100), ValueError),
    ((0, 10, 0, 10), (1.0, 0), (100, 100), TypeError),
    ((0, 10, 0, 10), (np.bool_(True), 0), (100, 100), TypeError),
    ((0, 10, 0, 10), (1,), (100, 100), TypeError),
    ((0, 10, 0, 10), 1, (100, 100), TypeError),
    ((0.0, 10, 0, 10), (1, 1), (100, 100), TypeError),
    ((False, 10, 0, 10), (1, 1), (100, 100), TypeError),
    ((0, 10, 0), (1, 1), (100, 100), TypeError),
    (np.array([0, 10, 0, 10]), (1, 1), (100, 100), TypeError),
    ((0, 10, 0, 10), (1, 1), (100.0, 100), TypeError),
    ((0, 10, 0, 10), (1, 1), (100, 100, 3), TypeError),
    ((0, 10, 0, 10), (1, 1), (0, 100), ValueError),
    ((5, 3, 0, 10), (1, 1), (100, 100), ValueError),
    ((-1, 10, 0, 10), (1, 1), (100, 100), ValueError),
    ((0, 101, 0, 10), (1, 1), (100, 100), ValueError),
])
def test_refusals(bounds, margin, shape, error):
    with pytest.raises(error):
        context_bounds(bounds, margin, shape)


def test_value_refusals_are_patch_bounds_errors():
    with pytest.raises(PatchBoundsError, match="never negative"):
        context_bounds((0, 10, 0, 10), (-1, 0), (100, 100))
    with pytest.raises(PatchBoundsError, match="leave"):
        context_bounds((0, 10, 0, 110), (1, 1), (100, 100))


def test_numpy_integers_are_accepted_and_the_record_holds_plain_ints():
    record = context_bounds(
        (np.int64(2), np.uint16(6), 1, 4), [np.int32(3), 1], (np.int64(8), 5)
    )
    assert record == WindowRecord(
        context_bounds=(0, 8, 0, 5),
        origin=(0, 0),
        margin_requested=(3, 3, 1, 1),
        margin_clipped=(2, 2, 1, 1),
        local_interior=(2, 6, 1, 4),
        plane_shape=(8, 5),
    )
    values = [v for field in (
        record.context_bounds, record.origin, record.margin_requested,
        record.margin_clipped, record.local_interior, record.plane_shape,
    ) for v in field]
    assert all(type(v) is int for v in values)


def test_the_record_is_frozen():
    record = context_bounds((10, 20, 30, 40), (2, 2), (100, 100))
    with pytest.raises(AttributeError):
        record.origin = (0, 0)


def test_random_rectangles_and_margins():
    """The window contains the patch, lies in the plane, and converts back exactly."""
    rng = random.Random(20261114)
    for _ in range(2000):
        height, width = rng.randint(1, 300), rng.randint(1, 300)
        y0 = rng.randrange(height)
        y1 = rng.randint(y0 + 1, height)
        x0 = rng.randrange(width)
        x1 = rng.randint(x0 + 1, width)
        my, mx = rng.randint(0, 120), rng.randint(0, 120)
        record = context_bounds((y0, y1, x0, x1), (my, mx), (height, width))
        cy0, cy1, cx0, cx1 = record.context_bounds
        assert 0 <= cy0 <= y0 < y1 <= cy1 <= height
        assert 0 <= cx0 <= x0 < x1 <= cx1 <= width
        assert record.origin == (cy0, cx0)
        assert record.margin_requested == (my, my, mx, mx)
        assert all(
            0 <= kept <= asked
            for kept, asked in zip(record.margin_clipped, record.margin_requested, strict=True)
        )
        # The margin kept is exactly what lies between the patch and the window, and a side is
        # clipped only where the plane edge cuts it.
        assert record.margin_clipped == (y0 - cy0, cy1 - y1, x0 - cx0, cx1 - x1)
        assert (cy0, cy1, cx0, cx1) == (
            max(0, y0 - my), min(height, y1 + my), max(0, x0 - mx), min(width, x1 + mx)
        )
        top, bottom, left, right = record.margin_clipped
        assert top == my or cy0 == 0
        assert bottom == my or cy1 == height
        assert left == mx or cx0 == 0
        assert right == mx or cx1 == width
        ly0, ly1, lx0, lx1 = record.local_interior
        origin_y, origin_x = record.origin
        assert (ly0 + origin_y, ly1 + origin_y, lx0 + origin_x, lx1 + origin_x) == (y0, y1, x0, x1)
        assert 0 <= ly0 < ly1 <= cy1 - cy0 and 0 <= lx0 < lx1 <= cx1 - cx0


# Partly clipped sides, corners, a whole-plane patch and a margin wider than the plane, each with
# every field written out (review N, B2): bounds, margin, shape -> context_bounds, margin_clipped,
# local_interior.
EXACT = {
    "top partly clipped": (
        (2, 10, 20, 30), (5, 3), (100, 100), (0, 15, 17, 33), (2, 5, 3, 3), (2, 10, 3, 13)
    ),
    "bottom partly clipped": (
        (90, 98, 20, 30), (5, 3), (100, 100), (85, 100, 17, 33), (5, 2, 3, 3), (5, 13, 3, 13)
    ),
    "left partly clipped": (
        (20, 30, 1, 10), (3, 4), (100, 100), (17, 33, 0, 14), (3, 3, 1, 4), (3, 13, 1, 10)
    ),
    "right partly clipped": (
        (20, 30, 90, 99), (3, 4), (100, 100), (17, 33, 86, 100), (3, 3, 4, 1), (3, 13, 4, 13)
    ),
    "top-left corner": ((1, 5, 2, 6), (3, 4), (50, 60), (0, 8, 0, 10), (1, 3, 2, 4), (1, 5, 2, 6)),
    "top-right corner": (
        (1, 5, 55, 59), (3, 4), (50, 60), (0, 8, 51, 60), (1, 3, 4, 1), (1, 5, 4, 8)
    ),
    "bottom-left corner": (
        (46, 49, 2, 6), (3, 4), (50, 60), (43, 50, 0, 10), (3, 1, 2, 4), (3, 6, 2, 6)
    ),
    "bottom-right corner": (
        (46, 49, 55, 59), (3, 4), (50, 60), (43, 50, 51, 60), (3, 1, 4, 1), (3, 6, 4, 8)
    ),
    "whole plane": ((0, 50, 0, 60), (7, 9), (50, 60), (0, 50, 0, 60), (0, 0, 0, 0), (0, 50, 0, 60)),
    "margin wider than the plane": (
        (10, 20, 30, 40), (500, 500), (50, 60), (0, 50, 0, 60), (10, 30, 30, 20), (10, 20, 30, 40)
    ),
}


@pytest.mark.parametrize("case", EXACT.values(), ids=list(EXACT))
def test_exact_windows_at_edges_and_corners(case):
    bounds, (my, mx), shape, window, kept, interior = case
    record = context_bounds(bounds, (my, mx), shape)
    assert record == WindowRecord(
        context_bounds=window,
        origin=(window[0], window[2]),
        margin_requested=(my, my, mx, mx),
        margin_clipped=kept,
        local_interior=interior,
        plane_shape=shape,
    )


VALID = {
    "context_bounds": (8, 22, 28, 42),
    "origin": (8, 28),
    "margin_requested": (2, 2, 2, 2),
    "margin_clipped": (2, 2, 2, 2),
    "local_interior": (2, 12, 2, 12),
    "plane_shape": (100, 100),
}


def test_a_record_built_directly_is_checked_and_equals_the_computed_one():
    assert WindowRecord(**VALID) == context_bounds((10, 20, 30, 40), (2, 2), (100, 100))


@pytest.mark.parametrize("field, value", [
    ("context_bounds", [8, 22, 28, 42]),  # a list could be changed after the check
    ("origin", (np.int64(8), 28)),
    ("margin_requested", (2, 2, 2, True)),
    ("margin_clipped", (2.0, 2, 2, 2)),
    ("local_interior", (2, 12, 2)),
    ("plane_shape", None),
])
def test_a_record_of_other_types_is_refused(field, value):
    with pytest.raises(TypeError, match=field):
        WindowRecord(**{**VALID, field: value})


@pytest.mark.parametrize("field, value", [
    ("context_bounds", (0, 22, 28, 42)),
    ("origin", (99, 99)),
    ("margin_requested", (2, 3, 2, 2)),  # not (my, my, mx, mx)
    ("margin_requested", (3, 3, 2, 2)),
    ("margin_clipped", (0, 2, 2, 2)),
    ("local_interior", (2, 12, 2, 13)),
    ("plane_shape", (100, 41)),
    ("plane_shape", (100, 30)),  # the patch would leave the plane
])
def test_a_record_whose_fields_disagree_is_refused(field, value):
    with pytest.raises(PatchBoundsError):
        WindowRecord(**{**VALID, field: value})
