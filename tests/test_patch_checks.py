"""check_patch_bounds (D86 API C3); check_region_mask moved to normalization (Spec ADR 0005 BM7).

Four feature bricks copied these checks; MorphoEmbed's copy took float bounds
such as (0.0, 5.0, 0, 4). The shared rule is strict: integers only, a
non-empty rectangle, inside the plane.
"""

import numpy as np
import pytest

import cellpatcher
from cellpatcher import PatchBoundsError, check_patch_bounds


def test_integer_bounds_inside_the_plane_are_returned_as_ints():
    bounds = check_patch_bounds((np.int64(0), np.uint16(5), 0, 4), (5, 4))
    assert bounds == (0, 5, 0, 4) and all(type(v) is int for v in bounds)
    assert check_patch_bounds([1, 2, 3, 4], (2, 4)) == (1, 2, 3, 4)


@pytest.mark.parametrize("bounds, match", [
    ((0.0, 5.0, 0, 4), "not 4 integers"),          # MorphoEmbed's copy accepted these
    ((0, 5, 0), "not 4 integers"),
    ((True, 5, 0, 4), "not 4 integers"),
    (("0", "5", "0", "4"), "not 4 integers"),
    (np.array([0, 5, 0, 4]), "not 4 integers"),
    (None, "not 4 integers"),
    ((2, 2, 0, 4), "empty"),
    ((0, 5, 3, 1), "empty"),
    ((-1, 5, 0, 4), "leave"),
    ((0, 6, 0, 4), "leave"),
    ((0, 5, 0, 5), "leave"),
])
def test_bounds_that_are_not_a_rectangle_inside_the_plane_are_refused(bounds, match):
    with pytest.raises(PatchBoundsError, match=match):
        check_patch_bounds(bounds, (5, 4), name="patch 'p1'")


def test_the_message_names_the_patch():
    with pytest.raises(ValueError, match=r"^patch 'p1': bounds \[0, 9, 0, 4\] leave the \(5, 4\) plane$"):
        check_patch_bounds((0, 9, 0, 4), (5, 4), name="patch 'p1'")


@pytest.mark.parametrize("shape", [(5,), (5, 4, 3), (0, 4), (5.0, 4), "54"])
def test_a_shape_that_is_not_a_plane_is_refused(shape):
    with pytest.raises(PatchBoundsError, match="plane shape"):
        check_patch_bounds((0, 1, 0, 1), shape)


def test_check_region_mask_is_removed_for_normalizations_check_region():
    # Spec ADR 0005 BM7: the one region checker is morphobricks_normalization.regions.check_region.
    assert not hasattr(cellpatcher, "check_region_mask")
    assert "check_region_mask" not in cellpatcher.__all__
    import cellpatcher.patch_checks as patch_checks

    assert not hasattr(patch_checks, "check_region_mask")
