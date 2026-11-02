"""Patches before and after normalization, as the notebook's preview shows them (CP-3b).

cellpatcher.normalization_review pairs the patches of the run without normalization with the
same patches of the run with it, by patch_id, and bins each shown plane with the normalization
brick's compute_plane_histogram (the maths stays with its owner, design v0.4 §9).
"""

import numpy as np
import pytest
from morphobricks_normalization import compute_plane_histogram

from cellpatcher.normalization_review import DISPLAY_BINS, before_after, display_plane
from cellpatcher.types import PatchRegion


def _patch(patch_id, axes="CYX", c=None, label_id=None):
    return PatchRegion(patch_id=patch_id, image_id="image_0000", mode="grid", y0=0, y1=4, x0=0, x1=5,
                       output_axes=axes, c=c, label_id=label_id)


def test_the_plane_shown_is_the_chosen_channel_of_the_first_plane():
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)  # ZCYX
    assert np.array_equal(display_plane(array, "ZCYX", channel=2), array[0, 2])
    assert np.array_equal(display_plane(array[0, 1], "YX"), array[0, 1])


@pytest.mark.parametrize(("axes", "channel", "message"), [
    ("CXY", 0, "put X before Y"),
    ("YX", 0, "do not describe"),
    ("CYX", 3, "is not a channel"),
])
def test_a_plane_is_never_guessed(axes, channel, message):
    with pytest.raises(ValueError, match=message):
        display_plane(np.zeros((3, 4, 5)), axes, channel)


def test_patches_are_paired_by_patch_id_not_by_order():
    raw = [np.full((2, 4, 5), value, np.uint16) for value in (10, 20, 30)]
    normalized = [np.full((2, 4, 5), value, np.float32) for value in (0.3, 0.1)]
    raw_patches = [_patch(name) for name in ("a", "b", "c")]
    # "b" was discarded by the normalization (its own pixels had no spread); "c" and "a" remain.
    pairs = before_after(raw_patches, raw, [_patch("c"), _patch("a")], normalized)
    assert [pair.patch_id for pair in pairs] == ["c", "a"]
    assert [int(pair.raw[0, 0]) for pair in pairs] == [30, 10]
    assert [float(pair.normalized[0, 0]) for pair in pairs] == [np.float32(0.3), np.float32(0.1)]


def test_the_histograms_are_the_brick_s():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 4000, (2, 4, 5)).astype(np.uint16)
    normalized = ((raw - 100.0) / 3000.0).astype(np.float32)
    normalized[1, 0, 0] = np.nan  # a NaN mask fill is left out by the brick and counted
    (pair,) = before_after([_patch("a")], [raw], [_patch("a")], [normalized], channel=1)
    assert pair.channel == 1
    for plane, histogram in ((raw[1], pair.raw_histogram), (normalized[1], pair.normalized_histogram)):
        expected = compute_plane_histogram(plane, DISPLAY_BINS)
        assert np.array_equal(histogram.counts, expected.counts)
        assert np.array_equal(histogram.representatives, expected.representatives)
    assert pair.normalized_histogram.valid_pixel_count == 19


def test_a_split_channel_patch_shows_its_own_channel():
    (pair,) = before_after([_patch("a", "YX", c=1)], [np.ones((4, 5), np.uint8)],
                           [_patch("a", "YX", c=1, label_id=7)], [np.ones((4, 5), np.float32)])
    assert (pair.channel, pair.label_id) == (1, 7)


def test_only_the_first_pairs_are_kept():
    patches = [_patch(f"p{index}") for index in range(6)]
    arrays = [np.zeros((2, 4, 5), np.float32)] * 6
    assert len(before_after(patches, arrays, patches, arrays)) == 4
    assert len(before_after(patches, arrays, patches, arrays, limit=2)) == 2


def test_runs_without_their_arrays_or_with_other_axes_are_refused():
    with pytest.raises(ValueError, match="the arrays were not kept"):
        before_after([_patch("a")], [], [_patch("a")], [np.zeros((2, 4, 5))])
    with pytest.raises(ValueError, match="without normalization and"):
        before_after([_patch("a", "YX")], [np.zeros((4, 5))], [_patch("a")], [np.zeros((2, 4, 5))])
