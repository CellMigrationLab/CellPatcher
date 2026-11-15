"""Mask fractions from a mask window reader (``WindowedMask``), never the whole mask.

Every case is run twice, with the mask as an array and as a reader slicing that array, and the
patches must be the same to the bit: rectangles, order, ``patch_id`` and every ``mask_fractions``
value. A spy reader checks that each pixel is read at most once per mask and that no read holds
more than the memory limit with its foreground.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from cellpatcher import (
    CellConfig,
    CellPatcherConfig,
    GridConfig,
    RandomConfig,
    WholeFOVConfig,
    WindowedMask,
    run_patching,
    windowed_foreground_counts,
)
from cellpatcher.mask_windows import mask_blocks

LARGE = 1 << 30


def _masks(shape):
    rng = np.random.default_rng(7)
    binary = rng.random(shape) < 0.4
    instances = np.zeros(shape, dtype=np.uint16)
    for label in range(1, 30):
        y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        instances[max(0, y - 3):y + 3, max(0, x - 4):x + 4] = label
    parents = np.zeros(shape, dtype=np.int32)
    parents[: shape[0] // 2, : shape[1] // 3] = 70000
    parents[shape[0] // 2:, shape[1] // 2:] = 3
    floats = np.where(binary, 0.5, 0.0).astype(np.float32)
    floats[0, 0] = -0.0
    return {
        "binary": binary,
        "binary_u8": binary.astype(np.uint8),
        "instances": instances,
        "parent_labels": parents,
        "empty": np.zeros(shape, dtype=np.uint8),
        "floats": floats,
    }


class Spy:
    """A reader slicing *array* that records what each read covered and held."""

    def __init__(self, array, limit):
        self.array, self.limit = array, limit
        self.coverage = np.zeros(array.shape, dtype=np.int64)
        self.reads = []

    def __call__(self, bounds):
        assert all(type(v) is int for v in bounds)
        y0, y1, x0, x1 = bounds
        assert 0 <= y0 < y1 <= self.array.shape[0] and 0 <= x0 < x1 <= self.array.shape[1]
        self.coverage[y0:y1, x0:x1] += 1
        window = self.array[y0:y1, x0:x1].copy()
        self.reads.append((bounds, window.nbytes + window.size))  # the window and its foreground
        return window

    def mask(self):
        return WindowedMask(read=self, shape=self.array.shape, dtype=self.array.dtype, memory_limit=self.limit)


def _cut(mode, shape, fraction_masks, masks=None, **sections):
    config = CellPatcherConfig(mode=mode, **sections)
    return run_patching(
        images=[np.broadcast_to(np.uint8(0), shape)], masks=masks, config=config,
        image_metadata=[{"position_id": "pos0"}], fraction_masks=[fraction_masks],
    ).patches


def _bits(patches):
    """Every patch's row and its fractions as the bytes of each float64."""
    return [
        (patch.to_dict(), {name: struct.pack("<d", value) for name, value in patch.mask_fractions.items()})
        for patch in patches
    ]


def _assert_read_once(spies, patches):
    union = np.zeros(next(iter(spies.values())).array.shape, dtype=bool)
    for patch in patches:
        union[patch.y0:patch.y1, patch.x0:patch.x1] = True
    for spy in spies.values():
        assert spy.coverage.max(initial=0) <= 1  # each pixel at most once per mask
        assert (spy.coverage[union] == 1).all()  # every patch pixel was counted
        assert all(held <= spy.limit for _, held in spy.reads)


def _compare(mode, shape, limit, masks=None, **sections):
    arrays = _masks(shape)
    whole = _cut(mode, shape, arrays, masks=masks, **sections)
    spies = {name: Spy(array, limit) for name, array in arrays.items()}
    windowed = _cut(mode, shape, {name: spy.mask() for name, spy in spies.items()}, masks=masks, **sections)
    assert _bits(windowed) == _bits(whole)
    for patch in whole:
        assert list(patch.mask_fractions)[-len(arrays):] == list(arrays)
        for name, array in arrays.items():
            region = array[patch.y0:patch.y1, patch.x0:patch.x1]
            assert patch.mask_fractions[name] == np.count_nonzero(region) / region.size
    _assert_read_once(spies, whole)
    return whole, spies


GRIDS = [
    {"tile_size": 1},
    {"tile_size": 5},
    {"tile_size": 16},
    {"tile_size": 16, "overlap": 0.5},
    {"tile_size": 16, "include_partial": True},
    {"tile_size": 16, "include_partial": True, "partial_mode": "clipped"},
    {"tile_size": 7, "overlap": 0.25, "include_partial": True},
    {"tile_size": 37},
    {"tile_size": 64, "include_partial": True},
]


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: "-".join(f"{k}={v}" for k, v in g.items()))
@pytest.mark.parametrize("shape", [(37, 53), (64, 64), (1, 40), (40, 3)])
@pytest.mark.parametrize("limit", [9, 17 * 7, 5 * 53 * 9, LARGE])
def test_grid_fractions_by_windows_are_bitwise_the_whole_arrays(grid, shape, limit):
    if not grid.get("include_partial") and grid["tile_size"] > min(shape):
        assert _cut("grid", shape, {}, grid=GridConfig(**grid)) == []
        return
    patches, spies = _compare("grid", shape, limit, grid=GridConfig(**grid))
    assert patches
    if limit == LARGE:  # one read per mask: the patches' bounding box
        assert all(len(spy.reads) == 1 for spy in spies.values())


def test_pixels_outside_the_patches_bounding_box_are_never_read():
    patches, spies = _compare("grid", (37, 53), LARGE, grid=GridConfig(tile_size=16))
    for spy in spies.values():
        assert spy.coverage[32:].sum() == 0 and spy.coverage[:, 48:].sum() == 0
        assert spy.reads[0][0] == (0, 32, 0, 48)


def test_a_small_limit_splits_rows_and_still_reads_each_pixel_once():
    _, spies = _compare("grid", (20, 30), 36, grid=GridConfig(tile_size=10, overlap=0.5))
    for spy in spies.values():
        per_read = 36 // spy.mask().pixel_bytes  # under 30: pieces of single rows
        assert per_read < 30
        assert len(spy.reads) == 20 * -(-30 // per_read)


def test_overlapping_random_draws_read_each_pixel_once():
    _compare("random", (50, 60), 6 * 60 * 9, random=RandomConfig(patch_size=12, n_patches=40, seed=3))


def test_random_sampling_with_a_mask_is_unchanged():
    sampling = _masks((50, 60))["binary"]
    _compare(
        "random", (50, 60), 200, masks=[sampling],
        random=RandomConfig(patch_size=8, n_patches=25, seed=11, min_mask_fraction=0.3),
    )


def test_cell_and_whole_fov_patches():
    cells = _masks((40, 50))["instances"]
    _compare("cell", (40, 50), 300, masks=[cells], cell=CellConfig(patch_size=9, exclude_border=False))
    _compare("whole_fov", (40, 50), 300, whole_fov=WholeFOVConfig(square_cover=True))


def test_counts_equal_the_slices_for_random_rectangles():
    rng = np.random.default_rng(1)
    mask = rng.integers(0, 3, size=(41, 29)).astype(np.int16)
    rectangles = []
    for _ in range(200):
        y0, x0 = int(rng.integers(0, 41)), int(rng.integers(0, 29))
        rectangles.append((y0, int(rng.integers(y0 + 1, 42)), x0, int(rng.integers(x0 + 1, 30))))
    for limit in (3, 50, 3 * 29 * 3, LARGE):
        spy = Spy(mask, limit)
        counts = windowed_foreground_counts(spy.mask(), rectangles)
        assert counts == [int(np.count_nonzero(mask[a:b, c:d])) for a, b, c, d in rectangles]
        assert spy.coverage.max() <= 1
        assert all(held <= limit for _, held in spy.reads)
    assert windowed_foreground_counts(Spy(mask, LARGE).mask(), []) == []


def test_blocks_partition_the_box():
    for box, pixels in [((2, 9, 3, 11), 1), ((2, 9, 3, 11), 5), ((2, 9, 3, 11), 8), ((0, 4, 0, 4), 100)]:
        cover = np.zeros((12, 12), dtype=int)
        for y0, y1, x0, x1 in mask_blocks(box, pixels):
            assert (y1 - y0) * (x1 - x0) <= pixels
            cover[y0:y1, x0:x1] += 1
        expected = np.zeros_like(cover)
        expected[box[0]:box[1], box[2]:box[3]] = 1
        assert (cover == expected).all()


def _reader(array):
    return lambda bounds: array[bounds[0]:bounds[1], bounds[2]:bounds[3]]


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (np.zeros((20, 20), dtype=np.uint16), "dtype uint8"),
        (np.zeros((20, 21), dtype=np.uint8), "shape"),
        (np.full((20, 20), np.nan, dtype=np.float32), "NaN"),
    ],
)
def test_a_window_that_is_not_the_declared_one_is_refused(window, message):
    dtype = np.float32 if window.dtype == np.float32 else np.uint8
    mask = WindowedMask(read=lambda bounds: window, shape=(20, 20), dtype=dtype, memory_limit=LARGE)
    with pytest.raises(ValueError, match=message):
        _cut("grid", (20, 20), {"tissue": mask}, grid=GridConfig(tile_size=20))


def test_a_list_is_not_a_window():
    mask = WindowedMask(read=lambda bounds: [[0]], shape=(1, 1), dtype=np.uint8, memory_limit=2)
    with pytest.raises(ValueError, match="ndarray"):
        windowed_foreground_counts(mask, [(0, 1, 0, 1)])


def test_a_windowed_mask_of_another_shape_is_refused():
    mask = WindowedMask(read=_reader(np.zeros((10, 20), np.uint8)), shape=(10, 20), dtype=np.uint8, memory_limit=LARGE)
    with pytest.raises(ValueError, match="WindowedMask of shape"):
        _cut("grid", (20, 20), {"tissue": mask}, grid=GridConfig(tile_size=10))


def test_a_rectangle_outside_the_plane_is_refused():
    mask = WindowedMask(read=_reader(np.zeros((10, 10), np.uint8)), shape=(10, 10), dtype=np.uint8, memory_limit=LARGE)
    for rectangle in [(0, 11, 0, 5), (3, 3, 0, 5), (-1, 2, 0, 5)]:
        with pytest.raises(ValueError, match="not a non-empty part"):
            windowed_foreground_counts(mask, [rectangle])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"read": None}, "callable"),
        ({"shape": (10,)}, "shape"),
        ({"shape": (10, 0)}, "shape"),
        ({"shape": (10.0, 10)}, "shape"),
        ({"dtype": object}, "dtype"),
        ({"dtype": np.complex64}, "dtype"),
        ({"memory_limit": 2}, "memory_limit"),
        ({"memory_limit": 1024.0}, "memory_limit"),
        ({"memory_limit": True}, "memory_limit"),
    ],
)
def test_a_malformed_windowed_mask_is_refused(fields, message):
    arguments = {"read": _reader(np.zeros((10, 10))), "shape": (10, 10), "dtype": np.uint16, "memory_limit": 64}
    arguments.update(fields)
    with pytest.raises(ValueError, match=message):
        WindowedMask(**arguments)


def test_the_windowed_mask_records_plain_values():
    mask = WindowedMask(read=_reader(np.zeros((4, 5))), shape=(np.int64(4), 5), dtype="uint8", memory_limit=np.int64(2))
    assert mask.shape == (4, 5) and type(mask.shape[0]) is int
    assert mask.dtype == np.dtype(np.uint8) and mask.memory_limit == 2 and mask.pixel_bytes == 2
