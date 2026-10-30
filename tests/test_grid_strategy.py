import numpy as np
import pytest

from cellpatcher.config import GridConfig
from cellpatcher.strategies.grid import GridStrategy


def test_grid_100_100_tile50_no_overlap_no_partial_is_4_tiles():
    s = GridStrategy(GridConfig(tile_size=50, overlap=0.0, include_partial=False))
    regions = s.generate_regions(image_shape=(100, 100), image_id="im")
    assert len(regions) == 4


def test_grid_overlap_expected_tile_count():
    s = GridStrategy(GridConfig(tile_size=50, overlap=0.5, include_partial=False))
    regions = s.generate_regions(image_shape=(100, 100))
    assert len(regions) == 9


def test_grid_large_tile_no_partial_returns_empty():
    s = GridStrategy(GridConfig(tile_size=200, include_partial=False))
    assert s.generate_regions(image_shape=(100, 100)) == []


def test_grid_large_tile_with_partial_returns_one_tile():
    s = GridStrategy(GridConfig(tile_size=200, include_partial=True))
    regions = s.generate_regions(image_shape=(100, 100))
    assert len(regions) == 1
    assert (regions[0].y0, regions[0].y1, regions[0].x0, regions[0].x1) == (0, 100, 0, 100)


def test_grid_overlap_very_high_still_generates_tiles():
    s = GridStrategy(GridConfig(tile_size=5, overlap=0.99, include_partial=True))
    regions = s.generate_regions(image_shape=(6, 6))
    assert len(regions) > 0
    assert all(0 <= r.y0 < r.y1 <= 6 for r in regions)
    assert all(0 <= r.x0 < r.x1 <= 6 for r in regions)


def test_grid_overlap_very_high_avoids_duplicate_coordinates():
    s = GridStrategy(GridConfig(tile_size=5, overlap=0.99, include_partial=True))
    regions = s.generate_regions(image_shape=(6, 6))
    coords = [(r.y0, r.y1, r.x0, r.x1) for r in regions]
    assert len(coords) == len(set(coords))


def test_grid_large_tile_partial_with_high_overlap_still_one_tile():
    s = GridStrategy(GridConfig(tile_size=200, overlap=0.9, include_partial=True))
    regions = s.generate_regions(image_shape=(100, 100))
    assert len(regions) == 1
    assert (regions[0].y0, regions[0].y1, regions[0].x0, regions[0].x1) == (0, 100, 0, 100)


def test_grid_include_partial_covers_edge():
    s = GridStrategy(GridConfig(tile_size=30, overlap=0.0, include_partial=True))
    regions = s.generate_regions(image_shape=(100, 100))
    assert any(r.y1 == 100 for r in regions)
    assert any(r.x1 == 100 for r in regions)


def test_grid_patch_ids_deterministic():
    s = GridStrategy(GridConfig(tile_size=50, overlap=0.0, include_partial=False))
    regions = s.generate_regions(image_shape=(100, 100), image_id="img")
    # The last level of patch_id: the tile's own rectangle (cellpatcher.ids).
    assert [r.patch_id for r in regions] == [
        "grid-y0-x0-h50-w50",
        "grid-y0-x50-h50-w50",
        "grid-y50-x0-h50-w50",
        "grid-y50-x50-h50-w50",
    ]


def test_grid_touches_border_for_border_and_interior_tiles():
    s = GridStrategy(GridConfig(tile_size=20, overlap=0.0, include_partial=False))
    regions = s.generate_regions(image_shape=(60, 60))
    border_flags = {(r.y0, r.x0): r.touches_border for r in regions}
    assert border_flags[(20, 20)] is False
    assert border_flags[(0, 0)] is True


def test_grid_tile_size_equal_image_produces_single_full_tile():
    s = GridStrategy(GridConfig(tile_size=10, overlap=0.0, include_partial=False))
    regions = s.generate_regions(image_shape=(10, 10))
    assert len(regions) == 1
    assert (regions[0].y0, regions[0].y1, regions[0].x0, regions[0].x1) == (0, 10, 0, 10)


def _coverage(regions, shape):
    counts = np.zeros(shape, dtype=int)
    for r in regions:
        counts[r.y0:r.y1, r.x0:r.x1] += 1
    return counts


def _overlaps(a, b):
    return a.y0 < b.y1 and b.y0 < a.y1 and a.x0 < b.x1 and b.x0 < a.x1


@pytest.mark.parametrize("shape", [(100, 100), (100, 70), (31, 64), (7, 5), (60, 60), (128, 129)])
def test_grid_clipped_partial_tiles_cover_every_pixel_once(shape):
    s = GridStrategy(GridConfig(tile_size=30, overlap=0.0, include_partial=True, partial_mode="clipped"))
    regions = s.generate_regions(image_shape=shape)
    assert not any(
        _overlaps(a, b) for i, a in enumerate(regions) for b in regions[i + 1:]
    ), "no two clipped tiles may overlap"
    assert (_coverage(regions, shape) == 1).all()
    # Edge tiles are clipped (never a full tile shifted back to the edge).
    assert all(r.y0 % 30 == 0 and r.x0 % 30 == 0 for r in regions)
    assert all(r.y1 - r.y0 == min(30, shape[0] - r.y0) for r in regions)
    assert all(r.x1 - r.x0 == min(30, shape[1] - r.x0) for r in regions)


def test_grid_clipped_partial_tile_records_requested_full_tile():
    s = GridStrategy(GridConfig(tile_size=30, include_partial=True, partial_mode="clipped"))
    corner = [r for r in s.generate_regions(image_shape=(100, 100)) if r.y0 == 90 and r.x0 == 90]
    assert len(corner) == 1
    r = corner[0]
    assert (r.y1, r.x1, r.area_px) == (100, 100, 100)
    assert (r.requested_y1, r.requested_x1) == (120, 120)
    assert r.patch_id == "grid-y90-x90-h10-w10"


def test_grid_flush_partial_default_is_unchanged():
    # The historical include_partial=True behaviour stays the default: the
    # clipped tile at 90 and a full tile flush with the edge at 70 (overlapping).
    assert GridConfig().partial_mode == "flush"
    s = GridStrategy(GridConfig(tile_size=30, include_partial=True))
    starts = sorted({r.y0 for r in s.generate_regions(image_shape=(100, 100))})
    assert starts == [0, 30, 60, 70, 90]
    assert (_coverage(s.generate_regions(image_shape=(100, 100)), (100, 100)) > 1).any()


def test_grid_clipped_and_flush_agree_when_tiles_divide_the_image():
    def coords(mode):
        s = GridStrategy(GridConfig(tile_size=25, include_partial=True, partial_mode=mode))
        return [(r.y0, r.y1, r.x0, r.x1) for r in s.generate_regions(image_shape=(100, 50))]

    assert coords("clipped") == coords("flush")


def test_grid_partial_mode_is_validated():
    with pytest.raises(ValueError, match="partial_mode must be one of"):
        GridConfig(include_partial=True, partial_mode="pad")
    with pytest.raises(ValueError, match="include_partial=True"):
        GridConfig(include_partial=False, partial_mode="clipped")


def test_grid_partial_mode_set_after_construction_is_checked_by_the_strategy():
    config = GridConfig(tile_size=30)
    config.partial_mode = "clipped"  # include_partial stays False
    with pytest.raises(ValueError, match="include_partial=True"):
        GridStrategy(config).generate_regions(image_shape=(100, 100))


@pytest.mark.parametrize("mode", ["clipped", "bogus"])
def test_grid_partial_mode_is_checked_even_when_the_tile_exceeds_the_image(mode):
    """#48 review: the early empty return for a too-large tile must not skip the check."""
    config = GridConfig(tile_size=64)
    config.partial_mode = mode  # include_partial stays False
    with pytest.raises(ValueError):
        GridStrategy(config).generate_regions(image_shape=(10, 10))


def test_grid_clipped_partial_through_the_pipeline():
    from cellpatcher.config import CellPatcherConfig
    from cellpatcher.orchestration import apply_overrides
    from cellpatcher.pipeline import run_patching
    from tests.helpers import named

    config = apply_overrides(
        CellPatcherConfig(mode="grid"),
        {"grid.tile_size": 16, "grid.include_partial": True, "grid.partial_mode": "clipped"},
    )
    image = np.zeros((40, 50), dtype=np.uint8)
    result = run_patching(images=[image], config=config, image_metadata=named([image]))
    assert (_coverage(result.patches, (40, 50)) == 1).all()
