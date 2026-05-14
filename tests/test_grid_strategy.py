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
    s = GridStrategy(GridConfig(tile_size=20, overlap=0.99, include_partial=True))
    regions = s.generate_regions(image_shape=(30, 30))
    assert len(regions) > 0


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
    assert [r.patch_id for r in regions] == [
        "img_grid_00000",
        "img_grid_00001",
        "img_grid_00002",
        "img_grid_00003",
    ]


def test_grid_touches_border_for_border_and_interior_tiles():
    s = GridStrategy(GridConfig(tile_size=20, overlap=0.0, include_partial=False))
    regions = s.generate_regions(image_shape=(60, 60))
    border_flags = {(r.y0, r.x0): r.touches_border for r in regions}
    assert border_flags[(20, 20)] is False
    assert border_flags[(0, 0)] is True
