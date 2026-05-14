import pytest

from cellpatcher.strategies.whole_fov import WholeFOVStrategy


def test_whole_fov_one_2d_shape_produces_one_region():
    strategy = WholeFOVStrategy()
    regions = strategy.generate_regions(image_shape=(64, 32), image_id="img")
    assert len(regions) == 1


def test_whole_fov_cyx_uses_last_two_dims():
    strategy = WholeFOVStrategy()
    region = strategy.generate_regions(image_shape=(3, 40, 50), axes="CYX")[0]
    assert (region.y1, region.x1) == (40, 50)


def test_whole_fov_invalid_1d_shape_raises():
    strategy = WholeFOVStrategy()
    with pytest.raises(ValueError):
        strategy.generate_regions(image_shape=(10,))


def test_whole_fov_zero_size_dimensions_raise():
    strategy = WholeFOVStrategy()
    with pytest.raises(ValueError):
        strategy.generate_regions(image_shape=(0, 10))
    with pytest.raises(ValueError):
        strategy.generate_regions(image_shape=(10, 0))


def test_whole_fov_region_metadata_correct():
    strategy = WholeFOVStrategy()
    region = strategy.generate_regions(image_shape=(20, 10), image_id="a")[0]
    assert region.patch_id == "a_whole_fov_0"
    assert region.mode == "whole_fov"
    assert (region.y0, region.y1, region.x0, region.x1) == (0, 20, 0, 10)
    assert region.center_y == 10
    assert region.center_x == 5
    assert region.area_px == 200
    assert region.touches_border is True
