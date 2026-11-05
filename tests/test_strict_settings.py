"""Scientific settings are numbers and flags as they are written, never coerced (release blocker 8).

``float(True)`` is 1.0 and ``float("0.5")`` is 0.5, so a coercing check lets ``True``, ``False``
and text through as fractions. Each setting below is refused when constructed, and again where a
run uses it, because the config dataclasses are mutable (``cfg.grid.overlap = False``).
"""

import numpy as np
import pytest

from cellpatcher import (
    CellConfig,
    CellPatcherConfig,
    CommonConfig,
    ExtractionConfig,
    FilterConfig,
    GridConfig,
    NormalizationConfig,
    WholeFOVConfig,
    run_patching,
)
from cellpatcher.filters import compute_threshold, filter_regions
from cellpatcher.strategies.grid import GridStrategy

NOT_NUMBERS = [True, False, np.True_, "0.5", "text", float("nan")]
IMAGE = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
META = [{"position_id": "fov"}]

# (section, setting, values refused although "almost" valid, extra kwargs a valid construction needs)
REFUSED = [
    ("filters", "min_foreground_fraction", NOT_NUMBERS + [float("inf"), 1.5, -0.1]),
    ("filters", "min_mean_intensity", NOT_NUMBERS[:3] + [float("nan"), float("inf"), [1]]),
    ("filters", "foreground_threshold", NOT_NUMBERS[:3] + [float("nan"), float("inf"), [1]]),
    ("whole_fov", "square_cover", [1, 0, "False", None, np.True_]),
    ("common", "extract_arrays", [1, 0, "False", None, np.True_]),
    ("grid", "overlap", NOT_NUMBERS + [float("inf"), 1, 1.5, -0.1]),
    ("grid", "include_partial", [1, 0, "yes", "False", np.True_, None, 0.0]),
    ("grid", "tile_size", [True, 16.0, 16.5, "16", float("nan"), 0, -1]),
    ("random", "patch_size", [True, 16.0, "16", float("nan"), 0]),
    ("random", "n_patches", [True, 3.0, "3", float("nan"), 0]),
    ("random", "seed", [True, 1.0, "1"]),
    ("cell", "patch_size", [True, 8.0, "bad", 0, -4]),
    ("cell", "min_area", [True, 2.0, "2", 0]),
    ("cell", "max_area", [True, 2.0, "2"]),
    ("cell", "exclude_border", [1, 0, "True", None]),
    ("extraction", "output_size", [True, 8.0, "8", (8, True), (8.5, 8), (8, "8"), (8,), 0, (0, 8)]),
    ("extraction", "resize_order", [True, 1.0, "1", -1]),
    ("extraction", "mask_fill_value", [True, "0", [0]]),
    ("extraction", "constant_values", [True, "0", [0]]),
    ("normalization", "percentile", [(True, 99.8), (1.0, "99"), (float("nan"), 99), "1,99", (1,), (0, 1, 2)]),
    ("normalization", "clip", [(True, 5.0), (0, "5"), (0, float("inf")), (float("nan"), 1), "01", (1, 0)]),
]
CONFIGS = {"common": CommonConfig, "whole_fov": WholeFOVConfig, "filters": FilterConfig, "grid": GridConfig,
           "cell": CellConfig, "extraction": ExtractionConfig,
           "normalization": NormalizationConfig}
CASES = [(section, setting, value) for section, setting, values in REFUSED for value in values]


def _config(section, setting, value):
    """A run config with the setting assigned after construction (``__post_init__`` never sees it)."""
    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 16
    setattr(getattr(config, section), setting, value)
    return config


@pytest.mark.parametrize(("section", "setting", "value"), CASES, ids=repr)
def test_a_setting_that_is_not_a_plain_number_or_flag_is_refused_when_constructed(section, setting, value):
    if section == "random":
        from cellpatcher import RandomConfig as config_class
    else:
        config_class = CONFIGS[section]
    with pytest.raises(ValueError, match=setting.split("_")[0]):
        config_class(**{setting: value})


@pytest.mark.parametrize(("section", "setting", "value"), CASES, ids=repr)
def test_a_setting_edited_after_construction_is_refused_by_run_patching(section, setting, value):
    # Former code: True, False and "0.5" were read as 1.0, 0.0 and 0.5 (or crashed mid-run).
    with pytest.raises(ValueError, match=setting.split("_")[0]):
        run_patching(images=[IMAGE], config=_config(section, setting, value), image_metadata=META)


def test_the_filter_and_grid_boundaries_refuse_an_edited_setting_without_the_pipeline():
    filters = FilterConfig()
    filters.min_foreground_fraction = True
    region_source = run_patching(images=[IMAGE], config=_config("grid", "overlap", 0.0), image_metadata=META)
    with pytest.raises(ValueError, match="min_foreground_fraction"):
        filter_regions(IMAGE, region_source.patches, filters)
    grid = GridConfig(tile_size=16)
    grid.overlap = False
    with pytest.raises(ValueError, match="overlap"):
        GridStrategy(grid).generate_regions(image_shape=IMAGE.shape)
    grid = GridConfig(tile_size=16)
    grid.include_partial = 1
    with pytest.raises(ValueError, match="include_partial"):
        GridStrategy(grid).generate_regions(image_shape=IMAGE.shape)
    for bad in (True, float("nan"), [1]):
        with pytest.raises(ValueError, match="threshold"):
            compute_threshold(IMAGE, bad)


def test_cell_strategy_refuses_an_edited_area_bound():
    from cellpatcher.strategies.cell import CellSegmentationStrategy

    cell = CellConfig(patch_size=8)
    cell.min_area = True
    with pytest.raises(ValueError, match="min_area"):
        CellSegmentationStrategy(cell).generate_regions(
            image_shape=(16, 16), labels=np.ones((16, 16), np.int32))


def test_a_settings_error_comes_before_any_file_is_read(tmp_path):
    from cellpatcher.orchestration import run_patching_pipeline

    config = _config("filters", "min_foreground_fraction", "0.5")
    config.common.preprocessing_run = str(tmp_path / "no_such_run")
    with pytest.raises(ValueError, match="min_foreground_fraction"):
        run_patching_pipeline(tmp_path, config=config)


@pytest.mark.parametrize(
    ("overlap", "stride"), [(0, 24), (0.0, 24), (0.25, 18), (0.5, 12), (np.float32(0.5), 12), (0.999, 1)]
)
def test_valid_grid_settings_are_unchanged(overlap, stride):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=24, overlap=overlap))
    patches = run_patching(images=[IMAGE], config=config, image_metadata=META).patches
    starts = list(range(0, 64 - 24 + 1, stride))
    assert sorted((p.y0, p.x0) for p in patches) == sorted((y, x) for y in starts for x in starts)
    partial = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=24, overlap=overlap, include_partial=True))
    assert len(run_patching(images=[IMAGE], config=partial, image_metadata=META).patches) > len(patches) - 1


@pytest.mark.parametrize("fraction", [None, 0, 0.0, 0.5, 1, 1.0, np.float32(0.25)])
def test_valid_filter_settings_are_unchanged(fraction):
    # The image is a ramp 0..4095; the foreground is the pixels above 100 and the first row of
    # tiles holds the only patches below it.
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16))
    config.filters = FilterConfig(min_foreground_fraction=fraction, foreground_threshold=100, min_mean_intensity=1)
    kept = {(p.y0, p.x0) for p in run_patching(images=[IMAGE], config=config, image_metadata=META).patches}
    expected = set()
    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            patch = IMAGE[y : y + 16, x : x + 16]
            if patch.mean() >= 1 and (fraction is None or (patch > 100).mean() >= fraction):
                expected.add((y, x))
    assert kept == expected
    assert len(expected) > 0


def test_valid_extraction_and_normalization_numbers_are_unchanged():
    extraction = ExtractionConfig(output_size=(4, 6), resize_order=0, mask_fill_value=float("nan"), constant_values=2)
    assert (extraction.output_size, extraction.resize_order) == ((4, 6), 0)
    assert NormalizationConfig(percentile=[1, 99], clip=[0, 1]).percentile == (1.0, 99.0)
    assert NormalizationConfig(percentile=(0, 100), clip=(-1, 1)).clip == (-1.0, 1.0)
    assert ExtractionConfig(output_size=np.int64(8)).output_size == 8
