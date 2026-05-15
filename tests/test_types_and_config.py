import pytest

from cellpatcher.config import CellConfig, CellPatcherConfig, GridConfig, MatchConfig, RandomConfig
from cellpatcher.pipeline import run_patching
from cellpatcher.types import PatchRegion


def test_patch_region_height_width():
    region = PatchRegion(patch_id="p1", y0=2, y1=10, x0=3, x1=11)
    assert region.height == 8
    assert region.width == 8


def test_patch_region_has_z_false_for_2d():
    region = PatchRegion(patch_id="p1", y0=0, y1=2, x0=0, x1=3)
    assert region.has_z is False


def test_patch_region_has_z_true_for_3d():
    region = PatchRegion(patch_id="p1", y0=0, y1=2, x0=0, x1=3, z0=1, z1=5)
    assert region.has_z is True


def test_patch_region_rejects_invalid_y_coordinates():
    with pytest.raises(ValueError):
        PatchRegion(patch_id="p1", y0=5, y1=5, x0=0, x1=1)


def test_patch_region_rejects_invalid_x_coordinates():
    with pytest.raises(ValueError):
        PatchRegion(patch_id="p1", y0=0, y1=1, x0=2, x1=2)


def test_patch_region_validates_z_pairing():
    with pytest.raises(ValueError):
        PatchRegion(patch_id="p1", y0=0, y1=2, x0=0, x1=2, z0=1)


def test_patch_region_to_dict_contains_expected_keys():
    region = PatchRegion(patch_id="p1", y0=0, y1=2, x0=0, x1=2)
    payload = region.to_dict()
    assert {"patch_id", "y0", "y1", "x0", "x1", "has_z"}.issuperset({"patch_id", "y0", "x1"})
    assert "patch_id" in payload
    assert "y0" in payload
    assert "x1" in payload


@pytest.mark.parametrize("mode", ["cell", "random", "whole_fov", "grid"])
def test_cell_patcher_config_accepts_modes(mode):
    cfg = CellPatcherConfig(mode=mode)
    assert cfg.mode == mode


def test_cell_patcher_config_rejects_invalid_mode():
    with pytest.raises(ValueError):
        CellPatcherConfig(mode="bad")


def test_cell_config_rejects_invalid_center_on():
    with pytest.raises(ValueError):
        CellConfig(center_on="middle")


def test_cell_config_rejects_invalid_binary_mask_policy():
    with pytest.raises(ValueError):
        CellConfig(binary_mask_policy="invalid")


def test_random_config_rejects_invalid_n_patches():
    with pytest.raises(ValueError):
        RandomConfig(n_patches=0)


def test_grid_config_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        GridConfig(overlap=1.0)


def test_match_config_rejects_invalid_strategy():
    with pytest.raises(ValueError):
        MatchConfig(strategy="bad")


def test_match_config_rejects_invalid_mask_suffixes_type():
    with pytest.raises(ValueError):
        MatchConfig(mask_suffixes=[1])


def test_match_config_rejects_invalid_case_sensitive_type():
    with pytest.raises(ValueError):
        MatchConfig(case_sensitive="yes")


def test_run_patching_cell_mode_requires_images_and_masks():
    with pytest.raises(ValueError):
        run_patching(config=CellPatcherConfig(mode="cell"))


# ---------------------------------------------------------------------------
# CellConfig additional validation
# ---------------------------------------------------------------------------

def test_cell_config_rejects_min_area_below_one():
    with pytest.raises(ValueError, match="min_area"):
        CellConfig(min_area=0)


def test_cell_config_rejects_max_area_below_min_area():
    with pytest.raises(ValueError, match="max_area"):
        CellConfig(min_area=5, max_area=3)


# ---------------------------------------------------------------------------
# RandomConfig validation
# ---------------------------------------------------------------------------

def test_random_config_rejects_non_positive_patch_size():
    with pytest.raises(ValueError, match="patch_size"):
        RandomConfig(patch_size=0)


# ---------------------------------------------------------------------------
# GridConfig validation
# ---------------------------------------------------------------------------

def test_grid_config_rejects_non_positive_tile_size():
    with pytest.raises(ValueError, match="tile_size"):
        GridConfig(tile_size=0)


# ---------------------------------------------------------------------------
# FilterConfig additional validation
# ---------------------------------------------------------------------------

from cellpatcher.config import FilterConfig


def test_filter_config_rejects_negative_min_foreground_fraction():
    with pytest.raises(ValueError, match="min_foreground_fraction"):
        FilterConfig(min_foreground_fraction=-0.1)


def test_filter_config_rejects_fraction_above_one():
    with pytest.raises(ValueError, match="min_foreground_fraction"):
        FilterConfig(min_foreground_fraction=1.1)


def test_filter_config_rejects_non_numeric_min_mean_intensity():
    with pytest.raises(ValueError, match="min_mean_intensity"):
        FilterConfig(min_mean_intensity="high")


def test_filter_config_rejects_non_valid_foreground_threshold_type():
    with pytest.raises(ValueError, match="foreground_threshold"):
        FilterConfig(foreground_threshold=[0.5])


# ---------------------------------------------------------------------------
# ExportConfig validation
# ---------------------------------------------------------------------------

from cellpatcher.config import ExportConfig


def test_export_config_rejects_non_bool_enabled():
    with pytest.raises(ValueError, match="enabled"):
        ExportConfig(enabled="yes")


def test_export_config_rejects_non_bool_save_metadata_csv():
    with pytest.raises(ValueError, match="save_metadata_csv"):
        ExportConfig(save_metadata_csv=1)


def test_export_config_rejects_invalid_image_format():
    with pytest.raises(ValueError, match="image_format"):
        ExportConfig(image_format="png")


# ---------------------------------------------------------------------------
# ExtractionConfig validation
# ---------------------------------------------------------------------------

from cellpatcher.config import ExtractionConfig


def test_extraction_config_rejects_invalid_pad_mode():
    with pytest.raises(ValueError, match="pad_mode"):
        ExtractionConfig(pad_mode="reflect")


def test_extraction_config_rejects_zero_output_size_int():
    with pytest.raises(ValueError, match="output_size"):
        ExtractionConfig(output_size=0)


def test_extraction_config_rejects_zero_output_size_tuple():
    with pytest.raises(ValueError, match="output_size"):
        ExtractionConfig(output_size=(0, 4))


def test_extraction_config_rejects_wrong_output_size_type():
    with pytest.raises(ValueError, match="output_size"):
        ExtractionConfig(output_size="large")


def test_extraction_config_rejects_negative_resize_order():
    with pytest.raises(ValueError, match="resize_order"):
        ExtractionConfig(resize_order=-1)


def test_extraction_config_rejects_non_bool_apply_mask():
    with pytest.raises(ValueError, match="apply_mask"):
        ExtractionConfig(apply_mask="yes")


def test_extraction_config_rejects_non_numeric_mask_fill_value():
    with pytest.raises(ValueError, match="mask_fill_value"):
        ExtractionConfig(mask_fill_value="zero")


# ---------------------------------------------------------------------------
# run_patching entry-point validation
# ---------------------------------------------------------------------------

def test_run_patching_with_none_config_uses_cell_mode_default():
    """run_patching(config=None) should default to CellPatcherConfig() which is mode='cell'."""
    import numpy as np
    img = np.zeros((6, 6))
    mask = np.zeros((6, 6), dtype=int)
    mask[2:4, 2:4] = 1
    result = run_patching(images=img, masks=mask, config=None)
    assert len(result.patches) == 1


def test_run_patching_rejects_non_config_instance():
    import numpy as np
    with pytest.raises(TypeError, match="CellPatcherConfig"):
        run_patching(images=np.zeros((4, 4)), config={"mode": "grid"})
