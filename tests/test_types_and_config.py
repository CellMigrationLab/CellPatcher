import json

import pytest

from cellpatcher.config import CellConfig, CellPatcherConfig, GridConfig, MatchConfig, RandomConfig
from cellpatcher.orchestration import apply_overrides, save_config_for_mode, load_config, save_default_config
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
from tests.helpers import named


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
    result = run_patching(images=img, masks=mask, config=None, image_metadata=named(img))
    assert len(result.patches) == 1


def test_run_patching_rejects_non_config_instance():
    import numpy as np
    with pytest.raises(TypeError, match="CellPatcherConfig"):
        run_patching(images=np.zeros((4, 4)), config={"mode": "grid"}, image_metadata=named(np.zeros((4, 4))))


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

def test_apply_overrides_top_level_mode():
    cfg = CellPatcherConfig(mode="cell")
    result = apply_overrides(cfg, {"mode": "grid"})
    assert result.mode == "grid"
    assert cfg.mode == "cell"  # original is not mutated


def test_apply_overrides_section_field():
    cfg = CellPatcherConfig(mode="grid")
    result = apply_overrides(cfg, {"grid.tile_size": 64})
    assert result.grid.tile_size == 64
    assert cfg.grid.tile_size == 128  # original is not mutated


def test_apply_overrides_multiple_keys():
    cfg = CellPatcherConfig(mode="random")
    result = apply_overrides(cfg, {"random.n_patches": 50, "random.seed": 42, "common.image_axes": "CYX"})
    assert result.random.n_patches == 50
    assert result.random.seed == 42
    assert result.common.image_axes == "CYX"


def test_apply_overrides_empty_dict_returns_equal_config():
    cfg = CellPatcherConfig(mode="grid")
    result = apply_overrides(cfg, {})
    assert result.mode == cfg.mode
    assert result.grid.tile_size == cfg.grid.tile_size


def test_apply_overrides_rejects_unknown_top_level_field():
    cfg = CellPatcherConfig()
    with pytest.raises(ValueError, match="Unknown top-level config field"):
        apply_overrides(cfg, {"nonexistent": 1})


def test_apply_overrides_rejects_unknown_section():
    cfg = CellPatcherConfig()
    with pytest.raises(ValueError, match="Unknown config section"):
        apply_overrides(cfg, {"bogus.field": 1})


def test_apply_overrides_rejects_unknown_field_in_section():
    cfg = CellPatcherConfig()
    with pytest.raises(ValueError, match="Unknown field"):
        apply_overrides(cfg, {"grid.nonexistent": 1})


# ---------------------------------------------------------------------------
# save_config_for_mode
# ---------------------------------------------------------------------------

def test_save_config_for_mode_grid_contains_only_expected_sections(tmp_path):
    out = save_config_for_mode("grid", tmp_path / "grid_cfg.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["mode"] == "grid"
    # common sections always present
    for section in ("common", "filters", "export", "extraction"):
        assert section in data, f"Expected section '{section}' in grid config"
    # grid-specific section present
    assert "grid" in data
    # other mode sections must NOT be present
    for absent in ("cell", "random", "matching"):
        assert absent not in data, f"Section '{absent}' should not appear in grid config"


def test_save_config_for_mode_cell_contains_cell_and_matching(tmp_path):
    out = save_config_for_mode("cell", tmp_path / "cell_cfg.json")
    data = json.loads(out.read_text())
    assert data["mode"] == "cell"
    assert "cell" in data
    assert "matching" in data
    assert "random" not in data
    assert "grid" not in data


def test_save_config_for_mode_random_contains_random_and_matching(tmp_path):
    out = save_config_for_mode("random", tmp_path / "random_cfg.json")
    data = json.loads(out.read_text())
    assert data["mode"] == "random"
    assert "random" in data
    assert "matching" in data
    assert "cell" not in data
    assert "grid" not in data


def test_save_config_for_mode_whole_fov_has_no_mode_specific_sections(tmp_path):
    out = save_config_for_mode("whole_fov", tmp_path / "wfov_cfg.json")
    data = json.loads(out.read_text())
    assert data["mode"] == "whole_fov"
    for absent in ("cell", "random", "grid", "matching"):
        assert absent not in data


def test_save_config_for_mode_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        save_config_for_mode("bad_mode", tmp_path / "cfg.json")


def test_load_config_handles_mode_specific_file_roundtrip(tmp_path):
    """load_config must reconstruct a full CellPatcherConfig from a mode-sparse file."""
    out = save_config_for_mode("grid", tmp_path / "grid_cfg.json")
    cfg = load_config(out)
    assert isinstance(cfg, CellPatcherConfig)
    assert cfg.mode == "grid"
    assert cfg.grid.tile_size == 128  # default
    assert cfg.cell.patch_size == "auto"  # load_config initializes missing sections from dataclass defaults


def test_save_config_for_mode_default_values_match_dataclass_defaults(tmp_path):
    out = save_config_for_mode("grid", tmp_path / "cfg.json")
    cfg = load_config(out)
    expected = CellPatcherConfig(mode="grid")
    assert cfg.grid.tile_size == expected.grid.tile_size
    assert cfg.common.image_axes == expected.common.image_axes
    assert cfg.export.enabled == expected.export.enabled
