import pytest

from cellpatcher.config import CellConfig, CellPatcherConfig, GridConfig, RandomConfig
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


def test_run_patching_cell_mode_requires_images_and_masks():
    with pytest.raises(ValueError):
        run_patching(config=CellPatcherConfig(mode="cell"))
