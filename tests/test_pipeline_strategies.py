import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig, RandomConfig
from cellpatcher.pipeline import run_patching


def test_pipeline_whole_fov_one_patch_per_image():
    cfg = CellPatcherConfig(mode="whole_fov")
    images = [np.zeros((10, 20)), np.zeros((5, 7))]
    result = run_patching(images=images, config=cfg)
    assert len(result.patches) == 2


def test_pipeline_grid_returns_grid_patches():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=50, overlap=0.0, include_partial=False))
    image = np.zeros((100, 100))
    result = run_patching(images=image, config=cfg)
    assert len(result.patches) == 4


def test_pipeline_rejects_images_none_for_implemented_modes():
    with pytest.raises(ValueError):
        run_patching(images=None, config=CellPatcherConfig(mode="whole_fov"))


def test_pipeline_cell_single_image_mask_returns_patches():
    cfg = CellPatcherConfig(mode="cell")
    img = np.zeros((6, 6))
    mask = np.zeros((6, 6), dtype=int)
    mask[2:4, 2:4] = 1
    result = run_patching(images=img, masks=mask, config=cfg)
    assert len(result.patches) == 1


def test_pipeline_cell_list_images_masks_works():
    cfg = CellPatcherConfig(mode="cell")
    imgs = [np.zeros((6, 6)), np.zeros((6, 6))]
    masks = [np.pad(np.ones((2, 2), dtype=int), 2), np.pad(np.ones((2, 2), dtype=int), 2)]
    result = run_patching(images=imgs, masks=masks, config=cfg)
    assert len(result.patches) >= 2


def test_pipeline_cell_rejects_missing_masks():
    cfg = CellPatcherConfig(mode="cell")
    with pytest.raises(ValueError):
        run_patching(images=np.zeros((5, 5)), masks=None, config=cfg)


def test_pipeline_cell_rejects_mismatched_lengths():
    cfg = CellPatcherConfig(mode="cell")
    with pytest.raises(ValueError):
        run_patching(images=[np.zeros((5, 5)), np.zeros((5, 5))], masks=[np.zeros((5, 5))], config=cfg)


def test_pipeline_random_single_image_returns_patches():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=4, seed=0))
    result = run_patching(images=np.zeros((32, 32)), config=cfg)
    assert len(result.patches) == 4


def test_pipeline_random_list_images_works():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=3, seed=0))
    result = run_patching(images=[np.zeros((32, 32)), np.zeros((32, 32))], config=cfg)
    assert len(result.patches) == 6


def test_pipeline_random_multi_image_seeded_unique_coordinates():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=3, seed=0))
    result = run_patching(images=[np.zeros((32, 32)), np.zeros((32, 32))], config=cfg)
    image_ids = sorted({str(patch.image_id) for patch in result.patches})
    assert len(image_ids) == 2
    by_image: dict[str, list[tuple[int, int]]] = {image_id: [] for image_id in image_ids}
    for patch in result.patches:
        by_image[str(patch.image_id)].append((patch.y0, patch.x0))
    assert by_image[image_ids[0]] != by_image[image_ids[1]]


def test_pipeline_rejects_empty_image_list():
    with pytest.raises(ValueError):
        run_patching(images=[], config=CellPatcherConfig(mode="grid"))


def test_pipeline_random_matching_mask_list_works():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=4, n_patches=2, seed=0))
    imgs = [np.zeros((16, 16)), np.zeros((16, 16))]
    masks = [np.ones((16, 16), dtype=bool), np.ones((16, 16), dtype=bool)]
    result = run_patching(images=imgs, masks=masks, config=cfg)
    assert len(result.patches) == 4


def test_pipeline_random_rejects_single_mask_with_multiple_images():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=4, n_patches=2, seed=0))
    imgs = [np.zeros((16, 16)), np.zeros((16, 16))]
    with pytest.raises(ValueError):
        run_patching(images=imgs, masks=np.ones((16, 16), dtype=bool), config=cfg)


def test_pipeline_random_rejects_mismatched_lengths():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=4, n_patches=2, seed=0))
    imgs = [np.zeros((16, 16)), np.zeros((16, 16))]
    masks = [np.ones((16, 16), dtype=bool)]
    with pytest.raises(ValueError):
        run_patching(images=imgs, masks=masks, config=cfg)


def test_pipeline_extract_arrays_whole_fov_returns_patch_arrays():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    image = np.arange(20).reshape(4, 5)
    result = run_patching(images=image, config=cfg)
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == 1
    assert result.patch_arrays[0].shape == (4, 5)


def test_pipeline_extract_arrays_grid_one_array_per_patch():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    image = np.arange(16).reshape(4, 4)
    result = run_patching(images=image, config=cfg)
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == len(result.patches)


def test_pipeline_extract_arrays_false_keeps_none():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = False
    result = run_patching(images=np.zeros((4, 4)), config=cfg)
    assert result.patch_arrays is None


def test_pipeline_extract_arrays_uses_extraction_pad_mode_constant():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=3, n_patches=1, seed=0))
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.extraction.pad_mode = "constant"
    cfg.extraction.constant_values = -5

    image = np.arange(25).reshape(5, 5)
    result = run_patching(images=image, config=cfg)
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == 1


def test_pipeline_extract_arrays_resized_output_size():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.extraction.output_size = 3
    cfg.extraction.resize_order = 0

    image = np.arange(20).reshape(4, 5)
    result = run_patching(images=image, config=cfg)
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (3, 3)


def test_pipeline_extract_arrays_resized_non_yx_axes_cyx():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "CYX"
    cfg.extraction.output_size = (3, 3)
    cfg.extraction.resize_order = 0

    image = np.arange(2 * 4 * 5).reshape(2, 4, 5)
    result = run_patching(images=image, config=cfg)
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (2, 3, 3)


def test_pipeline_grid_filters_out_dark_patches():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.filters.min_mean_intensity = 1.0
    image = np.array([[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]])
    result = run_patching(images=image, config=cfg)
    assert len(result.patches) == 2


def test_pipeline_filters_applied_before_extraction_patch_arrays_count_matches_kept():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.filters.min_mean_intensity = 1.0
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    image = np.array([[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]])
    result = run_patching(images=image, config=cfg)
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == len(result.patches)


def test_pipeline_export_enabled_writes_files_and_metadata(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True

    result = run_patching(images=np.zeros((4, 4)), config=cfg)
    assert len(result.exported_files) == 1
    assert (tmp_path / "patches.csv").exists()
    assert result.metadata_file is not None


def test_pipeline_export_enabled_without_output_dir_raises():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    with pytest.raises(ValueError):
        run_patching(images=np.zeros((4, 4)), config=cfg)


def test_pipeline_export_enabled_with_extract_arrays_false_still_exports(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = False
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"

    result = run_patching(images=np.zeros((4, 4)), config=cfg)
    assert len(result.exported_files) == 1
    assert result.patch_arrays is None


def test_pipeline_export_disabled_writes_no_files(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = False
    result = run_patching(images=np.zeros((4, 4)), config=cfg)
    assert result.exported_files == []


def test_pipeline_zero_patches_export_writes_metadata_only(tmp_path):
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=10, overlap=0.0, include_partial=False))
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True
    cfg.common.output_dir = str(tmp_path)
    cfg.common.extract_arrays = False

    result = run_patching(images=np.zeros((4, 4)), config=cfg)
    assert result.patches == []
    assert result.exported_files == []
    assert result.metadata_file is not None
    assert (tmp_path / "patches.csv").exists()
    assert list(tmp_path.glob("*.npy")) == []

def test_pipeline_cell_apply_mask_true_clears_outside_label():
    cfg = CellPatcherConfig(mode="cell")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.extraction.apply_mask = True
    cfg.extraction.mask_fill_value = 0
    cfg.cell.patch_size = 5

    image = np.full((8, 8), 9, dtype=np.int32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:5, 2:5] = 1

    result = run_patching(images=image, masks=labels, config=cfg)
    assert result.patch_arrays is not None
    patch = result.patch_arrays[0]
    assert np.all(patch[0, :] == 0)
    assert np.all(patch[:, 0] == 0)
    assert patch[1, 1] == 9


def test_pipeline_cell_apply_mask_before_resize_runs():
    cfg = CellPatcherConfig(mode="cell")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.extraction.apply_mask = True
    cfg.extraction.output_size = (10, 10)
    cfg.extraction.resize_order = 0

    image = np.full((8, 8), 5, dtype=np.int32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:5, 2:5] = 1

    result = run_patching(images=image, masks=labels, config=cfg)
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (10, 10)


def test_pipeline_export_path_apply_mask_true_exports_masked_npy(tmp_path):
    cfg = CellPatcherConfig(mode="cell")
    cfg.common.extract_arrays = False
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.extraction.apply_mask = True
    cfg.cell.patch_size = 5

    image = np.full((8, 8), 7, dtype=np.int32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:5, 2:5] = 1

    result = run_patching(images=image, masks=labels, config=cfg)
    assert len(result.exported_files) == 1
    arr = np.load(result.exported_files[0])
    assert arr[0, 0] == 0
    assert arr[1, 1] == 7


def test_pipeline_apply_mask_true_without_masks_raises():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True
    with pytest.raises(ValueError):
        run_patching(images=np.ones((4, 4)), masks=None, config=cfg)


def test_pipeline_apply_mask_true_non_cell_region_raises():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True
    with pytest.raises(ValueError):
        run_patching(images=np.ones((4, 4)), masks=[np.ones((4, 4), dtype=int)], config=cfg)
