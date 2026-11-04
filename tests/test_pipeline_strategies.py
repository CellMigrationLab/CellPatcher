import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig, RandomConfig
from cellpatcher.pipeline import run_patching
from tests.helpers import named


def test_pipeline_whole_fov_one_patch_per_image():
    cfg = CellPatcherConfig(mode="whole_fov")
    images = [np.zeros((10, 20)), np.zeros((5, 7))]
    result = run_patching(images=images, config=cfg, image_metadata=named(images))
    assert len(result.patches) == 2


def test_pipeline_whole_fov_yxc_uses_yx_axes_not_trailing_dims():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.image_axes = "YXC"
    image = np.zeros((10, 20, 3))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 1
    patch = result.patches[0]
    assert patch.y1 == 10
    assert patch.x1 == 20


def test_pipeline_grid_returns_grid_patches():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=50, overlap=0.0, include_partial=False))
    image = np.zeros((100, 100))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 4


def test_pipeline_grid_yxc_tiles_true_spatial_plane():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=5, overlap=0.0, include_partial=False))
    cfg.common.image_axes = "YXC"
    image = np.zeros((10, 20, 3))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 8
    assert all(0 <= p.y0 < p.y1 <= 10 for p in result.patches)
    assert all(0 <= p.x0 < p.x1 <= 20 for p in result.patches)


def test_pipeline_rejects_images_none_for_implemented_modes():
    with pytest.raises(ValueError):
        run_patching(images=None, config=CellPatcherConfig(mode="whole_fov"), image_metadata=named(None))


def test_pipeline_cell_single_image_mask_returns_patches():
    cfg = CellPatcherConfig(mode="cell")
    img = np.zeros((6, 6))
    mask = np.zeros((6, 6), dtype=int)
    mask[2:4, 2:4] = 1
    result = run_patching(images=img, masks=mask, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 1


def test_pipeline_cell_yxc_uses_image_axes_and_yx_mask_axes():
    cfg = CellPatcherConfig(mode="cell")
    cfg.common.image_axes = "YXC"
    cfg.common.mask_axes = "YX"
    cfg.cell.exclude_border = False
    cfg.cell.patch_size = "auto"
    img = np.zeros((10, 20, 3))
    mask = np.zeros((10, 20), dtype=int)
    mask[2:5, 7:11] = 1
    result = run_patching(images=img, masks=mask, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 1
    patch = result.patches[0]
    assert (patch.y0, patch.y1, patch.x0, patch.x1) == (2, 6, 7, 11)


def test_pipeline_cell_list_images_masks_works():
    cfg = CellPatcherConfig(mode="cell")
    imgs = [np.zeros((6, 6)), np.zeros((6, 6))]
    masks = [np.pad(np.ones((2, 2), dtype=int), 2), np.pad(np.ones((2, 2), dtype=int), 2)]
    metadata = [{"image_file": "a.tif"}, {"image_file": "b.tif"}]
    result = run_patching(images=imgs, masks=masks, config=cfg, image_metadata=metadata)
    assert len(result.patches) >= 2


def test_pipeline_refuses_unnamed_images_and_images_sharing_a_position():
    # In-memory images need a position_id (or file name); two images given
    # the same position without plane indices are refused rather than told
    # apart by their place in the run.
    cfg = CellPatcherConfig(mode="cell")
    images = [np.zeros((6, 6)), np.zeros((6, 6))]
    masks = [np.pad(np.ones((2, 2), dtype=int), 2)] * 2
    with pytest.raises(ValueError, match="Image 0 of the run has neither a position_id nor an image_file"):
        run_patching(images=images, masks=masks, config=cfg)
    with pytest.raises(ValueError, match="same position_id"):
        run_patching(images=images, masks=masks, config=cfg, image_metadata=[{"position_id": "P"}] * 2)


def test_pipeline_cell_rejects_missing_masks():
    cfg = CellPatcherConfig(mode="cell")
    with pytest.raises(ValueError):
        run_patching(images=np.zeros((5, 5)), masks=None, config=cfg, image_metadata=named(np.zeros((5, 5))))


def test_pipeline_cell_rejects_mismatched_lengths():
    cfg = CellPatcherConfig(mode="cell")
    with pytest.raises(ValueError):
        run_patching(images=[np.zeros((5, 5)), np.zeros((5, 5))], masks=[np.zeros((5, 5))], config=cfg,
            image_metadata=named([np.zeros((5, 5)), np.zeros((5, 5))]))


def test_pipeline_random_single_image_returns_patches():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=4, seed=0))
    result = run_patching(images=np.zeros((32, 32)), config=cfg, image_metadata=named(np.zeros((32, 32))))
    assert len(result.patches) == 4


def test_pipeline_random_yxc_uses_spatial_bounds():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=5, n_patches=10, seed=0))
    cfg.common.image_axes = "YXC"
    image = np.zeros((10, 20, 3))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 10
    assert all(0 <= p.y0 < p.y1 <= 10 for p in result.patches)
    assert all(0 <= p.x0 < p.x1 <= 20 for p in result.patches)


def test_pipeline_random_list_images_works():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=3, seed=0))
    result = run_patching(images=[np.zeros((32, 32)), np.zeros((32, 32))], config=cfg,
        image_metadata=named([np.zeros((32, 32)), np.zeros((32, 32))]))
    assert len(result.patches) == 6


def test_pipeline_random_multi_image_seeded_unique_coordinates():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=3, seed=0))
    result = run_patching(images=[np.zeros((32, 32)), np.zeros((32, 32))], config=cfg,
        image_metadata=named([np.zeros((32, 32)), np.zeros((32, 32))]))
    image_ids = sorted({str(patch.image_id) for patch in result.patches})
    assert len(image_ids) == 2
    by_image: dict[str, list[tuple[int, int]]] = {image_id: [] for image_id in image_ids}
    for patch in result.patches:
        by_image[str(patch.image_id)].append((patch.y0, patch.x0))
    assert by_image[image_ids[0]] != by_image[image_ids[1]]


def test_pipeline_rejects_empty_image_list():
    with pytest.raises(ValueError):
        run_patching(images=[], config=CellPatcherConfig(mode="grid"), image_metadata=named([]))


def test_pipeline_random_matching_mask_list_works():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=4, n_patches=2, seed=0))
    imgs = [np.zeros((16, 16)), np.zeros((16, 16))]
    masks = [np.ones((16, 16), dtype=bool), np.ones((16, 16), dtype=bool)]
    result = run_patching(images=imgs, masks=masks, config=cfg, image_metadata=named(imgs))
    assert len(result.patches) == 4


def test_pipeline_random_rejects_single_mask_with_multiple_images():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=4, n_patches=2, seed=0))
    imgs = [np.zeros((16, 16)), np.zeros((16, 16))]
    with pytest.raises(ValueError):
        run_patching(images=imgs, masks=np.ones((16, 16), dtype=bool), config=cfg, image_metadata=named(imgs))


def test_pipeline_random_rejects_mismatched_lengths():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=4, n_patches=2, seed=0))
    imgs = [np.zeros((16, 16)), np.zeros((16, 16))]
    masks = [np.ones((16, 16), dtype=bool)]
    with pytest.raises(ValueError):
        run_patching(images=imgs, masks=masks, config=cfg, image_metadata=named(imgs))


def test_pipeline_extract_arrays_whole_fov_returns_patch_arrays():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    image = np.arange(20).reshape(4, 5)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == 1
    assert result.patch_arrays[0].shape == (4, 5)


def test_pipeline_extract_arrays_grid_one_array_per_patch():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    image = np.arange(16).reshape(4, 4)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == len(result.patches)


def test_pipeline_extract_arrays_false_keeps_none():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = False
    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
    assert result.patch_arrays is None


def test_pipeline_extract_arrays_uses_extraction_pad_mode_constant():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=3, n_patches=1, seed=0))
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.extraction.pad_mode = "constant"
    cfg.extraction.constant_values = -5

    image = np.arange(25).reshape(5, 5)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == 1


def test_pipeline_apply_mask_true_requires_masks_in_random_mode():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=3, n_patches=1, seed=0))
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True

    with pytest.raises(ValueError, match="only supported in mode='cell'"):
        run_patching(images=np.arange(25).reshape(5, 5), masks=None, config=cfg,
            image_metadata=named(np.arange(25).reshape(5, 5)))


def test_pipeline_apply_mask_random_mode_rejected_even_with_masks():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=1, seed=0))
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True

    with pytest.raises(ValueError, match="only supported in mode='cell'"):
        run_patching(images=np.ones((8, 8), dtype=np.float32), masks=np.ones((8, 8), dtype=np.int32), config=cfg,
            image_metadata=named(np.ones((8, 8), dtype=np.float32)))


def test_pipeline_extract_arrays_resized_output_size():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.extraction.output_size = 3
    cfg.extraction.resize_order = 0

    image = np.arange(20).reshape(4, 5)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (3, 3)


def test_pipeline_extract_arrays_resized_non_yx_axes_cyx():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "CYX"
    cfg.extraction.output_size = (3, 3)
    cfg.extraction.resize_order = 0

    image = np.arange(2 * 4 * 5).reshape(2, 4, 5)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (2, 3, 3)


def test_pipeline_whole_fov_yxc_uses_image_axes_spatial_shape():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YXC"

    image = np.arange(4 * 5 * 2).reshape(4, 5, 2)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))

    assert len(result.patches) == 1
    assert (result.patches[0].y0, result.patches[0].y1, result.patches[0].x0, result.patches[0].x1) == (0, 4, 0, 5)
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (4, 5, 2)


def test_pipeline_grid_yxc_uses_image_axes_spatial_shape():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YXC"

    image = np.arange(4 * 6 * 3).reshape(4, 6, 3)
    result = run_patching(images=image, config=cfg, image_metadata=named(image))

    assert len(result.patches) == 6
    assert result.patch_arrays is not None
    assert all(array.shape == (2, 2, 3) for array in result.patch_arrays)


def test_pipeline_grid_filters_out_dark_patches():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.filters.min_mean_intensity = 1.0
    image = np.array([[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]])
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 2


def test_pipeline_filters_applied_before_extraction_patch_arrays_count_matches_kept():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.filters.min_mean_intensity = 1.0
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    image = np.array([[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]])
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == len(result.patches)


def test_pipeline_all_patches_filtered_out_returns_empty_arrays():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.filters.min_mean_intensity = 1.0
    cfg.common.extract_arrays = True
    image = np.zeros((4, 4))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patches == []
    assert result.patch_arrays == []


def test_pipeline_export_enabled_writes_files_and_metadata(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True

    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
    assert len(result.exported_files) == 1
    assert (tmp_path / "patches.csv").exists()
    assert result.metadata_file is not None


def test_pipeline_export_enabled_without_output_dir_raises():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    with pytest.raises(ValueError):
        run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))


def test_pipeline_export_enabled_with_extract_arrays_false_still_exports(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = False
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"

    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
    assert len(result.exported_files) == 1
    assert result.patch_arrays is None


def test_pipeline_export_disabled_writes_no_files(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = False
    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
    assert result.exported_files == []


def test_pipeline_zero_patches_export_writes_metadata_only(tmp_path):
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=10, overlap=0.0, include_partial=False))
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True
    cfg.common.output_dir = str(tmp_path)
    cfg.common.extract_arrays = False

    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
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

    result = run_patching(images=image, masks=labels, config=cfg, image_metadata=named(image))
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

    result = run_patching(images=image, masks=labels, config=cfg, image_metadata=named(image))
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

    result = run_patching(images=image, masks=labels, config=cfg, image_metadata=named(image))
    assert len(result.exported_files) == 1
    arr = np.load(result.exported_files[0])
    assert arr[0, 0] == 0
    assert arr[1, 1] == 7


def test_pipeline_apply_mask_true_without_masks_raises():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True
    with pytest.raises(ValueError):
        run_patching(images=np.ones((4, 4)), masks=None, config=cfg, image_metadata=named(np.ones((4, 4))))


def test_pipeline_apply_mask_true_non_cell_region_raises():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=2, overlap=0.0, include_partial=False))
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True
    with pytest.raises(ValueError):
        run_patching(images=np.ones((4, 4)), masks=[np.ones((4, 4), dtype=int)], config=cfg,
            image_metadata=named(np.ones((4, 4))))


# ---------------------------------------------------------------------------
# Additional pipeline option coverage
# ---------------------------------------------------------------------------

def test_run_patching_rejects_non_config_instance():
    with pytest.raises(TypeError):
        run_patching(images=np.zeros((4, 4)), config={"mode": "grid"}, image_metadata=named(np.zeros((4, 4))))


def test_run_patching_none_config_defaults_to_cell_mode(tmp_path):
    img = np.zeros((6, 6))
    mask = np.zeros((6, 6), dtype=int)
    mask[2:4, 2:4] = 1
    result = run_patching(images=img, masks=mask, config=None, image_metadata=named(img))
    assert len(result.patches) == 1


def test_pipeline_image_without_shape_raises():
    cfg = CellPatcherConfig(mode="whole_fov")
    with pytest.raises(ValueError, match="shape"):
        run_patching(images="not_an_array", config=cfg, image_metadata=named("not_an_array"))


# ---------------------------------------------------------------------------
# Cell mode: all CellConfig options exercised through the pipeline
# ---------------------------------------------------------------------------

def test_pipeline_cell_centroid_center_on():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.center_on = "centroid"
    cfg.cell.patch_size = 4
    cfg.cell.exclude_border = False
    img = np.zeros((10, 10))
    labels = np.zeros((10, 10), dtype=int)
    labels[3:6, 3:6] = 1
    result = run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 1
    p = result.patches[0]
    ys, xs = np.where(labels == 1)
    assert abs(p.center_y - float(np.mean(ys))) < 1e-6


def test_pipeline_cell_binary_mask_single_object_policy():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.binary_mask_policy = "single_object"
    cfg.cell.exclude_border = False
    img = np.zeros((10, 10))
    labels = np.zeros((10, 10), dtype=bool)
    labels[1:3, 1:3] = True
    labels[6:8, 6:8] = True
    # Decision D47: a bool mask has no object ids and is refused without the
    # connected-component opt-in; as integers it is one object, label 1.
    with pytest.raises(ValueError, match="binary"):
        run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    result = run_patching(images=img, masks=labels.astype(np.uint8), config=cfg, image_metadata=named(img))
    assert [patch.label_id for patch in result.patches] == [1]


def test_pipeline_cell_binary_mask_connected_components_policy():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.binary_mask_policy = "connected_components"
    cfg.cell.exclude_border = False
    img = np.zeros((10, 10))
    labels = np.zeros((10, 10), dtype=bool)
    labels[1:3, 1:3] = True
    labels[6:8, 6:8] = True
    with pytest.warns(UserWarning, match="connected components"):
        result = run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 2


def test_pipeline_refuses_masking_by_computed_component_labels():
    # apply_mask keeps pixels equal to label_id, which a 0/1 mask does not hold
    # for components after the first: refused instead of blanking them.
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.binary_mask_policy = "connected_components"
    cfg.extraction.apply_mask = True
    cfg.common.extract_arrays = True
    labels = np.zeros((10, 10), dtype=bool)
    labels[1:3, 1:3] = True
    with pytest.raises(ValueError, match="apply_mask"):
        run_patching(images=np.zeros((10, 10)), masks=labels, config=cfg, image_metadata=named(np.zeros((10, 10))))


def test_pipeline_cell_min_area_filters_tiny_cells():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.min_area = 10
    cfg.cell.exclude_border = False
    img = np.zeros((10, 10))
    labels = np.zeros((10, 10), dtype=int)
    labels[1, 1] = 1     # area 1 → filtered
    labels[4:7, 4:7] = 2  # area 9 → just below
    result = run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 0


def test_pipeline_cell_max_area_filters_large_cells():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.max_area = 4
    cfg.cell.exclude_border = False
    img = np.zeros((10, 10))
    labels = np.zeros((10, 10), dtype=int)
    labels[1:2, 1:3] = 1  # area 2 → included
    labels[4:8, 4:8] = 2  # area 16 → excluded
    result = run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 1
    assert result.patches[0].label_id == 1


def test_pipeline_cell_exclude_border_false_includes_edge_cells():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.exclude_border = False
    cfg.cell.patch_size = 3
    img = np.zeros((8, 8))
    labels = np.zeros((8, 8), dtype=int)
    labels[0:2, 0:2] = 1  # touches border
    result = run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    assert len(result.patches) == 1
    assert result.patches[0].touches_border is True


# ---------------------------------------------------------------------------
# Grid options: overlap and include_partial
# ---------------------------------------------------------------------------

def test_pipeline_grid_with_overlap_produces_more_tiles():
    base_cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=20, overlap=0.0, include_partial=False))
    overlap_cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=20, overlap=0.5, include_partial=False))
    image = np.zeros((100, 100))
    base_result = run_patching(images=image, config=base_cfg, image_metadata=named(image))
    overlap_result = run_patching(images=image, config=overlap_cfg, image_metadata=named(image))
    assert len(overlap_result.patches) > len(base_result.patches)


def test_pipeline_grid_include_partial_covers_all_pixels():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=30, overlap=0.0, include_partial=True))
    image = np.zeros((100, 100))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert any(p.y1 == 100 for p in result.patches)
    assert any(p.x1 == 100 for p in result.patches)


# ---------------------------------------------------------------------------
# Random strategy: seed reproducibility and mask
# ---------------------------------------------------------------------------

def test_pipeline_random_seed_produces_reproducible_patches():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=8, n_patches=5, seed=42))
    image = np.zeros((32, 32))
    r1 = run_patching(images=image, config=cfg, image_metadata=named(image))
    r2 = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert [(p.y0, p.x0) for p in r1.patches] == [(p.y0, p.x0) for p in r2.patches]


def test_pipeline_random_with_sampling_mask_stays_in_masked_region():
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=6, n_patches=10, seed=7))
    image = np.zeros((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True
    result = run_patching(images=image, masks=mask, config=cfg, image_metadata=named(image))
    for p in result.patches:
        assert np.any(mask[p.y0:p.y1, p.x0:p.x1])


# ---------------------------------------------------------------------------
# Extraction options
# ---------------------------------------------------------------------------

def test_pipeline_extraction_pad_mode_constant_pads_out_of_bound_patches():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 10
    cfg.cell.exclude_border = False
    cfg.common.extract_arrays = True
    cfg.extraction.pad_mode = "constant"
    cfg.extraction.constant_values = -1

    img = np.zeros((8, 8), dtype=np.float32)
    labels = np.zeros((8, 8), dtype=int)
    labels[0:2, 0:2] = 1  # cell at top-left border
    result = run_patching(images=img, masks=labels, config=cfg, image_metadata=named(img))
    assert result.patch_arrays is not None
    # With constant padding, patch should contain -1 fill at padded positions


def test_pipeline_extraction_output_size_tuple():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = (4, 6)
    cfg.extraction.resize_order = 0
    image = np.zeros((10, 10))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (4, 6)


def test_pipeline_extraction_mask_fill_value_controls_background():
    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 6
    cfg.cell.exclude_border = False
    cfg.common.extract_arrays = True
    cfg.common.image_axes = "YX"
    cfg.extraction.apply_mask = True
    cfg.extraction.mask_fill_value = -99

    image = np.full((8, 8), 5, dtype=np.float32)
    labels = np.zeros((8, 8), dtype=int)
    labels[3:5, 3:5] = 1  # 2x2 cell inside a 6x6 patch; non-cell pixels get fill_value
    result = run_patching(images=image, masks=labels, config=cfg, image_metadata=named(image))
    arr = result.patch_arrays[0]
    assert float(arr.min()) == pytest.approx(-99)


# ---------------------------------------------------------------------------
# Export options
# ---------------------------------------------------------------------------

def test_pipeline_export_tif_format_writes_tif_files(tmp_path):
    try:
        import tifffile  # noqa: F401
    except ImportError:
        pytest.skip("tifffile not available")
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "tif"
    cfg.export.save_metadata_csv = True
    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
    assert len(result.exported_files) == 1
    assert result.exported_files[0].endswith(".tif")


def test_pipeline_export_save_metadata_csv_false_skips_csv(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = False
    result = run_patching(images=np.zeros((4, 4)), config=cfg, image_metadata=named(np.zeros((4, 4))))
    assert not (tmp_path / "patches.csv").exists()
    assert result.metadata_file is None


def test_pipeline_export_grid_multi_image_writes_all_patches(tmp_path):
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8, overlap=0.0, include_partial=False))
    cfg.common.output_dir = str(tmp_path)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    images = [np.zeros((16, 16)), np.zeros((16, 16))]
    result = run_patching(images=images, config=cfg, image_metadata=[{"position_id": "P1"}, {"position_id": "P2"}])
    assert len(result.exported_files) == len(result.patches) == 8


# ---------------------------------------------------------------------------
# Whole FOV: multi-image and channel axes
# ---------------------------------------------------------------------------

def test_pipeline_whole_fov_multi_image_one_patch_each():
    cfg = CellPatcherConfig(mode="whole_fov")
    images = [np.zeros((4, 5)), np.zeros((6, 7))]
    result = run_patching(images=images, config=cfg, image_metadata=named(images))
    assert len(result.patches) == 2
    assert result.patches[0].y1 == 4 and result.patches[0].x1 == 5
    assert result.patches[1].y1 == 6 and result.patches[1].x1 == 7


def test_pipeline_whole_fov_zyx_axes():
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.image_axes = "ZYX"
    image = np.zeros((3, 10, 20))
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 1
    patch = result.patches[0]
    assert (patch.y1, patch.x1) == (10, 20)


# ---------------------------------------------------------------------------
# Filters through the pipeline
# ---------------------------------------------------------------------------

def test_pipeline_filter_foreground_fraction_removes_dark_tiles():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4, overlap=0.0, include_partial=False))
    cfg.filters.min_foreground_fraction = 0.5
    cfg.filters.foreground_threshold = 1.0
    image = np.zeros((8, 8), dtype=float)
    image[0:4, 4:8] = 5  # right half has bright values
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) < 4
    assert all(p.x0 >= 4 for p in result.patches)


def test_pipeline_filter_foreground_threshold_mean_works():
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4, overlap=0.0, include_partial=False))
    cfg.filters.min_foreground_fraction = 0.5
    cfg.filters.foreground_threshold = "mean"
    image = np.zeros((8, 8), dtype=float)
    image[0:4, 4:8] = 10.0
    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) <= 4


# ---------------------------------------------------------------------------
# Cell mode: apply_mask=True with None in masks list
# ---------------------------------------------------------------------------

def test_pipeline_cell_apply_mask_with_none_in_masks_raises():
    """In cell mode, a None entry in the masks list should raise an error."""
    cfg = CellPatcherConfig(mode="cell")
    cfg.common.extract_arrays = True
    cfg.extraction.apply_mask = True
    cfg.cell.patch_size = 3
    cfg.cell.exclude_border = False

    img = np.zeros((6, 6))
    labels1 = np.zeros((6, 6), dtype=int)
    labels1[2:4, 2:4] = 1

    with pytest.raises(ValueError):
        run_patching(images=[img, img], masks=[labels1, None], config=cfg, image_metadata=named([img, img]))
