import csv

import numpy as np

from cellpatcher.cell_scan import scan_cell_patch_size
from cellpatcher.config import CellPatcherConfig, GridConfig, RandomConfig
from cellpatcher.io import read_array, validate_image_and_mask_folders, write_array
from cellpatcher.pipeline import run_patching
from tests.helpers import make_synthetic_cell_image_and_labels
from tests.helpers import named


def test_e2e_cell_scan_mask_resize_export(tmp_path):
    image, labels, metadata = make_synthetic_cell_image_and_labels(seed=1)

    summary = scan_cell_patch_size(
        image_shapes=image.shape,
        labels_list=labels,
    )
    assert summary.n_included == len(metadata["expected_valid_labels"]) + len(metadata["expected_tiny_labels"])
    assert summary.n_skipped_border == len(metadata["expected_border_labels"])

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = summary.recommended_size("max")
    cfg.cell.exclude_border = True
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.extraction.apply_mask = True
    cfg.extraction.mask_fill_value = 0
    cfg.extraction.output_size = 32
    cfg.extraction.resize_order = 0
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True
    cfg.common.output_dir = str(tmp_path)
    cfg.common.extract_arrays = False

    result = run_patching(images=image, masks=labels, config=cfg, image_metadata=named(image))

    assert cfg.cell.patch_size == summary.recommended_size("max")
    assert len(result.exported_files) == summary.n_included
    csv_path = tmp_path / "patches.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == summary.n_included

    arr = np.load(result.exported_files[0])
    assert arr.shape == (32, 32)
    assert np.any(arr == 0)
    assert np.any(arr != 0)


def test_e2e_grid_filter_resize_export(tmp_path):
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 10.0

    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8, overlap=0.0, include_partial=False))
    cfg.filters.min_mean_intensity = 1.0
    cfg.common.image_axes = "YX"
    cfg.extraction.output_size = 16
    cfg.extraction.resize_order = 0
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path)

    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) < 16
    assert len(result.exported_files) == len(result.patches)
    loaded = np.load(result.exported_files[0])
    assert loaded.shape == (16, 16)


def test_e2e_random_mask_resize_export(tmp_path):
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    sampling_mask = np.zeros((64, 64), dtype=bool)
    sampling_mask[16:48, 16:48] = True

    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=10, n_patches=5, seed=11))
    cfg.common.image_axes = "YX"
    cfg.extraction.output_size = 16
    cfg.extraction.resize_order = 0
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path)

    result = run_patching(images=image, masks=sampling_mask, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 5
    assert len(result.exported_files) == 5
    assert all(np.load(path).shape == (16, 16) for path in result.exported_files)


def test_e2e_whole_fov_export(tmp_path):
    image = np.ones((12, 20), dtype=np.float32)
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True
    cfg.common.output_dir = str(tmp_path)

    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 1
    assert len(result.exported_files) == 1
    assert (tmp_path / "patches.csv").exists()
    assert result.metadata_file == str(tmp_path / "patches.csv")

    with (tmp_path / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "whole_fov"
    assert row["image_id"] == "image_0000"
    assert int(row["y0"]) == 0
    assert int(row["x0"]) == 0
    assert int(row["y1"]) == image.shape[0]
    assert int(row["x1"]) == image.shape[1]


def test_e2e_yxc_cell_pipeline():
    y, x, c = 20, 24, 3
    image = np.zeros((y, x, c), dtype=np.float32)
    image[5:11, 8:14, :] = 2.0
    labels = np.zeros((y, x), dtype=np.int32)
    labels[5:11, 8:14] = 1

    cfg = CellPatcherConfig(mode="cell")
    cfg.common.image_axes = "YXC"
    cfg.common.mask_axes = "YX"
    cfg.common.extract_arrays = True
    cfg.cell.patch_size = 6
    cfg.extraction.output_size = 12
    cfg.extraction.resize_order = 0

    result = run_patching(images=image, masks=labels, config=cfg, image_metadata=named(image))
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == 1
    assert result.patch_arrays[0].shape == (12, 12, 3)


def test_e2e_folder_matching_to_cell_pipeline(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=2)
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    img_path = img_dir / "image001.npy"
    mask_path = mask_dir / "image001_mask.npy"
    write_array(img_path, image)
    write_array(mask_path, labels)

    matched = validate_image_and_mask_folders(img_dir, mask_dir)
    assert len(matched) == 1

    image_arr = read_array(next(iter(matched.keys())))
    mask_arr = read_array(next(iter(matched.values())))

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 24
    cfg.cell.exclude_border = True
    cfg.common.extract_arrays = True

    result = run_patching(images=image_arr, masks=mask_arr, config=cfg, image_metadata=named(image_arr))
    assert len(result.patches) > 0
    assert result.patch_arrays is not None

    # confirm metadata remains available when exporting afterwards
    cfg.common.extract_arrays = False
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True
    cfg.common.output_dir = str(tmp_path / "out")
    result2 = run_patching(images=image_arr, masks=mask_arr, config=cfg, image_metadata=named(image_arr))
    assert (tmp_path / "out" / "patches.csv").exists()
    with (tmp_path / "out" / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result2.patches)
