import csv
import json

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.io import write_array
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.types import PatchRegion
from tests.helpers import make_synthetic_cell_image_and_labels
from tests.helpers import named


def test_to_dict_includes_height_and_width():
    patch = PatchRegion(patch_id="p1", y0=2, y1=10, x0=3, x1=9)
    row = patch.to_dict()
    assert row["height"] == 8
    assert row["width"] == 6


def test_condition_and_repeat_stamped_on_every_patch():
    image = np.ones((16, 16), dtype=np.float32)
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.condition = "treated"
    cfg.common.biological_repeat = "2026_01_01"

    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    assert len(result.patches) == 1
    assert result.patches[0].condition == "treated"
    assert result.patches[0].biological_repeat == "2026_01_01"


def test_missing_labels_default_to_single():
    image = np.ones((16, 16), dtype=np.float32)
    result = run_patching(images=image, config=CellPatcherConfig(mode="whole_fov"), image_metadata=named(image))
    patch = result.patches[0]
    assert (patch.condition, patch.biological_repeat, patch.technical_repeat) == ("single", "single", "single")

    # A label set to None (or null in an older run_config.json) is "single" too.
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.condition, cfg.common.technical_repeat = "treated", None
    patch = run_patching(images=image, config=cfg, image_metadata=named(image)).patches[0]
    assert (patch.condition, patch.biological_repeat, patch.technical_repeat) == ("treated", "single", "single")


def test_output_height_width_reflect_resize_not_raw_bbox():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 10.0
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8, overlap=0.0, include_partial=False))
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = 16
    cfg.extraction.resize_order = 0

    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    patch = result.patches[0]
    assert patch.height == 8 and patch.width == 8
    assert patch.output_height == 16 and patch.output_width == 16


def test_output_height_width_none_when_arrays_not_extracted():
    image = np.ones((16, 16), dtype=np.float32)
    result = run_patching(images=image, config=CellPatcherConfig(mode="whole_fov"), image_metadata=named(image))
    assert result.patches[0].output_height is None
    assert result.patches[0].output_width is None


def test_config_file_written_and_referenced_when_export_enabled(tmp_path):
    image = np.ones((16, 16), dtype=np.float32)
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path)

    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    config_path = tmp_path / RUN_CONFIG_FILENAME
    assert config_path.exists()
    assert result.patches[0].config_file == RUN_CONFIG_FILENAME

    with config_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["mode"] == "whole_fov"
    assert saved["common"]["output_dir"] == str(tmp_path)


def test_config_file_captures_seed_for_deterministic_reconstruction(tmp_path):
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    cfg = CellPatcherConfig(mode="random")
    cfg.random.seed = 42
    cfg.random.n_patches = 3
    cfg.random.patch_size = 8
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path)

    run_patching(images=image, config=cfg, image_metadata=named(image))
    with (tmp_path / RUN_CONFIG_FILENAME).open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["random"]["seed"] == 42


def test_patch_file_matches_actual_exported_filename(tmp_path):
    image = np.ones((16, 16), dtype=np.float32)
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path)

    result = run_patching(images=image, config=cfg, image_metadata=named(image))
    patch = result.patches[0]
    assert patch.patch_file is not None
    assert (tmp_path / patch.patch_file).exists()
    assert str(tmp_path / patch.patch_file) in result.exported_files


def test_exported_csv_includes_new_provenance_columns(tmp_path):
    image = np.ones((16, 16), dtype=np.float32)
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.condition = "control"
    cfg.common.biological_repeat = "2026_02_02"
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path)

    run_patching(images=image, config=cfg, image_metadata=named(image))
    with (tmp_path / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    for col in ("condition", "biological_repeat", "image_file", "mask_file", "patch_file", "config_file", "height", "width"):
        assert col in row
    assert row["condition"] == "control"
    assert row["biological_repeat"] == "2026_02_02"
    assert row["patch_file"]
    assert row["config_file"] == RUN_CONFIG_FILENAME


def test_image_metadata_stamps_plane_t_and_z():
    images = [np.ones((8, 8), dtype=np.float32), np.ones((8, 8), dtype=np.float32)]
    # One position seen at two planes: the plane indices tell their patches apart.
    image_metadata = [{"t": 0, "z": 2, "position_id": "P"}, {"t": 1, "z": 5, "position_id": "P"}]

    result = run_patching(images=images, config=CellPatcherConfig(mode="whole_fov"), image_metadata=image_metadata)
    assert len(result.patches) == 2
    first, second = result.patches
    assert first.t == 0 and first.z0 == 2 and first.z1 == 3
    assert second.t == 1 and second.z0 == 5 and second.z1 == 6


def test_image_metadata_length_mismatch_raises():
    images = [np.ones((8, 8), dtype=np.float32), np.ones((8, 8), dtype=np.float32)]
    with pytest.raises(ValueError, match="image_metadata"):
        run_patching(images=images, config=CellPatcherConfig(mode="whole_fov"), image_metadata=[{"t": 0}])


def test_run_patching_pipeline_stamps_image_and_mask_file_before_export(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=3)
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "sample001.tif", image)
    write_array(mask_dir / "sample001_mask.tif", labels)

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 24
    cfg.cell.exclude_border = True
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    result = run_patching_pipeline(str(img_dir), mask_folder_path=str(mask_dir), config=cfg)
    assert len(result.patches) > 0
    for patch in result.patches:
        assert patch.image_file == "sample001.tif"
        assert patch.mask_file == "sample001_mask.tif"

    # The CSV on disk must already carry these values (not just the in-memory
    # PatchRegion objects) -- this is the ordering guarantee that motivated
    # threading image_metadata into run_patching instead of stamping after.
    with (tmp_path / "out" / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(row["image_file"] == "sample001.tif" for row in rows)
    assert all(row["mask_file"] == "sample001_mask.tif" for row in rows)


def test_run_patching_pipeline_grid_mode_stamps_image_file_only(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "tile_source.tif", np.ones((16, 16), dtype=np.float32))

    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    result = run_patching_pipeline(str(img_dir), config=cfg)
    assert len(result.patches) > 0
    assert all(p.image_file == "tile_source.tif" for p in result.patches)
    assert all(p.mask_file is None for p in result.patches)


def test_image_metadata_stamps_position_id_across_planes():
    images = [
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=np.float32),
    ]
    image_metadata = [
        {"position_id": "P1", "z": 0, "image_file": "P1_z000.tif"},
        {"position_id": "P1", "z": 1, "image_file": "P1_z001.tif"},
    ]
    cfg = CellPatcherConfig(mode="whole_fov")
    result = run_patching(images=images, config=cfg, image_metadata=image_metadata)

    assert [patch.position_id for patch in result.patches] == ["P1", "P1"]
    assert [patch.z0 for patch in result.patches] == [0, 1]
    assert [patch.image_file for patch in result.patches] == [
        "P1_z000.tif",
        "P1_z001.tif",
    ]
    rows = [patch.to_dict() for patch in result.patches]
    assert [row["position_id"] for row in rows] == ["P1", "P1"]


def test_technical_repeat_from_config_and_per_image_metadata():
    images = [np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32)]
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.condition, cfg.common.biological_repeat = "control", "2026_02_02"
    cfg.common.technical_repeat = "dish_1"
    result = run_patching(images=images, config=cfg, image_metadata=[{"image_file": "a.tif"}, {"image_file": "b.tif"}])
    assert {p.technical_repeat for p in result.patches} == {"dish_1"}
    # Per image, the metadata wins: one biological-repeat folder can hold several wells.
    result = run_patching(
        images=images,
        config=cfg,
        image_metadata=[{"technical_repeat": "B02", "image_file": "a.tif"}, {"image_file": "a.tif"}],
    )
    # The same file name in two wells is two positions: one position_id (the image name, D85),
    # two technical repeats, two patch_ids.
    assert [p.position_id for p in result.patches] == ["a", "a"]
    assert len({p.patch_id for p in result.patches}) == len(result.patches)
    assert [p.technical_repeat for p in result.patches] == ["B02", "dish_1"]
    # Unset, it is "single", as condition and biological_repeat are.
    unset = run_patching(images=images[0], config=CellPatcherConfig(mode="whole_fov"), image_metadata=named(images[0]))
    assert unset.patches[0].technical_repeat == "single"
