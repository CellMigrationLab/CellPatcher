"""``PatchRegion.source_relative_path``: the image's real path under the experiment root.

See docs/normalization_contract.md §2 and §10.
"""

import csv
from dataclasses import fields

import numpy as np

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.pipeline import run_patching
from cellpatcher.types import PatchRegion

# patches.csv header before source_relative_path was added; existing columns
# must keep these names and this order.
_EARLIER_HEADER = [
    "patch_id", "image_id", "mode", "condition", "biological_repeat", "technical_repeat",
    "position_id", "image_file", "mask_file", "y0", "y1", "x0", "x1", "height", "width",
    "z0", "z1", "t", "c", "label_id", "center_y", "center_x", "center_z", "touches_border",
    "area_px", "patch_file", "output_height", "output_width", "config_file", "valid", "reason",
]


def _write_image(path, value=1.0, shape=(16, 16)):
    write_array(path, np.full(shape, value, dtype=np.float32))


def _grid_config(output_dir=None):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    if output_dir is not None:
        config.common.output_dir = str(output_dir)
        config.export.enabled = True
    return config


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def test_field_is_appended_after_all_existing_fields():
    names = [f.name for f in fields(PatchRegion)]
    assert names[-1] == "source_relative_path"
    assert PatchRegion(patch_id="p", y1=1, x1=1).source_relative_path is None


def test_csv_appends_the_column_and_keeps_earlier_columns(tmp_path):
    _write_image(tmp_path / "images" / "a.tif")
    run_patching_pipeline(tmp_path / "images", config=_grid_config(tmp_path / "out"))

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert header == [*_EARLIER_HEADER, "source_relative_path"]
    assert {row["source_relative_path"] for row in rows} == {"a.tif"}


def test_flat_folder_run_uses_the_file_name(tmp_path):
    _write_image(tmp_path / "images" / "b.tif")
    _write_image(tmp_path / "images" / "a.tif")

    result = run_patching_pipeline(tmp_path / "images", config=_grid_config())
    assert {patch.source_relative_path for patch in result.patches} == {"a.tif", "b.tif"}
    for patch in result.patches:
        assert patch.source_relative_path == patch.image_file


def test_tree_run_uses_the_real_location_not_the_output_folder(tmp_path):
    root = tmp_path / "experiment"
    _write_image(root / "control" / "img.tif")
    _write_image(root / "treated" / "2026_01_01" / "img.tif")
    _write_image(root / "treated" / "2026_01_02" / "B02" / "field.tif")
    _write_image(root / "treated" / "2026_01_02" / "B03" / "field.tif")

    results = run_patching_condition_repeat_tree(root, config=_grid_config())

    by_folder = {
        leaf.relative_folder: {patch.source_relative_path for patch in leaf.result.patches}
        for leaf in results
    }
    assert by_folder == {
        # Output folder control/single, but the image sits directly in control/.
        "control/single": {"control/img.tif"},
        "treated/2026_01_01": {"treated/2026_01_01/img.tif"},
        # Same file name in two technical repeats: different paths.
        "treated/2026_01_02/B02": {"treated/2026_01_02/B02/field.tif"},
        "treated/2026_01_02/B03": {"treated/2026_01_02/B03/field.tif"},
    }


def test_flat_root_tree_run_uses_the_file_name(tmp_path):
    root = tmp_path / "experiment"
    _write_image(root / "a.tif")

    results = run_patching_condition_repeat_tree(root, config=_grid_config())
    assert {patch.source_relative_path for patch in results[0].result.patches} == {"a.tif"}


def test_combined_tree_csv_records_the_column(tmp_path):
    root = tmp_path / "experiment"
    _write_image(root / "control" / "img.tif")
    _write_image(root / "treated" / "2026_01_01" / "img.tif")

    run_patching_condition_repeat_tree(root, config=_grid_config(tmp_path / "out"), combine_metadata=True)

    _, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert {row["source_relative_path"] for row in rows} == {"control/img.tif", "treated/2026_01_01/img.tif"}


def test_run_patching_takes_it_from_image_metadata():
    images = [np.ones((16, 16), dtype=np.float32)] * 2
    metadata = [{"source_relative_path": "control/a.tif"}, {}]

    result = run_patching(images=images, config=_grid_config(), image_metadata=metadata)
    paths = {patch.image_id: patch.source_relative_path for patch in result.patches}
    assert paths == {"image_0000": "control/a.tif", "image_0001": None}


def test_run_patching_without_metadata_leaves_it_empty():
    result = run_patching(images=[np.ones((16, 16), dtype=np.float32)], config=_grid_config())
    assert all(patch.source_relative_path is None for patch in result.patches)


def test_split_channel_patches_keep_it(tmp_path):
    root = tmp_path / "experiment"
    write_array(root / "control" / "rgb.tif", np.ones((2, 16, 16), dtype=np.float32))
    config = _grid_config()
    config.common.image_axes = "CYX"
    config.extraction.split_channels = True

    results = run_patching_condition_repeat_tree(root, config=config)
    patches = results[0].result.patches
    assert {patch.c for patch in patches} == {0, 1}
    assert {patch.source_relative_path for patch in patches} == {"control/rgb.tif"}
