import csv
from dataclasses import fields

import numpy as np
import pytest

from cellpatcher.config import CellConfig, CellPatcherConfig, CommonConfig, ExportConfig, ExtractionConfig, GridConfig
from cellpatcher.export import save_metadata_csv
from cellpatcher.extract import extract_patch
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import _padding, run_patching
from cellpatcher.types import PatchRegion

from tests.helpers import make_synthetic_cell_image_and_labels
from tests.helpers import named

GEOMETRY_COLUMNS = [
    "image_height", "image_width", "pad_top", "pad_bottom", "pad_left", "pad_right", "resized", "output_axes",
]


def _stack(n_channels=2, shape=(40, 48)):
    # Channel k holds k * 1000 plus the pixel's flat index, so every crop is identifiable.
    plane = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    return np.stack([plane + 1000 * channel for channel in range(n_channels)])


@pytest.mark.parametrize("mode", ["cell", "grid", "random", "whole_fov"])
def test_image_size_is_recorded_in_every_mode_without_arrays(mode):
    image, labels, _ = make_synthetic_cell_image_and_labels()
    config = CellPatcherConfig(mode=mode, grid=GridConfig(tile_size=32))
    config.random.patch_size = 16
    config.random.n_patches = 3
    masks = [labels] if mode in ("cell", "random") else None
    result = run_patching(images=[image], masks=masks, config=config, image_metadata=named([image]))

    assert result.patches
    assert {(patch.image_height, patch.image_width) for patch in result.patches} == {image.shape}
    # Without arrays nothing describes an array, as for output_height/output_width.
    for patch in result.patches:
        assert patch.output_height is None
        assert (patch.pad_top, patch.pad_bottom, patch.pad_left, patch.pad_right) == (None, None, None, None)
        assert patch.resized is None and patch.output_axes is None


def test_exported_grid_patches_record_axes_padding_and_no_resize(tmp_path):
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="CYX", output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )
    result = run_patching(images=[_stack()], config=config, image_metadata=named([_stack()]))

    with (tmp_path / "patches.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # object_set (decision D44) was appended after the geometry columns.
    assert reader.fieldnames[-len(GEOMETRY_COLUMNS) - 1:] == [*GEOMETRY_COLUMNS, "object_set"]
    assert len(rows) == len(result.patches) == 6
    for row in rows:
        assert (row["image_height"], row["image_width"]) == ("40", "48")
        assert (row["pad_top"], row["pad_bottom"], row["pad_left"], row["pad_right"]) == ("0", "0", "0", "0")
        assert row["resized"] == "False"
        assert row["output_axes"] == "CYX"
        array = np.load(tmp_path / row["patch_file"])
        assert array.shape == (2, int(row["output_height"]), int(row["output_width"]))


def test_earlier_columns_keep_their_order():
    # normalization_parameters, the last field, is written as per-channel columns, not as itself.
    header = [f.name for f in fields(PatchRegion)][:-1]
    assert fields(PatchRegion)[-1].name == "normalization_parameters"
    assert header.index("normalization_exact") + 1 == header.index("image_height")
    assert header[-len(GEOMETRY_COLUMNS) - 1:] == [*GEOMETRY_COLUMNS, "object_set"]


def test_resized_is_true_only_where_resizing_changed_the_shape():
    image, labels, _ = make_synthetic_cell_image_and_labels()
    config = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(extract_arrays=True),
        cell=CellConfig(patch_size=20, exclude_border=False),
        extraction=ExtractionConfig(output_size=20),
    )
    resized = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))
    config.extraction.output_size = None
    plain = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))

    flags = {patch.label_id: patch.resized for patch in resized.patches}
    for patch, array, crop in zip(resized.patches, resized.patch_arrays, plain.patch_arrays):
        clipped = (patch.height, patch.width) != (20, 20)  # cut short by the image border
        assert patch.resized is clipped
        if not clipped:
            # A zoom by 1 leaves the pixels exactly as cropped.
            assert np.array_equal(array, crop)
    assert True in flags.values() and False in flags.values()
    assert all(patch.resized is False for patch in plain.patches)


def test_split_channel_patches_have_no_channel_axis():
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="CYX", extract_arrays=True),
        grid=GridConfig(tile_size=16),
        extraction=ExtractionConfig(split_channels=True),
    )
    result = run_patching(images=[_stack(n_channels=3)], config=config, image_metadata=named([_stack(n_channels=3)]))

    assert {patch.output_axes for patch in result.patches} == {"YX"}
    assert {patch.c for patch in result.patches} == {0, 1, 2}
    assert all(array.ndim == 2 for array in result.patch_arrays)


@pytest.mark.parametrize(
    "z_policy, z_index, expected_axes",
    [("keep", None, "ZCYX"), ("select", 1, "CYX"), ("max", None, "CYX"), ("all", None, "CYX")],
)
def test_output_axes_follow_the_plane_policy(z_policy, z_index, expected_axes):
    stack = np.stack([_stack(), _stack() + 1])  # ZCYX
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="ZCYX", extract_arrays=True, z_policy=z_policy, z_index=z_index),
        grid=GridConfig(tile_size=16),
    )
    result = run_patching(images=[stack], config=config, image_metadata=named([stack]))

    assert {patch.output_axes for patch in result.patches} == {expected_axes}
    assert {(patch.image_height, patch.image_width) for patch in result.patches} == {(40, 48)}
    for patch, array in zip(result.patches, result.patch_arrays):
        assert array.ndim == len(patch.output_axes)


@pytest.mark.parametrize("pad_mode", ["constant", "clip"])
def test_padding_matches_what_extraction_pads(pad_mode):
    image = np.arange(1, 10 * 12 + 1, dtype=np.uint16).reshape(10, 12)  # no zero inside the image
    region = PatchRegion(patch_id="p", y0=-2, y1=6, x0=9, x1=15, image_height=10, image_width=12)
    array = extract_patch(image, region, pad_mode=pad_mode, constant_values=0)
    top, bottom, left, right = _padding(region, pad_mode)

    if pad_mode == "clip":
        assert (top, bottom, left, right) == (0, 0, 0, 0)
        assert array.shape == (6, 3)
        return
    assert (top, bottom, left, right) == (2, 0, 0, 3)
    assert array.shape == (region.height, region.width)
    inside = array[top : array.shape[0] - bottom, left : array.shape[1] - right]
    assert np.array_equal(inside, image[0:6, 9:12])
    assert np.count_nonzero(array == 0) == array.size - inside.size


def test_combined_tree_csv_records_each_image_size(tmp_path):
    for well, shape in (("A01", (32, 32)), ("A02", (32, 48))):
        write_array(tmp_path / "raw" / "control" / "2024_01_01" / well / "img.tif", _stack(shape=shape))
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="CYX", output_dir=str(tmp_path / "out")),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True),
    )
    run_patching_condition_repeat_tree(str(tmp_path / "raw"), config=config, combine_metadata=True)

    with (tmp_path / "out" / "patches.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sizes = {row["technical_repeat"]: (row["image_height"], row["image_width"]) for row in rows}
    assert sizes == {"A01": ("32", "32"), "A02": ("32", "48")}
    assert {row["output_axes"] for row in rows} == {"CYX"}


def test_csv_without_arrays_leaves_the_array_columns_empty(tmp_path):
    region = PatchRegion(patch_id="p", y0=0, y1=4, x0=0, x1=4, image_height=8, image_width=8)
    save_metadata_csv([region], tmp_path / "patches.csv")
    with (tmp_path / "patches.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert (row["image_height"], row["image_width"]) == ("8", "8")
    assert all(row[name] == "" for name in GEOMETRY_COLUMNS[2:])
