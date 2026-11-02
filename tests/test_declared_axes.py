"""Declared axes against file headers: the preprocessing brick's axis check (design v0.4 §3).

Since CP-2a folder and tree runs declare no axes: they read each file in the roles the
preprocessing run recorded, and refuse ``common.image_axes``/``mask_axes`` other than the
default. Since CP-2b lazy delivery reads no file either (it reads through the run), so
CellPatcher checks no declared axes against a file header: the preprocessing run does. Since
preprocessing v0.4.0 a Y/X swap is refused like any other contradiction: nothing is ever read
transposed. In-memory masks with X before Y are refused.
"""

import csv

import numpy as np
import pytest
import tifffile

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.export import IMAGE_TABLE_FILENAME
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import run_patching
from tests.helpers import (
    file_run_config,
    make_synthetic_cell_image_and_labels,
    named,
    preprocessed_folder,
    write_tiff,
)


def _grid_config(image_axes, output_dir=None):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = image_axes
    config.common.extract_arrays = True
    if output_dir is not None:
        config.export.enabled = True
        config.common.output_dir = str(output_dir)
    return config


def _image_rows(folder):
    with open(folder / IMAGE_TABLE_FILENAME, newline="", encoding="utf-8") as handle:
        return {row["image_file"]: row for row in csv.DictReader(handle)}


def test_y_and_x_declared_swapped_are_refused_never_read_transposed(tmp_path):
    # Before preprocessing v0.4 (CellPatcher v0.1.2), a declaration swapping the header's named
    # Y and X was accepted and the image read transposed (image_read_transposed). Design v0.4
    # §3: nothing is ever read transposed; the swap is an error naming every file.
    image = np.arange(3 * 16 * 24, dtype=np.uint16).reshape(3, 16, 24)  # Z, Y, X in the header
    write_tiff(tmp_path / "exp" / "a.tif", image, "ZYX")
    write_tiff(tmp_path / "exp" / "b.tif", image, "ZYX")
    # A file run (CP-2a) declares nothing: the declaration is refused, and the image is read
    # in the roles the preprocessing run recorded, Y before X. Lazy delivery no longer reads a
    # file with declared axes (CP-2b), so no CellPatcher-side declaration of a file remains.
    config = file_run_config(_grid_config("ZXY", tmp_path / "out"), tmp_path / "exp", "ZYX")
    config.common.z_policy, config.common.z_index = "select", 1
    with pytest.raises(ValueError, match=r"common.image_axes='ZXY': each image's \(and mask's\) axes are the roles"):
        run_patching_pipeline(str(tmp_path / "exp"), config=config)
    assert not (tmp_path / "out").exists()
    config.common.image_axes = "YX"
    result = run_patching_pipeline(str(tmp_path / "exp"), config=config)
    patch = result.patches[0]
    assert np.array_equal(result.patch_arrays[0], image[1, patch.y0:patch.y1, patch.x0:patch.x1])


def test_the_image_table_has_no_transposition_column(tmp_path):
    # images.csv had a last column image_read_transposed (v0.1.2), always false or empty since
    # preprocessing v0.4 refuses every Y/X swap; removed in v0.2.0 (CP-2b).
    write_tiff(tmp_path / "exp" / "a.tif", np.zeros((3, 16, 24), dtype=np.uint8), "ZYX")
    config = _grid_config("YX", tmp_path / "out")
    config.common.z_policy = "all"
    preprocessed_folder(tmp_path / "exp", config=config, axes="ZYX")
    images = [np.zeros((16, 16), dtype=np.uint8)]
    run_patching(images=images, config=_grid_config("YX", tmp_path / "memory"), image_metadata=named(images))
    for out in (tmp_path / "out", tmp_path / "memory"):
        (row,) = _image_rows(out).values()
        assert list(row)[-1] == "n_normalized" and "image_read_transposed" not in row


def test_a_mask_declared_swapped_is_refused(tmp_path):
    # A file run declares no mask axes at all since CP-2a: a declared swap is refused.
    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(96, 96))
    write_tiff(tmp_path / "exp" / "a.tif", image, "YX")
    write_tiff(tmp_path / "masks" / "a_cells.tif", np.ascontiguousarray(labels.T.astype(np.uint16)), "YX")
    config = file_run_config(_cell_config("YX", "XY"), tmp_path / "exp", masks=tmp_path / "masks")
    with pytest.raises(ValueError, match=r"common.mask_axes='XY': each image's \(and mask's\) axes"):
        run_patching_pipeline(str(tmp_path / "exp"), config=config, mask_folder_path=str(tmp_path / "masks"))


@pytest.mark.parametrize("mode", ["cell", "random"])
def test_an_in_memory_mask_with_x_before_y_is_refused(mode):
    # Before v0.1.2, mask_axes='XY' was ignored for an in-memory mask: the strategies took the
    # label plane as Y, X in its stored order, so with image_axes='XY' too the patches missed
    # their objects without a word. A mask is never read transposed: refused, with the fix.
    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    config = _cell_config("XY", "XY")
    config.mode = mode
    with pytest.raises(ValueError, match=r"common.mask_axes='XY' puts X before Y, but CellPatcher cannot read a "
                                         r"mask transposed.*To fix the masks"):
        run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))


def test_a_mask_file_whose_header_names_x_before_y_is_refused(tmp_path):
    # Declared as its header names it, but CellPatcher would still take its first dimension as Y.
    # The preprocessing pairing refuses it first (mask_layout): masks are never transposed.
    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    write_tiff(tmp_path / "exp" / "a.tif", image, "YX")
    (tmp_path / "masks").mkdir()
    tifffile.imwrite(tmp_path / "masks" / "a_cells.tif", labels.astype(np.uint16), metadata={"axes": "XY"})
    # Since CP-2a the preprocessing run pairs it; rc1 records it without axes, and a file run refuses that.
    config = file_run_config(_cell_config("YX", "YX"), tmp_path / "exp", masks=tmp_path / "masks")
    with pytest.raises(ValueError, match=r"recorded no axes for the mask a_cells.tif: its header was not usable"):
        run_patching_pipeline(str(tmp_path / "exp"), config=config, mask_folder_path=str(tmp_path / "masks"))


def test_lazy_delivery_refuses_a_mask_file(tmp_path):
    # Before CP-2b a mask path was read with common.mask_axes and checked against its header by
    # CellPatcher; a file is now read only through the preprocessing run.
    from cellpatcher.delivery import patch_array

    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    write_tiff(tmp_path / "m.tif", labels.astype(np.uint16), "YX")
    config = _cell_config("YX", "YX")
    patch = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image])).patches[0]
    config.extraction.apply_mask = True
    with pytest.raises(TypeError, match=r"takes the masks of run_patching as arrays; got the path .*m.tif"):
        patch_array(patch, image, config, mask=tmp_path / "m.tif")


def _cell_config(image_axes, mask_axes, output_dir=None):
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 6
    config.cell.exclude_border = False
    config.common.image_axes = image_axes
    config.common.mask_axes = mask_axes
    config.common.extract_arrays = True
    if output_dir is not None:
        config.export.enabled = True
        config.common.output_dir = str(output_dir)
    return config
