"""Declared axes against file headers: the preprocessing brick's check_declared_axes (D86, v0.1.2).

CellPatcher keeps no copy of the rule. A declaration may name what a header
leaves open and never contradict it, except that the named Y and X may be
declared swapped (owner decision of 31 October 2026): an image is then read
transposed, and images.csv records it. CellPatcher cannot read a mask
transposed, so a mask declared swapped is refused.
"""

import csv

import numpy as np
import pytest
import tifffile

import cellpatcher.io
from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.export import IMAGE_TABLE_FILENAME
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import run_patching
from tests.helpers import make_synthetic_cell_image_and_labels, named, write_tiff


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


def test_the_shared_rule_is_the_preprocessing_brick_s(tmp_path, monkeypatch):
    import morphobricks_preprocessing

    calls = []
    check = morphobricks_preprocessing.check_declared_axes

    def recording(meta, declared):
        calls.append((meta.relative_path, declared))
        return check(meta, declared)

    monkeypatch.setattr(morphobricks_preprocessing, "check_declared_axes", recording)
    write_tiff(tmp_path / "exp" / "a.tif", np.zeros((3, 16, 16), dtype=np.uint8), "ZYX")
    run_patching_pipeline(str(tmp_path / "exp"), config=_grid_config("ZYX"))
    assert ("a.tif", "ZYX") in calls


def test_y_and_x_declared_swapped_are_read_transposed_and_recorded(tmp_path):
    image = np.arange(3 * 16 * 24, dtype=np.uint16).reshape(3, 16, 24)  # Z, Y, X in the header
    write_tiff(tmp_path / "exp" / "a.tif", image, "ZYX")
    write_tiff(tmp_path / "exp" / "b.tif", image, "ZYX")
    config = _grid_config("ZXY", tmp_path / "out")
    config.common.z_policy = "select"
    config.common.z_index = 1
    with pytest.warns(UserWarning, match=r"common.image_axes declares Y and X swapped .* 2 image file\(s\) "
                                         r"\(a.tif, b.tif\): they are read transposed"):
        result = run_patching_pipeline(str(tmp_path / "exp"), config=config)
    rows = _image_rows(tmp_path / "out")
    assert {name: row["image_read_transposed"] for name, row in rows.items()} == {"a.tif": "true", "b.tif": "true"}
    assert (rows["a.tif"]["image_height"], rows["a.tif"]["image_width"]) == ("24", "16")
    patch = next(p for p in result.patches if p.image_file == "a.tif" and p.y0 == 8 and p.x0 == 0)
    array = result.patch_arrays[result.patches.index(patch)]
    # The rectangle is in the transposed image; the array keeps the declared axis order
    # (delivery.patch_axes: "XY" here), so read as Y, X it is the transposed image's patch.
    from cellpatcher.delivery import patch_axes

    assert patch_axes(config, patch) == "XY"
    np.testing.assert_array_equal(np.transpose(array), image[1].T[8:16, 0:8])


def test_axes_as_named_are_recorded_as_not_transposed(tmp_path):
    write_tiff(tmp_path / "exp" / "a.tif", np.zeros((3, 16, 24), dtype=np.uint8), "ZYX")
    config = _grid_config("ZYX", tmp_path / "out")
    config.common.z_policy = "all"
    run_patching_pipeline(str(tmp_path / "exp"), config=config)
    assert _image_rows(tmp_path / "out")["a.tif"]["image_read_transposed"] == "false"


def test_a_mask_declared_swapped_is_refused(tmp_path):
    # The shared rule accepts a Y/X swap, but CellPatcher takes a label plane's dimensions as
    # Y, X in their stored order, so it cannot read a mask transposed. Before v0.1.2 the swap
    # was accepted and the mask read as stored: on a square image, misaligned without a word.
    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(96, 96))
    write_tiff(tmp_path / "exp" / "a.tif", image, "YX")
    write_tiff(tmp_path / "masks" / "a_cells.tif", np.ascontiguousarray(labels.T.astype(np.uint16)), "YX")
    with pytest.raises(ValueError, match=r"common.mask_axes='XY' declares Y and X swapped relative to the header "
                                         r"of 1 mask file\(s\): a_cells.tif. CellPatcher cannot read a mask transposed"):
        run_patching_pipeline(str(tmp_path / "exp"), config=_cell_config("YX", "XY"),
                              mask_folder_path=str(tmp_path / "masks"))


def test_lazy_delivery_refuses_a_mask_declared_swapped(tmp_path):
    from cellpatcher.delivery import patch_array

    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    write_tiff(tmp_path / "a.tif", image, "YX")
    write_tiff(tmp_path / "m.tif", labels.astype(np.uint16), "YX")
    config = _cell_config("YX", "YX")
    patch = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image])).patches[0]
    config.common.mask_axes = "XY"
    config.extraction.apply_mask = True
    with pytest.raises(ValueError, match="CellPatcher cannot read a mask transposed"):
        patch_array(patch, tmp_path / "a.tif", config, mask=tmp_path / "m.tif")


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


def test_an_in_memory_run_records_no_transposition(tmp_path):
    images = [np.zeros((16, 16), dtype=np.uint8)]
    config = _grid_config("YX", tmp_path / "out")
    run_patching(images=images, config=config, image_metadata=named(images))
    with open(tmp_path / "out" / IMAGE_TABLE_FILENAME, newline="", encoding="utf-8") as handle:
        (row,) = csv.DictReader(handle)
    assert row["image_read_transposed"] == ""


@pytest.mark.parametrize("header, declared, problem", [
    ("ZYX", "YZX", "dimension 0 is"),       # anything but a Y/X swap is refused
    ("ZYX", "TXY", "dimension 0 is"),       # a swap does not excuse another disagreement
    ("ZYX", "YX", "2 declared for 3 dimensions"),
])
def test_other_disagreements_are_refused(tmp_path, header, declared, problem):
    write_tiff(tmp_path / "exp" / "a.tif", np.zeros((3, 16, 16), dtype=np.uint8), header)
    with pytest.raises(ValueError, match=rf"common.image_axes='{declared}'.*a.tif.*{problem}"):
        run_patching_pipeline(str(tmp_path / "exp"), config=_grid_config(declared))


def test_a_plain_stack_may_be_declared_with_its_channels_last(tmp_path):
    # preprocessing v0.3.4: a plain stack's positional Y and X are open, so a (Y, X, C)
    # array written without metadata (tifffile: QYX) may be declared YXC, and is not transposed.
    (tmp_path / "exp").mkdir()
    pixels = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
    tifffile.imwrite(tmp_path / "exp" / "a.tif", pixels, photometric="minisblack", planarconfig="contig", metadata=None)
    problem, transposed = cellpatcher.io.header_axes_disagreement(tmp_path / "exp" / "a.tif", "YXC")
    assert (problem, transposed) == (None, False)
