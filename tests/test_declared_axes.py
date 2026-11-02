"""Declared axes against file headers: the preprocessing brick's axis check (design v0.4 §3).

CellPatcher keeps no copy of the rule: the declared axes are checked as the
file's roles with the brick's ``review_axes`` (``check_mapping`` and
``header_axis_issue``). A
declaration may name what a header leaves open and never contradict it. Since
preprocessing v0.4.0 a Y/X swap is refused like any other contradiction:
nothing is ever read transposed (``check_declared_axes`` and
``declared_axes_transposed`` were removed).
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
    check = morphobricks_preprocessing.review_axes

    def recording(files, mapping):
        calls.extend((meta.relative_path, mapping.roles_string) for meta in files)
        return check(files, mapping)

    monkeypatch.setattr(morphobricks_preprocessing, "review_axes", recording)
    write_tiff(tmp_path / "exp" / "a.tif", np.zeros((3, 16, 16), dtype=np.uint8), "ZYX")
    run_patching_pipeline(str(tmp_path / "exp"), config=_grid_config("ZYX"))
    assert ("a.tif", "ZYX") in calls


def test_y_and_x_declared_swapped_are_refused_never_read_transposed(tmp_path):
    # Before preprocessing v0.4 (CellPatcher v0.1.2), a declaration swapping the header's named
    # Y and X was accepted and the image read transposed (image_read_transposed). Design v0.4
    # §3: nothing is ever read transposed; the swap is an error naming every file.
    image = np.arange(3 * 16 * 24, dtype=np.uint16).reshape(3, 16, 24)  # Z, Y, X in the header
    write_tiff(tmp_path / "exp" / "a.tif", image, "ZYX")
    write_tiff(tmp_path / "exp" / "b.tif", image, "ZYX")
    config = _grid_config("ZXY", tmp_path / "out")
    config.common.z_policy = "select"
    config.common.z_index = 1
    with pytest.raises(ValueError, match=r"common.image_axes='ZXY' does not fit the header of 2 image file\(s\): "
                                         r"a.tif: \[axes_yx_swapped\].* b.tif: \[axes_yx_swapped\]") as error_info:
        run_patching_pipeline(str(tmp_path / "exp"), config=config)
    assert "nothing is read transposed" in str(error_info.value)
    assert not (tmp_path / "out").exists()


def test_axes_as_named_are_recorded_as_not_transposed(tmp_path):
    write_tiff(tmp_path / "exp" / "a.tif", np.zeros((3, 16, 24), dtype=np.uint8), "ZYX")
    config = _grid_config("ZYX", tmp_path / "out")
    config.common.z_policy = "all"
    run_patching_pipeline(str(tmp_path / "exp"), config=config)
    assert _image_rows(tmp_path / "out")["a.tif"]["image_read_transposed"] == "false"


def test_a_mask_declared_swapped_is_refused(tmp_path):
    # A mask is checked with the same rule as an image: a Y/X swap is refused.
    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(96, 96))
    write_tiff(tmp_path / "exp" / "a.tif", image, "YX")
    write_tiff(tmp_path / "masks" / "a_cells.tif", np.ascontiguousarray(labels.T.astype(np.uint16)), "YX")
    with pytest.raises(ValueError, match=r"common.mask_axes='XY' does not fit the header of 1 mask file\(s\): "
                                         r"a_cells.tif: \[axes_yx_swapped\]"):
        run_patching_pipeline(str(tmp_path / "exp"), config=_cell_config("YX", "XY"),
                              mask_folder_path=str(tmp_path / "masks"))


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
    with pytest.raises(ValueError, match=r"\[mask_layout\].*Masks are never transposed"):
        run_patching_pipeline(str(tmp_path / "exp"), config=_cell_config("YX", "XY"),
                              mask_folder_path=str(tmp_path / "masks"))


def test_a_swap_the_preprocessing_run_did_not_declare_is_refused_with_its_statistics(tmp_path):
    # With a CSV normalization scope, the preprocessing brick's check_sources compares
    # common.image_axes with the roles the preprocessing run recorded for each file. A Y/X
    # swap CellPatcher accepts against the header is refused there unless that run declared it
    # too: the statistics were computed with the other roles (kept as is, owner decision of
    # 31 October 2026).
    from cellpatcher.orchestration import run_patching_condition_repeat_tree
    from cellpatcher.statistics import PreprocessingCheckError
    from tests.test_normalization import _config, _preprocess, _write_tiff

    root = tmp_path / "experiment"
    rng = np.random.default_rng(0)
    for path in ("control/bio1/a.tif", "treated/bio1/b.tif"):
        _write_tiff(root / path, rng.integers(0, 200, (2, 16, 16), dtype=np.uint8), "ZYX")
    csv_path = _preprocess(root, tmp_path / "preprocessing", "ZYX")
    with pytest.raises(PreprocessingCheckError) as error_info:
        run_patching_condition_repeat_tree(root, config=_config(csv_path, image_axes="ZXY", scope="image"))
    codes = {issue.code for issue in error_info.value.issues}
    assert codes == {"axis_mapping_mismatch"}
    assert "given ZXY, preprocessing ZYX" in str(error_info.value)


def test_lazy_delivery_refuses_a_mask_declared_swapped(tmp_path):
    from cellpatcher.delivery import patch_array

    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    write_tiff(tmp_path / "a.tif", image, "YX")
    write_tiff(tmp_path / "m.tif", labels.astype(np.uint16), "YX")
    config = _cell_config("YX", "YX")
    patch = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image])).patches[0]
    config.common.mask_axes = "XY"
    config.extraction.apply_mask = True
    with pytest.raises(ValueError, match=r"m.tif: \[axes_yx_swapped\]"):
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
    ("ZYX", "YZX", r"\[axes_contradict_header\]"),
    ("ZYX", "TXY", r"\[axes_contradict_header\]"),  # a swap does not excuse another disagreement
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
