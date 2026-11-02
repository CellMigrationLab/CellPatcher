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
    # (Since v0.2.0 the image is refused too: test_an_in_memory_image_with_x_before_y_is_refused.)
    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    config = _cell_config("YX", "XY")
    config.mode = mode
    with pytest.raises(ValueError, match=r"common.mask_axes='XY' puts X before Y, but CellPatcher cannot read a "
                                         r"mask transposed.*To fix the masks"):
        run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))


@pytest.mark.parametrize("mode", ["grid", "whole_fov", "random", "cell"])
@pytest.mark.parametrize(("axes", "shape"), [("XY", (12, 20)), ("CXY", (2, 12, 20)), ("XYZ", (12, 20, 3))])
def test_an_in_memory_image_with_x_before_y_is_refused(mode, axes, shape):
    # Before v0.2.0 an in-memory image declared 'XY' was patched in the transposed frame: a
    # 12 x 20 array gave the whole-FOV patch y0..y1 = 0..20 and x0..x1 = 0..12, so patches.csv
    # held the coordinates of the transposed image (CP-2b review follow-up; design v0.4 §6:
    # in-memory arrays take a canonical-only layout, never swapped).
    image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    config = CellPatcherConfig(mode=mode)
    config.common.image_axes = axes
    config.cell.patch_size = 4
    config.random.patch_size = 4
    config.grid.tile_size = 4
    masks = [np.ones(shape[-2:] if axes != "XYZ" else shape[:2], dtype=np.int32)] if mode == "cell" else None
    with pytest.raises(ValueError, match=rf"common.image_axes='{axes}' puts X before Y, but CellPatcher takes "
                                         r"in-memory images only with Y before X .*Nothing is ever read transposed"):
        run_patching(images=[image], masks=masks, config=config, image_metadata=named([image]))


def test_an_in_memory_image_with_y_before_x_is_patched_as_given():
    # The canonical layout is unchanged: a 12 x 20 'YX' array gives y0..y1 = 0..12, x0..x1 = 0..20.
    image = np.zeros((12, 20), dtype=np.uint16)
    config = CellPatcherConfig(mode="whole_fov")
    (patch,) = run_patching(images=[image], config=config, image_metadata=named([image])).patches
    assert (patch.y0, patch.y1, patch.x0, patch.x1) == (0, 12, 0, 20)


def test_lazy_delivery_refuses_an_in_memory_image_declared_swapped():
    # Lazy delivery takes the same canonical layout as run_patching.
    from cellpatcher.delivery import patch_array

    image = np.zeros((12, 20), dtype=np.uint16)
    config = CellPatcherConfig(mode="whole_fov")
    (patch,) = run_patching(images=[image], config=config, image_metadata=named([image])).patches
    config.common.image_axes = "XY"
    with pytest.raises(ValueError, match=r"common.image_axes='XY' puts X before Y"):
        patch_array(patch, image, config)


@pytest.mark.parametrize(("image_axes", "mask_axes", "refused"), [("YX", "XY", "mask"), ("XY", "YX", "image"),
                                                                ("XY", "XY", "image")])
def test_the_object_scan_refuses_in_memory_arrays_with_x_before_y(image_axes, mask_axes, refused):
    # Before (CP-3 review), scan_cell_patch_size accepted mask_axes='XY' on a square image and
    # gave the transposed mask's sizes: an object 3 rows high and 7 columns wide, declared 'XY'
    # (so 7 high, 3 wide), was reported as bbox_height 3, bbox_width 7. Refused, as run_patching does.
    from cellpatcher import scan_cell_patch_size
    from cellpatcher.cell_scan import scan_cell_objects

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 3:10] = 1
    match = rf"common.{refused}_axes='XY' puts X before Y"
    with pytest.raises(ValueError, match=match):
        scan_cell_patch_size([(16, 16)], [labels], image_axes=image_axes, mask_axes=mask_axes)
    with pytest.raises(ValueError, match=match):
        scan_cell_objects(image_shape=(16, 16), labels=labels, image_axes=image_axes, mask_axes=mask_axes)
    (record,) = scan_cell_objects(image_shape=(16, 16), labels=labels)
    assert (record.bbox_height, record.bbox_width) == (3, 7)


def test_the_cell_strategy_refuses_mask_axes_with_x_before_y():
    from cellpatcher.strategies import CellSegmentationStrategy

    with pytest.raises(ValueError, match=r"common.mask_axes='ZXY' puts X before Y"):
        CellSegmentationStrategy(mask_axes="ZXY")


def test_a_mask_file_whose_header_names_x_before_y_is_refused(tmp_path):
    # Declared as its header names it, but CellPatcher would still take its first dimension as Y.
    # Masks are never transposed. With preprocessing rc1 the run was written with no axes for this
    # mask, and the file run refused that record; since rc2 the preprocessing run itself refuses
    # the mask (and RunInputs.open an rc1 record: tests/test_unrecorded_files.py).
    import morphobricks_preprocessing as mbp

    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    write_tiff(tmp_path / "exp" / "a.tif", image, "YX")
    (tmp_path / "masks").mkdir()
    tifffile.imwrite(tmp_path / "masks" / "a_cells.tif", labels.astype(np.uint16), metadata={"axes": "XY"})
    result = mbp.run_experiment(
        tmp_path / "exp", mbp.AxisMapping("YX", "YX"), output_folder=tmp_path / "run", masks_folder=tmp_path / "masks"
    )
    assert not result.written
    assert any("a_cells.tif: header axes XY: X is stored before Y" in str(issue) for issue in result.issues)


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


def test_lazy_delivery_refuses_an_in_memory_mask_declared_swapped():
    # The in-memory half of the removed test_lazy_delivery_refuses_a_mask_declared_swapped (its
    # file half is test_lazy_delivery_refuses_a_mask_file): an array mask with X before Y.
    from cellpatcher.delivery import patch_array

    image, labels, _ = make_synthetic_cell_image_and_labels(shape=(32, 32))
    config = _cell_config("YX", "YX")
    patch = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image])).patches[0]
    config.common.mask_axes = "XY"
    config.extraction.apply_mask = True
    with pytest.raises(ValueError, match=r"common.mask_axes='XY' puts X before Y"):
        patch_array(patch, image, config, mask=labels)


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
