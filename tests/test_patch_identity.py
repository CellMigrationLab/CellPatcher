"""Run-independent patch identity (owner decisions D44, D45, D47; 2026-10-26).

``patch_id`` is built from the FOV identity (``position_id``), the object set,
the plane indices and the patch's own label or rectangle (``cellpatcher.ids``),
never from the image's place in the run; ``position_id`` is always filled.
"""

import importlib.util
import json

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExtractionConfig, GridConfig
from cellpatcher.ids import DEFAULT_OBJECT_SET, object_set_from_mask_file
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from tests.helpers import write_ngff, write_ngff_labels

needs_zarr = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None or importlib.util.find_spec("zarr") is None,
    reason="needs the 'preprocessing' extra with zarr",
)


def _image(seed):
    return np.random.default_rng(seed).integers(0, 1000, (32, 32), dtype=np.uint16)


def _labels(seed):
    """Two square objects whose place and label values depend on *seed*."""
    labels = np.zeros((32, 32), dtype=np.uint16)
    offset = seed % 5
    labels[4 + offset:10 + offset, 4:10] = 3 + seed
    labels[18:24, 16 + offset:22 + offset] = 40 + seed
    return labels


def _write_pair(folder, mask_folder, name, seed):
    write_array(folder / f"{name}.tif", _image(seed))
    write_array(mask_folder / f"{name}_cells.tif", _labels(seed))


def _cell_config(out=None):
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 8
    config.cell.exclude_border = False
    config.matching.mask_suffixes = ("_cells",)
    if out is not None:
        config.export.enabled = True
        config.export.image_format = "npy"
        config.common.output_dir = str(out)
    return config


def _ids_of(result, image_file):
    return [patch.patch_id for patch in result.patches if patch.image_file == image_file]


# ---------------------------------------------------------------------------
# D44: the same file gets the same ids, whatever else is in the run
# ---------------------------------------------------------------------------


def test_same_file_alone_in_ten_files_and_after_an_earlier_file_has_the_same_patch_ids(tmp_path):
    alone, alone_masks = tmp_path / "alone" / "images", tmp_path / "alone" / "masks"
    _write_pair(alone, alone_masks, "m_target", 7)
    alone_result = run_patching_pipeline(alone, mask_folder_path=alone_masks, config=_cell_config())
    expected = _ids_of(alone_result, "m_target.tif")
    assert expected == ["single/single/single/m_target/cells/label10", "single/single/single/m_target/cells/label47"]

    ten, ten_masks = tmp_path / "ten" / "images", tmp_path / "ten" / "masks"
    for index in range(9):
        _write_pair(ten, ten_masks, f"n{index}", index)
    _write_pair(ten, ten_masks, "m_target", 7)
    result = run_patching_pipeline(ten, mask_folder_path=ten_masks, config=_cell_config())
    assert _ids_of(result, "m_target.tif") == expected

    # A file that sorts first shifts every image_id, but no patch_id.
    _write_pair(ten, ten_masks, "a_first", 11)
    shifted = run_patching_pipeline(ten, mask_folder_path=ten_masks, config=_cell_config())
    target_image_ids = {p.image_id for p in shifted.patches if p.image_file == "m_target.tif"}
    assert target_image_ids != {p.image_id for p in result.patches if p.image_file == "m_target.tif"}
    assert _ids_of(shifted, "m_target.tif") == expected
    for name in (f"n{index}.tif" for index in range(9)):
        assert _ids_of(shifted, name) == _ids_of(result, name)


def test_grid_ids_do_not_depend_on_the_other_files(tmp_path):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16))
    write_array(tmp_path / "one" / "img.tif", _image(1))
    alone = run_patching_pipeline(tmp_path / "one", config=config)
    for name, seed in (("aaa.tif", 2), ("img.tif", 1), ("zzz.tif", 3)):
        write_array(tmp_path / "three" / name, _image(seed))
    three = run_patching_pipeline(tmp_path / "three", config=config)
    assert _ids_of(three, "img.tif") == _ids_of(alone, "img.tif") == [
        f"single/single/single/img/grid-y{y}-x{x}-h16-w16" for y in (0, 16) for x in (0, 16)
    ]


def test_two_wells_with_the_same_file_names_get_unique_ids_in_a_tree_run(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    for well in ("B02", "B03"):
        _write_pair(images / "control" / "2026_01_01" / well, masks / "control" / "2026_01_01" / well, "img", 1)
    leaves = run_patching_condition_repeat_tree(
        images, mask_root_path=masks, config=_cell_config(tmp_path / "out"), combine_metadata=True
    )

    ids = [patch.patch_id for leaf in leaves for patch in leaf.result.patches]
    assert len(ids) == len(set(ids)) == 4
    assert {patch.position_id for leaf in leaves for patch in leaf.result.patches} == {
        "control/2026_01_01/B02/img", "control/2026_01_01/B03/img"
    }
    assert ids[0] == "control/2026_01_01/B02/img/cells/label4"
    rows = (tmp_path / "out" / "patches.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 4
    assert "B02/control_2026_01_01_B02_img_cells_label4.npy" in rows[1]


def test_a_rerun_writes_a_byte_identical_patches_csv(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    for index in range(3):
        _write_pair(images, masks, f"img{index}", index)
    for out in ("first", "second"):
        run_patching_pipeline(images, mask_folder_path=masks, config=_cell_config(tmp_path / out))
    for table in ("patches.csv", "images.csv"):
        assert (tmp_path / "first" / table).read_bytes() == (tmp_path / "second" / table).read_bytes()


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------


def test_split_channel_time_series_ids_name_the_time_point_and_channel():
    stack = np.random.default_rng(0).integers(0, 100, (2, 3, 16, 16), dtype=np.uint16)  # T, C, Y, X
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[4:9, 4:9] = 5
    config = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(image_axes="TCYX", mask_axes="YX", t_policy="all", condition="ctrl",
                            biological_repeat="2026_01_01", technical_repeat="B02", object_set="nuclei"),
        extraction=ExtractionConfig(split_channels=True),
    )
    config.cell.exclude_border = False
    result = run_patching(images=stack, masks=labels, config=config, image_metadata=[{"image_file": "fov1.tif"}])
    assert [patch.patch_id for patch in result.patches] == [
        f"ctrl/2026_01_01/B02/fov1/nuclei/t{t:04d}/c{c}/label5" for t in range(2) for c in range(3)
    ]
    assert {patch.object_set for patch in result.patches} == {"nuclei"}


def test_repeated_random_draws_are_numbered_in_sampling_order():
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches = 8, 40
    result = run_patching(images=np.zeros((9, 9)), config=config, image_metadata=[{"position_id": "P"}])
    ids = [patch.patch_id for patch in result.patches]
    assert len(set(ids)) == 40
    first = ids[0]
    assert first == f"P/random-y{result.patches[0].y0}-x{result.patches[0].x0}-h8-w8"
    assert ids.count(first) == 1 and f"{first}-n2" in ids


def test_random_draws_of_two_images_sharing_a_position_are_refused_not_numbered():
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches = 8, 40
    with pytest.raises(ValueError, match="position_id"):
        run_patching(images=[np.zeros((9, 9))] * 2, config=config, image_metadata=[{"position_id": "P"}] * 2)


def test_object_set_set_after_construction_is_checked():
    config = _cell_config()
    config.common.object_set = "a/b"
    with pytest.raises(ValueError, match="object_set"):
        run_patching(images=_image(0), masks=_labels(0), config=config)


def test_whole_fov_and_supplied_position_id():
    config = CellPatcherConfig(mode="whole_fov")
    result = run_patching(images=np.zeros((4, 4)), config=config, image_metadata=[{"position_id": "plate1/A01/f3"}])
    assert [patch.patch_id for patch in result.patches] == ["plate1/A01/f3/whole_fov"]


# ---------------------------------------------------------------------------
# D45: position_id is always filled
# ---------------------------------------------------------------------------


def test_folder_runs_fill_position_id_without_split_channels(tmp_path):
    write_array(tmp_path / "img.ome.tif", _image(0))
    config = CellPatcherConfig(mode="whole_fov", common=CommonConfig(condition="treated", biological_repeat="r1"))
    result = run_patching_pipeline(tmp_path, config=config)
    assert [patch.position_id for patch in result.patches] == ["treated/r1/single/img.ome"]
    assert [record.position_id for record in result.image_records] == ["treated/r1/single/img.ome"]


def test_an_unnamed_in_memory_image_is_named_by_its_pixels_wherever_it_is_in_the_run():
    config = CellPatcherConfig(mode="whole_fov")
    alone = run_patching(images=_image(1), config=config).patches[0].patch_id
    second = run_patching(images=[_image(2), _image(1)], config=config).patches[1].patch_id
    assert alone == second
    assert alone.startswith("single/single/single/array-") and alone.endswith("/whole_fov")


# ---------------------------------------------------------------------------
# object_set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image_file", "mask_file", "expected"),
    [
        ("img.tif", "img_cells.tif", "cells"),
        ("img.tif", "IMG-Nuclei.tif", "Nuclei"),
        ("img.ome.zarr", "img_mask.tif", "mask"),
        ("img.tif", "img.tif", None),
        ("img1.tif", "cells_img1.tif", None),
    ],
)
def test_object_set_from_the_mask_file_name(image_file, mask_file, expected):
    assert object_set_from_mask_file(image_file, mask_file) == expected


def test_object_set_precedence(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    write_array(images / "img.tif", _image(0))
    write_array(masks / "img.tif", _labels(0))  # named like its image: no name given
    result = run_patching_pipeline(images, mask_folder_path=masks, config=_cell_config())
    assert {patch.object_set for patch in result.patches} == {DEFAULT_OBJECT_SET} == {"mask"}

    config = _cell_config()
    config.common.object_set = "cytoplasm"
    result = run_patching_pipeline(images, mask_folder_path=masks, config=config)
    assert {patch.patch_id.split("/")[4] for patch in result.patches} == {"cytoplasm"}

    # In memory, image_metadata wins over the config; other modes have none.
    result = run_patching(images=_image(0), masks=_labels(0), config=config, image_metadata=[{"object_set": "cells"}])
    assert {patch.object_set for patch in result.patches} == {"cells"}
    grid_config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16))
    grid = run_patching(images=_image(0), config=grid_config, image_metadata=[{"object_set": "x"}])
    assert {patch.object_set for patch in grid.patches} == {None}


@pytest.mark.parametrize("value", ["", "a/b", "two words", 3])
def test_object_set_must_be_one_safe_level(value):
    with pytest.raises(ValueError, match="object_set"):
        CommonConfig(object_set=value)


@needs_zarr
def test_object_set_is_the_ngff_label_name(tmp_path):
    path = write_ngff(tmp_path / "control" / "a.ome.zarr", np.ones((16, 16), dtype=np.uint16), "yx")
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[3:7, 3:7] = 9
    write_ngff_labels(path, {"nuclei": labels}, "yx")
    config = _cell_config()
    config.common.mask_source = "ngff_labels"
    (leaf,) = run_patching_condition_repeat_tree(tmp_path, config=config)
    assert [patch.patch_id for patch in leaf.result.patches] == ["control/single/single/a/nuclei/label9"]


# ---------------------------------------------------------------------------
# D47 is recorded; multi-image stores are refused
# ---------------------------------------------------------------------------


def test_the_connected_components_opt_in_is_recorded_in_run_config(tmp_path):
    config = _cell_config(tmp_path)
    config.cell.binary_mask_policy = "connected_components"
    mask = _labels(0) > 0
    with pytest.warns(UserWarning, match="connected components"):
        run_patching(images=_image(0), masks=mask, config=config)
    saved = json.loads((tmp_path / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["cell"]["binary_mask_policy"] == "connected_components"


@needs_zarr
def test_a_store_with_several_images_is_refused_when_reading_planes(tmp_path):
    write_ngff(tmp_path / "plate.ome.zarr", np.zeros((2, 16, 16), dtype=np.uint16), "zyx", images=2)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.z_policy = "ZYX", "all"
    with pytest.raises(ValueError, match="plate.ome.zarr holds 2 images"):
        run_patching_pipeline(tmp_path, config=config)
