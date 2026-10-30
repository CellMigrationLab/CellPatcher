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

needs_preprocessing = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs the 'preprocessing' extra"
)
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
    # D85: position_id is the image name; the two wells differ by technical_repeat, and the
    # patch_id, led by the four FOV key fields, stays unique across the tree.
    assert {(p.technical_repeat, p.position_id) for leaf in leaves for p in leaf.result.patches} == {
        ("B02", "img"), ("B03", "img")
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
    assert first == f"single/single/single/P/random-y{result.patches[0].y0}-x{result.patches[0].x0}-h8-w8"
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
        run_patching(images=_image(0), masks=_labels(0), config=config, image_metadata=[{"position_id": "P"}])


def test_whole_fov_and_supplied_position_id():
    config = CellPatcherConfig(mode="whole_fov")
    result = run_patching(images=np.zeros((4, 4)), config=config, image_metadata=[{"position_id": "plate1/A01/f3"}])
    # A supplied position_id is kept verbatim; the FOV key leads the patch_id (D85).
    assert [patch.position_id for patch in result.patches] == ["plate1/A01/f3"]
    assert [patch.patch_id for patch in result.patches] == ["single/single/single/plate1/A01/f3/whole_fov"]


# ---------------------------------------------------------------------------
# D45: position_id is always filled
# ---------------------------------------------------------------------------


def test_folder_runs_fill_position_id_without_split_channels(tmp_path):
    write_array(tmp_path / "img.ome.tif", _image(0))
    config = CellPatcherConfig(mode="whole_fov", common=CommonConfig(condition="treated", biological_repeat="r1"))
    result = run_patching_pipeline(tmp_path, config=config)
    # D85: the image name without its whole extension (preprocessing's rule: .ome.tif -> img).
    assert [patch.position_id for patch in result.patches] == ["img"]
    assert [record.position_id for record in result.image_records] == ["img"]
    assert [patch.patch_id for patch in result.patches] == ["treated/r1/single/img/whole_fov"]


def test_an_in_memory_image_without_position_id_or_file_name_is_refused():
    config = CellPatcherConfig(mode="whole_fov")
    message = 'Image 1 of the run has neither a position_id nor an image_file.*"position_id"'
    with pytest.raises(ValueError, match=message):
        run_patching(images=[_image(2), _image(1)], config=config, image_metadata=[{"position_id": "P"}, {}])
    # A file name alone names the position (one file is one position).
    named = run_patching(images=_image(1), config=config, image_metadata=[{"image_file": "fov.tif"}])
    assert [patch.patch_id for patch in named.patches] == ["single/single/single/fov/whole_fov"]


# ---------------------------------------------------------------------------
# Random mode is seeded per FOV
# ---------------------------------------------------------------------------


def _random_config(seed=0):
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches, config.random.seed = 8, 5, seed
    return config


def _rectangles(result, image_file):
    return [(p.patch_id, p.y0, p.x0) for p in result.patches if p.image_file == image_file]


def test_random_rectangles_do_not_depend_on_the_place_in_the_run(tmp_path):
    write_array(tmp_path / "alone" / "m_target.tif", _image(7))
    alone = _rectangles(run_patching_pipeline(tmp_path / "alone", config=_random_config()), "m_target.tif")
    assert len(alone) == 5

    ten = tmp_path / "ten"
    for index in range(9):
        write_array(ten / f"n{index}.tif", _image(index))
    write_array(ten / "m_target.tif", _image(7))
    assert _rectangles(run_patching_pipeline(ten, config=_random_config()), "m_target.tif") == alone
    write_array(ten / "a_first.tif", _image(11))
    assert _rectangles(run_patching_pipeline(ten, config=_random_config()), "m_target.tif") == alone

    # Another seed draws other rectangles; so does another position.
    reseeded = run_patching_pipeline(tmp_path / "alone", config=_random_config(seed=1))
    assert _rectangles(reseeded, "m_target.tif") != alone
    other = run_patching(images=_image(7), config=_random_config(), image_metadata=[{"image_file": "other.tif"}])
    assert [(p.y0, p.x0) for p in other.patches] != [(y0, x0) for _, y0, x0 in alone]


def test_random_planes_are_seeded_by_their_own_indices():
    config = _random_config()
    config.common.image_axes, config.common.z_policy = "ZYX", "all"
    stack = np.stack([_image(0)] * 3)
    full = run_patching(images=stack, config=config, image_metadata=[{"position_id": "P"}])
    # Plane z=2 alone (from metadata) draws what it drew inside the stack.
    alone = run_patching(images=stack[2], config=_random_config(), image_metadata=[{"position_id": "P", "z": 2}])
    assert [(p.patch_id, p.y0, p.x0) for p in full.patches if p.z0 == 2] == [
        (p.patch_id, p.y0, p.x0) for p in alone.patches
    ]
    assert len({tuple((p.y0, p.x0) for p in full.patches if p.z0 == z) for z in range(3)}) == 3


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
    assert {patch.object_set for patch in result.patches} == {"cytoplasm"}

    # In memory, image_metadata wins over the config; other modes have none.
    result = run_patching(
        images=_image(0), masks=_labels(0), config=config, image_metadata=[{"object_set": "cells", "position_id": "P"}]
    )
    assert {patch.object_set for patch in result.patches} == {"cells"}
    grid_config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16))
    grid = run_patching(images=_image(0), config=grid_config, image_metadata=[{"object_set": "x", "position_id": "P"}])
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
        run_patching(images=_image(0), masks=mask, config=config, image_metadata=[{"position_id": "P"}])
    saved = json.loads((tmp_path / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["cell"]["binary_mask_policy"] == "connected_components"


@pytest.mark.parametrize("z_policy", ["keep", "max"])
def test_a_multi_series_tiff_read_whole_is_refused(tmp_path, z_policy):
    import tifffile

    path = tmp_path / "images" / "plate.tif"
    path.parent.mkdir()
    tifffile.imwrite(path, np.zeros((2, 16, 16), dtype=np.uint16))
    tifffile.imwrite(path, np.ones((2, 16, 16), dtype=np.uint16), append=True)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.z_policy = "ZYX", z_policy
    with pytest.raises(ValueError, match="plate.tif holds 2 image series"):
        run_patching_pipeline(path.parent, config=config)


def test_a_multi_series_tiff_is_refused_without_the_preprocessing_extra(tmp_path, monkeypatch):
    import tifffile

    import cellpatcher.statistics as statistics

    def unavailable():
        raise statistics.PreprocessingUnavailableError("not installed")

    monkeypatch.setattr(statistics, "require_preprocessing", unavailable)
    path = tmp_path / "plate.tif"
    tifffile.imwrite(path, np.zeros((2, 16, 16), dtype=np.uint16))
    tifffile.imwrite(path, np.ones((2, 16, 16), dtype=np.uint16), append=True)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.z_policy = "ZYX", "all"
    with pytest.raises(ValueError, match="plate.tif holds 2 image series"):
        run_patching_pipeline(tmp_path, config=config)


@needs_preprocessing
def test_a_multi_series_tiff_is_refused_when_reading_planes(tmp_path):
    import tifffile

    path = tmp_path / "plate.tif"
    tifffile.imwrite(path, np.zeros((2, 16, 16), dtype=np.uint16))
    tifffile.imwrite(path, np.ones((2, 16, 16), dtype=np.uint16), append=True)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.z_policy = "ZYX", "all"
    with pytest.raises(ValueError, match="plate.tif holds 2 images"):
        run_patching_pipeline(tmp_path, config=config)


@needs_zarr
def test_a_store_with_several_images_is_refused_when_reading_planes(tmp_path):
    write_ngff(tmp_path / "plate.ome.zarr", np.zeros((2, 16, 16), dtype=np.uint16), "zyx", images=2)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.z_policy = "ZYX", "all"
    with pytest.raises(ValueError, match="plate.ome.zarr holds 2 images"):
        run_patching_pipeline(tmp_path, config=config)


def test_positions_differing_only_in_case_are_refused_in_one_repeat():
    """D85: position_id is unique within its technical repeat whatever its case."""
    config = CellPatcherConfig(mode="whole_fov")
    with pytest.raises(ValueError, match="position_id"):
        run_patching(images=[np.zeros((4, 4))] * 2, config=config,
                     image_metadata=[{"position_id": "Img"}, {"position_id": "img"}])
    other_wells = [{"position_id": "img", "technical_repeat": "A"}, {"position_id": "img", "technical_repeat": "B"}]
    result = run_patching(images=[np.zeros((4, 4))] * 2, config=config, image_metadata=other_wells)
    assert [p.patch_id for p in result.patches] == ["single/single/A/img/whole_fov", "single/single/B/img/whole_fov"]


def test_a_duplicate_position_is_refused_even_when_it_yields_no_patch():
    """Audit of #41: D85's uniqueness is checked per source image, before geometry; the first
    image (empty mask) gives no patch, so patch ids alone would never collide."""
    images, masks = [_image(0), _image(1)], [np.zeros((32, 32), np.uint16), _labels(1)]
    with pytest.raises(ValueError, match="same position_id"):
        run_patching(images=images, masks=masks, config=_cell_config(),
                     image_metadata=[{"position_id": "Img"}, {"position_id": "img"}])


def test_a_duplicate_position_is_refused_whatever_tokens_its_patches_get():
    """Cell labels or random rectangles that differ between the two images do not hide the duplicate."""
    with pytest.raises(ValueError, match="same position_id"):
        run_patching(images=[_image(0), _image(1)], masks=[_labels(0), _labels(1)], config=_cell_config(),
                     image_metadata=[{"position_id": "Img"}, {"position_id": "img"}])
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches, config.random.seed = 8, 2, 3
    with pytest.raises(ValueError, match="same position_id"):
        run_patching(images=[_image(0), _image(1)], config=config,
                     image_metadata=[{"position_id": "Img"}, {"position_id": "img"}])


def test_a_duplicate_position_is_refused_in_a_folder_run(tmp_path):
    """The lazily read folder route (one source at a time) checks it too, before the image's planes."""
    write_array(tmp_path / "Img.tif", _image(0))
    write_array(tmp_path / "img.tiff", _image(1))
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.z_policy = "all"
    with pytest.raises(ValueError, match="same position_id"):
        run_patching_pipeline(tmp_path, config=config)


def test_only_the_position_is_case_insensitive():
    """Technical repeats "A" and "a" are distinct labels: one position name in each is legal, and
    images of one position told apart by their planes are legal too."""
    config = CellPatcherConfig(mode="whole_fov")
    wells = [{"position_id": "img", "technical_repeat": "A"}, {"position_id": "img", "technical_repeat": "a"}]
    result = run_patching(images=[np.zeros((4, 4))] * 2, config=config, image_metadata=wells)
    assert [p.patch_id for p in result.patches] == ["single/single/A/img/whole_fov", "single/single/a/img/whole_fov"]
    planes = [{"position_id": "img", "t": 0}, {"position_id": "IMG", "t": 1}]
    assert len(run_patching(images=[np.zeros((4, 4))] * 2, config=config, image_metadata=planes).patches) == 2


@pytest.mark.parametrize("axis", ["t", "z", "c"])
def test_plane_indices_given_as_text_are_the_same_plane(axis):
    """Re-audit of #41: uniqueness compares the indices as patches carry them (int), so 1 and "1"
    are one plane and two case-differing positions there are refused."""
    config = CellPatcherConfig(mode="whole_fov")
    metadata = [{"position_id": "Img", axis: 1}, {"position_id": "img", axis: "1"}]
    with pytest.raises(ValueError, match="same position_id"):
        run_patching(images=[np.zeros((4, 4))] * 2, config=config, image_metadata=metadata)


@pytest.mark.parametrize("value", [7, "", "  ", 1.5])
def test_a_supplied_position_id_must_be_text(value):
    """D85: a supplied position_id is kept verbatim, so a number or an empty one is refused, not
    converted (also for an image yielding no patch)."""
    with pytest.raises(ValueError, match="must be non-empty text"):
        run_patching(images=[_image(0)], masks=[np.zeros((32, 32), np.uint16)], config=_cell_config(),
                     image_metadata=[{"position_id": value}])


@pytest.mark.parametrize("value", [True, 1.5, 1.0, -1, "1.0", "-1", "01", " 1"])
@pytest.mark.parametrize("axis", ["t", "z", "c"])
def test_a_plane_index_is_never_coerced(axis, value):
    """Re-audit of #41: a plane index is a non-negative integer or its decimal text; anything
    int() would round or convert is refused rather than becoming another plane."""
    config = CellPatcherConfig(mode="whole_fov")
    with pytest.raises(ValueError, match="plane index is a non-negative integer"):
        run_patching(images=[np.zeros((4, 4))], config=config,
                     image_metadata=[{"position_id": "p1", axis: value}])


def test_integer_plane_indices_of_numpy_and_text_are_accepted():
    config = CellPatcherConfig(mode="whole_fov")
    metadata = [{"position_id": "p1", "t": np.int64(2)}, {"position_id": "p1", "t": "3"}]
    result = run_patching(images=[np.zeros((4, 4))] * 2, config=config, image_metadata=metadata)
    assert [patch.t for patch in result.patches] == [2, 3]


@pytest.mark.parametrize("mode", ["grid", "random"])
def test_a_supplied_channel_is_refused_with_split_channels(mode, monkeypatch):
    """Re-audit of #41: split_channels generates c and would overwrite a supplied c (D85); the
    pair is refused before any geometry, so a supplied c never seeds random rectangles."""
    from cellpatcher import pipeline

    def no_seed(*args, **kwargs):
        raise AssertionError("the random seed was computed before the refusal")

    monkeypatch.setattr(pipeline, "random_seed", no_seed)
    config = CellPatcherConfig(mode=mode, grid=GridConfig(tile_size=8))
    config.random.patch_size, config.random.n_patches = 8, 2
    config.common.image_axes = "CYX"
    config.extraction.split_channels = True
    image = np.zeros((2, 16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="split_channels=True generates c"):
        run_patching(images=[image], config=config, image_metadata=[{"position_id": "p1", "c": 5}])


@pytest.mark.parametrize("policy", ["max", "mean", "all", "first"])
def test_a_supplied_z_is_refused_with_a_z_policy(policy, monkeypatch):
    """Re-audit of #41: a projection is no single plane, so a supplied z with z_policy="max"
    would label it one; any policy but "keep" with a supplied z is refused before seeding."""
    from cellpatcher import pipeline

    def no_seed(*args, **kwargs):
        raise AssertionError("the random seed was computed before the refusal")

    monkeypatch.setattr(pipeline, "random_seed", no_seed)
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches = 8, 2
    config.common.image_axes, config.common.z_policy = "ZYX", policy
    image = np.zeros((3, 16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="z_policy=.* also decides its planes"):
        run_patching(images=[image], config=config, image_metadata=[{"position_id": "p1", "z": 2}])


def test_a_supplied_t_is_refused_with_a_t_policy():
    config = CellPatcherConfig(mode="whole_fov")
    config.common.image_axes, config.common.t_policy = "TYX", "all"
    with pytest.raises(ValueError, match="t_policy='all' also decides its planes"):
        run_patching(images=[np.zeros((2, 8, 8))], config=config,
                     image_metadata=[{"position_id": "p1", "t": 0}])


@pytest.mark.parametrize(
    ("where", "value"),
    [("technical_repeat", 7), ("object_set", 7), ("technical_repeat", "")],
)
def test_supplied_identity_labels_must_be_text(where, value):
    """Re-audit of #41: a per-image label is identity, never converted (7 is not "7")."""
    with pytest.raises(ValueError, match="must be"):
        run_patching(images=[_image(0)], masks=[_labels(0)], config=_cell_config(),
                     image_metadata=[{"position_id": "p1", where: value}])


@pytest.mark.parametrize("field", ["condition", "biological_repeat", "technical_repeat"])
def test_config_fov_labels_must_be_text(field):
    config = CellPatcherConfig(mode="whole_fov")
    setattr(config.common, field, 2026)
    with pytest.raises(ValueError, match=f"common.{field} must be non-empty text"):
        run_patching(images=[np.zeros((4, 4))], config=config, image_metadata=[{"position_id": "p1"}])


@pytest.mark.parametrize("field", ["condition", "biological_repeat", "technical_repeat", "object_set"])
@pytest.mark.parametrize("split", [False, True])
def test_a_folder_run_refuses_config_identity_before_reading(tmp_path, monkeypatch, field, split):
    """Re-audit of #41: an invalid config label (also from a reloaded JSON config) is refused
    before any image or mask is listed or read, on the eager and the plane-split routes."""
    from cellpatcher import orchestration
    from cellpatcher.orchestration import load_config, save_config_for_mode

    def no_read(*args, **kwargs):
        raise AssertionError("a file was read before the identity refusal")

    images, masks = tmp_path / "images", tmp_path / "masks"
    write_array(images / "img.tif", _image(0))
    write_array(masks / "img.tif", _labels(0))
    monkeypatch.setattr(orchestration, "read_image", no_read)
    monkeypatch.setattr(orchestration, "list_supported_files", no_read)
    monkeypatch.setattr(orchestration, "_mask_loaders", no_read)
    path = save_config_for_mode("cell", tmp_path / "config.json")
    data = json.loads(path.read_text())
    data["common"][field] = 7
    if split:
        data["common"]["z_policy"] = "all"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"{field} must be"):
        run_patching_pipeline(images, mask_folder_path=masks, config=load_config(path))


@pytest.mark.parametrize("planes", ["keep", "all"])
def test_colliding_positions_are_refused_before_any_read(tmp_path, monkeypatch, planes):
    """Re-audit of #41: Img.tif and img.tiff collide (D85); the refusal comes from the names,
    before any image or mask is read, on the eager and the plane-split routes."""
    from cellpatcher import orchestration

    def no_read(*args, **kwargs):
        raise AssertionError("a file was read before the identity refusal")

    images, masks = tmp_path / "images", tmp_path / "masks"
    for name in ("Img.tif", "img.tiff"):
        write_array(images / name, _image(0))
        write_array(masks / name, _labels(0))
    monkeypatch.setattr(orchestration, "read_image", no_read)
    monkeypatch.setattr(orchestration, "_mask_loaders", no_read)
    config = _cell_config()
    config.common.z_policy = planes
    with pytest.raises(ValueError, match="same position_id"):
        run_patching_pipeline(images, mask_folder_path=masks, config=config)


def test_a_tree_refuses_a_later_leaf_before_reading_any_leaf(tmp_path, monkeypatch):
    """Every leaf's identity is checked before the first leaf is read."""
    from cellpatcher import orchestration

    def no_read(*args, **kwargs):
        raise AssertionError("a leaf was read before the identity refusal")

    write_array(tmp_path / "ctrl" / "r1" / "a.tif", _image(0))
    for name in ("Img.tif", "img.tiff"):
        write_array(tmp_path / "ctrl" / "r2" / name, _image(1))
    monkeypatch.setattr(orchestration, "read_image", no_read)
    monkeypatch.setattr(orchestration, "_run_folder_patching", no_read)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    with pytest.raises(ValueError, match="same position_id"):
        run_patching_condition_repeat_tree(tmp_path, config=config)


def _refuse_mask_object_set(monkeypatch):
    """Make every derived mask object set invalid, and every image or mask read fail loudly.

    A mask file name always gives a valid name today (object_set_from_mask_file
    replaces other characters), so the refusal is forced to show *when* it is checked.
    """
    from cellpatcher import orchestration

    def no_read(*args, **kwargs):
        raise AssertionError("a file was read before the object_set refusal")

    monkeypatch.setattr(orchestration, "object_set_from_mask_file", lambda image, mask: "bad/name")
    monkeypatch.setattr(orchestration, "read_image", no_read)
    monkeypatch.setattr(orchestration, "_read_mask_file", no_read)


@pytest.mark.parametrize("planes", ["keep", "all"])
def test_a_mask_named_object_set_is_refused_before_any_read(tmp_path, monkeypatch, planes):
    """Issue #42: the object set from a matched mask's name is checked from the names alone."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    write_array(images / "img.tif", _image(0))
    write_array(masks / "img_cells.tif", _labels(0))
    _refuse_mask_object_set(monkeypatch)
    config = _cell_config()
    config.common.z_policy = planes
    with pytest.raises(ValueError, match="object set from mask 'img_cells.tif'"):
        run_patching_pipeline(images, mask_folder_path=masks, config=config)


def test_a_tree_refuses_a_later_leaf_mask_object_set_before_reading_any_leaf(tmp_path, monkeypatch):
    """Issue #42: a later leaf's mask-derived object set is refused before the first leaf is read."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    for leaf in ("r1", "r2"):
        write_array(images / "ctrl" / leaf / "img.tif", _image(0))
        write_array(masks / "ctrl" / leaf / "img_cells.tif", _labels(0))
    from cellpatcher import orchestration

    real = orchestration.object_set_from_mask_file
    _refuse_mask_object_set(monkeypatch)
    calls = []

    def bad_in_second_leaf(image, mask):
        calls.append(mask)
        return "bad/name" if len(calls) > 1 else real(image, mask)

    monkeypatch.setattr(orchestration, "object_set_from_mask_file", bad_in_second_leaf)
    with pytest.raises(ValueError, match="object set from mask 'img_cells.tif'"):
        run_patching_condition_repeat_tree(images, config=_cell_config(), mask_root_path=masks)
