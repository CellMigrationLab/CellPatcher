"""OME-Zarr images (folders ending in ``.zarr``): discovery, reading, axes and patching.

See docs/normalization_contract.md §2. Reading needs the ``zarr`` extra;
those tests skip without it.
"""

import importlib.util
import sys

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.io import (
    conform_axes,
    discover_condition_repeat_folders,
    image_stem,
    list_supported_files,
    read_array,
    read_image,
    validate_image_and_mask_folders,
    write_array,
)
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.statistics import PreprocessingUnavailableError
from tests.helpers import write_ngff

needs_zarr = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None or importlib.util.find_spec("zarr") is None,
    reason="needs the 'zarr' extra",
)

def _fake_store(path):
    """A folder that is named like an OME-Zarr image (enough for discovery)."""
    (path / "0").mkdir(parents=True)
    (path / ".zattrs").write_text("{}", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Discovery and names (no extra needed)
# ---------------------------------------------------------------------------


def test_zarr_folders_are_listed_as_images(tmp_path):
    _fake_store(tmp_path / "b.ome.zarr")
    write_array(tmp_path / "a.tif", np.zeros((4, 4), dtype=np.uint8))
    (tmp_path / "notes").mkdir()
    assert [p.name for p in list_supported_files(tmp_path)] == ["a.tif", "b.ome.zarr"]


def test_discovery_treats_a_store_as_an_image_and_never_enters_it(tmp_path):
    root = tmp_path / "experiment"
    _fake_store(root / "control" / "a.ome.zarr")
    _fake_store(root / "treated" / "bio1" / "wellA" / "b.zarr")

    leaves = discover_condition_repeat_folders(root)
    assert [(leaf.condition, leaf.biological_repeat, leaf.technical_repeat) for leaf in leaves] == [
        ("control", "single", "single"),
        ("treated", "bio1", "wellA"),
    ]
    assert leaves[0].image_folder == root / "control"


def test_a_store_next_to_subfolders_is_a_mixed_folder(tmp_path):
    root = tmp_path / "experiment"
    _fake_store(root / "control" / "a.ome.zarr")
    (root / "control" / "bio1").mkdir()
    with pytest.raises(ValueError, match=r"\[tiff_outside_leaf\].*control/a.ome.zarr"):
        discover_condition_repeat_folders(root)


@needs_zarr
def test_store_stems_match_tiff_masks(tmp_path):
    with pytest.warns(DeprecationWarning, match="removed in CellPatcher v0.2.0"):
        assert image_stem("img.ome.zarr") == "img"
    with pytest.warns(DeprecationWarning, match="position_id"):
        assert image_stem("img.ome.tif") == "img.ome"
    # A real store: the mask's header is checked against the image's Y and X sizes
    # (find_masks' mask_size_mismatch, refused for the mask type used).
    write_ngff(tmp_path / "images" / "a.ome.zarr", np.zeros((4, 4), dtype=np.uint16), "yx")
    write_array(tmp_path / "masks" / "a_mask.tif", np.zeros((4, 4), dtype=np.uint8))
    selected = validate_image_and_mask_folders(tmp_path / "images", tmp_path / "masks")
    assert {k.name: v.name for k, v in selected.items()} == {"a.ome.zarr": "a_mask.tif"}


def test_writing_a_store_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        write_array(tmp_path / "a.ome.zarr", np.zeros((4, 4)))


def test_reading_a_store_without_the_extra_names_the_install_command(tmp_path, monkeypatch):
    # The preprocessing brick is a core dependency; only its zarr reader is the "zarr" extra (CP-6).
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(PreprocessingUnavailableError, match="cellpatcher\\[zarr\\]"):
        read_array(_fake_store(tmp_path / "a.ome.zarr"))


# ---------------------------------------------------------------------------
# Axes by role (no extra needed)
# ---------------------------------------------------------------------------


def test_conform_axes_reorders_drops_and_adds_length_one_axes():
    data = np.arange(2 * 1 * 3 * 4).reshape(2, 1, 3, 4)  # C, Z (length 1), Y, X
    out = conform_axes(data, "CZYX", "YXC")
    assert out.shape == (3, 4, 2)
    np.testing.assert_array_equal(out[..., 1], data[1, 0])
    assert conform_axes(data, "CZYX", "TCYX").shape == (1, 2, 3, 4)


def test_conform_axes_rejects_axes_longer_than_one_without_a_role():
    with pytest.raises(ValueError, match="without a known role"):
        conform_axes(np.zeros((2, 3, 4)), "QYX", "YX")
    with pytest.raises(ValueError, match="'Z', length 2"):
        conform_axes(np.zeros((2, 3, 4)), "ZYX", "YX")


# ---------------------------------------------------------------------------
# Reading and patching (needs the extra)
# ---------------------------------------------------------------------------


@needs_zarr
def test_read_array_and_read_image(tmp_path):
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    path = write_ngff(tmp_path / "a.ome.zarr", data, "czyx")
    np.testing.assert_array_equal(read_array(path), data)
    as_zcyx = read_image(path, "ZCYX")
    assert as_zcyx.shape == (3, 2, 4, 5)
    np.testing.assert_array_equal(as_zcyx[1, 0], data[0, 1])


@needs_zarr
def test_a_store_with_several_images_is_an_error(tmp_path):
    path = write_ngff(tmp_path / "a.ome.zarr", np.zeros((4, 4), dtype=np.uint8), "yx", images=2)
    with pytest.raises(ValueError, match="holds 2 images"):
        read_array(path)


@needs_zarr
def test_folder_that_is_not_ome_zarr_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="cannot be read as an OME-Zarr image"):
        read_array(_fake_store(tmp_path / "a.ome.zarr"))


@needs_zarr
def test_tree_run_patches_ome_zarr_and_tiff_images_alike(tmp_path):
    root = tmp_path / "experiment"
    data = np.arange(16 * 16, dtype=np.uint16).reshape(1, 16, 16)
    write_ngff(root / "control" / "a.ome.zarr", data, "zyx")  # Z of length 1 is dropped for "YX"
    write_array(root / "control" / "b.tif", data[0])
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.extract_arrays = True

    (leaf,) = run_patching_condition_repeat_tree(root, config=config)
    by_file = {}
    for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays):
        by_file.setdefault(patch.image_file, []).append(array)
        assert patch.source_relative_path == f"control/{patch.image_file}"
    assert set(by_file) == {"a.ome.zarr", "b.tif"}
    for zarr_patch, tiff_patch in zip(by_file["a.ome.zarr"], by_file["b.tif"]):
        np.testing.assert_array_equal(zarr_patch, tiff_patch)


@needs_zarr
def test_cell_mode_matches_tiff_masks_to_stores(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    image = np.ones((16, 16), dtype=np.uint16)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[4:9, 4:9] = 1
    write_ngff(images / "control" / "a.ome.zarr", image, "yx")
    write_array(masks / "control" / "a_mask.tif", labels)
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 8

    (leaf,) = run_patching_condition_repeat_tree(images, config=config, mask_root_path=masks)
    assert [(p.image_file, p.mask_file, p.label_id) for p in leaf.result.patches] == [
        ("a.ome.zarr", "a_mask.tif", 1)
    ]


@needs_zarr
def test_ome_zarr_planes_link_to_preprocessing_rows(tmp_path):
    import morphobricks_preprocessing as mbp

    from cellpatcher.statistics import SourcePlane, check_images, check_planes, load_plane_statistics

    root = tmp_path / "experiment"
    data = np.random.default_rng(0).integers(0, 255, (2, 3, 16, 16), dtype=np.uint8)
    write_ngff(root / "control" / "a.ome.zarr", data, "czyx")
    result = mbp.run_experiment(root, mbp.AxisMapping("CZYX", "CZYX"), output_folder=tmp_path / "pre")
    assert result.written, [str(issue) for issue in result.issues]
    statistics = load_plane_statistics(result.csv_path)

    config = CellPatcherConfig(mode="whole_fov")
    config.common.image_axes, config.common.z_policy = "CZYX", "all"
    config.extraction.split_channels = True
    (leaf,) = run_patching_condition_repeat_tree(root, config=config)
    patches = leaf.result.patches
    assert {p.source_relative_path for p in patches} == {"control/a.ome.zarr"}

    check_images(statistics, {"control/a.ome.zarr": "CZYX"}, experiment_root=root)
    planes = [
        (SourcePlane(p.source_relative_path, p.condition, p.biological_repeat, p.technical_repeat,
                     z=p.z0, c=p.c, axis_lengths={"C": 2, "Z": 3}), data[p.c, p.z0])
        for p in patches
    ]
    records = check_planes(statistics, planes)
    assert sorted((r.plane_id, r.c) for r in records) == sorted(
        (f"control/a.ome.zarr::z={z},t=0", c) for z in range(3) for c in range(2)
    )


@needs_zarr
def test_an_ome_zarr_image_is_matched_by_role_not_by_header_position(tmp_path):
    # OME-Zarr axes are put into common.image_axes by role (conform_axes), so its stored order
    # (ZCYX with Z of length 1) is not compared position by position with the declared CYX.
    from cellpatcher.orchestration import run_patching_pipeline

    data = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(1, 2, 16, 16)
    write_ngff(tmp_path / "exp" / "a.zarr", data, axes="zcyx")
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = "CYX"
    config.common.extract_arrays = True
    result = run_patching_pipeline(str(tmp_path / "exp"), config=config)
    assert result.patch_arrays[0].shape == (2, 8, 8)
