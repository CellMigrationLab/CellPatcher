
import numpy as np
import pytest

from cellpatcher.file_runs import validate_image_and_mask_folders
from cellpatcher.io import (
    list_supported_files,
    read_array,
    write_array,
)
from tests.helpers import file_run_config, preprocess, write_tiff


def test_list_supported_files_sorted_and_filtered(tmp_path):
    (tmp_path / "b.TIFF").write_bytes(b"x")
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x")
    files = list_supported_files(tmp_path)
    assert [p.name for p in files] == ["a.tif", "b.TIFF"]


def test_list_supported_files_refuses_the_image_formats_no_longer_read(tmp_path):
    # Owner decision of 2026-10-30 (#45): images are TIFF or OME-Zarr, as the preprocessing
    # brick discovers them. The former formats are refused by name, never skipped.
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "b.npy").write_bytes(b"x")
    (tmp_path / "c.PNG").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"2 image\(s\) in a format CellPatcher no longer reads: b.npy, c.PNG"):
        list_supported_files(tmp_path)


def test_list_supported_files_rejects_missing_folder(tmp_path):
    with pytest.raises(ValueError):
        list_supported_files(tmp_path / "missing")


def test_read_write_array_npy_roundtrip(tmp_path):
    arr = np.arange(6).reshape(2, 3)
    p = tmp_path / "arr.npy"
    write_array(p, arr)
    out = read_array(p)
    assert np.array_equal(out, arr)


def test_read_array_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "arr.xyz"
    p.write_text("x")
    with pytest.raises(ValueError):
        read_array(p)


def test_write_array_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError):
        write_array(tmp_path / "arr.xyz", np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Additional IO edge cases
# ---------------------------------------------------------------------------

from cellpatcher.io import read_array_shape


def test_read_array_rejects_nonexistent_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        read_array(tmp_path / "missing.npy")


def test_read_array_shape_returns_correct_shape(tmp_path):
    arr = np.arange(12).reshape(3, 4)
    p = tmp_path / "a.npy"
    write_array(p, arr)
    shape = read_array_shape(p)
    assert shape == (3, 4)


def test_write_read_tiff_roundtrip(tmp_path):
    """Write and read back a TIFF file using tifffile."""
    arr = np.arange(9).reshape(3, 3).astype(np.uint16)
    p = tmp_path / "img.tif"
    try:
        write_array(p, arr)
        loaded = read_array(p)
        assert np.array_equal(loaded, arr)
    except ImportError:
        pytest.skip("tifffile not available")


def test_list_supported_files_with_custom_extensions(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"x")
    (tmp_path / "b.csv").write_text("x")
    files = list_supported_files(tmp_path, supported_extensions={".csv"})
    assert [p.name for p in files] == ["b.csv"]


from cellpatcher import CellPatcherConfig
from cellpatcher.config import CommonConfig, GridConfig
from cellpatcher.io import run_patching_from_folders


def test_run_patching_from_folders_grid_mode(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.arange(64).reshape(8, 8))

    config = file_run_config(CellPatcherConfig(mode="grid"), img_dir)
    config.grid.tile_size = 4

    result = run_patching_from_folders(img_dir, config=config)
    assert len(result.patches) == 4


def test_run_patching_from_folders_cell_mode(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((16, 16), dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:12, 5:13] = 1

    write_array(img_dir / "sample.tif", image)
    write_array(mask_dir / "sample_mask.tif", labels)

    config = file_run_config(CellPatcherConfig(mode="cell"), img_dir, masks=mask_dir)
    config.cell.patch_size = 8

    result = run_patching_from_folders(img_dir, mask_folder_path=mask_dir, config=config)
    assert len(result.patches) == 1


def test_run_patching_from_folders_cell_requires_mask_folder(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((8, 8), dtype=np.float32))

    config = file_run_config(CellPatcherConfig(mode="cell"), img_dir)  # a run without masks
    with pytest.raises(ValueError, match="mode='cell' reads masks, but the preprocessing run .* recorded none"):
        run_patching_from_folders(img_dir, config=config)

from cellpatcher.orchestration import (
    load_config,
    run_patching_condition_repeat_tree,
    run_patching_pipeline,
    save_default_config,
)


def test_run_patching_pipeline_alias_grid_mode(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.arange(64).reshape(8, 8))

    config = file_run_config(CellPatcherConfig(mode="grid"), img_dir)
    config.grid.tile_size = 4
    result = run_patching_pipeline(img_dir, config=config)
    assert len(result.patches) == 4


@pytest.mark.parametrize("mode", ["grid", "whole_fov"])
def test_grid_and_whole_fov_folder_runs_take_a_moved_masks_folder(tmp_path, mode):
    """Owner decision 3 (2026-11-02): as in the tree run, mask_folder_path says where the run's masks
    folder is now; the masks are checked there and never read. Old main refused it for these modes."""
    import shutil

    from cellpatcher.statistics import PreprocessingCheckError

    img_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    write_array(img_dir / "a.tif", np.zeros((8, 8), dtype=np.uint16))
    write_array(mask_dir / "a_mask.tif", np.ones((8, 8), dtype=np.uint16))
    config = file_run_config(CellPatcherConfig(mode=mode), img_dir, masks=mask_dir)
    config.grid.tile_size = 4
    shutil.move(mask_dir, tmp_path / "moved")
    with pytest.raises(PreprocessingCheckError, match="1 mask changed or missing since the run"):
        run_patching_pipeline(img_dir, config=config)  # checked where the run recorded it: gone
    result = run_patching_pipeline(img_dir, config=config, mask_folder_path=tmp_path / "moved")
    assert result.patches and all(patch.mask_file is None for patch in result.patches)


def test_run_patching_pipeline_random_mode_with_optional_mask(tmp_path):
    """random mode accepts an optional mask folder to restrict sampling region."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    img = np.zeros((32, 32), dtype=np.float32)
    # mask restricts sampling to top half
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[:16, :] = 1
    write_array(img_dir / "a.tif", img)
    write_array(mask_dir / "a_mask.tif", mask)

    binary = {"mask": "binary"}  # a sampling region is a binary mask, and the description says so
    config = file_run_config(CellPatcherConfig(mode="random"), img_dir, masks=mask_dir, representations=binary)
    config.random.patch_size = 4
    config.random.n_patches = 5
    config.random.seed = 0
    result = run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)
    assert len(result.patches) == 5


def test_run_patching_pipeline_random_mode_without_mask(tmp_path):
    """random mode works without a mask folder."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((32, 32), dtype=np.float32))

    config = file_run_config(CellPatcherConfig(mode="random"), img_dir)
    config.random.patch_size = 4
    config.random.n_patches = 3
    result = run_patching_pipeline(img_dir, config=config)
    assert len(result.patches) == 3


def test_save_and_load_default_config(tmp_path):
    config_path = tmp_path / "cellpatcher_config.json"
    save_default_config(config_path)
    loaded = load_config(config_path)
    assert isinstance(loaded, CellPatcherConfig)
    assert loaded.mode == "cell"


def test_load_config_nested_types_are_dataclasses(tmp_path):
    """load_config must reconstruct all nested sub-configs as dataclass instances."""
    config_path = tmp_path / "cfg.json"
    save_default_config(config_path)
    loaded = load_config(config_path)

    assert isinstance(loaded.common, CommonConfig)
    assert isinstance(loaded.grid, GridConfig)
    # Nested field access must work without AttributeError
    assert loaded.common.image_axes == "YX"
    assert loaded.grid.tile_size == 128


def test_load_config_is_usable_for_patching(tmp_path):
    """A config saved and loaded must work end-to-end with run_patching_pipeline."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.arange(64).reshape(8, 8))

    config_path = tmp_path / "cfg.json"
    save_default_config(config_path)
    loaded = load_config(config_path)
    # Switch to grid so no mask folder is needed
    loaded.mode = "grid"
    loaded.grid.tile_size = 4
    file_run_config(loaded, img_dir)

    result = run_patching_pipeline(img_dir, config=loaded)
    assert len(result.patches) == 4


# ---------------------------------------------------------------------------
# Issue #45/#46: ambiguous pairs are refused; system files are skipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["._a.tif", ".hidden.tif"])
def test_list_supported_files_skips_system_files(tmp_path, name):
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_supported_files(tmp_path)] == ["a.tif"]


def _folders(tmp_path, images, masks):
    """An image folder and a mask folder beside it, with real (tiny) files."""
    img_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for name in images:
        write_array(img_dir / name, np.zeros((4, 4), np.uint8))
    for name in masks:
        (mask_dir / name).parent.mkdir(parents=True, exist_ok=True)
        write_array(mask_dir / name, np.ones((4, 4), np.uint16))
    return img_dir, mask_dir


# ---------------------------------------------------------------------------
# validate_image_and_mask_folders reads the preprocessing run's pairing (v0.2.0,
# CP-2b): CellPatcher pairs no file itself. Its former checks (ambiguous names,
# formats, header problems, masks inside the images) are the preprocessing run's.
# ---------------------------------------------------------------------------


def _validated(tmp_path, images, masks):
    """Preprocess an image folder with its masks folder; return both folders and the run."""
    img_dir, mask_dir = _folders(tmp_path, images, masks)
    return img_dir, mask_dir, preprocess(img_dir, masks=mask_dir)


def test_validate_needs_a_preprocessing_run(tmp_path):
    from cellpatcher.statistics import PreprocessingCheckError

    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    # Old main paired the folders itself (find_masks) and returned {a.tif: a_cells.tif}.
    with pytest.raises(PreprocessingCheckError, match=r"(?s)validate_image_and_mask_folders reads files only through a clean preprocessing run.*preprocessing_run= is not set.*morphobricks-preprocessing check"):
        validate_image_and_mask_folders(img_dir, mask_dir)


@pytest.mark.parametrize(
    ("masks", "mask_type", "expected"),
    [
        (["a_cells.tif"], None, "a_cells.tif"),
        (["nuclei/a.tif"], None, "nuclei/a.tif"),
        (["a_cells.tif", "a_nuclei.tif"], "nuclei", "a_nuclei.tif"),
    ],
)
def test_validate_returns_the_runs_pairing(tmp_path, masks, mask_type, expected):
    img_dir, mask_dir, run = _validated(tmp_path, ["a.tif"], masks)
    out = validate_image_and_mask_folders(img_dir, mask_dir, preprocessing_run=run, mask_type=mask_type)
    assert out == {img_dir.resolve() / "a.tif": mask_dir.resolve() / expected}


def test_validate_reads_a_moved_experiment_where_it_is_now_and_refuses_a_changed_mask(tmp_path):
    import shutil

    from cellpatcher.statistics import PreprocessingCheckError

    img_dir, mask_dir, run = _validated(tmp_path, ["a.tif"], ["a_cells.tif"])
    moved = tmp_path / "moved"
    for folder in (img_dir, mask_dir):
        shutil.copytree(folder, moved / folder.name, copy_function=shutil.copy2)
        shutil.rmtree(folder)
    out = validate_image_and_mask_folders(moved / "images", moved / "masks", preprocessing_run=run)
    assert out == {(moved / "images").resolve() / "a.tif": (moved / "masks").resolve() / "a_cells.tif"}
    write_array(moved / "masks" / "a_cells.tif", np.full((4, 4), 2, np.uint16))
    with pytest.raises(PreprocessingCheckError, match=r"cannot be used: 1 mask changed or missing since the run: a_cells.tif"):
        validate_image_and_mask_folders(moved / "images", moved / "masks", preprocessing_run=run)


def test_validate_needs_a_mask_type_choice_and_a_mask_for_every_image(tmp_path):
    img_dir, mask_dir, run = _validated(tmp_path, ["a.tif", "b.tif"], ["a_cells.tif", "b_cells.tif", "a_nuclei.tif"])
    with pytest.raises(ValueError, match=r"recorded 2 mask types \(cells, nuclei\); set matching.mask_type"):
        validate_image_and_mask_folders(img_dir, mask_dir, preprocessing_run=run)
    out = validate_image_and_mask_folders(img_dir, mask_dir, preprocessing_run=run, mask_type="cells")
    assert {image.name: mask.name for image, mask in out.items()} == {"a.tif": "a_cells.tif", "b.tif": "b_cells.tif"}
    # An image without a mask of the chosen type is refused, not left out.
    with pytest.raises(ValueError, match=r"recorded no 'nuclei' mask for b.tif; every image needs one"):
        validate_image_and_mask_folders(img_dir, mask_dir, preprocessing_run=run, mask_type="nuclei")
    with pytest.raises(ValueError, match=r"'Cells' is not a mask type of the preprocessing run"):
        validate_image_and_mask_folders(img_dir, mask_dir, preprocessing_run=run, mask_type="Cells")


def test_a_mask_the_run_did_not_pair_is_refused_not_dropped(tmp_path):
    # masks/a.tif is no image's mask for the preprocessing run (a mask has a type): its run
    # records no mask, so validation refuses instead of returning an empty pairing.
    img_dir, mask_dir, run = _validated(tmp_path, ["a.tif"], ["a.tif"])
    with pytest.raises(ValueError, match="reads masks, but the preprocessing run .* recorded none"):
        validate_image_and_mask_folders(img_dir, mask_dir, preprocessing_run=run)


def test_a_folder_run_refuses_a_folder_of_image_subfolders(tmp_path):
    img_dir, _ = _folders(tmp_path, ["ctrl/a.tif", "treated/b.tif"], [])
    config = file_run_config(CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4)), img_dir)
    with pytest.raises(ValueError, match=r"run has 2 conditions or biological repeats.*run_patching_condition_repeat_tree"):
        run_patching_pipeline(img_dir, config=config)


def test_the_former_matching_options_are_refused_by_name(tmp_path):
    # Removed in v0.1.1 (D86, owner decision): refused whatever their value, old defaults included.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_mask.tif"])
    with pytest.raises(ValueError, match="strategy were removed in CellPatcher v0.1.1.*mask_type"):
        validate_image_and_mask_folders(img_dir, mask_dir, strategy="exact_or_suffix")
    with pytest.raises(ValueError, match="set matching.mask_type='cells'"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_suffixes=("_cells",), ignore_unrequested_masks=True)
    with pytest.raises(ValueError, match="case_sensitive were removed"):
        run_patching_pipeline(img_dir, mask_folder_path=mask_dir, case_sensitive=False)
    with pytest.raises(ValueError, match="mask_suffixes were removed"):
        run_patching_condition_repeat_tree(img_dir, mask_root_path=mask_dir, mask_suffixes=("_mask",))
    with pytest.raises(TypeError, match=r"unknown option\(s\) \['colour'\]"):
        validate_image_and_mask_folders(img_dir, mask_dir, colour="red")


def test_the_former_matchers_are_gone():
    import cellpatcher
    from cellpatcher import io

    for name in ("find_candidate_matches", "filter_shared_matches", "select_requested_masks", "image_stem",
                 "LEGACY_MATCHING_DEFAULTS", "check_legacy_matching", "warn_legacy_matching_removed"):
        assert not hasattr(io, name), name
        assert name not in cellpatcher.__all__
    # CellPatcher's own pairing, removed in v0.2.0 (CP-2b): validation reads the run's pairing.
    for name in ("validate_image_and_mask_folders", "pair_masks", "MaskPairing", "discover_folder_images"):
        assert not hasattr(io, name), name



@pytest.mark.parametrize("mask_axes", ["CYX", "YXC"])
def test_mask_axes_cannot_read_a_named_z_stack_as_label_channels(tmp_path, mask_axes):
    # A mask whose header names a Z axis passes find_masks. common.mask_axes with a C axis
    # used to override that header and read the Z slices as label channels (with the former
    # cell.mask_channel). Since CP-2a file runs read masks in the roles the preprocessing run
    # recorded, and refuse any common.mask_axes but the default with a migration message.
    from cellpatcher import CellPatcherConfig

    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], [])
    labels = np.zeros((4, 4), np.uint16)
    labels[1:3, 1:3] = 1
    write_tiff(mask_dir / "a_cells.tif", np.stack([labels, labels]), "ZYX")
    config = CellPatcherConfig(mode="cell")
    config.common.mask_axes = mask_axes
    config.cell.exclude_border = False
    with pytest.raises(ValueError, match=rf"common\.mask_axes='{mask_axes}': each image's \(and mask's\) axes are the roles"):
        run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)
