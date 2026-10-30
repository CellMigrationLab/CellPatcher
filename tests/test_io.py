from pathlib import Path

import numpy as np
import pytest

from cellpatcher.io import (
    filter_shared_matches,
    find_candidate_matches,
    list_supported_files,
    read_array,
    validate_image_and_mask_folders,
    write_array,
)


def test_list_supported_files_sorted_and_filtered(tmp_path):
    (tmp_path / "b.npy").write_bytes(b"x")
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x")
    files = list_supported_files(tmp_path)
    assert [p.name for p in files] == ["a.tif", "b.npy"]


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


def test_find_candidate_matches_exact_strategy():
    images = [Path("img1.tif")]
    masks = [Path("img1.tif"), Path("other.tif")]
    matches = find_candidate_matches(images, masks, strategy="exact")
    assert matches[images[0]] == [masks[0]]


def test_find_candidate_matches_default_exact_or_suffix():
    images = [Path("img1.tif")]
    masks = [Path("img1_mask.tif"), Path("other.tif")]
    matches = find_candidate_matches(images, masks)
    assert matches[images[0]] == [masks[0]]


def test_find_candidate_matches_default_does_not_match_substring_pitfall():
    images = [Path("img1.tif")]
    masks = [Path("img10_mask.tif")]
    matches = find_candidate_matches(images, masks)
    assert matches[images[0]] == []


def test_find_candidate_matches_substring_compatibility_mode():
    images = [Path("img1.tif")]
    masks = [Path("img10_mask.tif")]
    matches = find_candidate_matches(images, masks, strategy="substring")
    assert matches[images[0]] == [masks[0]]


def test_find_candidate_matches_case_insensitive_by_default():
    images = [Path("IMG1.TIF")]
    masks = [Path("img1_mask.TIF")]
    matches = find_candidate_matches(images, masks)
    assert matches[images[0]] == [masks[0]]


def test_find_candidate_matches_case_sensitive_option():
    images = [Path("IMG1.TIF")]
    masks = [Path("img1_mask.TIF")]
    matches = find_candidate_matches(images, masks, case_sensitive=True)
    assert matches[images[0]] == []


def test_find_candidate_matches_invalid_strategy_raises():
    images = [Path("img1.tif")]
    masks = [Path("img1_mask.tif")]
    with pytest.raises(ValueError):
        find_candidate_matches(images, masks, strategy="bad")


def test_find_candidate_matches_invalid_mask_suffixes_raises():
    images = [Path("img1.tif")]
    masks = [Path("img1_mask.tif")]
    with pytest.raises(ValueError):
        find_candidate_matches(images, masks, mask_suffixes=[1])


def test_find_candidate_matches_invalid_case_sensitive_raises():
    images = [Path("img1.tif")]
    masks = [Path("img1_mask.tif")]
    with pytest.raises(ValueError):
        find_candidate_matches(images, masks, case_sensitive="yes")


def test_filter_shared_matches_prefers_exclusive_candidates():
    img1 = Path("a.tif")
    img2 = Path("ab.tif")
    m1 = Path("a_mask.tif")
    m2 = Path("ab_mask.tif")
    candidate = {img1: [m1, m2], img2: [m2]}
    out = filter_shared_matches(candidate)
    assert out[img1] == [m1]
    assert out[img2] == [m2]


def test_filter_shared_matches_iterative_pruning_converges():
    # img1 starts with three candidates; img2 and img3 each hold one exclusively.
    # First pass prunes img1 to its own exclusive candidate (m1).
    # Second pass finds the result is already stable.
    img1 = Path("img1.tif")
    img2 = Path("img2.tif")
    img3 = Path("img3.tif")
    m1 = Path("m1.tif")
    m2 = Path("m2.tif")
    m3 = Path("m3.tif")
    candidate = {img1: [m1, m2, m3], img2: [m2], img3: [m3]}
    out = filter_shared_matches(candidate)
    assert out[img1] == [m1]
    assert out[img2] == [m2]
    assert out[img3] == [m3]


def test_validate_image_and_mask_folders_empty_image_folder_raises(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(mask_dir / "a_mask.tif", np.ones((4, 4), np.uint8))
    with pytest.raises(ValueError, match="image folder"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_empty_mask_folder_raises(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="No mask in .* pairs with an image"):
        validate_image_and_mask_folders(img_dir, mask_dir)


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
    write_array(img_dir / "a.npy", np.arange(64).reshape(8, 8))

    config = CellPatcherConfig(mode="grid")
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

    write_array(img_dir / "sample.npy", image)
    write_array(mask_dir / "sample_mask.tif", labels)

    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 8

    result = run_patching_from_folders(img_dir, mask_folder_path=mask_dir, config=config)
    assert len(result.patches) == 1


def test_run_patching_from_folders_cell_requires_mask_folder(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.npy", np.zeros((8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="mask_folder_path"):
        run_patching_from_folders(img_dir, config=CellPatcherConfig(mode="cell"))

from cellpatcher.orchestration import load_config, run_patching_pipeline, save_default_config


def test_run_patching_pipeline_alias_grid_mode(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.npy", np.arange(64).reshape(8, 8))

    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 4
    result = run_patching_pipeline(img_dir, config=config)
    assert len(result.patches) == 4


def test_run_patching_pipeline_grid_raises_if_mask_folder_provided(tmp_path):
    """grid mode must not silently ignore a provided mask folder."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "a.npy", np.zeros((8, 8)))
    write_array(mask_dir / "a_mask.npy", np.zeros((8, 8), dtype=np.int32))

    config = CellPatcherConfig(mode="grid")
    with pytest.raises(ValueError, match="mask_folder_path"):
        run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)


def test_run_patching_pipeline_whole_fov_raises_if_mask_folder_provided(tmp_path):
    """whole_fov mode must not silently ignore a provided mask folder."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "a.npy", np.zeros((8, 8)))
    write_array(mask_dir / "a_mask.npy", np.zeros((8, 8), dtype=np.int32))

    config = CellPatcherConfig(mode="whole_fov")
    with pytest.raises(ValueError, match="mask_folder_path"):
        run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)


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
    write_array(img_dir / "a.npy", img)
    write_array(mask_dir / "a_mask.tif", mask)

    config = CellPatcherConfig(mode="random")
    config.random.patch_size = 4
    config.random.n_patches = 5
    config.random.seed = 0
    result = run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)
    assert len(result.patches) == 5


def test_run_patching_pipeline_random_mode_without_mask(tmp_path):
    """random mode works without a mask folder."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.npy", np.zeros((32, 32), dtype=np.float32))

    config = CellPatcherConfig(mode="random")
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
    write_array(img_dir / "a.npy", np.arange(64).reshape(8, 8))

    config_path = tmp_path / "cfg.json"
    save_default_config(config_path)
    loaded = load_config(config_path)
    # Switch to grid so no mask folder is needed
    loaded.mode = "grid"
    loaded.grid.tile_size = 4

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


def _preprocessing_pairs(img_dir, mask_dir):
    """The preprocessing brick's own pairing of the same folders, run as the brick runs it.

    ``discover_experiment`` + ``inspect_images`` + ``find_masks`` on TIFF
    images: ``{image name: (mask type, mask path)}``, or the error codes.
    """
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.metadata import inspect_images

    discovery = mbp.discover_experiment(img_dir)
    images = inspect_images(img_dir, discovery.images).files
    inventory = mbp.find_masks(mask_dir, discovery, images, ome_zarr_labels=False)
    errors = sorted(issue.code for issue in inventory.issues if issue.is_error)
    if errors:
        return errors
    return {f.image: (f.mask_type, f.path) for f in inventory.files}


# ---------------------------------------------------------------------------
# Issue #45: CellPatcher pairs images and masks with the preprocessing brick's
# rule (find_masks), so both bricks pair the same files. Each case below was
# paired differently by CellPatcher's former rule (exact_or_suffix with the
# suffixes _mask/_labels/_label/_seg/_segmentation), shown in the comment.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("masks", "mask_type", "expected"),
    [
        # Any mask type after "<image>_": former rule: "Images with no mask match".
        (["a_cells.tif"], None, {"a.tif": ("cells", "a_cells.tif")}),
        # The <mask type>/ folder layout: former rule: no mask files found (no recursion).
        (["nuclei/a.tif"], None, {"a.tif": ("nuclei", "nuclei/a.tif")}),
        # Several types, one chosen: former rule: refused unless ignore_unrequested_masks.
        (["a_cells.tif", "a_nuclei.tif"], "nuclei", {"a.tif": ("nuclei", "a_nuclei.tif")}),
    ],
)
def test_pairing_is_the_preprocessing_bricks(tmp_path, masks, mask_type, expected):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], masks)
    preprocessing = _preprocessing_pairs(img_dir, mask_dir)
    assert {image: pair for image, pair in preprocessing.items() if mask_type in (None, pair[0])} == expected
    options = {} if mask_type is None else {"mask_type": mask_type}
    out = validate_image_and_mask_folders(img_dir, mask_dir, **options)
    assert {image.name: mask.relative_to(mask_dir).as_posix() for image, mask in out.items()} == {
        image: path for image, (_, path) in expected.items()
    }


def test_a_mask_name_starting_with_another_images_name_is_refused_like_preprocessing(tmp_path):
    # a_seg_mask.tif is "a" + "_seg_mask" and "a_seg" + "_mask". The former rule
    # paired it with a_seg (and a_mask.tif with a); preprocessing refuses it as
    # ambiguous, and so does CellPatcher now.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif", "a_seg.tif"], ["a_mask.tif", "a_seg_mask.tif"])
    assert _preprocessing_pairs(img_dir, mask_dir) == ["ambiguous_mask"]
    with pytest.raises(ValueError, match=r"\[ambiguous_mask\].*a_seg_mask.tif"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_a_mask_named_exactly_like_its_image_no_longer_pairs(tmp_path):
    # Former rule ("exact" and the default "exact_or_suffix"): masks/a.tif paired with a.tif.
    # Preprocessing reads it as no image's mask (a mask has a type), so it is refused, never dropped.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a.tif"])
    assert _preprocessing_pairs(img_dir, mask_dir) == {}
    with pytest.raises(ValueError, match=r"\[mask_without_image\].*\(a.tif\)"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_masks_that_are_not_tiff_files_are_refused_not_dropped(tmp_path):
    # Former rule: a_mask.npy paired with a.npy. Preprocessing pairs TIFF masks only.
    img_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "a.npy", np.zeros((4, 4), np.float32))
    write_array(mask_dir / "a_mask.npy", np.ones((4, 4), np.uint16))
    with pytest.raises(ValueError, match=r"\[mask_outside_type_folder\].*a_mask.npy"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_images_of_other_formats_pair_with_tiff_masks_by_position_id(tmp_path):
    # Images may still be PNG/NumPy/...: an image's name is its position_id (D85),
    # and both mask layouts may be mixed.
    img_dir, mask_dir = _folders(tmp_path, ["a.npy", "b.png"], ["a_cells.tif", "cells/b.tif"])
    out = validate_image_and_mask_folders(img_dir, mask_dir)
    assert {image.name: mask.relative_to(mask_dir).as_posix() for image, mask in out.items()} == {
        "a.npy": "a_cells.tif",
        "b.png": "cells/b.tif",
    }


def test_several_mask_types_need_a_choice(tmp_path):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif", "b.tif"], ["a_cells.tif", "b_cells.tif", "a_nuclei.tif"])
    with pytest.raises(ValueError, match=r"2 mask types \(cells, nuclei\); set matching.mask_type"):
        validate_image_and_mask_folders(img_dir, mask_dir)
    out = validate_image_and_mask_folders(img_dir, mask_dir, mask_type="cells")
    assert {image.name: mask.name for image, mask in out.items()} == {"a.tif": "a_cells.tif", "b.tif": "b_cells.tif"}
    # An image without a mask of the chosen type is refused, not left out.
    with pytest.raises(ValueError, match=r"Images with no 'nuclei' mask: b.tif"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="nuclei")
    with pytest.raises(ValueError, match=r"No mask of type 'Cells'.*cells, nuclei"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="Cells")


def test_a_mask_folder_inside_the_image_folder_is_refused_like_preprocessing(tmp_path):
    # Former rule: accepted (the image folder listing skips subfolders).
    img_dir = tmp_path / "images"
    write_array(img_dir / "a.tif", np.zeros((4, 4), np.float32))
    write_array(img_dir / "masks" / "a_mask.tif", np.ones((4, 4), np.uint16))
    with pytest.raises(ValueError, match=r"\[masks_inside_experiment\]"):
        validate_image_and_mask_folders(img_dir, img_dir / "masks")


def test_the_former_matching_options_are_refused_unless_at_their_old_defaults(tmp_path):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_mask.tif"])
    with pytest.raises(ValueError, match="strategy='substring' is no longer supported"):
        validate_image_and_mask_folders(img_dir, mask_dir, strategy="substring")
    with pytest.raises(ValueError, match="set matching.mask_type='cells'"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_suffixes=("_cells",), ignore_unrequested_masks=True)
    with pytest.warns(DeprecationWarning, match="no longer have any effect"):
        out = validate_image_and_mask_folders(img_dir, mask_dir, strategy="exact_or_suffix", case_sensitive=False)
    assert [mask.name for mask in out.values()] == ["a_mask.tif"]
    with pytest.raises(ValueError, match="case_sensitive=True"):
        run_patching_pipeline(img_dir, mask_folder_path=mask_dir, case_sensitive=True)


def test_the_former_matchers_warn_that_they_are_unused():
    with pytest.warns(DeprecationWarning, match="no longer used by CellPatcher"):
        find_candidate_matches(["a.tif"], ["a_mask.tif"])
    with pytest.warns(DeprecationWarning, match="no longer used by CellPatcher"):
        filter_shared_matches({})


def test_validate_ignores_macos_resource_forks_in_both_folders(tmp_path):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    for folder in (img_dir, mask_dir):
        (folder / "._a.tif").write_bytes(b"x")
        (folder / "._a_cells.tif").write_bytes(b"x")
    (mask_dir / "__MACOSX").mkdir()
    (mask_dir / "__MACOSX" / "._a_cells.tif").write_bytes(b"x")
    out = validate_image_and_mask_folders(img_dir, mask_dir)
    assert {image.name: mask.name for image, mask in out.items()} == {"a.tif": "a_cells.tif"}
