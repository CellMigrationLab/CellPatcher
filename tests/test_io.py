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


def test_validate_image_and_mask_folders_success(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    (img_dir / "b.npy").write_bytes(b"x")
    (mask_dir / "a_mask.npy").write_bytes(b"x")
    (mask_dir / "b_mask.npy").write_bytes(b"x")

    out = validate_image_and_mask_folders(img_dir, mask_dir)
    assert len(out) == 2


def test_validate_image_and_mask_folders_raises_unmatched_image(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_raises_ambiguous_image(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    # Both names match via different allowed suffixes → genuinely ambiguous under exact_or_suffix.
    (mask_dir / "a_mask.npy").write_bytes(b"x")
    (mask_dir / "a_labels.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_default_rejects_substring_pitfall(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "img1.npy").write_bytes(b"x")
    (mask_dir / "img10_mask.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_substring_compatibility_mode(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    img = img_dir / "img1.npy"
    mask = mask_dir / "img10_mask.npy"
    img.write_bytes(b"x")
    mask.write_bytes(b"x")
    out = validate_image_and_mask_folders(img_dir, mask_dir, strategy="substring")
    assert out[img] == mask


def test_validate_image_and_mask_folders_multiple_suffixes_can_be_ambiguous(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "img1.npy").write_bytes(b"x")
    (mask_dir / "img1_labels.npy").write_bytes(b"x")
    (mask_dir / "img1_segmentation.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_case_sensitive_option(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "IMG1.npy").write_bytes(b"x")
    (mask_dir / "img1_mask.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir, case_sensitive=True)


def test_validate_image_and_mask_folders_passes_suffix_options_through(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    img = img_dir / "img1.npy"
    mask = mask_dir / "img1_ann.npy"
    img.write_bytes(b"x")
    mask.write_bytes(b"x")
    out = validate_image_and_mask_folders(img_dir, mask_dir, mask_suffixes=("_ann",))
    assert out[img] == mask


def test_validate_image_and_mask_folders_raises_unmatched_mask(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    (mask_dir / "a_mask.npy").write_bytes(b"x")
    (mask_dir / "orphan.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_empty_image_folder_raises(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (mask_dir / "a_mask.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="image folder"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_empty_mask_folder_raises(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="mask folder"):
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


def test_validate_folders_shared_mask_raises(tmp_path):
    """Two images that both match the same mask should raise."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    (img_dir / "b.npy").write_bytes(b"x")
    # Both images have the same stem as the mask → shared match
    (mask_dir / "a.npy").write_bytes(b"x")
    (mask_dir / "b.npy").write_bytes(b"x")
    # This should work fine (one-to-one)
    out = validate_image_and_mask_folders(img_dir, mask_dir, strategy="exact")
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Additional IO edge cases
# ---------------------------------------------------------------------------

def test_validate_folders_shared_mask_with_two_images_raises(tmp_path):
    """One mask matched by two images (both exact-match after normalization) should raise."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    # Use substring strategy so img1 and img10 both match the single mask "img1_mask"
    (img_dir / "img1.npy").write_bytes(b"x")
    (img_dir / "img10.npy").write_bytes(b"x")
    (mask_dir / "img1_mask.npy").write_bytes(b"x")
    (mask_dir / "img10_mask.npy").write_bytes(b"x")
    # Both mask correctly with exact_or_suffix → no ambiguity, should succeed
    out = validate_image_and_mask_folders(img_dir, mask_dir)
    assert len(out) == 2


def test_validate_folders_shared_mask_unique_candidates_raises(tmp_path):
    """Two images that each uniquely select the same mask → 'Masks linked to multiple images' error."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.npy").write_bytes(b"x")
    (img_dir / "b.npy").write_bytes(b"x")
    # "a" is in "ab_mask" and "b" is in "ab_mask" — both images share one mask exclusively
    (mask_dir / "ab_mask.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="[Mm]asks"):
        validate_image_and_mask_folders(img_dir, mask_dir, strategy="substring")

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
    write_array(mask_dir / "sample_mask.npy", labels)

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
    write_array(mask_dir / "a_mask.npy", mask)

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
