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
