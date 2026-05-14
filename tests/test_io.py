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


def test_find_candidate_matches_substring_stems():
    images = [Path("img001.tif")]
    masks = [Path("img001_mask.tif"), Path("other.tif")]
    matches = find_candidate_matches(images, masks)
    assert matches[images[0]] == [masks[0]]


def test_filter_shared_matches_prefers_exclusive_candidates():
    img1 = Path("a.tif")
    img2 = Path("ab.tif")
    m1 = Path("a_mask.tif")
    m2 = Path("ab_mask.tif")
    candidate = {img1: [m1, m2], img2: [m2]}
    out = filter_shared_matches(candidate)
    assert out[img1] == [m1]
    assert out[img2] == [m2]


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
    (mask_dir / "a_mask1.npy").write_bytes(b"x")
    (mask_dir / "a_mask2.npy").write_bytes(b"x")
    with pytest.raises(ValueError):
        validate_image_and_mask_folders(img_dir, mask_dir)


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
