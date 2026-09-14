import csv
from pathlib import Path

import numpy as np
import pytest

from cellpatcher.export import (
    export_patches,
    patch_metadata_to_rows,
    sanitize_filename,
    save_metadata_csv,
    save_patch_array,
)
from cellpatcher.types import PatchRegion


def test_sanitize_filename_replaces_unsafe_characters():
    assert sanitize_filename("a b/c?d") == "a_b_c_d"


def test_patch_metadata_to_rows_returns_dicts():
    patches = [PatchRegion(patch_id="p1", y0=0, y1=1, x0=0, x1=1)]
    rows = patch_metadata_to_rows(patches)
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    assert rows[0]["patch_id"] == "p1"


def test_save_metadata_csv_writes_headers_and_rows(tmp_path):
    patches = [PatchRegion(patch_id="p1", y0=0, y1=1, x0=0, x1=1)]
    out = tmp_path / "meta.csv"
    save_metadata_csv(patches, out)
    with out.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert "patch_id" in reader.fieldnames
    assert rows[0]["patch_id"] == "p1"


def test_save_metadata_csv_empty_patch_list(tmp_path):
    out = tmp_path / "meta.csv"
    save_metadata_csv([], out)
    text = out.read_text(encoding="utf-8")
    assert "patch_id" in text


def test_save_patch_array_saves_npy(tmp_path):
    arr = np.arange(4)
    out = tmp_path / "a.npy"
    save_patch_array(arr, out)
    loaded = np.load(out)
    assert np.array_equal(loaded, arr)


def test_save_patch_array_tiff_optional_dependency_behavior(tmp_path):
    arr = np.arange(4).reshape(2, 2)
    out = tmp_path / "a.tif"
    try:
        save_patch_array(arr, out)
        assert out.exists()
    except ImportError as exc:
        assert "tifffile" in str(exc)


def test_export_patches_writes_one_file_per_patch(tmp_path):
    patches = [PatchRegion(patch_id="p1", y0=0, y1=1, x0=0, x1=1), PatchRegion(patch_id="p2", y0=0, y1=1, x0=0, x1=1)]
    arrays = [np.zeros((1, 1)), np.ones((1, 1))]
    files = export_patches(patches, arrays, tmp_path, image_format="npy", save_metadata=False)
    assert len(files) == 2
    assert all(Path(f).exists() for f in files)


def test_export_patches_writes_metadata_csv(tmp_path):
    patches = [PatchRegion(patch_id="p1", y0=0, y1=1, x0=0, x1=1)]
    arrays = [np.zeros((1, 1))]
    export_patches(patches, arrays, tmp_path, image_format="npy", save_metadata=True)
    assert (tmp_path / "patches.csv").exists()


def test_export_patches_rejects_mismatched_lengths(tmp_path):
    patches = [PatchRegion(patch_id="p1", y0=0, y1=1, x0=0, x1=1)]
    arrays = []
    with pytest.raises(ValueError):
        export_patches(patches, arrays, tmp_path)


def test_export_patches_rejects_unsupported_image_format(tmp_path):
    patches = [PatchRegion(patch_id="p1", y0=0, y1=1, x0=0, x1=1)]
    arrays = [np.zeros((1, 1))]
    with pytest.raises(ValueError):
        export_patches(patches, arrays, tmp_path, image_format="png")


def test_export_patches_rejects_duplicate_filenames_after_sanitization(tmp_path):
    patches = [
        PatchRegion(patch_id="a/b", y0=0, y1=1, x0=0, x1=1),
        PatchRegion(patch_id="a?b", y0=0, y1=1, x0=0, x1=1),
    ]
    arrays = [np.zeros((1, 1)), np.ones((1, 1))]
    with pytest.raises(ValueError, match="Duplicate export filenames"):
        export_patches(patches, arrays, tmp_path, image_format="npy")
