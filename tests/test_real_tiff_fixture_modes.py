from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cellpatcher.cell_scan import scan_cell_patch_size
from cellpatcher.config import CellPatcherConfig, GridConfig, RandomConfig
from cellpatcher.pipeline import run_patching
from tests.fixtures.helpers import (
    REAL_TIFF_IMAGE_PATH,
    REAL_TIFF_LABELS_PATH,
    load_real_tiff_fixture,
    summarize_real_tiff_fixture,
)

EXPECTED_FIXTURE_SUMMARY = {
    "image_shape": (1024, 1024),
    "label_shape": (1024, 1024),
    "image_dtype": "uint16",
    "labels_dtype": "uint8",
    "n_foreground_labels": 10,
    "foreground_pixel_count": 686973,
    "image_min": 368,
    "image_max": 1463,
}

EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER = 2
EXPECTED_CELL_PATCH_COUNT_INCLUDE_BORDER = 10
EXPECTED_CELL_LABEL_IDS_INCLUDE_BORDER = set(range(1, 11))
EXPECTED_GRID_COUNT_64_NO_OVERLAP = 256
EXPECTED_GRID_COUNT_64_OVERLAP_PARTIAL = 484

# When `labels > 0` is converted to binary and connected-components labeled, the
# fixture's 10 cells merge spatially into exactly 2 large blobs — both of which
# touch the image border.  With exclude_border=False all 2 components are kept;
# with exclude_border=True none survive.  This constant is distinct from
# EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER (which counts per-label non-border
# cells in the *labeled* mask) even though both equal 2 for this fixture.
EXPECTED_BINARY_CC_PATCH_COUNT = 2


def _cell_config(*, patch_size="auto", center_on="bbox_center", apply_mask=False, binary_mask_policy="connected_components", exclude_border=True):
    cfg = CellPatcherConfig(mode="cell")
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.common.extract_arrays = True
    cfg.cell.patch_size = patch_size
    cfg.cell.exclude_border = exclude_border
    cfg.cell.min_area = 5
    cfg.cell.center_on = center_on
    cfg.cell.binary_mask_policy = binary_mask_policy
    cfg.extraction.output_size = 32
    cfg.extraction.apply_mask = apply_mask
    cfg.extraction.mask_fill_value = 0
    return cfg


def test_real_tiff_fixture_loads_and_has_expected_properties():
    image, labels = load_real_tiff_fixture()
    summary = summarize_real_tiff_fixture()

    assert image.ndim == 2
    assert labels.ndim == 2
    assert image.shape == labels.shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert 0 in np.unique(labels)
    assert np.any(labels > 0)
    assert REAL_TIFF_IMAGE_PATH.stat().st_size <= 4 * 1024 * 1024
    assert REAL_TIFF_LABELS_PATH.stat().st_size <= 4 * 1024 * 1024

    for key, expected in EXPECTED_FIXTURE_SUMMARY.items():
        assert summary[key] == expected


def test_real_tiff_cell_mode_auto_extract():
    image, labels = load_real_tiff_fixture()
    result = run_patching(images=image, masks=labels, config=_cell_config(exclude_border=True))

    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == len(result.patches) == EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)
    assert all(int(p.label_id) > 0 for p in result.patches if p.label_id is not None)


def test_real_tiff_cell_mode_all_labels_when_border_included():
    image, labels = load_real_tiff_fixture()
    result = run_patching(images=image, masks=labels, config=_cell_config(exclude_border=False))
    label_ids = {int(p.label_id) for p in result.patches if p.label_id is not None}

    assert len(result.patches) == EXPECTED_CELL_PATCH_COUNT_INCLUDE_BORDER
    assert label_ids == EXPECTED_CELL_LABEL_IDS_INCLUDE_BORDER


def test_real_tiff_cell_mode_scan_global_size_apply_mask():
    image, labels = load_real_tiff_fixture()
    summary = scan_cell_patch_size(image_shapes=image.shape, labels_list=labels, config=_cell_config().cell, image_axes="YX", mask_axes="YX")
    chosen_size = summary.recommended_size("p95") if summary.recommended_size("p95") > 0 else summary.recommended_size("max")

    result = run_patching(images=image, masks=labels, config=_cell_config(patch_size=chosen_size, apply_mask=True))

    assert summary.n_included == EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER
    assert len(result.patches) == summary.n_included
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == summary.n_included
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)
    assert any(np.any(arr == 0) for arr in result.patch_arrays)


@pytest.mark.parametrize("center_on", ["bbox_center", "centroid"])
def test_real_tiff_cell_mode_centroid_and_bbox_center_both_work(center_on):
    image, labels = load_real_tiff_fixture()
    result = run_patching(images=image, masks=labels, config=_cell_config(center_on=center_on, exclude_border=True))

    assert len(result.patches) == EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER
    assert result.patch_arrays is not None
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)


def test_real_tiff_cell_mode_binary_policy_connected_components():
    image, labels = load_real_tiff_fixture()
    binary = labels > 0
    with pytest.warns(UserWarning, match="connected components"):
        config = _cell_config(exclude_border=False, binary_mask_policy="connected_components")
        result = run_patching(images=image, masks=binary, config=config)

    # Both binary CCs in this fixture touch the image border; exclude_border=False
    # means they are kept, giving EXPECTED_BINARY_CC_PATCH_COUNT patches.
    assert len(result.patches) == EXPECTED_BINARY_CC_PATCH_COUNT
    assert result.patch_arrays is not None
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)
    # Confirm that exclude_border=False actually retains border-touching objects.
    assert any(p.touches_border for p in result.patches)


@pytest.mark.parametrize("resize_order", [0, 1])
def test_real_tiff_resize_orders_smoke(resize_order):
    image, labels = load_real_tiff_fixture()
    cfg = _cell_config(exclude_border=True)
    cfg.extraction.resize_order = resize_order
    result = run_patching(images=image, masks=labels, config=cfg)

    assert len(result.patches) == EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER
    assert result.patch_arrays is not None
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)


def test_real_tiff_random_mode_unconstrained():
    image, _ = load_real_tiff_fixture()
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=32, n_patches=8, seed=42))
    cfg.common.image_axes = "YX"
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = 32

    r1 = run_patching(images=image, config=cfg)
    r2 = run_patching(images=image, config=cfg)

    assert len(r1.patches) == len(r1.patch_arrays) == 8
    assert all(arr.shape == (32, 32) for arr in r1.patch_arrays)
    assert [(p.y0, p.x0, p.y1, p.x1) for p in r1.patches] == [(p.y0, p.x0, p.y1, p.x1) for p in r2.patches]


def test_real_tiff_random_mode_label_constrained():
    image, labels = load_real_tiff_fixture()
    cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=16, n_patches=4, seed=42))
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = 32

    mask = labels > 0
    result = run_patching(images=image, masks=mask, config=cfg)
    assert len(result.patches) == len(result.patch_arrays) == 4
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)
    # Every patch must overlap the foreground mask (mask-constrained sampling).
    assert all(np.any(mask[p.y0:p.y1, p.x0:p.x1]) for p in result.patches)


def test_real_tiff_grid_mode_no_overlap():
    image, _ = load_real_tiff_fixture()
    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=64, overlap=0.0, include_partial=False))
    cfg.common.image_axes = "YX"
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = 32

    result = run_patching(images=image, config=cfg)
    expected_count = (image.shape[0] // 64) * (image.shape[1] // 64)
    assert len(result.patches) == expected_count == EXPECTED_GRID_COUNT_64_NO_OVERLAP
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)


def test_real_tiff_grid_mode_overlap_partial():
    image, _ = load_real_tiff_fixture()
    base = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=64, overlap=0.0, include_partial=False))
    base.common.image_axes = "YX"
    over = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=64, overlap=0.25, include_partial=True))
    over.common.image_axes = "YX"
    over.common.extract_arrays = True
    over.extraction.output_size = 32

    base_result = run_patching(images=image, config=base)
    result = run_patching(images=image, config=over)

    coords = [(p.y0, p.x0, p.y1, p.x1) for p in result.patches]
    assert len(result.patches) > len(base_result.patches)
    assert len(result.patches) == EXPECTED_GRID_COUNT_64_OVERLAP_PARTIAL
    assert len(coords) == len(set(coords))
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)


def test_real_tiff_whole_fov_mode():
    image, _ = load_real_tiff_fixture()
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.image_axes = "YX"
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = 64
    result = run_patching(images=image, config=cfg)

    assert len(result.patches) == len(result.patch_arrays) == 1
    assert result.patch_arrays[0].shape == (64, 64)
    p = result.patches[0]
    assert (p.y0, p.x0, p.y1, p.x1) == (0, 0, image.shape[0], image.shape[1])


def test_real_tiff_filtering_smoke():
    image, labels = load_real_tiff_fixture()
    unfiltered = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=64, overlap=0.0, include_partial=True))
    unfiltered.common.image_axes = "YX"
    base_result = run_patching(images=image, config=unfiltered)

    filtered = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=64, overlap=0.0, include_partial=True))
    filtered.common.image_axes = "YX"
    filtered.common.mask_axes = "YX"
    filtered.filters.min_mean_intensity = float(np.mean(image) + 10.0)
    filtered.filters.min_foreground_fraction = 0.05
    filtered.filters.foreground_threshold = "mean"
    filtered_result = run_patching(images=image, masks=(labels > 0), config=filtered)

    # The intensity threshold (mean + 10) must eliminate a meaningful fraction of
    # patches — require strictly fewer, not merely ≤, to confirm filtering is active.
    assert len(filtered_result.patches) < len(base_result.patches)
    assert len(filtered_result.patches) > 0


@pytest.mark.parametrize("mode", ["cell", "random", "grid", "whole_fov"])
def test_real_tiff_export_all_modes_npy(tmp_path, mode):
    image, labels = load_real_tiff_fixture()
    out_dir = tmp_path / mode

    if mode == "cell":
        summary = scan_cell_patch_size(image_shapes=image.shape, labels_list=labels, config=_cell_config().cell, image_axes="YX", mask_axes="YX")
        cfg = _cell_config(patch_size=summary.recommended_size("p95") or summary.recommended_size("max"))
        expected = summary.n_included
        masks = labels
    elif mode == "random":
        cfg = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=32, n_patches=3, seed=42))
        cfg.common.image_axes = "YX"
        expected = 3
        masks = None
    elif mode == "grid":
        cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=128, overlap=0.0, include_partial=False))
        cfg.common.image_axes = "YX"
        expected = (image.shape[0] // 128) * (image.shape[1] // 128)
        masks = None
    else:
        cfg = CellPatcherConfig(mode="whole_fov")
        cfg.common.image_axes = "YX"
        expected = 1
        masks = None

    cfg.common.extract_arrays = False
    cfg.common.output_dir = str(out_dir)
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True
    cfg.extraction.output_size = 32 if mode != "whole_fov" else 64

    result = run_patching(images=image, masks=masks, config=cfg)
    assert len(result.patches) == expected
    assert result.patch_arrays is None  # extract_arrays=False; arrays written to disk, not returned
    assert len(result.exported_files) == len(result.patches)
    assert all(Path(p).exists() for p in result.exported_files)
    assert (out_dir / "patches.csv").exists()


def test_real_tiff_cell_mode_single_object_and_area_filters():
    image, labels = load_real_tiff_fixture()
    cfg = _cell_config(exclude_border=False, binary_mask_policy="single_object")
    cfg.cell.min_area = 10000
    cfg.cell.max_area = 800000
    # A binary mask as integers is one object (decision D47: a bool mask is refused).
    result = run_patching(images=image, masks=(labels > 0).astype(np.uint8), config=cfg)
    assert len(result.patches) == 1
    assert result.patch_arrays is not None
    assert result.patch_arrays[0].shape == (32, 32)


def test_real_tiff_cell_mode_mask_channel_noop_for_2d_labels():
    image, labels = load_real_tiff_fixture()
    cfg = _cell_config(exclude_border=True)
    cfg.cell.mask_channel = 0
    result = run_patching(images=image, masks=labels, config=cfg)
    assert len(result.patches) == EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER


@pytest.mark.parametrize("pad_mode,constant_values", [("clip", 0.0), ("constant", 7.0)])
def test_real_tiff_padding_modes_smoke(pad_mode, constant_values):
    image, labels = load_real_tiff_fixture()
    cfg = _cell_config(exclude_border=True)
    cfg.extraction.pad_mode = pad_mode
    cfg.extraction.constant_values = constant_values
    result = run_patching(images=image, masks=labels, config=cfg)
    # Smoke test: verify the config is accepted and patches are produced.
    # Note: run_patching clips all PatchRegion coordinates to image bounds before
    # extraction, so constant padding (constant_values) is never triggered through
    # this code path.  Behavioral coverage for out-of-bounds constant padding is
    # provided by tests/test_extract.py (e.g. test_extract_patch_constant_padding_preserves_requested_shape).
    assert len(result.patches) == EXPECTED_CELL_PATCH_COUNT_EXCLUDE_BORDER
    assert result.patch_arrays is not None


def test_real_tiff_output_size_tuple_smoke():
    image, labels = load_real_tiff_fixture()
    cfg = _cell_config(exclude_border=True)
    cfg.extraction.output_size = (24, 40)
    result = run_patching(images=image, masks=labels, config=cfg)
    assert result.patch_arrays is not None
    assert all(arr.shape == (24, 40) for arr in result.patch_arrays)


def test_real_tiff_export_tiff_and_no_metadata_csv(tmp_path):
    image, _ = load_real_tiff_fixture()
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.common.image_axes = "YX"
    cfg.common.output_dir = str(tmp_path)
    cfg.common.extract_arrays = False
    cfg.export.enabled = True
    cfg.export.image_format = "tiff"
    cfg.export.save_metadata_csv = False
    result = run_patching(images=image, config=cfg)
    assert len(result.patches) == 1
    assert len(result.exported_files) == 1
    assert Path(result.exported_files[0]).suffix in {".tif", ".tiff"}
    assert not (tmp_path / "patches.csv").exists()
