import csv

import numpy as np

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.pipeline import run_patching
from tests.helpers import make_synthetic_cell_image_and_labels


def test_cell_pipeline_extracts_expected_cells():
    image, labels, meta = make_synthetic_cell_image_and_labels()

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.exclude_border = True
    cfg.cell.min_area = 10
    cfg.cell.patch_size = 32
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = None

    result = run_patching(images=image, masks=labels, config=cfg)
    got_labels = sorted([p.label_id for p in result.patches])
    assert got_labels == meta["expected_valid_labels"]
    assert result.patch_arrays is not None
    assert len(result.patch_arrays) == len(result.patches)
    assert all(arr.shape == (32, 32) for arr in result.patch_arrays)


def test_cell_pipeline_resize():
    image, labels, _ = make_synthetic_cell_image_and_labels()

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.exclude_border = True
    cfg.cell.min_area = 10
    cfg.cell.patch_size = 32
    cfg.common.extract_arrays = True
    cfg.extraction.output_size = 64
    cfg.extraction.resize_order = 0

    result = run_patching(images=image, masks=labels, config=cfg)
    assert result.patch_arrays is not None
    assert all(arr.shape == (64, 64) for arr in result.patch_arrays)


def test_cell_pipeline_export_metadata(tmp_path):
    image, labels, meta = make_synthetic_cell_image_and_labels()

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.exclude_border = True
    cfg.cell.min_area = 10
    cfg.cell.patch_size = 32
    cfg.common.output_dir = str(tmp_path)
    cfg.common.extract_arrays = False
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_metadata_csv = True

    result = run_patching(images=image, masks=labels, config=cfg)
    assert len(result.exported_files) == len(meta["expected_valid_labels"])
    assert all(np.load(path).shape == (32, 32) for path in result.exported_files)

    csv_path = tmp_path / "patches.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    label_ids = sorted(int(r["label_id"]) for r in rows if r["label_id"] not in ("", "None"))
    assert label_ids == meta["expected_valid_labels"]


def test_background_filter_integration_reduces_grid_patches():
    image, _, _ = make_synthetic_cell_image_and_labels()

    base = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16, overlap=0.0, include_partial=False))
    unfiltered = run_patching(images=image, config=base)

    filtered_cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16, overlap=0.0, include_partial=False))
    filtered_cfg.filters.min_foreground_fraction = 0.1
    filtered_cfg.filters.foreground_threshold = "mean"
    filtered_cfg.common.extract_arrays = True

    filtered = run_patching(images=image, config=filtered_cfg)
    assert len(filtered.patches) < len(unfiltered.patches)
    assert filtered.patch_arrays is not None
    img_mean = float(np.mean(image))
    for arr in filtered.patch_arrays:
        assert float(np.mean(arr > img_mean)) >= 0.1
