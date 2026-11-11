"""The per-position table ``positions.csv`` written beside ``images.csv`` and ``patches.csv`` (issue #26).

One row per position (FOV key), grouping every source image, plane and channel of it, with a
status and reason from what CellPatcher observed and the counts behind ``n_patches``.
"""

import csv
import importlib.util
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExportConfig, GridConfig, NormalizationConfig
from cellpatcher.export import IMAGE_TABLE_FILENAME, export_patches
from cellpatcher.file_runs import fov_metadata
from cellpatcher.pipeline import run_patching
from cellpatcher.positions import (
    POSITION_STATUSES,
    POSITION_TABLE_COLUMNS,
    POSITION_TABLE_FILENAME,
    POSITION_TABLE_VERSION,
    position_records,
)
from cellpatcher.types import IMAGE_TABLE_COLUMNS, ImageRecord, PatchRegion
from tests.helpers import cell_folders, cell_tree, preprocessed_folder, preprocessed_tree

KEY = ("condition", "biological_repeat", "technical_repeat", "position_id")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _assert_positions_join(folder):
    """Every position's row is unique by its key and counts exactly its rows of patches.csv."""
    _, patch_rows = _read_csv(folder / "patches.csv")
    header, rows = _read_csv(folder / POSITION_TABLE_FILENAME)
    assert header == list(POSITION_TABLE_COLUMNS)
    positions = {tuple(row[column] for column in KEY): row for row in rows}
    assert len(positions) == len(rows), "the FOV key must name one position each"
    counts = {}
    for row in patch_rows:
        key = tuple(row[column] for column in KEY)
        assert key in positions, f"{row['patch_file']} comes from a position missing in positions.csv: {key}"
        counts[key] = counts.get(key, 0) + 1
    for key, row in positions.items():
        assert int(row["n_patches"]) == counts.get(key, 0)
        assert row["status"] in POSITION_STATUSES and row["schema_version"] == POSITION_TABLE_VERSION
        assert (row["status"] == "analysed") == (row["reason"] == "")
    return patch_rows, rows



def _cell_config(output_dir):
    config = CellPatcherConfig(mode="cell", common=CommonConfig(output_dir=str(output_dir)))
    config.cell.patch_size = 24
    config.export = ExportConfig(enabled=True, image_format="npy")
    return config


def _grid_config(output_dir, **common):
    return CellPatcherConfig(
        mode="grid",
        common=CommonConfig(output_dir=str(output_dir), **common),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )


# ---------------------------------------------------------------------------
# Folder run
# ---------------------------------------------------------------------------


def test_folder_run_writes_one_row_per_position(tmp_path):
    images, masks = cell_folders(tmp_path)
    result = preprocessed_folder(images, mask_folder_path=masks, config=_cell_config(tmp_path / "out"))

    out = tmp_path / "out"
    assert result.position_table_file == str(out / POSITION_TABLE_FILENAME)
    patch_rows, rows = _assert_positions_join(out)
    assert len(patch_rows) == 8
    assert [row["position_id"] for row in rows] == ["a", "b", "c"]
    assert rows[0] == {
        "condition": "single", "biological_repeat": "single", "technical_repeat": "single",
        "position_id": "a", "repeat_group": "single", "condition_id": "single", "technical_repeat_id": "single",
        "n_source_images": "1", "source_relative_paths": '["a.tif"]', "image_files": '["a.tif"]',
        "mask_files": '["a_mask.tif"]', "status": "analysed", "reason": "",
        "n_regions": "4", "n_kept": "4", "n_normalized": "4", "n_patches": "4",
        "schema_version": "1",
    }
    # The position whose mask holds no cell is a row of its own, analysed with zero regions.
    assert (rows[1]["status"], rows[1]["n_regions"], rows[1]["n_patches"]) == ("analysed_no_regions", "0", "0")
    assert rows[1]["reason"] == "the patch mode generated no region in this image"
    # images.csv is the unchanged per-image table.
    header, _ = _read_csv(out / IMAGE_TABLE_FILENAME)
    assert header == list(IMAGE_TABLE_COLUMNS) and "repeat_group" not in header


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------


def test_zero_regions_all_filtered_and_analysed_are_told_apart(tmp_path):
    config = _grid_config(tmp_path)
    config.filters.min_mean_intensity = 50
    bright = np.full((32, 32), 100, np.uint16)
    dark = np.zeros((32, 32), np.uint16)
    tiny = np.full((8, 8), 100, np.uint16)  # smaller than a tile: the grid generates no region
    run_patching(
        images=[bright, dark, tiny], config=config,
        image_metadata=[{"position_id": "P1"}, {"position_id": "P2"}, {"position_id": "P3"}],
    )

    _, rows = _assert_positions_join(tmp_path)
    assert [(r["position_id"], r["status"], r["n_regions"], r["n_kept"], r["n_patches"]) for r in rows] == [
        ("P1", "analysed", "4", "4", "4"),
        ("P2", "all_regions_filtered", "4", "0", "0"),
        ("P3", "analysed_no_regions", "0", "0", "0"),
    ]
    assert rows[1]["reason"] == "the filters removed all 4 region(s)"


@pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs morphobricks-preprocessing"
)
def test_all_regions_rejected_by_normalization(tmp_path):
    varied = np.arange(256, dtype=np.uint8).reshape(16, 16)
    constant = np.full((16, 16), 7, np.uint8)
    config = _grid_config(tmp_path, image_axes="YX")
    config.grid = GridConfig(tile_size=8)
    config.normalization = NormalizationConfig(enabled=True, method="minmax", scope="patch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_patching(
            images=[varied, constant], config=config,
            image_metadata=[{"position_id": "P1"}, {"position_id": "P2"}],
        )

    _, rows = _assert_positions_join(tmp_path)
    assert [(r["position_id"], r["status"]) for r in rows] == [
        ("P1", "analysed"), ("P2", "all_rejected_by_normalization"),
    ]
    assert rows[1]["reason"].startswith("normalization discarded all 4 patch(es)")
    assert (rows[1]["n_regions"], rows[1]["n_kept"], rows[1]["n_normalized"], rows[1]["n_patches"]) == (
        "4", "4", "0", "0",
    )


def test_an_image_without_planes_is_a_skipped_position(tmp_path):
    config = _grid_config(tmp_path, image_axes="TYX", t_policy="all")
    stack = np.ones((2, 32, 32), np.uint8)
    empty = np.ones((0, 32, 32), np.uint8)
    run_patching(
        images=[stack, empty], config=config, image_metadata=[{"position_id": "P1"}, {"position_id": "P2"}]
    )

    _, rows = _assert_positions_join(tmp_path)
    assert [(r["position_id"], r["status"], r["n_patches"]) for r in rows] == [
        ("P1", "analysed", "8"), ("P2", "skipped", "0"),
    ]
    assert "no plane to patch" in rows[1]["reason"]


def test_status_of_a_position_with_one_skipped_source_image():
    records = [
        ImageRecord(image_id="image_0000", position_id="P", n_regions=2, n_kept=2, n_normalized=2, n_patches=2),
        ImageRecord(image_id="image_0001", position_id="P", skip_reason="nothing to patch"),
    ]
    (position,) = position_records(records)
    # Not a clean "analysed": a source image of the position was not.
    assert position.status == "partly_skipped"
    assert position.reason == "1 of 2 source image(s) skipped: nothing to patch"
    assert (position.n_source_images, position.n_patches) == (2, 2)


# ---------------------------------------------------------------------------
# Grouping and identity
# ---------------------------------------------------------------------------


def test_one_position_split_over_several_time_point_files_is_one_row(tmp_path):
    config = _grid_config(tmp_path)
    image = np.full((32, 32), 100, np.uint16)
    result = run_patching(
        images=[image, image, image], config=config,
        image_metadata=[
            {"position_id": "P1", "image_file": "P1_t0.tif", "t": 0},
            {"position_id": "P1", "image_file": "P1_t1.tif", "t": 1},
            {"position_id": "P2", "image_file": "P2_t0.tif", "t": 0},
        ],
    )

    _, image_rows = _read_csv(tmp_path / IMAGE_TABLE_FILENAME)
    assert len(image_rows) == 3  # images.csv keeps one row per input image
    patch_rows, rows = _assert_positions_join(tmp_path)
    assert len(patch_rows) == 12
    assert [(r["position_id"], r["n_source_images"], r["n_patches"]) for r in rows] == [
        ("P1", "2", "8"), ("P2", "1", "4"),
    ]
    assert json.loads(rows[0]["image_files"]) == ["P1_t0.tif", "P1_t1.tif"]
    assert len(result.image_records) == 3


def test_same_basename_and_position_in_two_technical_repeats_stay_two_rows(tmp_path):
    config = _grid_config(tmp_path)
    image = np.full((32, 32), 100, np.uint16)
    run_patching(
        images=[image, image], config=config,
        image_metadata=[
            {"image_file": "c.tif", "technical_repeat": "B02"},
            {"image_file": "c.tif", "technical_repeat": "B03"},
        ],
    )

    _, rows = _assert_positions_join(tmp_path)
    assert [(r["technical_repeat"], r["position_id"], r["n_patches"]) for r in rows] == [
        ("B02", "c", "4"), ("B03", "c", "4"),
    ]


def test_the_canonical_ids_of_a_description_reach_the_row(tmp_path):
    config = _grid_config(tmp_path)
    image = np.full((32, 32), 100, np.uint16)
    run_patching(
        images=[image, image], config=config,
        image_metadata=[
            {"position_id": "P1", "repeat_group": "rg1", "condition_id": "cond1", "technical_repeat_id": "tr1"},
            {"position_id": "P2"},
        ],
    )

    _, rows = _assert_positions_join(tmp_path)
    assert [(r["repeat_group"], r["condition_id"], r["technical_repeat_id"]) for r in rows] == [
        ("rg1", "cond1", "tr1"), ("", "", ""),
    ]
    _, image_rows = _read_csv(tmp_path / IMAGE_TABLE_FILENAME)
    assert list(image_rows[0]) == list(IMAGE_TABLE_COLUMNS)  # images.csv does not carry them


def test_a_file_runs_fov_metadata_carries_the_ids_of_its_description():
    key = ("treated", "r1", "single", "P1")
    described = SimpleNamespace(
        key=key, relative_path="treated/r1/P1.tif", repeat_group="rg1", condition_id="c1", technical_repeat_id="tr1",
        channel_ids=("C0",),
    )
    config = CellPatcherConfig(mode="grid")
    run = SimpleNamespace(mask_type=None, channel_indices={})

    meta = fov_metadata(run, described, config)
    assert (meta["repeat_group"], meta["condition_id"], meta["technical_repeat_id"]) == ("rg1", "c1", "tr1")
    # A FOV without an ID is refused, never left empty (every run is described; tests/test_described_ids.py).
    with pytest.raises(ValueError, match="has no repeat_group"):
        fov_metadata(run, SimpleNamespace(**{**vars(described), "repeat_group": None}), config)


def test_two_source_images_of_a_position_cannot_name_two_repeat_groups():
    records = [
        ImageRecord(image_id="image_0000", position_id="P", repeat_group="rg1"),
        ImageRecord(image_id="image_0001", position_id="P", repeat_group="rg2"),
    ]
    with pytest.raises(ValueError, match="two repeat_group values"):
        position_records(records)


def test_records_without_a_position_id_are_never_merged(tmp_path):
    patches = [
        PatchRegion(patch_id=f"p{i}", image_id=f"image_000{i}", y1=4, x1=4, condition="c", image_file="same.tif")
        for i in range(2)
    ]
    export_patches(patches, [np.zeros((4, 4))] * 2, tmp_path, image_format="npy")

    _, rows = _read_csv(tmp_path / POSITION_TABLE_FILENAME)
    assert [(r["position_id"], r["n_source_images"], r["n_patches"]) for r in rows] == [("", "1", "1")] * 2


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_export_patches_writes_positions_of_the_records_it_is_given(tmp_path):
    patches = [PatchRegion(patch_id="p0", image_id="image_0000", y1=4, x1=4, position_id="A")]
    records = [
        ImageRecord(image_id="image_0000", position_id="A", n_regions=1, n_kept=1, n_normalized=1, n_patches=1),
        ImageRecord(image_id="image_0001", position_id="B"),
    ]
    export_patches(patches, [np.zeros((4, 4))], tmp_path, image_format="npy", image_records=records)

    _, rows = _read_csv(tmp_path / POSITION_TABLE_FILENAME)
    assert [(r["position_id"], r["status"]) for r in rows] == [("A", "analysed"), ("B", "analysed_no_regions")]


def test_no_positions_table_without_the_metadata_csv(tmp_path):
    config = _grid_config(tmp_path)
    config.export = ExportConfig(enabled=True, image_format="npy", save_metadata_csv=False)
    result = run_patching(
        images=[np.ones((32, 32), np.uint8)], config=config, image_metadata=[{"position_id": "P1"}]
    )
    assert not (tmp_path / POSITION_TABLE_FILENAME).exists()
    assert result.position_table_file is None



EXPECTED_TREE_ROWS = [
    ("control", "2026_01_01", "B02", "c", "analysed", "4"),
    ("control", "2026_01_01", "B03", "c", "analysed_no_regions", "0"),
    ("treated", "2026_01_01", "single", "a", "analysed", "4"),
    ("treated", "2026_01_01", "single", "b", "analysed_no_regions", "0"),
]


def _summary(rows):
    return [(*(r[column] for column in KEY), r["status"], r["n_patches"]) for r in rows]


def test_tree_run_writes_positions_in_every_leaf_and_keeps_equal_basenames_apart(tmp_path):
    raw, masks = cell_tree(tmp_path)
    results = preprocessed_tree(raw, mask_root_path=masks, config=_cell_config(tmp_path / "out"))

    rows = []
    for leaf in results:
        folder = tmp_path / "out" / leaf.relative_folder
        assert leaf.result.position_table_file == str(folder / POSITION_TABLE_FILENAME)
        rows.extend(_assert_positions_join(folder)[1])
    assert [(r["condition"], r["biological_repeat"], r["technical_repeat"], r["position_id"], r["status"],
             r["n_patches"]) for r in rows] == EXPECTED_TREE_ROWS


def test_combined_tree_run_writes_one_positions_table(tmp_path):
    raw, masks = cell_tree(tmp_path)
    results = preprocessed_tree(
        raw, mask_root_path=masks, config=_cell_config(tmp_path / "out"), combine_metadata=True
    )

    out = tmp_path / "out"
    _, rows = _assert_positions_join(out)
    assert _summary(rows) == EXPECTED_TREE_ROWS
    assert {leaf.result.position_table_file for leaf in results} == {str(out / POSITION_TABLE_FILENAME)}
    assert sorted(path.relative_to(out).as_posix() for path in out.rglob("*.csv")) == [
        "images.csv", "patches.csv", "positions.csv",
    ]
