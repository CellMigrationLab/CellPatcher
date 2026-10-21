"""The exported metadata must look like the MorphoBricks shared CSV contract (AGENTS.md §5).

These checks read ``patches.csv`` as a downstream repository would, from the
file alone, and assert what the contract promises: the shared columns exist
with exactly their names; in a tree run condition, repeat and technical_repeat
are never empty; patch_file is a relative path that resolves to the exported
file from the CSV's folder; no two rows point to the same file; and each
per-leaf CSV holds a single technical repeat.
"""

import csv
from pathlib import Path

import numpy as np

from cellpatcher.config import CellPatcherConfig
from cellpatcher.io import SINGLE_TECHNICAL_REPEAT_LABEL, write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree

SHARED_COLUMNS = (
    "condition",
    "biological_repeat",
    "technical_repeat",
    "position_id",
    "image_file",
    "mask_file",
    "patch_file",
)
ALWAYS_FILLED_IN_TREE_RUNS = (
    "condition",
    "biological_repeat",
    "technical_repeat",
    "image_file",
    "patch_file",
)


def _read(csv_path: Path) -> tuple[list[str], list[dict]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _assert_contract(csv_path: Path) -> list[dict]:
    header, rows = _read(csv_path)
    assert rows, "no rows exported"
    for column in SHARED_COLUMNS:
        assert column in header, f"shared column {column!r} missing from {csv_path.name}"
    for row in rows:
        for column in ALWAYS_FILLED_IN_TREE_RUNS:
            assert row[column], f"{column!r} empty in a tree run: {row}"
        patch_file = Path(row["patch_file"])
        assert not patch_file.is_absolute() and ".." not in patch_file.parts
        assert (csv_path.parent / patch_file).is_file()
    assert len({row["patch_file"] for row in rows}) == len(rows)
    return rows


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "raw_data"
    image = np.ones((32, 32), dtype=np.float32)
    for well in ("B02", "B03"):
        write_array(root / "treated" / "2026_01_01" / well / "field_001.npy", image)
        write_array(root / "treated" / "2026_01_01" / well / "field_002.npy", image)
    write_array(root / "control" / "2026_01_01" / "field_001.npy", image)
    write_array(root / "mock" / "field_009.npy", image)  # no repeat level at all
    return root


def _config(tmp_path: Path) -> CellPatcherConfig:
    cfg = CellPatcherConfig(mode="grid")
    cfg.grid.tile_size = 16
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")
    return cfg


def test_combined_tree_export_meets_the_shared_csv_contract(tmp_path):
    run_patching_condition_repeat_tree(_tree(tmp_path), config=_config(tmp_path), combine_metadata=True)
    rows = _assert_contract(tmp_path / "out" / "patches.csv")
    wells = {row["technical_repeat"] for row in rows if row["condition"] == "treated"}
    assert wells == {"B02", "B03"}
    assert {row["technical_repeat"] for row in rows if row["condition"] != "treated"} == {
        SINGLE_TECHNICAL_REPEAT_LABEL
    }


def test_per_leaf_tree_export_meets_the_shared_csv_contract(tmp_path):
    run_patching_condition_repeat_tree(_tree(tmp_path), config=_config(tmp_path))
    csvs = sorted((tmp_path / "out").rglob("patches.csv"))
    assert [p.parent.relative_to(tmp_path / "out").as_posix() for p in csvs] == [
        "control/2026_01_01",
        "mock/single",
        "treated/2026_01_01/B02",
        "treated/2026_01_01/B03",
    ]
    for csv_path in csvs:
        rows = _assert_contract(csv_path)
        assert len({row["technical_repeat"] for row in rows}) == 1
