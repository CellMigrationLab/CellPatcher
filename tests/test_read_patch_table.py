"""read_patch_table: the typed inverse of save_metadata_csv (D86 API C2).

MorphoRelate (io/cellpatcher.py) and fiberfeatures (patches.py) parsed
patches.csv themselves; they need the rectangle, the identity columns as text,
label_id/z/t/c as integers, the resized flag and output size, mask/patch
files, and the run_config.json its config_file names.
"""

import csv
import json
import warnings

import numpy as np
import pytest

from cellpatcher import PatchTableError, read_patch_table
from cellpatcher.config import (
    CellConfig,
    CellPatcherConfig,
    CommonConfig,
    ExportConfig,
    ExtractionConfig,
    GridConfig,
    NormalizationConfig,
)
from cellpatcher.export import save_metadata_csv
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.types import PatchRegion
from tests.helpers import make_synthetic_cell_image_and_labels, named


def _exported_cell_run(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels()
    config = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(image_axes="YX", mask_axes="YX", output_dir=str(tmp_path)),
        cell=CellConfig(patch_size=24, exclude_border=False),
        extraction=ExtractionConfig(output_size=(16, 16)),
        export=ExportConfig(enabled=True, image_format="npy"),
        normalization=NormalizationConfig(enabled=True, method="minmax", scope="patch"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))
    return result.patches, tmp_path / "patches.csv"


def test_an_exported_run_reads_back_as_its_patches(tmp_path):
    patches, csv_path = _exported_cell_run(tmp_path)
    assert patches
    read, run_config = read_patch_table(csv_path)
    assert run_config == json.loads((tmp_path / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert [patch.to_dict() for patch in read] == [patch.to_dict() for patch in patches]
    first = read[0]
    assert isinstance(first, PatchRegion)
    assert type(first.label_id) is int and type(first.resized) is bool and first.resized
    assert first.mask_fractions["own_object"] == patches[0].mask_fractions["own_object"]
    assert first.normalization_parameters is not None
    assert first.normalization_parameters == patches[0].normalization_parameters


def test_labels_stay_text_in_a_tree_run(tmp_path):
    root = tmp_path / "experiment"
    for relative in ("control/001/wellA/img.tif", "control/001/wellB/img.tif"):
        write_array(root / relative, np.arange(256, dtype=np.uint16).reshape(16, 16))
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8), export=ExportConfig(enabled=True))
    config.common.output_dir = str(tmp_path / "out")
    leaves = run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)
    patches, run_config = read_patch_table(tmp_path / "out" / "patches.csv")
    assert run_config is not None
    assert {(p.biological_repeat, p.technical_repeat, p.position_id) for p in patches} == {
        ("001", "wellA", "img"), ("001", "wellB", "img"),
    }
    assert len(patches) == sum(len(leaf.result.patches) for leaf in leaves)


def _write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_a_table_without_config_reads_with_defaults_for_columns_it_lacks(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id", "condition", "y0", "y1", "x0", "x1", "label_id"],
                  [["p1", "NA-free", "0", "4", "2", "6", "7"]])
    (patch,), run_config = read_patch_table(path)
    assert run_config is None
    assert (patch.patch_id, patch.y0, patch.y1, patch.x0, patch.x1, patch.label_id) == ("p1", 0, 4, 2, 6, 7)
    assert patch.touches_border is False and patch.resized is None and patch.mask_fractions is None


@pytest.mark.parametrize("header, row, match", [
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "0", "4.0", "0", "4"], r"y1='4.0' is not an integer"),
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "0", "", "0", "4"], "y1 is empty"),
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "4", "0", "0", "4"], "must be greater"),
    (["patch_id", "y0", "y1", "x0", "x1", "resized"], ["p", "0", "4", "0", "4", "yes"], "not true or false"),
    (["patch_id", "y0", "y1", "x0", "x1", "height"], ["p", "0", "4", "0", "4", "5"], "not the rectangle's 4"),
    (["patch_id", "y0", "y1", "x0"], ["p", "0", "4", "0"], r"missing columns \['x1'\]"),
    (["patch_id", "y0", "y1", "x0", "x1", "colour"], ["p", "0", "4", "0", "4", "red"], "does not write"),
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "0", "4", "0", "4", "extra"], "more cell"),
    (["patch_id", "y0", "y1", "x0", "x1", "config_file"], ["p", "0", "4", "0", "4", "gone.json"], "does not exist"),
    (["patch_id", "y0", "y1", "x0", "x1", "mask_fraction_cells"], ["p", "0", "4", "0", "4", "nan"], "finite"),
])
def test_cells_that_are_not_what_cellpatcher_writes_are_refused(tmp_path, header, row, match):
    path = _write(tmp_path / "patches.csv", header, [row])
    with pytest.raises(PatchTableError, match=match):
        read_patch_table(path)


def test_every_problem_is_listed_with_its_line(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id", "y0", "y1", "x0", "x1"],
                  [["a", "0", "x", "0", "4"], ["b", "0", "4", "0", "4"], ["b", "0", "4", "0", "4"], ["c", "", "4", "0", "4"]])
    with pytest.raises(PatchTableError) as error:
        read_patch_table(path)
    message = str(error.value)
    assert "line 2 (patch_id 'a')" in message and "line 5 (patch_id 'c')" in message
    assert "patch_id values on several rows: ['b']" in message


def test_parameters_need_the_run_config(tmp_path):
    _, csv_path = _exported_cell_run(tmp_path)
    (tmp_path / RUN_CONFIG_FILENAME).unlink()
    with pytest.raises(PatchTableError, match="does not exist"):
        read_patch_table(csv_path)
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["config_file"] = ""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(PatchTableError, match="without the run_config.json"):
        read_patch_table(csv_path)


def test_an_old_repeat_column_reads_as_biological_repeat(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id", "condition", "repeat", "y0", "y1", "x0", "x1"],
                  [["p", "ctrl", "2026_01_01", "0", "4", "0", "4"]])
    with pytest.warns(DeprecationWarning, match="renamed"):
        (patch,), _ = read_patch_table(path)
    assert patch.biological_repeat == "2026_01_01"


def test_written_patches_round_trip_without_a_run(tmp_path):
    patches = [
        PatchRegion(patch_id="a", condition="001", y0=0, y1=3, x0=1, x1=5, z0=2, z1=3, t=0, c=1,
                    center_y=1.25, touches_border=True, mask_fractions={"cells": 0.5}),
        PatchRegion(patch_id="b", y0=2, y1=3, x0=1, x1=2, valid=True),
    ]
    save_metadata_csv(patches, tmp_path / "patches.csv")
    read, _ = read_patch_table(tmp_path / "patches.csv")
    assert [patch.to_dict() for patch in read] == [patch.to_dict() for patch in patches]


def test_every_unknown_column_is_named(tmp_path):
    # Owner decision: unknown columns are refused, and the message names each of them.
    header = ["patch_id", "colour", "y0", "y1", "x0", "x1", "notes", "mask_fraction_cells", "Normalization_Offset_C0x"]
    path = _write(tmp_path / "patches.csv", header, [["p", "red", "0", "4", "0", "4", "n", "0.5", "1"]])
    with pytest.raises(PatchTableError) as error:
        read_patch_table(path)
    assert "columns CellPatcher does not write: ['colour', 'notes', 'Normalization_Offset_C0x']" in str(error.value)
