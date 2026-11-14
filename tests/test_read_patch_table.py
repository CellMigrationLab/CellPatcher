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
from cellpatcher.export import save_metadata_csv, written_columns
from cellpatcher.io import write_array
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.types import PatchRegion
from tests.helpers import make_synthetic_cell_image_and_labels, named, preprocessed_tree


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
    leaves = preprocessed_tree(root, config=config, combine_metadata=True)
    patches, run_config = read_patch_table(tmp_path / "out" / "patches.csv")
    assert run_config is not None
    assert {(p.biological_repeat, p.technical_repeat, p.position_id) for p in patches} == {
        ("001", "wellA", "img"), ("001", "wellB", "img"),
    }
    assert len(patches) == sum(len(leaf.result.patches) for leaf in leaves)


# What save_metadata_csv writes in a column a hand-made table leaves out (an empty cell is None).
_WRITTEN_DEFAULTS = {"touches_border": "false", "valid": "true"}


def _complete(header, rows):
    """*header* and *rows* with every column CellPatcher writes that *header* lacks appended, as written.

    ``height``/``width`` are the rectangle's when its cells are integers; other added cells are
    ``touches_border``/``valid``'s defaults or empty. Since the compatibility sweep (Q-CP-a)
    read_patch_table refuses a table without one of these columns.
    """
    missing = [name for name in written_columns() if name not in header]
    completed = []
    for row in rows:
        cells = dict(zip(header, row[: len(header)], strict=True))  # a longer row stays as given
        added = []
        for name in missing:
            value = _WRITTEN_DEFAULTS.get(name, "")
            low, high = {"height": ("y0", "y1"), "width": ("x0", "x1")}.get(name, (None, None))
            if low is not None and str(cells.get(low, "")).isdigit() and str(cells.get(high, "")).isdigit():
                value = str(int(cells[high]) - int(cells[low]))
            added.append(value)
        completed.append(list(row) + added)
    return list(header) + missing, completed


def _write(path, header, rows, *, complete=True):
    if complete:
        header, rows = _complete(header, rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_a_hand_made_table_without_config_reads(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id", "condition", "y0", "y1", "x0", "x1", "label_id"],
                  [["p1", "NA-free", "0", "4", "2", "6", "7"]])
    (patch,), run_config = read_patch_table(path)
    assert run_config is None
    assert (patch.patch_id, patch.y0, patch.y1, patch.x0, patch.x1, patch.label_id) == ("p1", 0, 4, 2, 6, 7)
    assert patch.touches_border is False and patch.resized is None and patch.mask_fractions is None


def test_a_table_missing_a_written_column_is_refused_not_defaulted(tmp_path):
    # Q-CP-a (compatibility sweep): a column the table lacks no longer leaves its field at the default,
    # so position_id or biological_repeat is never None because a table was short.
    header, rows = _complete(["patch_id", "y0", "y1", "x0", "x1"], [["p", "0", "4", "0", "4"]])
    for dropped in ("position_id", "biological_repeat", "resized"):
        index = header.index(dropped)
        kept_header = header[:index] + header[index + 1:]
        kept_rows = [row[:index] + row[index + 1:] for row in rows]
        path = _write(tmp_path / f"{dropped}.csv", kept_header, kept_rows, complete=False)
        with pytest.raises(PatchTableError, match=rf"missing columns \['{dropped}'\]"):
            read_patch_table(path)
    path = _write(tmp_path / "short.csv", ["patch_id", "y0", "y1", "x0", "x1"], [["p", "0", "4", "0", "4"]],
                  complete=False)
    with pytest.raises(PatchTableError, match=r"missing columns \['image_id', 'mode', 'condition'"):
        read_patch_table(path)


@pytest.mark.parametrize("header, row, match", [
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "0", "4.0", "0", "4"], r"y1='4.0' is not an integer"),
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "0", "", "0", "4"], "y1 is empty"),
    (["patch_id", "y0", "y1", "x0", "x1"], ["p", "4", "0", "0", "4"], "must be greater"),
    (["patch_id", "y0", "y1", "x0", "x1", "resized"], ["p", "0", "4", "0", "4", "yes"], "not true or false"),
    (["patch_id", "y0", "y1", "x0", "x1", "height"], ["p", "0", "4", "0", "4", "5"], "not the rectangle's 4"),
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


def test_an_old_repeat_column_is_refused(tmp_path):
    # CP-1 (Spec ADR 0002 AR11 (6)): never read as biological_repeat.
    path = _write(tmp_path / "patches.csv", ["patch_id", "condition", "repeat", "y0", "y1", "x0", "x1"],
                  [["p", "ctrl", "2026_01_01", "0", "4", "0", "4"]])
    with pytest.raises(PatchTableError, match=r"columns CellPatcher does not write: \['repeat'\]"):
        read_patch_table(path)


def test_the_old_semicolon_parameter_columns_are_refused(tmp_path):
    # CP-5: the ;-separated normalization_offset/normalization_scale are unknown columns.
    path = _write(tmp_path / "patches.csv", ["patch_id", "y0", "y1", "x0", "x1", "normalization_offset"],
                  [["p", "0", "4", "0", "4", "0.0;1.0"]])
    with pytest.raises(PatchTableError, match=r"does not write: \['normalization_offset'\]"):
        read_patch_table(path)


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
    # Owner decision: unknown columns are refused, and the message names each of them (0x is not a channel ID).
    header = ["patch_id", "colour", "y0", "y1", "x0", "x1", "notes", "mask_fraction_cells", "Normalization_Offset_0x"]
    path = _write(tmp_path / "patches.csv", header, [["p", "red", "0", "4", "0", "4", "n", "0.5", "1"]])
    with pytest.raises(PatchTableError) as error:
        read_patch_table(path)
    assert "columns CellPatcher does not write: ['colour', 'notes', 'Normalization_Offset_0x']" in str(error.value)


# CSV structure (owner decision of 2026-10-31, CellPatcher v0.1.2): MorphoRelate
# checked it itself before calling read_patch_table. Nothing is filled in, shifted or lost.


def _write_text(path, text):
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_a_short_row_is_refused_not_read_with_empty_cells(tmp_path):
    # Before v0.1.2 the missing condition was read as None.
    path = _write_text(tmp_path / "patches.csv", "patch_id,y0,y1,x0,x1,condition\np,0,4,0,4,ctrl\nq,0,4,0,4\n")
    with pytest.raises(PatchTableError, match=r"line 3: 1 fewer cell\(s\) than the header has columns \(5 for 6\)"):
        read_patch_table(path)


def test_an_unclosed_quote_that_swallows_the_next_rows_is_refused(tmp_path):
    # Before v0.1.2 rows q and r became part of p's condition: one patch read, two lost.
    path = _write_text(
        tmp_path / "patches.csv",
        'patch_id,y0,y1,x0,x1,condition\np,0,4,0,4,"ctrl\nq,0,4,0,4,ctrl\nr,0,4,0,4,ctrl"\n',
    )
    with pytest.raises(PatchTableError, match=r"line 2, column 'condition': a quoted cell runs over several lines"):
        read_patch_table(path)


def test_an_unclosed_quote_at_the_end_of_the_table_is_refused(tmp_path):
    # Before v0.1.2 the last row's condition was read as 'ctrl\n'.
    path = _write_text(tmp_path / "patches.csv", 'patch_id,y0,y1,x0,x1,condition\np,0,4,0,4,"ctrl\n')
    with pytest.raises(PatchTableError, match=r"line 2: not a well-formed CSV row .*quote is not closed"):
        read_patch_table(path)


def test_a_padded_column_name_is_named_as_padded(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id", "y0", "y1", "x0", "x1", " condition"],
                  [["p", "0", "4", "0", "4", "ctrl"]])
    with pytest.raises(PatchTableError) as error:
        read_patch_table(path)
    message = str(error.value)
    assert "column 6 ' condition' has leading or trailing whitespace" in message
    assert "does not write" not in message


def test_a_padded_required_column_name_is_named_as_padded(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id ", "y0", "y1", "x0", "x1"], [["p", "0", "4", "0", "4"]])
    with pytest.raises(PatchTableError, match=r"column 1 'patch_id ' has leading or trailing whitespace"):
        read_patch_table(path)


def test_a_duplicate_column_is_named_with_its_positions(tmp_path):
    path = _write(tmp_path / "patches.csv", ["patch_id", "y0", "y1", "x0", "x1", "condition", "condition"],
                  [["p", "0", "4", "0", "4", "a", "b"]])
    with pytest.raises(PatchTableError, match=r"duplicate column 'condition' \(columns 6, 7\)"):
        read_patch_table(path)


def test_every_structural_problem_is_listed(tmp_path):
    path = _write_text(
        tmp_path / "patches.csv",
        "patch_id,y0,y1,x0,x1,x1, label_id\np,0,4,0,4,4,1\nq,0,4\nr,0,4,0,4,4,1,9\n",
    )
    with pytest.raises(PatchTableError) as error:
        read_patch_table(path)
    message = str(error.value)
    for part in ("column 7 ' label_id'", "duplicate column 'x1' (columns 5, 6)", "line 3: 4 fewer",
                 "line 4: 1 more"):
        assert part in message


def test_blank_lines_are_still_skipped(tmp_path):
    header, (row_p, row_q) = _complete(["patch_id", "y0", "y1", "x0", "x1"],
                                       [["p", "0", "4", "0", "4"], ["q", "0", "4", "0", "4"]])
    text = "\n".join([",".join(header), ",".join(row_p), "", ",".join(row_q)]) + "\n"
    path = _write_text(tmp_path / "patches.csv", text)
    patches, _ = read_patch_table(path)
    assert [patch.patch_id for patch in patches] == ["p", "q"]
