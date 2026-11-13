"""The ``biological_repeat`` column; the old ``repeat`` name is refused.

``repeat`` was renamed ``biological_repeat`` (PatchRegion field, CommonConfig
field, patches.csv column). Its transition alias was removed in the
compatibility sweep (CP-1, Spec ADR 0002 AR11 (6)): the old name is refused
everywhere, never read as the biological repeat.
"""

import csv
import dataclasses
import json
import warnings

import numpy as np
import pytest

import cellpatcher
from cellpatcher.config import CellPatcherConfig, CommonConfig
from cellpatcher.io import (
    SINGLE_BIOLOGICAL_REPEAT_LABEL,
    SINGLE_CONDITION_LABEL,
    write_array,
)
from cellpatcher.orchestration import ConditionRepeatResult, apply_overrides, load_config
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.types import PatchRegion
from tests.helpers import preprocessed_tree as _tree


def _run(config):
    """Patch one in-memory image named a.tif (config labels apply to in-memory runs; file runs
    take their labels from the preprocessing run since CP-2a)."""
    return run_patching(images=[np.ones((16, 16), np.float32)], config=config, image_metadata=[{"image_file": "a.tif"}])


def _write_image(path, shape=(16, 16)):
    write_array(path, np.ones(shape, dtype=np.float32))


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _export_config(output_dir):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(output_dir)
    return cfg


def _deprecations(records):
    return [record for record in records if issubclass(record.category, DeprecationWarning)]


def _build_tree(root):
    # treated has two dated biological repeats; control holds its images
    # directly (no biological-repeat subfolder -> "single").
    _write_image(root / "treated" / "2026_01_01" / "a.tif")
    _write_image(root / "treated" / "2026_02_02" / "b.tif")
    _write_image(root / "control" / "c.tif")


# --- new column in patches.csv ---------------------------------------------


def test_pipeline_csv_has_biological_repeat_column_and_no_repeat_column(tmp_path):
    cfg = _export_config(tmp_path / "out")
    cfg.common.condition = "treated"
    cfg.common.biological_repeat = "2026_01_01"

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _run(cfg)
    assert _deprecations(records) == []

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert "biological_repeat" in header
    assert "repeat" not in header
    # Same position the old column had: right after condition.
    assert header.index("biological_repeat") == header.index("condition") + 1
    assert [row["biological_repeat"] for row in rows] == ["2026_01_01"]
    run_config = json.loads((tmp_path / "out" / RUN_CONFIG_FILENAME).read_text())
    assert run_config["common"]["biological_repeat"] == "2026_01_01"
    assert "repeat" not in run_config["common"]


def test_tree_per_leaf_csvs_have_biological_repeat_column(tmp_path):
    _build_tree(tmp_path / "images")
    cfg = _export_config(tmp_path / "out")

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        results = _tree(tmp_path / "images", config=cfg)
    assert _deprecations(records) == []

    assert {(r.condition, r.biological_repeat) for r in results} == {
        ("treated", "2026_01_01"),
        ("treated", "2026_02_02"),
        ("control", SINGLE_BIOLOGICAL_REPEAT_LABEL),
    }
    for result in results:
        header, rows = _read_csv(tmp_path / "out" / result.condition / result.biological_repeat / "patches.csv")
        assert "biological_repeat" in header and "repeat" not in header
        assert rows and all(row["biological_repeat"] == result.biological_repeat for row in rows)
    _, control_rows = _read_csv(tmp_path / "out" / "control" / "single" / "patches.csv")
    assert [row["biological_repeat"] for row in control_rows] == ["single"]


def test_tree_combined_csv_has_biological_repeat_column(tmp_path):
    _build_tree(tmp_path / "images")
    cfg = _export_config(tmp_path / "out")

    _tree(tmp_path / "images", config=cfg, combine_metadata=True)

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert "biological_repeat" in header and "repeat" not in header
    assert sorted((row["condition"], row["biological_repeat"]) for row in rows) == [
        ("control", "single"),
        ("treated", "2026_01_01"),
        ("treated", "2026_02_02"),
    ]
    for row in rows:
        assert row["patch_file"].startswith(f"{row['condition']}/{row['biological_repeat']}/")
        assert (tmp_path / "out" / row["patch_file"]).exists()


def test_tree_flat_root_is_single_condition_and_single_biological_repeat(tmp_path):
    _write_image(tmp_path / "images" / "a.tif")
    cfg = _export_config(tmp_path / "out")

    _tree(tmp_path / "images", config=cfg, combine_metadata=True)

    _, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert [(row["condition"], row["biological_repeat"]) for row in rows] == [("single", "single")]
    assert (tmp_path / "out" / "single" / "single" / rows[0]["patch_file"].split("/")[-1]).exists()


def test_single_label_constants():
    assert SINGLE_BIOLOGICAL_REPEAT_LABEL == "single"
    assert SINGLE_CONDITION_LABEL == "single"
    assert cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL == "single"
    assert "SINGLE_BIOLOGICAL_REPEAT_LABEL" in cellpatcher.__all__
    assert not hasattr(cellpatcher, "SINGLE_REPEAT_LABEL")  # CP-2: the old name is gone


# --- the removed `repeat` name is refused ------------------------------------


def test_the_repeat_keyword_is_refused():
    with pytest.raises(TypeError, match="repeat"):
        CommonConfig(condition="treated", repeat="2026_01_01")
    with pytest.raises(TypeError, match="repeat"):
        PatchRegion(patch_id="p", y1=4, x1=4, repeat="2026_01_01")


def test_the_repeat_attribute_is_refused():
    common = CommonConfig(biological_repeat="2026_01_01")
    with pytest.raises(AttributeError):
        _ = common.repeat
    with pytest.raises(AttributeError):
        common.repeat = "2026_02_02"  # slots=True (Q-CP-c): never a silent new attribute
    patch = PatchRegion(patch_id="p", y1=4, x1=4, biological_repeat="2026_01_01")
    assert not hasattr(patch, "repeat")
    assert "repeat" not in patch.to_dict()
    assert "repeat" not in {field.name for field in dataclasses.fields(ConditionRepeatResult)}


def test_apply_overrides_refuses_the_repeat_key():
    with pytest.raises(ValueError, match="Unknown field 'repeat'"):
        apply_overrides(CellPatcherConfig(), {"common.repeat": "2026_01_01"})


def test_load_config_refuses_a_repeat_key(tmp_path):
    legacy = {"run_config_version": 2, "mode": "whole_fov", "common": {"condition": "treated", "repeat": "2026_01_01"}}
    path = tmp_path / "legacy_config.json"
    path.write_text(json.dumps(legacy))
    with pytest.raises(TypeError, match="repeat"):
        load_config(path)
