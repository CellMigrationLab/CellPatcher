"""The ``biological_repeat`` column and the deprecated ``repeat`` alias.

``repeat`` was renamed ``biological_repeat`` (PatchRegion field, CommonConfig
field, patches.csv column). The old name must keep working for a transition,
with exactly one DeprecationWarning per use and identical output.
"""

import csv
import json
import warnings

import numpy as np
import pytest

import cellpatcher
from cellpatcher._deprecation import upgrade_legacy_repeat_column
from cellpatcher.config import CellPatcherConfig, CommonConfig
from cellpatcher.io import (
    SINGLE_BIOLOGICAL_REPEAT_LABEL,
    SINGLE_CONDITION_LABEL,
    SINGLE_REPEAT_LABEL,
    ConditionRepeatFolder,
    discover_condition_repeat_folders,
    write_array,
)
from cellpatcher.orchestration import (
    ConditionRepeatResult,
    apply_overrides,
    load_config,
    run_patching_condition_repeat_tree,
    run_patching_pipeline,
)
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.types import PatchRegion
from tests.helpers import named


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
    _write_image(tmp_path / "images" / "a.tif")
    cfg = _export_config(tmp_path / "out")
    cfg.common.condition = "treated"
    cfg.common.biological_repeat = "2026_01_01"

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        run_patching_pipeline(str(tmp_path / "images"), config=cfg)
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
        results = run_patching_condition_repeat_tree(str(tmp_path / "images"), config=cfg)
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

    run_patching_condition_repeat_tree(str(tmp_path / "images"), config=cfg, combine_metadata=True)

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

    run_patching_condition_repeat_tree(str(tmp_path / "images"), config=cfg, combine_metadata=True)

    _, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert [(row["condition"], row["biological_repeat"]) for row in rows] == [("single", "single")]
    assert (tmp_path / "out" / "single" / "single" / rows[0]["patch_file"].split("/")[-1]).exists()


def test_single_label_constants():
    assert SINGLE_BIOLOGICAL_REPEAT_LABEL == "single"
    assert SINGLE_REPEAT_LABEL == SINGLE_BIOLOGICAL_REPEAT_LABEL
    assert SINGLE_CONDITION_LABEL == "single"
    assert cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL == cellpatcher.SINGLE_REPEAT_LABEL == "single"
    assert "SINGLE_BIOLOGICAL_REPEAT_LABEL" in cellpatcher.__all__
    assert "SINGLE_REPEAT_LABEL" in cellpatcher.__all__


# --- deprecated `repeat` alias ------------------------------------------------


def test_config_repeat_attribute_warns_once_and_gives_identical_output(tmp_path):
    _write_image(tmp_path / "images" / "a.tif")

    new_cfg = _export_config(tmp_path / "new")
    new_cfg.common.condition = "treated"
    new_cfg.common.biological_repeat = "2026_01_01"
    run_patching_pipeline(str(tmp_path / "images"), config=new_cfg)

    old_cfg = _export_config(tmp_path / "old")
    old_cfg.common.condition = "treated"
    with pytest.warns(DeprecationWarning, match="biological_repeat") as records:
        old_cfg.common.repeat = "2026_01_01"
    assert len(_deprecations(records)) == 1

    with warnings.catch_warnings(record=True) as run_records:
        warnings.simplefilter("always")
        run_patching_pipeline(str(tmp_path / "images"), config=old_cfg)
    assert _deprecations(run_records) == []

    assert (tmp_path / "old" / "patches.csv").read_text() == (tmp_path / "new" / "patches.csv").read_text()
    old_run_config = json.loads((tmp_path / "old" / RUN_CONFIG_FILENAME).read_text())
    new_run_config = json.loads((tmp_path / "new" / RUN_CONFIG_FILENAME).read_text())
    old_run_config["common"].pop("output_dir")
    new_run_config["common"].pop("output_dir")
    assert old_run_config == new_run_config


def test_common_config_repeat_keyword_warns_once():
    with pytest.warns(DeprecationWarning, match="CommonConfig") as records:
        common = CommonConfig(condition="treated", repeat="2026_01_01")
    assert len(_deprecations(records)) == 1
    assert common == CommonConfig(condition="treated", biological_repeat="2026_01_01")
    # The warning points at the caller, not at CellPatcher internals.
    assert records[0].filename == __file__


def test_config_repeat_attribute_read_warns_once():
    common = CommonConfig(biological_repeat="2026_01_01")
    with pytest.warns(DeprecationWarning) as records:
        assert common.repeat == "2026_01_01"
    assert len(_deprecations(records)) == 1


def test_conflicting_repeat_and_biological_repeat_raise():
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="conflicting"):
        CommonConfig(biological_repeat="2026_01_01", repeat="2026_02_02")


def test_apply_overrides_repeat_key_warns_once():
    with pytest.warns(DeprecationWarning) as records:
        cfg = apply_overrides(CellPatcherConfig(), {"common.repeat": "2026_01_01"})
    assert len(_deprecations(records)) == 1
    assert cfg.common.biological_repeat == "2026_01_01"


def test_pipeline_overrides_repeat_key_gives_identical_csv(tmp_path):
    _write_image(tmp_path / "images" / "a.tif")
    run_patching_pipeline(
        str(tmp_path / "images"),
        config=_export_config(tmp_path / "new"),
        overrides={"common.biological_repeat": "2026_01_01"},
    )
    with pytest.warns(DeprecationWarning) as records:
        run_patching_pipeline(
            str(tmp_path / "images"),
            config=_export_config(tmp_path / "old"),
            overrides={"common.repeat": "2026_01_01"},
        )
    assert len(_deprecations(records)) == 1
    assert (tmp_path / "old" / "patches.csv").read_text() == (tmp_path / "new" / "patches.csv").read_text()


def test_load_config_with_legacy_repeat_key_warns_once(tmp_path):
    legacy = {"mode": "whole_fov", "common": {"condition": "treated", "repeat": "2026_01_01"}}
    path = tmp_path / "legacy_config.json"
    path.write_text(json.dumps(legacy))

    with pytest.warns(DeprecationWarning) as records:
        cfg = load_config(path)
    assert len(_deprecations(records)) == 1
    assert cfg.common.biological_repeat == "2026_01_01"


def test_patch_region_repeat_keyword_and_attribute_warn_once_each():
    expected = PatchRegion(patch_id="p", y1=4, x1=4, biological_repeat="2026_01_01")
    with pytest.warns(DeprecationWarning, match="PatchRegion") as records:
        patch = PatchRegion(patch_id="p", y1=4, x1=4, repeat="2026_01_01")
    assert len(_deprecations(records)) == 1
    assert patch == expected
    assert patch.to_dict() == expected.to_dict()
    assert "repeat" not in patch.to_dict()

    with pytest.warns(DeprecationWarning) as records:
        patch.repeat = "2026_02_02"
    assert len(_deprecations(records)) == 1
    assert patch.biological_repeat == "2026_02_02"


def test_run_patching_with_repeat_config_gives_identical_patches():
    image = np.ones((16, 16), dtype=np.float32)
    new_cfg = CellPatcherConfig(mode="whole_fov", common=CommonConfig(biological_repeat="2026_01_01"))
    with pytest.warns(DeprecationWarning):
        old_cfg = CellPatcherConfig(mode="whole_fov", common=CommonConfig(repeat="2026_01_01"))
    assert run_patching(images=image, config=old_cfg,
        image_metadata=named(image)).patches == run_patching(images=image, config=new_cfg,
        image_metadata=named(image)).patches


def test_discovery_and_tree_results_expose_deprecated_repeat_alias(tmp_path):
    _build_tree(tmp_path / "images")
    leaves = discover_condition_repeat_folders(tmp_path / "images")
    assert isinstance(leaves[0], ConditionRepeatFolder)
    with pytest.warns(DeprecationWarning, match="ConditionRepeatFolder.repeat") as records:
        assert leaves[0].repeat == leaves[0].biological_repeat
    assert len(_deprecations(records)) == 1

    results = run_patching_condition_repeat_tree(str(tmp_path / "images"), config=CellPatcherConfig(mode="whole_fov"))
    assert isinstance(results[0], ConditionRepeatResult)
    with pytest.warns(DeprecationWarning, match="ConditionRepeatResult.repeat") as records:
        assert results[0].repeat == results[0].biological_repeat
    assert len(_deprecations(records)) == 1


# --- reading a pre-rename patches.csv ------------------------------------------


def test_legacy_repeat_column_is_read_as_biological_repeat_with_one_warning():
    fieldnames = ["patch_id", "condition", "repeat", "image_file"]
    rows = [
        {"patch_id": "p0", "condition": "treated", "repeat": "2026_01_01", "image_file": "a.tif"},
        {"patch_id": "p1", "condition": "treated", "repeat": "2026_02_02", "image_file": "b.tif"},
    ]
    with pytest.warns(DeprecationWarning, match="patches.csv") as records:
        upgraded = upgrade_legacy_repeat_column(fieldnames, rows, "old/patches.csv")
    assert len(_deprecations(records)) == 1
    assert [row["biological_repeat"] for row in upgraded] == ["2026_01_01", "2026_02_02"]
    assert all("repeat" not in row for row in upgraded)
    assert list(upgraded[0]) == ["patch_id", "condition", "biological_repeat", "image_file"]


def test_current_csv_rows_are_left_unchanged_without_warning():
    rows = [{"patch_id": "p0", "biological_repeat": "2026_01_01"}]
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        assert upgrade_legacy_repeat_column(["patch_id", "biological_repeat"], rows, "patches.csv") == rows
    assert _deprecations(records) == []


def test_torch_dataset_reads_legacy_repeat_column(tmp_path):
    pytest.importorskip("torch")
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    np.save(tmp_path / "p0.npy", np.ones((8, 8), dtype=np.float32))
    (tmp_path / "patches.csv").write_text("patch_id,condition,repeat,patch_file\np0,treated,2026_01_01,p0.npy\n")
    with pytest.warns(DeprecationWarning) as records:
        dataset = ExportedPatchTorchDataset(tmp_path / "patches.csv")
    assert len(_deprecations(records)) == 1
    assert dataset[0]["patch"]["biological_repeat"] == "2026_01_01"
