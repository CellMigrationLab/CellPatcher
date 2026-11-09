"""The canonical IDs of a described run reach ``positions.csv`` from the run, never from a guess (CP-1).

Runs are built with preprocessing's ``make_described_run`` from the vendored Spec fixtures
(``tests/spec_fixtures``); no test here opens a ``.toml``. CellPatcher sees the description only
through ``RunInputs`` (``Fov.repeat_group``, ``Fov.condition_id``, ``Fov.technical_repeat_id``).

- P32: the condition and technical-repeat folders are names on the forbidden-ID list, so the ID
  differs from the folder (``control`` / ``ctrl``) and a brick that took the folder for the ID is wrong.
- P25: the repeat folders differ across conditions (``v03``) or one folder name pairs with two
  repeats (``v30``); only ``repeat_group`` pairs them correctly.
- Absent IDs on a described run are an error; an undescribed run keeps its empty columns, byte for byte.
"""

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.file_runs import fov_metadata, open_file_run
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.statistics import PreprocessingCheckError

VALID = Path(__file__).parent / "spec_fixtures" / "fixtures" / "valid"
ID_COLUMNS = ("repeat_group", "condition_id", "technical_repeat_id")


def _fixture(name):
    """The bytes of the vendored valid fixture whose file name starts with *name* (``v33``)."""
    (path,) = VALID.glob(f"{name}_*.toml")
    return path.read_bytes()


def _described_run(tmp_path, name, **options):
    from morphobricks_preprocessing.testing_described import make_described_run

    return make_described_run(tmp_path / "run", _fixture(name), **options)


def _config(run, output):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.preprocessing_run, config.common.output_dir = str(run.output), str(output)
    config.export.enabled = True
    return config


def _positions(run, output):
    """Run the tree run on *run* and return the rows of its ``positions.csv``."""
    run_patching_condition_repeat_tree(run.experiment, config=_config(run, output), combine_metadata=True)
    with (output / "positions.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ids_by_folders(rows):
    """``{(condition, biological_repeat, technical_repeat): {(repeat_group, condition_id, technical_repeat_id)}}``."""
    found = {}
    for row in rows:
        key = (row["condition"], row["biological_repeat"], row["technical_repeat"])
        found.setdefault(key, set()).add(tuple(row[name] for name in ID_COLUMNS))
    return found


def _run_inputs_ids(run):
    """The IDs ``RunInputs`` gives each FOV, by position key: what ``positions.csv`` must carry."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    return {
        fov.key: (fov.repeat_group, fov.condition_id, fov.technical_repeat_id)
        for fov in RunInputs.open(run.output).fovs()
    }


def _row_ids_by_key(rows):
    return {
        (row["condition"], row["biological_repeat"], row["technical_repeat"], row["position_id"]):
        tuple(row[name] for name in ID_COLUMNS)
        for row in rows
    }


# P32 ------------------------------------------------------------------------------------------


def test_p32_v33_the_condition_id_is_the_runs_not_the_folder_name(tmp_path):
    run = _described_run(tmp_path, "v33")
    rows = _positions(run, tmp_path / "out")

    assert rows
    assert {row["condition"] for row in rows} == {"control", "DrugA"}  # the folder label stays in the key
    assert {(row["condition"], row["condition_id"]) for row in rows} == {("control", "ctrl"), ("DrugA", "DrugA")}
    assert {row["technical_repeat_id"] for row in rows} == {"single"}
    assert _row_ids_by_key(rows) == _run_inputs_ids(run)


def test_p32_v34_the_technical_repeat_id_is_the_runs_not_the_folder_name(tmp_path):
    run = _described_run(tmp_path, "v34")
    rows = _positions(run, tmp_path / "out")

    assert {(row["technical_repeat"], row["technical_repeat_id"]) for row in rows} == {
        ("files", "acq1"), ("overrides", "acq2"), ("single", "single"),
    }
    assert {row["condition_id"] for row in rows} == {"CTRL", "DrugA"}
    assert _row_ids_by_key(rows) == _run_inputs_ids(run)


# P25 ------------------------------------------------------------------------------------------


def test_p25_v03_repeats_pair_although_the_folder_names_differ(tmp_path):
    run = _described_run(tmp_path, "v03")
    rows = _positions(run, tmp_path / "out")

    groups = {}
    for row in rows:
        groups.setdefault(row["repeat_group"], set()).add((row["condition"], row["biological_repeat"]))
    assert groups == {
        "Day1": {("CTRL", "rep1"), ("DrugA", "experiment_23")},
        "Day2": {("CTRL", "rep2"), ("DrugA", "rep2_drug")},
    }
    assert _row_ids_by_key(rows) == _run_inputs_ids(run)


def test_p25_v30_one_folder_name_is_two_repeats(tmp_path):
    run = _described_run(tmp_path, "v30")
    rows = _positions(run, tmp_path / "out")

    assert _ids_by_folders(rows) == {
        ("CTRL", "rep1", "single"): {("Day1", "CTRL", "single")},
        ("DrugA", "rep2", "single"): {("Day1", "DrugA", "single")},
        ("CTRL", "rep2", "single"): {("Day2", "CTRL", "single")},
        ("DrugA", "rep1", "single"): {("Day2", "DrugA", "single")},
    }
    assert _row_ids_by_key(rows) == _run_inputs_ids(run)


@pytest.mark.parametrize("name", ["v03", "v30"])
def test_p25_normalization_pools_by_the_runs_repeat_group_not_the_folder_name(tmp_path, name):
    # The design the description derives (experiment-v1 section 11), given to the run, is what the
    # biological_repeat scope reads: its groups are exactly Fov.repeat_group.
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.design_derive import derive_design

    from cellpatcher.normalization import load_experiment_design
    from cellpatcher.statistics import load_plane_statistics

    run = _described_run(tmp_path, name)
    result = mbp.run_experiment(
        run.experiment, mbp.AxisMapping("CYX", "CYX"), output_folder=run.output, design=derive_design(run.description)
    )
    assert result.written, [str(issue) for issue in result.issues]
    histograms = str(run.output / "plane_histograms.csv")
    settings = NormalizationConfig(enabled=True, plane_histograms=histograms, scope="biological_repeat")

    groups = load_experiment_design(settings, load_plane_statistics(histograms)).repeat_groups

    assert groups == {key[:2]: ids[0] for key, ids in _run_inputs_ids(run).items()}
    assert len(set(groups.values())) == 2 and len(groups) == 4  # four folders, two repeats


def test_a_described_run_without_a_design_refuses_the_repeat_scope_instead_of_grouping_by_folder(tmp_path):
    run = _described_run(tmp_path, "v30")
    config = _config(run, tmp_path / "out")
    config.normalization = NormalizationConfig(
        enabled=True, plane_histograms=str(run.output / "plane_histograms.csv"), scope="biological_repeat"
    )

    with pytest.raises(PreprocessingCheckError, match="recorded no design"):
        run_patching_condition_repeat_tree(run.experiment, config=config)


# An absent ID on a described run is an error ---------------------------------------------------


def _file_run(run, tmp_path):
    from cellpatcher.run_session import statistics_session

    config = _config(run, tmp_path / "out")
    with statistics_session():
        return config, open_file_run(config, run.experiment, None, use_masks=False, where="test")


def test_a_described_run_is_described(tmp_path):
    described = _file_run(_described_run(tmp_path / "d", "v13"), tmp_path / "d")[1]
    assert described.described
    assert all(getattr(fov, name) for fov in described.fovs for name in ID_COLUMNS)


@pytest.mark.parametrize("name", ID_COLUMNS)
@pytest.mark.parametrize("missing", ["none", "empty", "absent"])
def test_a_described_fov_without_an_id_is_refused(tmp_path, name, missing):
    config, file_run = _file_run(_described_run(tmp_path, "v33"), tmp_path)
    fov = file_run.fovs[0]
    assert getattr(fov, name)
    if missing == "absent":  # a brick whose Fov has no such attribute (v0.4.0)
        fields = {field: getattr(fov, field) for field in fov.__dataclass_fields__ if field != name}
        fov = SimpleNamespace(**fields)
    else:
        fov = replace(fov, **{name: None if missing == "none" else ""})

    with pytest.raises(PreprocessingCheckError, match=f"has no {name}") as error:
        fov_metadata(file_run, fov, config)
    assert [issue.code for issue in error.value.issues] == ["described_run_missing_id"]
    assert fov.relative_path in str(error.value)


def test_an_undescribed_fov_without_ids_keeps_its_empty_columns(tmp_path):
    from morphobricks_preprocessing.testing import make_clean_run

    run = make_clean_run(tmp_path / "clean")
    config, file_run = _file_run(run, tmp_path)
    assert not file_run.described

    for fov in file_run.fovs:
        assert [getattr(fov, name) for name in ID_COLUMNS] == [None, None, None]
        assert not set(ID_COLUMNS) & set(fov_metadata(file_run, fov, config))
    for fov in (SimpleNamespace(**{k: getattr(file_run.fovs[0], k) for k in ("key", "relative_path")}),):
        assert not set(ID_COLUMNS) & set(fov_metadata(file_run, fov, config))  # no such attribute at all

    rows = _positions(run, tmp_path / "positions_out")
    assert rows
    assert {name: {row[name] for row in rows} for name in ID_COLUMNS} == {name: {""} for name in ID_COLUMNS}


# Legacy parity ------------------------------------------------------------------------------------


def test_v13_described_and_undescribed_runs_differ_only_in_the_three_id_columns(tmp_path):
    # v13: IDs = folder labels = header names. The undescribed twin runs preprocessing on the same
    # images without the description (a warning), so every CellPatcher output must be byte-identical
    # but the ID columns of positions.csv.
    import morphobricks_preprocessing as mbp

    described = _described_run(tmp_path / "d", "v13")
    twin_output = tmp_path / "u" / "output"
    with pytest.warns(UserWarning, match="No experiment description"):
        twin = mbp.run_experiment(described.experiment, mbp.AxisMapping("CYX", "CYX"), output_folder=twin_output)
    assert twin.written
    plain = SimpleNamespace(output=twin_output, experiment=described.experiment)

    described_rows = _positions(described, tmp_path / "d" / "out")
    plain_rows = _positions(plain, tmp_path / "u" / "out")

    for name in ("patches.csv", "images.csv"):
        assert (tmp_path / "d" / "out" / name).read_bytes() == (tmp_path / "u" / "out" / name).read_bytes()
    assert {name: {row[name] for row in plain_rows} for name in ID_COLUMNS} == {name: {""} for name in ID_COLUMNS}
    assert all(row["repeat_group"] and row["condition_id"] and row["technical_repeat_id"] for row in described_rows)
    without_ids = [{k: v for k, v in row.items() if k not in ID_COLUMNS} for row in described_rows]
    assert without_ids == [{k: v for k, v in row.items() if k not in ID_COLUMNS} for row in plain_rows]
    # v13: the IDs are the folder labels (condition_id and technical_repeat_id; the repeat folder is the repeat ID).
    assert all(
        (row["repeat_group"], row["condition_id"], row["technical_repeat_id"])
        == (row["biological_repeat"], row["condition"], row["technical_repeat"])
        for row in described_rows
    )
