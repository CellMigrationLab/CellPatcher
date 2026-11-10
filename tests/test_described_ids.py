"""The canonical IDs of a described run reach ``positions.csv`` from the run, never from a guess (CP-1).

Runs are built with preprocessing's ``make_described_run`` from the vendored Spec fixtures
(``tests/spec_fixtures``); no test here opens a ``.toml``. CellPatcher sees the description only
through ``RunInputs`` (``Fov.repeat_group``, ``Fov.condition_id``, ``Fov.technical_repeat_id``).

- P32: the condition and technical-repeat folders are names on the forbidden-ID list, so the ID
  differs from the folder (``control`` / ``ctrl``) and a brick that took the folder for the ID is wrong.
- P25: the repeat folders differ across conditions (``v03``) or one folder name pairs with two
  repeats (``v30``); only ``repeat_group`` pairs them correctly.
- Every run is described (Spec ADR 0002 AR10): a run without a published description is refused
  (``A_DESCRIPTION_REQUIRED``), and a FOV without one of the IDs is an error, never an empty cell.
- The ``biological_repeat`` scope pools by ``Fov.repeat_group``, never the derived
  ``experiment_design.csv`` (AR11 (3)).
"""

import csv
from dataclasses import replace
from pathlib import Path

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


def _repeat_scope(run, tmp_path):
    """Patch *run* at the ``biological_repeat`` scope; return ``{(condition, biological_repeat): group}``."""
    config = _config(run, tmp_path / "out")
    config.normalization = NormalizationConfig(
        enabled=True, plane_histograms=str(run.output / "plane_histograms.csv"), scope="biological_repeat"
    )
    results = run_patching_condition_repeat_tree(run.experiment, config=config)
    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert patches
    return {(patch.condition, patch.biological_repeat): patch.normalization_group for patch in patches}


@pytest.mark.parametrize("name", ["v03", "v30"])
def test_p25_normalization_pools_by_the_runs_repeat_group_not_the_folder_name(tmp_path, name):
    # The repeat groups are Fov.repeat_group through RunInputs (owner answer 3, AR11 (3)); the run has no
    # experiment_design.csv to read (make_described_run writes none), and before CP-1's trim the scope
    # refused it (missing_experiment_design).
    from cellpatcher.normalization import repeat_groups
    from cellpatcher.statistics import load_plane_statistics

    run = _described_run(tmp_path, name)
    assert not (run.output / "experiment_design.csv").exists()
    histograms = str(run.output / "plane_histograms.csv")

    groups = repeat_groups(load_plane_statistics(histograms))

    assert groups == {key[:2]: ids[0] for key, ids in _run_inputs_ids(run).items()}
    assert len(set(groups.values())) == 2 and len(groups) == 4  # four folders, two repeats
    assert _repeat_scope(run, tmp_path) == groups  # each patch's normalization_group is its repeat group


def test_the_repeat_scope_ignores_a_derived_design_that_pairs_by_folder_name(tmp_path):
    # v30: CTRL/rep1 pairs with DrugA/rep2 in the description. The run also writes the design that pairs
    # by folder name (rep1 with rep1); CellPatcher never reads it back as a grouping input.
    import morphobricks_preprocessing as mbp

    run = _described_run(tmp_path, "v30")
    design = mbp.default_design(mbp.preflight(run.experiment).dataset.repeats)
    for group in design.groups:
        design = design.with_confirmed(group, True, "2026-11-10")
    result = mbp.run_experiment(run.experiment, mbp.AxisMapping("CYX", "CYX"), output_folder=run.output, design=design)
    assert result.written, [str(issue) for issue in result.issues]
    assert (run.output / "experiment_design.csv").is_file()

    assert _repeat_scope(run, tmp_path) == {
        ("CTRL", "rep1"): "Day1", ("DrugA", "rep2"): "Day1", ("CTRL", "rep2"): "Day2", ("DrugA", "rep1"): "Day2",
    }


def test_a_missing_repeat_group_is_refused_never_grouped_by_folder(tmp_path, monkeypatch):
    from morphobricks_preprocessing.run_inputs import RunInputs

    from cellpatcher.normalization import repeat_groups
    from cellpatcher.statistics import load_plane_statistics

    run = _described_run(tmp_path, "v30")
    statistics = load_plane_statistics(run.output / "plane_histograms.csv")
    fovs = statistics.inputs.fovs()
    monkeypatch.setattr(RunInputs, "fovs", lambda self: (replace(fovs[0], repeat_group=None), *fovs[1:]))

    with pytest.raises(PreprocessingCheckError, match="have no repeat group") as error:
        repeat_groups(statistics)
    assert [issue.code for issue in error.value.issues] == ["missing_repeat_group"]
    assert fovs[0].relative_path in str(error.value)


def test_one_folder_with_two_repeat_groups_is_refused(tmp_path, monkeypatch):
    from morphobricks_preprocessing.run_inputs import RunInputs

    from cellpatcher.normalization import repeat_groups
    from cellpatcher.statistics import load_plane_statistics

    run = _described_run(tmp_path, "v03")
    statistics = load_plane_statistics(run.output / "plane_histograms.csv")
    fovs = statistics.inputs.fovs()
    first = fovs[0]
    twin = next(fov for fov in fovs[1:] if fov.key[:2] == first.key[:2])
    changed = tuple(replace(fov, repeat_group="Other") if fov is twin else fov for fov in fovs)
    monkeypatch.setattr(RunInputs, "fovs", lambda self: changed)

    with pytest.raises(PreprocessingCheckError, match="several repeat groups") as error:
        repeat_groups(statistics)
    assert [issue.code for issue in error.value.issues] == ["repeat_group_conflict"]


def test_a_saved_config_naming_an_experiment_design_is_refused():
    # normalization.experiment_design was removed (owner answer 3): the repeat groups are the run's
    # description's. A file written before holds None, which loads; one naming a file is refused.
    from cellpatcher.run_config_file import config_from_dict

    section = {"enabled": True, "scope": "biological_repeat"}
    loaded = config_from_dict({"run_config_version": 2, "normalization": {**section, "experiment_design": None}})
    assert loaded.normalization.scope == "biological_repeat"
    named = {**section, "experiment_design": "experiment_design.csv"}
    with pytest.raises(ValueError, match="experiment_design='experiment_design.csv' was removed"):
        config_from_dict({"run_config_version": 2, "normalization": named})


# Every run is described ---------------------------------------------------------------------------


def _file_run(run, tmp_path):
    from cellpatcher.run_session import statistics_session

    config = _config(run, tmp_path / "out")
    with statistics_session():
        return config, open_file_run(config, run.experiment, None, use_masks=False, where="test")


def test_every_fov_of_a_described_run_has_the_three_ids(tmp_path):
    _, file_run = _file_run(_described_run(tmp_path, "v13"), tmp_path)
    assert all(getattr(fov, name) for fov in file_run.fovs for name in ID_COLUMNS)


@pytest.mark.parametrize("name", ID_COLUMNS)
@pytest.mark.parametrize("missing", [None, ""])
def test_a_fov_without_an_id_is_refused(tmp_path, name, missing):
    config, file_run = _file_run(_described_run(tmp_path, "v33"), tmp_path)
    fov = replace(file_run.fovs[0], **{name: missing})

    with pytest.raises(PreprocessingCheckError, match=f"has no {name}") as error:
        fov_metadata(file_run, fov, config)
    assert [issue.code for issue in error.value.issues] == ["described_run_missing_id"]
    assert fov.relative_path in str(error.value)


def _undescribed_run(tmp_path):
    """A run of the v13 images without the description (preprocessing v0.5.0 still writes one, with a warning)."""
    import morphobricks_preprocessing as mbp

    described = _described_run(tmp_path / "d", "v13")
    output = tmp_path / "u" / "output"
    with pytest.warns(UserWarning, match="No experiment description"):
        assert mbp.run_experiment(described.experiment, mbp.AxisMapping("CYX", "CYX"), output_folder=output).written
    return described.experiment, output


def _assert_description_required(raised, output):
    assert [issue.code for issue in raised.value.issues] == ["A_DESCRIPTION_REQUIRED"]
    assert "no published experiment description (morphobricks.toml)" in str(raised.value)
    assert "publish its description in the preprocessing notebook" in str(raised.value)  # the fix is named
    assert str(output) in str(raised.value)


def test_a_tree_run_on_an_undescribed_run_is_refused_before_anything_is_written(tmp_path):
    # Spec ADR 0002 AR10: no undescribed path; before the trim the run patched with empty ID columns.
    experiment, output = _undescribed_run(tmp_path)
    config = _config(SimpleRun(output, experiment), tmp_path / "out")

    with pytest.raises(PreprocessingCheckError) as raised:
        run_patching_condition_repeat_tree(experiment, config=config, combine_metadata=True)
    _assert_description_required(raised, output)
    assert not (tmp_path / "out").exists()


def test_the_statistics_of_an_undescribed_run_are_refused(tmp_path):
    # A CSV scope of run_patching opens the run through its table: refused the same way.
    from cellpatcher.statistics import load_plane_statistics

    _, output = _undescribed_run(tmp_path)
    with pytest.raises(PreprocessingCheckError) as raised:
        load_plane_statistics(output / "plane_histograms.csv")
    _assert_description_required(raised, output)


class SimpleRun:
    """Where a run is, as ``make_described_run`` returns it (``output``, ``experiment``)."""

    def __init__(self, output, experiment):
        self.output, self.experiment = output, experiment
