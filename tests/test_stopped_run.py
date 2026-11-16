"""A run preprocessing refused on purpose is refused with preprocessing's code and fix (issue #126).

Found by the S3.4 adversarial suite (MorphoFeature draft PR #240). Spec fixture ``v29`` puts RGB units
beside fluorescence units; preprocessing refuses it on purpose (``R_HETEROGENEOUS_LAYOUT_NOT_ENABLED``)
and writes no run, only its report. CellPatcher said "Run preprocessing again", which stops the same
way: the advice looped. Spec ADR 0005 (construction and propagation): a refusal is passed on with its
code, message and fix unchanged, and CellPatcher advises a re-run only where one can help (a stale run).
CellPatcher gets the stopped run's issues from preprocessing's ``RunInputs.open`` (``RunStoppedError``,
``RUN_FOLDER_STOPPED``, preprocessing contract §9.1) and never reads ``discovery_report.json`` itself.
"""

import json
from pathlib import Path

import pytest
import tifffile

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.run_open import open_run
from cellpatcher.statistics import PreprocessingCheckError

VALID = Path(__file__).parent / "spec_fixtures" / "fixtures" / "valid"


def _v29_stopped(tmp_path):
    """Run preprocessing on Spec v29, as the G4 suite does; return (experiment, output, its issues)."""
    from morphobricks_preprocessing import run_experiment
    from morphobricks_preprocessing.axes import prefill_mapping
    from morphobricks_preprocessing.testing_described import description_of, write_described_tree

    (fixture,) = VALID.glob("v29_*.toml")
    parsed, data = description_of(fixture.read_bytes())
    experiment, output = tmp_path / "experiment", tmp_path / "output"
    write_described_tree(experiment, None, parsed)
    output.mkdir()
    (output / "morphobricks.toml").write_bytes(data)
    result = run_experiment(experiment, prefill_mapping("CYX"), output_folder=output)
    assert not result.written and not (output / "run_manifest.json").exists()
    return experiment, output, [issue for issue in result.issues if issue.is_error]


def _tree_run(experiment, output, target):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4))
    config.common.preprocessing_run, config.common.output_dir = str(output), str(target)
    config.export.enabled = True
    return run_patching_condition_repeat_tree(experiment, config=config, combine_metadata=True)


def test_v29_refusal_carries_preprocessings_code_message_and_fix_and_no_rerun_advice(tmp_path):
    experiment, output, refused = _v29_stopped(tmp_path)
    assert refused[0].code == "R_HETEROGENEOUS_LAYOUT_NOT_ENABLED"

    with pytest.raises(PreprocessingCheckError) as raised:
        _tree_run(experiment, output, tmp_path / "patches")
    issues = raised.value.issues
    # preprocessing's errors first, unchanged (code, message, paths, fix), then CellPatcher's naming the folder
    assert [(i.code, i.message, i.paths, i.action) for i in issues[:-1]] == [
        (i.code, i.message, i.paths, i.action) for i in refused
    ]
    assert issues[0].code == "R_HETEROGENEOUS_LAYOUT_NOT_ENABLED"
    from morphobricks_preprocessing import RunInputs, RunStoppedError

    with pytest.raises(RunStoppedError) as stopped:  # what CellPatcher passes on, equal and in order
        RunInputs.open(output)
    assert tuple(issues[:-1]) == stopped.value.issues
    assert issues[-1].code == "preprocessing_run_refused" and str(output) in issues[-1].message
    text = str(raised.value)
    assert "no run_manifest.json" in text  # what the G4 suite's A07 matches
    assert "preprocessing again" not in text.lower()  # the issue's acceptance: no re-run advice
    assert "again" not in issues[-1].action.lower() and "run preprocessing" not in issues[-1].action.lower()
    assert not (tmp_path / "patches").exists()


def test_a_folder_without_a_run_or_report_says_to_run_preprocessing_first_not_again(tmp_path):
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    assert [issue.code for issue in raised.value.issues] == ["preprocessing_run_refused"]
    assert "holds no run_manifest.json" in str(raised.value)
    assert "again" not in raised.value.issues[0].action.lower()
    assert "discovery_report.txt" in raised.value.issues[0].action


def test_the_issues_come_from_run_stopped_error_not_from_reading_the_report(tmp_path, monkeypatch):
    # The issues passed on are RunStoppedError's, in its order and unchanged (paths not re-sorted), and
    # CellPatcher opens no discovery_report.json of its own.
    from morphobricks_preprocessing import RunInputs, RunStoppedError
    from morphobricks_preprocessing.validation import Severity, ValidationIssue

    raised_issues = (
        ValidationIssue(Severity.ERROR, "B_SECOND", "second, listed first.", ("z.tif", "a.tif"), "Fix B."),
        ValidationIssue(Severity.ERROR, "A_FIRST", "first, listed second.", (), ""),
    )

    def stopped(*_args, **_kwargs):
        raise RunStoppedError("RUN_FOLDER_STOPPED: stopped.", raised_issues)

    monkeypatch.setattr(RunInputs, "open", stopped)
    (tmp_path / "discovery_report.json").write_text("this is not read", encoding="utf-8")
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    issues = raised.value.issues
    assert tuple(issues[:-1]) == raised_issues
    assert issues[-1].code == "preprocessing_run_refused" and "RUN_FOLDER_STOPPED" in issues[-1].message


@pytest.mark.parametrize(
    "report",
    [
        "not json",
        json.dumps({"report_version": "2", "status": "stopped", "issues": [{"code": "X", "severity": "error"}]}),
        json.dumps({"report_version": "1", "step": "run", "status": "stopped", "provisional": False, "issues": []}),
    ],
)
def test_an_unreadable_report_gives_preprocessings_coded_refusal(tmp_path, report):
    # Nothing is guessed from a report preprocessing cannot read: its code, naming the report.
    path = tmp_path / "discovery_report.json"
    path.write_text(report, encoding="utf-8")
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    (issue,) = raised.value.issues
    assert issue.code == "RUN_FOLDER_REPORT_UNREADABLE"
    assert issue.paths == (str(path),)
    assert "RUN_FOLDER_REPORT_UNREADABLE" in issue.message and str(tmp_path) in issue.message


@pytest.mark.parametrize("step, status, provisional", [("run", "written", False), ("check", "stopped", False),
                                                       ("run", "stopped", True)])
def test_a_report_of_no_stopped_run_gives_the_plain_missing_run_refusal(tmp_path, step, status, provisional):
    # A written, check or provisional report names no stopped run: the plain refusal of a folder without a run.
    report = {"report_version": "1", "step": step, "status": status, "provisional": provisional, "issues": [
        {"code": "X", "severity": "error", "message": "m", "paths": [], "action": "a"}]}
    (tmp_path / "discovery_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    assert [issue.code for issue in raised.value.issues] == ["preprocessing_run_refused"]
    assert "holds no run_manifest.json" in str(raised.value)
    assert "again" not in raised.value.issues[0].action.lower()


def test_a_stale_run_still_advises_running_preprocessing_again(tmp_path):
    # A run whose image changed since it was written is one a re-run fixes: that advice stays.
    from morphobricks_preprocessing.testing_described import make_described_run

    (fixture,) = VALID.glob("v13_*.toml")
    run = make_described_run(tmp_path / "run", fixture.read_bytes())
    image = next(run.experiment.rglob("*.tif"))
    tifffile.imwrite(image, tifffile.imread(image)[:, :4, :4], photometric="minisblack", metadata={"axes": "CYX"})
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(run.output)
    assert [issue.code for issue in raised.value.issues] == ["preprocessing_run_refused"]
    assert "Run preprocessing again" in raised.value.issues[0].action
