"""A run preprocessing stopped on purpose is refused with preprocessing's own refusal, unchanged (issue #126).

Found by the S3.4 adversarial suite (MorphoFeature draft PR #240, row A07). Spec fixture ``v29`` puts RGB
units beside fluorescence units; preprocessing stops it on purpose (``R_HETEROGENEOUS_LAYOUT_NOT_ENABLED``)
and writes no run, only its report. CellPatcher said "Run preprocessing again", which stops the same way:
its advice looped. Since preprocessing v0.7.0, ``RunInputs.open`` raises ``RunStoppedError``
(``RUN_FOLDER_STOPPED``, preprocessing contract §9.1) there, and ``RunReportUnreadableError``
(``RUN_FOLDER_REPORT_UNREADABLE``) for a report it cannot read. Spec ADR 0005 (construction and
propagation): CellPatcher passes each on with its code and message unchanged, adds no advice of its own,
and passes a stopped run's issues on unchanged and in order. It never reads ``discovery_report.json``.
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
SRC = Path(__file__).resolve().parents[1] / "src" / "cellpatcher"


def _v29_stopped(tmp_path):
    """Run preprocessing on Spec v29, as the G4 suite does; return (experiment, output, its errors)."""
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


def _fields(issue):
    return (issue.severity, issue.code, issue.message, issue.paths, issue.action)


def test_v29_passes_run_stopped_error_on_unchanged_with_no_advice_of_cellpatchers(tmp_path):
    from morphobricks_preprocessing import RunInputs, RunStoppedError

    experiment, output, refused = _v29_stopped(tmp_path)
    assert [issue.code for issue in refused] == ["R_HETEROGENEOUS_LAYOUT_NOT_ENABLED"]  # v29's single refusal
    with pytest.raises(RunStoppedError) as stopped:  # what CellPatcher passes on
        RunInputs.open(output)

    with pytest.raises(PreprocessingCheckError) as raised:
        _tree_run(experiment, output, tmp_path / "patches")
    first, *rest = raised.value.issues
    # The error itself: its code and its message, compared as strings, no action of CellPatcher's.
    assert first.code == RunStoppedError.code == "RUN_FOLDER_STOPPED"
    assert first.message == str(stopped.value)
    assert first.action == "" and first.paths == (str(output),)
    # Then the stopped run's issues, equal to RunStoppedError.issues and to the run's own errors, in order.
    assert tuple(rest) == stopped.value.issues
    assert [_fields(issue) for issue in rest] == [_fields(issue) for issue in refused]
    assert isinstance(raised.value.__cause__, RunStoppedError)
    # No text of CellPatcher's own: every message and action is preprocessing's.
    assert "cannot be used" not in str(raised.value)
    assert not (tmp_path / "patches").exists()


def test_the_issues_come_from_run_stopped_error_not_from_reading_the_report(tmp_path, monkeypatch):
    # The issues passed on are RunStoppedError's, in its order and unchanged (paths not re-sorted), and
    # CellPatcher opens no discovery_report.json of its own.
    from morphobricks_preprocessing import RunInputs, RunStoppedError
    from morphobricks_preprocessing.validation import Severity, ValidationIssue

    raised_issues = (
        ValidationIssue(Severity.ERROR, "B_SECOND", "second, listed first.", ("z.tif", "a.tif"), "Fix B."),
        ValidationIssue(Severity.ERROR, "A_FIRST", "first, listed second.", (), ""),
    )
    message = "RUN_FOLDER_STOPPED: stopped, as preprocessing words it."

    def stopped(*_args, **_kwargs):
        raise RunStoppedError(message, raised_issues)

    def no_report_read(self, *args, **kwargs):
        if self.name == "discovery_report.json":
            raise AssertionError("CellPatcher read discovery_report.json")
        return original(self, *args, **kwargs)

    original = Path.read_text
    monkeypatch.setattr(RunInputs, "open", stopped)
    monkeypatch.setattr(Path, "read_text", no_report_read)
    (tmp_path / "discovery_report.json").write_bytes(b"this is not read")
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    first, *rest = raised.value.issues
    assert (first.code, first.message, first.action) == ("RUN_FOLDER_STOPPED", message, "")
    assert tuple(rest) == raised_issues


@pytest.mark.parametrize(
    "report",
    [
        "not json",
        json.dumps({"report_version": "2", "status": "stopped", "issues": [{"code": "X", "severity": "error"}]}),
        json.dumps({"report_version": "1", "step": "run", "status": "stopped", "provisional": False, "issues": []}),
    ],
)
def test_an_unreadable_report_gives_preprocessings_refusal_unchanged(tmp_path, report):
    # Nothing is guessed from a report preprocessing cannot read: its code and message, naming the report.
    from morphobricks_preprocessing import RunInputs, RunReportUnreadableError

    path = tmp_path / "discovery_report.json"
    path.write_text(report, encoding="utf-8")
    with pytest.raises(RunReportUnreadableError) as unreadable:
        RunInputs.open(tmp_path)
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    (issue,) = raised.value.issues
    assert issue.code == RunReportUnreadableError.code == "RUN_FOLDER_REPORT_UNREADABLE"
    assert issue.message == str(unreadable.value)
    assert issue.paths == (str(path),) and issue.action == ""


def test_a_folder_without_a_run_or_report_says_to_run_preprocessing_first_not_again(tmp_path):
    with pytest.raises(PreprocessingCheckError) as raised:
        open_run(tmp_path)
    assert [issue.code for issue in raised.value.issues] == ["preprocessing_run_refused"]
    assert "holds no run_manifest.json" in str(raised.value)
    assert "again" not in raised.value.issues[0].action.lower()


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


def test_cellpatcher_constructs_no_run_folder_code_and_reads_no_run_report():
    # Construction is preprocessing's alone (Spec ADR 0005): no RUN_FOLDER_* text and no report reader here.
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert '"RUN_FOLDER_' not in text and "'RUN_FOLDER_" not in text, path
        assert "discovery_report.json" not in text, path
