"""Refusals before any read, and no patch file left without its record (issue #107).

A file run normalized at a CSV scope fits each group's parameters from ``plane_histograms.csv``,
which is on disk before the first image is read. A group that cannot be fitted (Spec fixture
``v17``: ``PHASE`` in 4 of 8 planes of the repeat group ``R002``, ``A_SCOPE_INCOMPLETE``) is refused
when the run is opened, before any ``RunInputs.read_plane`` call and before anything is written.

A run that stops later, for any reason, leaves no patch file behind: a run's files are written to a
staging folder beside ``output_dir`` and moved into it only once the run has succeeded.
"""

import csv
from pathlib import Path

import pytest
from morphobricks_preprocessing.run_inputs import RunInputs

from cellpatcher import export as export_module
from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from tests.test_described_ids import _described_run

_SCOPE_INCOMPLETE = r"(?s)R002: A_SCOPE_INCOMPLETE: .*channel PHASE is present in 4 of 8 planes"


def _config(run, output, *, scope="biological_repeat"):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.preprocessing_run, config.common.output_dir = str(run.output), str(output)
    config.export.enabled = True
    config.normalization = NormalizationConfig(enabled=True, method="minmax", scope=scope)
    return config


def _no_reads(monkeypatch):
    """Make every ``RunInputs.read_plane`` call fail the test; return the list of the calls made."""
    calls = []

    def read_plane(self, fov, **kwargs):
        calls.append(fov.relative_path)
        raise AssertionError(f"read {fov.relative_path} before the refusal")

    monkeypatch.setattr(RunInputs, "read_plane", read_plane)
    return calls


def _files(folder):
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("combine_metadata", [True, False])
def test_v17_partial_channel_in_a_repeat_group_is_refused_before_any_read_or_write(
    tmp_path, monkeypatch, combine_metadata
):
    run = _described_run(tmp_path, "v17")
    output = tmp_path / "out" / "patches"
    reads = _no_reads(monkeypatch)
    with pytest.raises(ValueError, match=_SCOPE_INCOMPLETE):
        run_patching_condition_repeat_tree(
            run.experiment, config=_config(run, output), combine_metadata=combine_metadata
        )
    assert reads == []
    assert not (tmp_path / "out").exists()  # neither output_dir nor its parent nor a staging folder


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_v17_image_scope_needs_no_partial_group_and_runs(tmp_path):
    # At the image scope every group is one image, whose planes all have its channels: nothing to refuse.
    run = _described_run(tmp_path, "v17")
    output = tmp_path / "out"
    config = _config(run, output, scope="image")
    run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=True)
    with (output / "patches.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all((output / row["patch_file"]).is_file() for row in rows)
    assert not [path for path in tmp_path.iterdir() if ".partial-" in path.name]


def _fail_on_export(monkeypatch, after):
    """Make ``save_patch_array`` fail once *after* files have been written."""
    real = export_module.save_patch_array
    written = []

    def save(array, path):
        if len(written) == after:
            raise OSError("disk full")
        real(array, path)
        written.append(path)

    monkeypatch.setattr(export_module, "save_patch_array", save)
    return written


@pytest.mark.parametrize("combine_metadata", [True, False])
def test_a_tree_run_stopped_after_some_leaves_leaves_no_patch_file(tmp_path, monkeypatch, combine_metadata):
    # v13 has several leaves: the export of a later leaf fails after earlier leaves were written in full.
    run = _described_run(tmp_path, "v13")
    output = tmp_path / "out"
    config = _config(run, output)
    config.normalization.enabled = False
    output.mkdir()
    (output / "kept.txt").write_text("an earlier file of the user's", encoding="utf-8")
    written = _fail_on_export(monkeypatch, after=6)
    with pytest.raises(OSError, match="disk full"):
        run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=combine_metadata)
    assert len(written) == 6  # files were written, to the staging folder
    assert _files(output) == ["kept.txt"]  # untouched; nothing of the failed run
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out", "run"]  # the staging folder is gone


def test_a_folder_run_stopped_mid_export_leaves_no_patch_file(tmp_path, monkeypatch):
    run = _described_run(tmp_path, "v35")
    output = tmp_path / "out"
    config = _config(run, output)
    config.normalization.enabled = False
    _fail_on_export(monkeypatch, after=1)
    with pytest.raises(OSError, match="disk full"):
        run_patching_pipeline(run.experiment, config=config)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run"]


def test_a_successful_run_writes_where_it_always_did(tmp_path):
    run = _described_run(tmp_path, "v13")
    output = tmp_path / "out"
    config = _config(run, output)
    config.normalization.enabled = False
    output.mkdir()
    (output / "kept.txt").write_text("an earlier file of the user's", encoding="utf-8")
    results = run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=True)
    files = _files(output)
    assert {"kept.txt", "patches.csv", "images.csv", "positions.csv", "run_config.json"} <= set(files)
    exported = [path for leaf in results for path in leaf.result.exported_files]
    assert exported and all(path.startswith(str(output)) for path in exported)
    assert all(Path(path).is_file() for path in exported)
    assert {leaf.result.metadata_file for leaf in results} == {str(output / "patches.csv")}
    with (output / "patches.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(row["patch_file"] for row in rows) == sorted(
        name for name in files if name.endswith(".tif")
    )
    assert '"output_dir": "' + str(output) in (output / "run_config.json").read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out", "run"]
