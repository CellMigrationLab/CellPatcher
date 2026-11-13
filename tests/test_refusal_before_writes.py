"""A group with a channel only some of its planes have is refused before any read (issue #107).

At a CSV normalization scope the normalization brick refuses a group whose planes do not all have every
channel (Spec fixture ``v17``: ``PHASE`` in 4 of 8 planes of the repeat group ``R002``, ``A_SCOPE_INCOMPLETE``).
``plane_histograms.csv`` decides it before CellPatcher reads any image, so folder and tree runs refuse it when
the run is opened: no ``RunInputs.read_plane`` call and nothing written. Nothing is fitted then: a group with no
finite pixels or zero spread is still refused by the patching, only when a kept patch uses it.
"""

import csv

import numpy as np
import pytest
import tifffile
from morphobricks_preprocessing.run_inputs import RunInputs

from cellpatcher.config import CellPatcherConfig, FilterConfig, GridConfig, NormalizationConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from tests.helpers import preprocess
from tests.test_described_ids import _described_run

_SCOPE_INCOMPLETE = r"(?s)R002: A_SCOPE_INCOMPLETE: channel PHASE is present in 4 of 8 planes"


def _config(run_output, output, *, scope="biological_repeat"):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.preprocessing_run, config.common.output_dir = str(run_output), str(output)
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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("combine_metadata", [True, False])
def test_v17_tree_run_refuses_the_partial_channel_before_any_read_or_write(tmp_path, monkeypatch, combine_metadata):
    run = _described_run(tmp_path, "v17")
    reads = _no_reads(monkeypatch)
    with pytest.raises(ValueError, match=_SCOPE_INCOMPLETE):
        run_patching_condition_repeat_tree(
            run.experiment, config=_config(run.output, tmp_path / "out" / "patches"), combine_metadata=combine_metadata
        )
    assert reads == []
    assert not (tmp_path / "out").exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_v17_folder_run_refuses_the_partial_channel_before_any_read_or_write(tmp_path, monkeypatch):
    run = _described_run(tmp_path, "v17")
    reads = _no_reads(monkeypatch)
    with pytest.raises(ValueError, match=_SCOPE_INCOMPLETE):
        run_patching_pipeline(run.experiment, config=_config(run.output, tmp_path / "out"))
    assert reads == []
    assert not (tmp_path / "out").exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_v17_image_scope_has_no_partial_group_and_runs(tmp_path):
    # At the image scope every group is one image, whose planes all have its channels: nothing to refuse.
    run = _described_run(tmp_path, "v17")
    output = tmp_path / "out"
    run_patching_condition_repeat_tree(
        run.experiment, config=_config(run.output, output, scope="image"), combine_metadata=True
    )
    with (output / "patches.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all((output / row["patch_file"]).is_file() for row in rows)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_blank_image_the_filters_empty_is_not_refused_up_front(tmp_path):
    # Review of #108: fitting every group when the run is opened refused this run ("zero intensity spread" for the
    # blank image), which v0.5.0 patches: the filters drop every patch of the blank image, so its group is never
    # fitted. Only scope completeness is checked up front; zero spread stays the patching's to refuse.
    rng = np.random.default_rng(0)
    root = tmp_path / "exp"
    for condition in ("ctrl", "drug"):
        folder = root / condition / "rep1"
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / "a.tif", rng.integers(10, 200, (32, 32), dtype=np.uint16))
        blank = condition == "drug"
        tifffile.imwrite(
            folder / "b.tif", np.zeros((32, 32), np.uint16) if blank else rng.integers(10, 200, (32, 32), np.uint16)
        )
    config = _config(preprocess(root, "YX"), tmp_path / "out", scope="image")
    config.filters = FilterConfig(min_mean_intensity=1.0)  # drops every patch of the blank image
    results = run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)
    sources = {patch.source_relative_path for leaf in results for patch in leaf.result.patches}
    assert sources == {"ctrl/rep1/a.tif", "ctrl/rep1/b.tif", "drug/rep1/a.tif"}
