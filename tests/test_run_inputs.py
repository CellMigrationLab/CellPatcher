"""Opening the preprocessing run through its ``RunInputs`` (CP-1, design v0.4 §6).

``load_plane_statistics`` reads only the run's own table, design and
``run_manifest.json``, in the run's output folder, and every refusal of
``RunInputs.open`` stops the run (``preprocessing_run_refused``). Moved here
from ``test_statistics.py``, which uses the same simulated experiment.
"""

import json
import os
import shutil

import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.statistics import PreprocessingCheckError, load_plane_statistics
from tests.test_statistics import STACK_IMAGES, _codes, _preprocess, _write_tiff


@pytest.fixture
def stack_experiment(tmp_path):
    """Three uint16 time series (TYX) and their preprocessing outputs, as in ``test_statistics``."""
    import numpy as np

    root = tmp_path / "experiment"
    rng = np.random.default_rng(0)
    for relative_path in STACK_IMAGES:
        _write_tiff(root / relative_path, rng.integers(1, 4000, (3, 20, 24), dtype=np.uint16), "TYX")
    return root, _preprocess(root, tmp_path / "preprocessing", "TYX")


def test_cellpatcher_pins_no_table_or_manifest_version_of_its_own():
    # CP-1: RunInputs refuses a run it cannot use, so CellPatcher's own version pin (which
    # accepted manifest version "1" only) is gone.
    import cellpatcher.statistics

    assert not hasattr(cellpatcher.statistics, "SUPPORTED_MANIFEST_VERSION")
    assert not hasattr(cellpatcher.statistics, "SUPPORTED_SCHEMA_VERSION")


def test_load_opens_the_run_through_run_inputs(stack_experiment, monkeypatch):
    import morphobricks_normalization as mbn
    from morphobricks_preprocessing.run_inputs import RunInputs

    _, output = stack_experiment
    opened = []
    original = RunInputs.open.__func__

    def recording(cls, folder, *args, **kwargs):
        opened.append(folder)
        return original(cls, folder, *args, **kwargs)

    monkeypatch.setattr(RunInputs, "open", classmethod(recording))
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    assert opened == [output]
    assert isinstance(statistics.inputs, RunInputs)
    assert statistics.records == statistics.inputs.histograms()


def test_load_refuses_a_table_without_its_manifest(stack_experiment, tmp_path):
    # Before CP-1 a table without run_manifest.json was read with a warning, and its images
    # were never checked against the run. RunInputs needs the run's manifest.
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(output / mbn.OUTPUT_FILENAME, alone / mbn.OUTPUT_FILENAME)

    with pytest.raises(PreprocessingCheckError, match="holds no run_manifest.json") as error_info:
        load_plane_statistics(alone / mbn.OUTPUT_FILENAME)
    assert _codes(error_info) == ["preprocessing_run_refused"]


def test_load_refuses_a_manifest_other_than_the_run_s(stack_experiment, tmp_path):
    # Before CP-1 normalization.manifest could name any copy of a manifest. Only the run's own,
    # in its output folder, is read now; naming it explicitly is still allowed.
    import morphobricks_normalization as mbn
    import morphobricks_preprocessing as mbp

    _, output = stack_experiment
    elsewhere = tmp_path / "elsewhere.json"
    shutil.copy(output / mbp.MANIFEST_FILENAME, elsewhere)

    with pytest.raises(PreprocessingCheckError, match="normalization.manifest names") as error_info:
        load_plane_statistics(output / mbn.OUTPUT_FILENAME, manifest=elsewhere)
    assert _codes(error_info) == ["not_the_run_output"]
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME, manifest=output / mbp.MANIFEST_FILENAME)
    assert statistics.manifest_path == output / mbp.MANIFEST_FILENAME


def test_load_refuses_a_renamed_table(stack_experiment):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    renamed = output / "my_histograms.csv"
    shutil.copy(output / mbn.OUTPUT_FILENAME, renamed)
    with pytest.raises(PreprocessingCheckError, match="normalization.plane_histograms names") as error_info:
        load_plane_statistics(renamed)
    assert _codes(error_info) == ["not_the_run_output"]


def test_load_detects_a_table_changed_after_the_manifest(stack_experiment):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    csv_path = output / mbn.OUTPUT_FILENAME
    records = mbn.read_plane_histograms(csv_path)
    mbn.write_plane_histograms(records[:-1], csv_path)

    with pytest.raises(PreprocessingCheckError, match="histogram table plane_histograms.csv is not the one") as error_info:
        load_plane_statistics(csv_path)
    assert _codes(error_info) == ["preprocessing_run_refused"]


def test_load_refuses_a_run_before_v040(stack_experiment):
    # A version 1 manifest is a run before the axis check (design v0.4 §6). Before CP-1,
    # CellPatcher read version 1 manifests only.
    import morphobricks_normalization as mbn
    import morphobricks_preprocessing as mbp

    _, output = stack_experiment
    manifest_path = output / mbp.MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["manifest_version"] = "1"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PreprocessingCheckError, match="written before v0.4.0") as error_info:
        load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    assert _codes(error_info) == ["preprocessing_run_refused"]


def test_load_refuses_a_run_whose_image_changed_since(stack_experiment):
    # RunInputs checks every image of the run when it opens, not only the ones patched.
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    image_path = root / "control" / "img1.tif"
    stat = image_path.stat()
    os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    with pytest.raises(PreprocessingCheckError, match="control/img1.tif") as error_info:
        load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    assert _codes(error_info) == ["preprocessing_run_refused"]


def test_a_clean_run_normalizes_a_tree_run_through_run_inputs(tmp_path):
    # Before CP-1, CellPatcher refused every run written by preprocessing v0.4 (manifest
    # version 2): it read only version 1 manifests.
    from morphobricks_preprocessing.testing import make_clean_run

    from cellpatcher.config import NormalizationConfig

    run = make_clean_run(tmp_path / "clean", design=True)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.normalization = NormalizationConfig(
        enabled=True, plane_histograms=str(run.output / "plane_histograms.csv"), scope="biological_repeat",
    )
    statistics = load_plane_statistics(config.normalization.plane_histograms)
    assert statistics.manifest.manifest_version == "2"
    config.common.preprocessing_run = str(run.output)  # a file run reads through the run (CP-2a)
    results = run_patching_condition_repeat_tree(run.experiment, config=config)
    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert patches
    assert {patch.normalization_scope for patch in patches} == {"biological_repeat"}
    assert {patch.source_relative_path for patch in patches} == {
        fov.relative_path for fov in statistics.inputs.fovs()
    }


def test_a_strict_run_s_version_3_manifest_patches_as_the_default_run(stack_experiment, tmp_path):
    # Preprocessing v0.4.0rc5: a --verify-content run writes manifest version 3 (the default run
    # still writes 2). CellPatcher reads it through RunInputs, records "3" in run_config.json and
    # gives the same normalized patches.
    import morphobricks_preprocessing as mbp
    import numpy as np

    from cellpatcher.config import NormalizationConfig

    root, default_output = stack_experiment
    strict_output = tmp_path / "strict"
    result = mbp.run_experiment(
        root, mbp.AxisMapping("TYX", "TYX"), output_folder=strict_output, mode=mbp.RunMode(verify_content=True)
    )
    assert result.written, [str(issue) for issue in result.issues]
    delivered, recorded = {}, {}
    for name, output in (("default", default_output), ("strict", strict_output)):
        config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
        config.common.extract_arrays, config.common.t_policy = True, "all"
        config.common.preprocessing_run, config.common.output_dir = str(output), str(tmp_path / f"{name}_patches")
        config.export.enabled = True
        config.normalization = NormalizationConfig(enabled=True, plane_histograms=str(output / "plane_histograms.csv"))
        results = run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)
        delivered[name] = {
            (patch.source_relative_path, patch.patch_id): array
            for leaf in results
            for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays, strict=True)
        }
        saved = json.loads((tmp_path / f"{name}_patches" / "run_config.json").read_text(encoding="utf-8"))
        recorded[name] = saved["normalization_inputs"]["manifest_version"]
    assert recorded == {"default": "2", "strict": "3"}
    assert delivered["strict"].keys() == delivered["default"].keys() and delivered["default"]
    for key, array in delivered["default"].items():
        np.testing.assert_array_equal(delivered["strict"][key], array)
