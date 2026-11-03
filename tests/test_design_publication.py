"""The experiment design of an unfinished but valid publication (audit round 2, A-m2).

A preprocessing run publishes a new generation by writing its manifest first; until the other files
are moved into place they are in ``<output>/.publish``, where the brick's ``published_path`` (and so
``RunInputs.open``) finds them. CellPatcher read ``<output>/experiment_design.csv`` itself.
"""

import json

import pytest

from cellpatcher.pipeline import RUN_CONFIG_FILENAME
from cellpatcher.statistics import PreprocessingCheckError
from tests.test_normalization import _config, _confirmed_design, _patches_by_image, _tree, _unpaired_experiment


def test_the_design_of_an_unfinished_valid_publication_is_read_as_run_inputs_reads_it(tmp_path):
    # Audit round 2, A-m2: a run whose publication is unfinished has the manifest's design only
    # in <output>/.publish (preprocessing's published_path finds it there, and RunInputs opens
    # the run). CellPatcher read <output>/experiment_design.csv itself and refused the run, its
    # SHA-256 not being the manifest's. The design read now is the published copy, SHA-checked.
    import hashlib

    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.manifest import file_sha256
    from morphobricks_preprocessing.run.generation import PUBLISH_FOLDER

    root, _, csv_path = _unpaired_experiment(tmp_path)  # control/rep_a and treated/rep_b unpaired
    pending = csv_path.parent / PUBLISH_FOLDER / "experiment_design.csv"
    pending.parent.mkdir()
    mbp.write_experiment_design(_confirmed_design(root, pairs={("treated", "rep_b"): "rep_a"}), pending)
    manifest_path = csv_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_design_sha256"] = file_sha256(pending)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    patches = _patches_by_image(_tree(root, config, combine_metadata=True))
    assert {p.normalization_group for p in patches.values()} == {"rep_a"}  # the pending design pairs them
    saved = json.loads((tmp_path / "out" / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["normalization_inputs"]["experiment_design_sha256"] == hashlib.sha256(pending.read_bytes()).hexdigest()

    pending.write_bytes(pending.read_bytes() + b"\n")  # the SHA-256 check stays
    with pytest.raises(PreprocessingCheckError):
        _tree(root, _config(csv_path, image_axes="YX", scope="biological_repeat"))
