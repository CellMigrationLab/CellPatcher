"""Outputs are those of CellPatcher v0.6.2 on Spec fixtures v07, v08 and v13 (Spec ADR 0005 step 3).

Spec ADR 0005 moves the labelling of a ``binary`` mask to preprocessing's ``RunInputs.read_objects``
and the region checks to normalization's ``check_region``. On instance masks nothing else may
change. ``tests/fixtures/v062_parity/digests.json`` holds, for every case below, the content
digest of every output file but ``run_config.json`` (patch pixels as dtype, shape and bytes;
CSV files as text), written by **CellPatcher v0.6.2** (``6f8c871``, preprocessing v0.5.6) running
this module::

    python tests/test_v062_output_parity.py <spec fixtures/valid> <work folder> > digests.json

The fixtures' images and masks are made by preprocessing's ``make_described_run``. The cases are
file runs in every mode: cell mode on each ``instances`` mask (plain, ``apply_mask``, patch-scope
normalization over the mask), random mode inside each mask, grid (with and without image-scope
normalization), whole FOV, and v08's ``binary`` ``TumorRegion`` as objects with
``binary_mask_policy="connected_components"`` (8-connected, the v0.6.2 default, so its patches are
unchanged too). ``run_config.json`` is left out: it records the ADR's new ``mask_conversions`` and
no ``cell.binary_mask_connectivity`` (``test_cell_mask_request.py`` checks it).
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

FIXTURES = {
    "v07": ("v07_instance_masks.toml", ("Nuclei", "Cells")),
    "v08": ("v08_binary_masks.toml", ("Nuclei", "TumorRegion")),
    "v13": ("v13_homogeneous_legacy.toml", ()),
}
RANDOM = {"random__patch_size": 8, "random__n_patches": 5, "random__seed": 3}
GOLDEN = Path(__file__).parent / "fixtures" / "v062_parity" / "digests.json"


def _cases(name: str, masks: tuple[str, ...]):
    yield "grid", "grid", {"grid__tile_size": 16}
    yield "whole_fov", "whole_fov", {}
    yield "random", "random", RANDOM
    yield "grid_norm_image", "grid", {"grid__tile_size": 16, "normalization__enabled": True,
                                      "normalization__scope": "image"}
    objects = [mask for mask in masks if mask != "TumorRegion"]
    for mask_id in objects:
        cell = {"matching__mask_id": mask_id, "cell__exclude_border": False}
        yield f"cell_{mask_id}", "cell", cell
        yield f"cell_{mask_id}_applymask", "cell", {**cell, "extraction__apply_mask": True}
        if name == "v07":
            yield f"cell_{mask_id}_norm_patch_mask", "cell", {
                **cell, "normalization__enabled": True, "normalization__scope": "patch",
                "normalization__patch_region": "mask"}
    for mask_type in {"v07": ("nuclei", "cells"), "v08": ("nuclei", "tumor_region")}.get(name, ()):
        yield f"random_in_{mask_type}", "random", {"matching__mask_type": mask_type, **RANDOM}
    if name == "v08":
        yield "BINARY_cell_TumorRegion_policy", "cell", {
            "matching__mask_id": "TumorRegion", "cell__exclude_border": False,
            "cell__binary_mask_policy": "connected_components", "extraction__apply_mask": True}


def _config(output: Path, mode: str, target: Path, settings: dict):
    from cellpatcher.config import CellPatcherConfig

    config = CellPatcherConfig(mode=mode)
    config.common.preprocessing_run = str(output)
    config.common.output_dir = str(target)
    config.export.enabled = True
    for dotted, value in settings.items():
        section, name = dotted.split("__")
        setattr(getattr(config, section), name, value)
    return config


def _digest(path: Path) -> str:
    """Return a file's content digest: pixels (dtype, shape, bytes) for an image, text for a table."""
    import tifffile

    if path.suffix not in (".tif", ".tiff", ".npy"):
        return hashlib.sha256(path.read_text("utf-8").replace("\r\n", "\n").encode()).hexdigest()
    array = np.ascontiguousarray(tifffile.imread(path) if path.suffix != ".npy" else np.load(path))
    head = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}).encode()
    return hashlib.sha256(head + b"\n" + array.tobytes()).hexdigest()


def output_digests(valid: Path, work: Path) -> dict[str, dict[str, str]]:
    """Run every case on the fixtures in *valid*, in *work*, and return each case's output digests."""
    from morphobricks_preprocessing.testing_described import make_described_run

    from cellpatcher.orchestration import run_patching_condition_repeat_tree

    digests = {}
    for name, (toml, masks) in FIXTURES.items():
        run = make_described_run(work / name, (valid / toml).read_bytes(), masks=masks)
        for case, mode, settings in _cases(name, masks):
            target = work / name / "patches" / case
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                run_patching_condition_repeat_tree(run.experiment, config=_config(run.output, mode, target, settings))
            digests[f"{name}/{case}"] = {
                path.relative_to(target).as_posix(): _digest(path)
                for path in sorted(target.rglob("*")) if path.is_file() and path.name != "run_config.json"
            }
    return digests


def test_outputs_are_v062s_on_v07_v08_v13(tmp_path):
    valid = Path(__file__).parent / "spec_fixtures" / "fixtures" / "valid"
    expected = json.loads(GOLDEN.read_text("utf-8"))
    found = output_digests(valid, tmp_path)
    assert sorted(found) == sorted(expected)
    for case in expected:
        assert found[case] == expected[case], case
    assert sum(len(files) for files in found.values()) == 907  # every patch, patches.csv, images.csv, positions.csv


if __name__ == "__main__":  # the golden's generator (see the module docstring)
    print(json.dumps(output_digests(Path(sys.argv[1]), Path(sys.argv[2])), indent=1, sort_keys=True))
