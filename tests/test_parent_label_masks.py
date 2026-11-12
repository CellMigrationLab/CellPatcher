"""Parent-label (grouped) masks on described runs (W3: preprocessing v0.5.2, resolver contract v2).

Spec experiment-v1 §4.5b (owner decision PL1, 11 November 2026): a ``parent_labels`` mask is never an
object set (``A_MASK_NOT_INSTANCES``, with or without a policy, never split or relabelled); as a region,
a consumer that does not declare ``parent_labels_support`` (CellPatcher) reads it as an area, every
non-zero pixel inside, with the recorded marker ``conversion = "parent_labels_as_area"``, which it
records and shows. CellPatcher's region is random mode's sampling mask.

Runs are built with preprocessing's ``make_described_run`` from the vendored Spec fixture ``v38``
(masks ``Cells``, instances, source type ``cells``; ``Mito``, parent_labels, source type ``mito``).
"""

import json
import warnings

import numpy as np
import pytest
import tifffile

from cellpatcher.config import CellPatcherConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.run_request import PARENT_LABELS_AS_AREA
from cellpatcher.statistics import PreprocessingCheckError
from tests.test_described_ids import _described_run

AREA_MESSAGE = "'Mito': parent_labels used as an area; grouping not kept"


@pytest.fixture(scope="module")
def v38(tmp_path_factory):
    return _described_run(tmp_path_factory.mktemp("v38"), "v38", masks=("Cells", "Mito"))


def _config(run, output, mode, **sections):
    config = CellPatcherConfig(mode=mode)
    config.common.preprocessing_run, config.common.output_dir = str(run.output), str(output)
    config.export.enabled = True
    for name, value in sections.items():
        section, attribute = name.split("__")
        setattr(getattr(config, section), attribute, value)
    return config


def _run(run, config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run_patching_condition_repeat_tree(
            run.experiment, config=config, mask_root_path=run.masks, combine_metadata=True
        )
    return results, [str(warning.message) for warning in caught]


def _random(run, output, mask_type, fraction=0.25):
    return _config(
        run, output, "random", matching__mask_type=mask_type,
        random__n_patches=3, random__patch_size=8, random__min_mask_fraction=fraction,
    )


def test_the_marker_is_the_resolvers():
    from morphobricks_preprocessing import PARENT_LABELS_AS_AREA as BRICK

    assert PARENT_LABELS_AS_AREA == BRICK


@pytest.mark.parametrize("matching", [{"matching__mask_id": "Mito"}, {"matching__mask_type": "mito"}])
def test_a_parent_labels_mask_is_never_the_cell_mode_object_set(v38, tmp_path, matching):
    with pytest.raises(PreprocessingCheckError) as raised:
        _run(v38, _config(v38, tmp_path / "out", "cell", **matching))
    assert [issue.code for issue in raised.value.issues] == ["A_MASK_NOT_INSTANCES"]
    message = str(raised.value)
    assert "parent_labels" in message and "never an object set" in message
    assert "binary_mask_policy='connected_components'" not in message  # no conversion is offered for it
    assert not (tmp_path / "out").exists()


def test_a_policy_on_a_parent_labels_mask_is_refused(v38, tmp_path):
    config = _config(v38, tmp_path / "out", "cell", matching__mask_id="Mito",
                     cell__binary_mask_policy="connected_components")
    with pytest.raises(ValueError, match=r"parent_labels.*never split, relabelled or converted") as raised:
        _run(v38, config)
    assert ".." not in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_the_instances_mask_of_the_same_run_is_the_object_set(v38, tmp_path):
    results, messages = _run(v38, _config(v38, tmp_path / "out", "cell", matching__mask_id="Cells"))
    assert [patch for leaf in results for patch in leaf.result.patches]
    assert not [message for message in messages if "parent_labels" in message]
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert "mask_conversions" not in saved["preprocessing"]


def test_random_mode_reads_a_parent_labels_mask_as_a_recorded_and_shown_area(v38, tmp_path):
    results, messages = _run(v38, _random(v38, tmp_path / "out", "mito"))
    assert AREA_MESSAGE in messages  # shown
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert saved["preprocessing"]["mask_conversions"] == [  # recorded with the mask ID
        {"mask_id": "Mito", "role": "region", "conversion": "parent_labels_as_area"}
    ]
    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert patches
    for patch in patches:  # every non-zero pixel is inside: the groups are not kept
        mask = tifffile.imread(v38.masks / "mito" / patch.source_relative_path)
        window = mask[patch.y0:patch.y1, patch.x0:patch.x1]
        assert np.count_nonzero(window) / window.size >= 0.25


def test_random_mode_with_an_instances_mask_records_no_conversion(v38, tmp_path):
    results, messages = _run(v38, _random(v38, tmp_path / "out", "cells", fraction=None))
    assert [patch for leaf in results for patch in leaf.result.patches]
    assert not [message for message in messages if "parent_labels" in message]
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert "mask_conversions" not in saved["preprocessing"]


def test_an_unchecked_parent_labels_area_is_refused_by_the_resolver(tmp_path):
    run = _described_run(tmp_path, "v38", masks=("Cells", "Mito"), mask_step=False)
    with pytest.raises(PreprocessingCheckError) as raised:
        _run(run, _random(run, tmp_path / "out", "mito"))
    assert [issue.code for issue in raised.value.issues] == ["A_MASK_UNCHECKED"]
    assert "random-mode sampling mask" in str(raised.value)
