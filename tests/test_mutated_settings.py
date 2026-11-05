"""Settings edited after construction are refused where a run uses them (release blockers 7 and 8).

The config dataclasses are mutable and documented as edited in place, so ``__post_init__`` alone
does not guard them: ``cfg.random.min_mask_fraction = True`` would reach the sampler as 1. The
checks are repeated in ``generate_regions`` and in a preflight (``check_run_settings``) that
``run_patching`` and the folder and tree routes pass through before any file is read.
"""

import numpy as np
import pytest

from cellpatcher import CellPatcherConfig, RandomConfig, run_patching
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.strategies.random import RandomPatchStrategy

BAD_FRACTIONS = [True, False, np.True_, "0.5", float("nan"), float("inf"), 1.5, -0.1]
IMAGE = np.zeros((64, 64), np.uint16)
MASK = np.ones((64, 64), np.uint8)
META = [{"position_id": "fov"}]


def _random_config(value):
    config = CellPatcherConfig(mode="random")
    config.random.patch_size = 16
    config.random.n_patches = 3
    config.random.min_mask_fraction = value  # assigned after construction: __post_init__ never sees it
    return config


@pytest.mark.parametrize("value", BAD_FRACTIONS, ids=repr)
def test_the_random_strategy_refuses_a_min_mask_fraction_edited_after_construction(value):
    # Former code: True reached the sampler as 1 and "0.5" raised a TypeError mid-draw.
    config = RandomConfig(patch_size=16, n_patches=3)
    config.min_mask_fraction = value
    with pytest.raises(ValueError, match="min_mask_fraction"):
        RandomPatchStrategy(config).generate_regions(image_shape=IMAGE.shape, labels=MASK)


@pytest.mark.parametrize("value", BAD_FRACTIONS, ids=repr)
def test_run_patching_refuses_an_edited_min_mask_fraction(value):
    with pytest.raises(ValueError, match="min_mask_fraction"):
        run_patching(images=[IMAGE], masks=[MASK], config=_random_config(value), image_metadata=META)


@pytest.mark.parametrize("value", BAD_FRACTIONS, ids=repr)
def test_run_patching_refuses_it_by_planes_and_when_no_plane_is_drawn(value):
    config = _random_config(value)
    config.common.z_policy = "all"
    with pytest.raises(ValueError, match="min_mask_fraction"):
        run_patching(images=[np.zeros((2, 64, 64), np.uint16)], masks=[MASK], config=config,
                     image_metadata=META)
    config = _random_config(value)
    config.mode = "grid"  # the preflight checks the whole config: the random section is not drawn here
    with pytest.raises(ValueError, match="min_mask_fraction"):
        run_patching(images=[IMAGE], config=config, image_metadata=META)


@pytest.mark.parametrize("run", [run_patching_pipeline, run_patching_condition_repeat_tree])
@pytest.mark.parametrize("value", [True, "0.5", float("nan")], ids=repr)
def test_folder_and_tree_runs_refuse_it_before_opening_the_preprocessing_run(run, value, tmp_path):
    # Former code: the setting was met only after the run was opened and a file read.
    config = _random_config(value)
    config.common.preprocessing_run = str(tmp_path / "no_such_run")
    with pytest.raises(ValueError, match="min_mask_fraction"):
        run(tmp_path, config=config)


@pytest.mark.parametrize("value", [None, 0, 0.0, 0.25, 1, 1.0, np.float32(0.5)])
def test_valid_fractions_are_unchanged(value):
    result = run_patching(images=[IMAGE], masks=[MASK], config=_random_config(value), image_metadata=META)
    assert len(result.patches) == 3
