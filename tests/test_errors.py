"""Typed per-plane refusals (issue #46): callers catch them without parsing messages."""

import pickle

import numpy as np
import pytest

import cellpatcher
from cellpatcher import CellPatcherRefusal, LabelRangeError, SamplingExhaustedError
from cellpatcher.config import CellConfig, CellPatcherConfig, RandomConfig
from cellpatcher.pipeline import run_patching
from tests.helpers import named


def test_refusals_are_exported_and_stay_value_errors():
    for name in ("CellPatcherRefusal", "SamplingExhaustedError", "LabelRangeError"):
        assert name in cellpatcher.__all__
    assert issubclass(CellPatcherRefusal, ValueError)
    assert issubclass(SamplingExhaustedError, CellPatcherRefusal)
    assert issubclass(LabelRangeError, CellPatcherRefusal)


def test_random_sampling_exhaustion_is_a_typed_refusal():
    config = CellPatcherConfig(mode="random", random=RandomConfig(patch_size=5, n_patches=3, seed=0))
    image = np.zeros((20, 20), dtype=np.uint8)
    empty_mask = np.zeros((20, 20), dtype=bool)
    with pytest.raises(SamplingExhaustedError) as info:
        run_patching(images=[image], masks=[empty_mask], config=config, image_metadata=named([image]))
    assert (info.value.requested, info.value.collected) == (3, 0)
    assert info.value.attempts == 1000


def test_labels_above_int32_max_are_a_typed_refusal():
    config = CellPatcherConfig(mode="cell", cell=CellConfig(patch_size=2, exclude_border=False))
    image = np.zeros((4, 4), dtype=np.uint8)
    labels = np.zeros((4, 4), dtype=np.uint32)
    labels[1:3, 1:3] = 2**31
    with pytest.raises(LabelRangeError, match="int32 maximum") as info:
        run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))
    assert info.value.label == 2**31


def test_negative_labels_are_a_typed_refusal():
    from cellpatcher.strategies.cell import CellSegmentationStrategy

    labels = np.zeros((4, 4), dtype=np.int16)
    labels[0, 0] = -3
    with pytest.raises(LabelRangeError, match="non-negative") as info:
        CellSegmentationStrategy(CellConfig(exclude_border=False)).generate_regions(
            image_shape=(4, 4), labels=labels
        )
    assert info.value.label == -3


def test_typed_refusals_survive_pickling_for_worker_processes():
    error = pickle.loads(pickle.dumps(SamplingExhaustedError("m", requested=3, collected=1, attempts=9)))
    assert isinstance(error, SamplingExhaustedError)
    assert (str(error), error.requested, error.collected, error.attempts) == ("m", 3, 1, 9)
    error = pickle.loads(pickle.dumps(LabelRangeError("m", label=-1)))
    assert isinstance(error, LabelRangeError) and error.label == -1
