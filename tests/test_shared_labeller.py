"""The one shared labeller, ``cellpatcher.label_objects(mask, policy, connectivity)`` (Stage 2 CP-2).

Spec ADR 0002 AR5 and experiment-v1 §4.5c: a ``binary`` mask is an object set only through the
explicit per-request ``binary_mask_policy``, applied by CellPatcher's labeller on every route.
The goldens in ``tests/fixtures/shared_labeller/golden.json`` are written by hand (not by the
labeller) and are what the cross-route tests of MorphoFeature, fiberfeatures and MorphoRelate
compare with.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cellpatcher
from cellpatcher.object_geometry import (
    BINARY_MASK_CONNECTIVITY,
    BINARY_MASK_POLICIES,
    CONNECTIVITIES,
    label_objects,
)

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "shared_labeller" / "golden.json").read_text("utf-8"))


def _cases():
    for case in GOLDEN["cases"]:
        for connectivity, labels in case["labels"].items():
            yield pytest.param(case, int(connectivity), labels, id=f"{case['name']}-connectivity{connectivity}")


@pytest.mark.parametrize(("case", "connectivity", "expected"), list(_cases()))
def test_the_labeller_gives_the_golden_label_image(case, connectivity, expected):
    mask = np.array(case["mask"], dtype=case["dtype"])
    labels = label_objects(mask, GOLDEN["policy"], connectivity)
    assert labels.dtype == np.int32 and labels.shape == mask.shape
    np.testing.assert_array_equal(labels, np.array(expected, dtype=np.int32))


def test_4_and_8_connectivity_differ_only_where_objects_touch_at_a_corner():
    corner = next(case for case in GOLDEN["cases"] if case["name"] == "corner_touching")
    mask = np.array(corner["mask"], dtype=np.uint8)
    four, eight = label_objects(mask, "connected_components", 1), label_objects(mask, "connected_components", 2)
    assert (four.max(), eight.max()) == (5, 3)  # the diagonal chain is 3 objects at 1, one at 2
    np.testing.assert_array_equal(four != 0, eight != 0)  # the foreground is mask != 0 either way
    assert CONNECTIVITIES == {1: "4-connected", 2: "8-connected"}
    assert BINARY_MASK_CONNECTIVITY == 2  # the shared 2-D routes' connectivity (8-connected)


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.uint16, np.int32])
def test_the_two_blob_mask_is_two_objects_whatever_its_binary_pixel_type(dtype):
    blob = next(case for case in GOLDEN["cases"] if case["name"] == "two_blob_0_1")
    mask = np.array(blob["mask"]).astype(dtype)
    if dtype is not bool:
        mask = mask * 255 if dtype == np.uint8 else mask  # 0/255 is binary too
    labels = label_objects(mask, "connected_components", BINARY_MASK_CONNECTIVITY)
    np.testing.assert_array_equal(labels, np.array(blob["labels"][str(BINARY_MASK_CONNECTIVITY)], dtype=np.int32))


def test_the_labeller_is_pure_and_deterministic():
    blob = np.array(GOLDEN["cases"][1]["mask"], dtype=np.uint8)
    before = blob.copy()
    first, second = (label_objects(blob, "connected_components", 1) for _ in range(2))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(blob, before)  # the input is not changed


@pytest.mark.parametrize("policy", [None, "", "single_object", "instances", "CONNECTED_COMPONENTS"])
def test_there_is_no_default_policy_and_no_other_one(policy):
    with pytest.raises(ValueError, match="binary_mask_policy must be one of"):
        label_objects(np.ones((3, 3), np.uint8), policy, 1)


@pytest.mark.parametrize("connectivity", [0, 3, True, 1.0, "1", None])
def test_connectivity_is_1_or_2(connectivity):
    with pytest.raises(ValueError, match="connectivity must be 1"):
        label_objects(np.ones((3, 3), np.uint8), "connected_components", connectivity)


@pytest.mark.parametrize(("mask", "match"), [
    (np.array([[0, 1], [2, 0]], np.uint8), "foreground values 1, 2"),  # labels name objects: instances
    (np.array([[0, -1]], np.int16), "foreground values -1"),
    (np.array([[0.0, 1.0]]), "pixel type float64"),
    (np.ones((2, 2, 2), np.uint8), "must be 2D"),
    (np.zeros((0, 3), np.uint8), "is empty"),
])
def test_a_mask_that_is_not_a_2d_binary_mask_is_refused(mask, match):
    with pytest.raises(ValueError, match=match):
        label_objects(mask, "connected_components", 1)


def test_an_empty_foreground_has_no_objects():
    labels = label_objects(np.zeros((4, 4), np.uint8), "connected_components", 1)
    assert labels.dtype == np.int32 and not labels.any()


def test_the_policies_are_the_resolver_contracts_and_the_labeller_is_public():
    from morphobricks_preprocessing.run_inputs_request import BINARY_MASK_POLICIES as RESOLVER_POLICIES

    assert BINARY_MASK_POLICIES == RESOLVER_POLICIES == ("connected_components",)
    assert cellpatcher.label_objects is label_objects
    assert "label_objects" in cellpatcher.__all__
