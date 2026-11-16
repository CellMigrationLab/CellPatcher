"""CellPatcher has no labeller left (Spec ADR 0005 BM1, BM2, BM6).

``label_objects``, ``BINARY_MASK_POLICIES``, ``CONNECTIVITIES`` and ``BINARY_MASK_CONNECTIVITY`` are
removed: the one labeller is preprocessing's ``masks.binary.label_binary``, applied on a file run
by ``RunInputs.read_objects`` with the conversion the resolver recorded, 8-connected and fixed.
The hand-written goldens in ``tests/fixtures/shared_labeller/golden.json`` stay: their
8-connected label images (key ``"2"``, SciPy's rank 2) are what ``label_binary`` gives, so the
labels a CellPatcher file run reads are the ones the removed labeller gave at its default.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cellpatcher
import cellpatcher.object_geometry as object_geometry

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "shared_labeller" / "golden.json").read_text("utf-8"))
REMOVED = ("label_objects", "BINARY_MASK_POLICIES", "CONNECTIVITIES", "BINARY_MASK_CONNECTIVITY")


@pytest.mark.parametrize("name", REMOVED)
def test_the_labeller_and_its_constants_are_removed(name):
    assert not hasattr(cellpatcher, name)
    assert name not in cellpatcher.__all__
    assert not hasattr(object_geometry, name)


def test_no_cellpatcher_source_labels_a_mask():
    # BM2: no other labeller. No module of the package calls scipy's label or names the removed API.
    source = Path(cellpatcher.__file__).parent
    for path in source.rglob("*.py"):
        text = path.read_text("utf-8")
        for call in ("label_objects(", "generate_binary_structure", "import label", "ndimage.label", "label_binary("):
            assert call not in text, (path, call)


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda case: case["name"])
def test_the_8_connected_goldens_are_label_binarys_labels(case):
    from morphobricks_preprocessing import label_binary

    mask = np.array(case["mask"], dtype=case["dtype"])
    labels = label_binary(mask)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, np.array(case["labels"]["2"], dtype=np.int32))
