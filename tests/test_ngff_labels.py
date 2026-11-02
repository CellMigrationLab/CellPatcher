"""Masks from the OME-NGFF label images of OME-Zarr images, read through the preprocessing run.

Since CP-2a a file run reads the masks the preprocessing run paired, a label image's type
being its name (``matching.mask_type``), with ``RunInputs.read_mask``; the former
``common.mask_source='ngff_labels'`` and ``mask_label_name`` were removed in v0.2.0 (CP-2b).
Needs the ``zarr`` extra; the tests skip without it.
"""

import importlib.util

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from tests.helpers import file_run_config, preprocess, write_ngff, write_ngff_labels

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None or importlib.util.find_spec("zarr") is None,
    reason="needs the 'zarr' extra",
)


def _labels(shape, squares):
    """Integer labels with one 3 × 3 square per ``(value, y, x)``."""
    labels = np.zeros(shape, dtype=np.uint16)
    for value, y, x in squares:
        labels[..., y:y + 3, x:x + 3] = value
    return labels


def _cell_config(**common):
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 6
    config.cell.exclude_border = False
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _tree(root, config, axes="YX", **preprocessing):
    return run_patching_condition_repeat_tree(root, config=file_run_config(config, root, axes, **preprocessing))


def _image(root, name="a.ome.zarr", shape=(16, 16), axes="yx", labels=None, label_axes=None):
    path = write_ngff(root / "control" / name, np.ones(shape, dtype=np.uint16), axes)
    if labels:
        write_ngff_labels(path, labels, label_axes or axes)
    return path


def test_cell_mode_reads_the_only_label_image(tmp_path):
    _image(tmp_path, labels={"cells": _labels((16, 16), [(1, 2, 2), (2, 9, 9)])})

    (leaf,) = _tree(tmp_path, _cell_config())
    assert [(p.label_id, p.mask_file, p.object_set) for p in leaf.result.patches] == [
        (1, "a.ome.zarr/labels/cells", "cells"),
        (2, "a.ome.zarr/labels/cells", "cells"),
    ]


def test_the_label_image_is_chosen_by_its_type(tmp_path):
    _image(tmp_path, labels={"cells": _labels((16, 16), [(1, 2, 2)]), "nuclei": _labels((16, 16), [(7, 9, 9)])})
    config = _cell_config()
    config.matching.mask_type = "nuclei"
    (leaf,) = _tree(tmp_path, config)
    assert [(p.label_id, p.mask_file) for p in leaf.result.patches] == [(7, "a.ome.zarr/labels/nuclei")]

    config.matching.mask_type = None
    with pytest.raises(ValueError, match=r"recorded 2 mask types \(cells, nuclei\); set matching.mask_type"):
        run_patching_condition_repeat_tree(tmp_path, config=config)
    config.matching.mask_type = "membranes"
    with pytest.raises(ValueError, match="'membranes' is not a mask type of the preprocessing run"):
        run_patching_condition_repeat_tree(tmp_path, config=config)


def test_label_planes_follow_or_are_shared_by_the_image_planes(tmp_path):
    per_plane = np.stack([_labels((16, 16), [(z + 1, 2 + 4 * z, 2 + 4 * z)]) for z in range(3)])
    _image(tmp_path, shape=(3, 16, 16), axes="zyx", labels={"cells": per_plane})
    (leaf,) = _tree(tmp_path, _cell_config(z_policy="all"), "ZYX")
    assert [(p.z0, p.label_id) for p in leaf.result.patches] == [(0, 1), (1, 2), (2, 3)]

    # One 2-D label plane serves every plane only when stated shared for its type (preprocessing).
    shared_root = tmp_path / "shared"
    _image(shared_root, shape=(3, 16, 16), axes="zyx", labels={"cells": per_plane[:1]})
    with pytest.raises(AssertionError, match="does not have the depth slices and time points of its image"):
        preprocess(shared_root, "ZYX")
    (leaf,) = _tree(shared_root, _cell_config(z_policy="all"), "ZYX", mask_axes={"cells": "YX"})
    assert [(p.z0, p.label_id) for p in leaf.result.patches] == [(0, 1), (1, 1), (2, 1)]


def test_a_label_image_that_does_not_fit_has_no_clean_run(tmp_path):
    # Former code: CellPatcher's own reader refused these; now the preprocessing run does, so no
    # file run reads them.
    _image(tmp_path, labels={"cells": _labels((12, 16), [(1, 2, 2)])})
    with pytest.raises(AssertionError, match="do not have the size"):
        preprocess(tmp_path)
    floats = tmp_path / "floats"
    _image(floats, labels={"cells": _labels((16, 16), [(1, 2, 2)]).astype(np.float32)})
    with pytest.raises(AssertionError, match="do not hold whole numbers"):
        preprocess(floats)


def test_every_image_needs_a_label_image_of_the_type(tmp_path):
    _image(tmp_path, labels={"cells": _labels((16, 16), [(1, 2, 2)])})
    write_array(tmp_path / "control" / "b.tif", np.ones((16, 16), dtype=np.uint16))
    # The TIFF has no label image.
    with pytest.raises(ValueError, match="recorded no 'cells' mask for control/b.tif"):
        _tree(tmp_path, _cell_config())


def test_random_mode_samples_inside_the_labels(tmp_path):
    _image(tmp_path, shape=(32, 32), labels={"cells": _labels((32, 32), [(1, 20, 20)])})
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches = 3, 5
    config.matching.mask_type = "cells"  # random mode reads masks when asked

    (leaf,) = _tree(tmp_path, config)
    assert leaf.result.patches
    assert {p.mask_file for p in leaf.result.patches} == {"a.ome.zarr/labels/cells"}
    assert all(17 <= p.y0 <= 23 and 17 <= p.x0 <= 23 for p in leaf.result.patches)
