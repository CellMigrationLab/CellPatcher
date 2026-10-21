"""``common.mask_source='ngff_labels'``: masks from the OME-NGFF labels of OME-Zarr images.

See docs/normalization_contract.md §4. Needs the ``preprocessing`` extra with
``zarr``; the tests skip without it.
"""

import importlib.util

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from tests.helpers import write_ngff, write_ngff_labels

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None or importlib.util.find_spec("zarr") is None,
    reason="needs the 'preprocessing' extra with zarr",
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
    config.common.mask_source = "ngff_labels"
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _image(root, name="a.ome.zarr", shape=(16, 16), axes="yx", labels=None, label_axes=None):
    path = write_ngff(root / "control" / name, np.ones(shape, dtype=np.uint16), axes)
    if labels:
        write_ngff_labels(path, labels, label_axes or axes)
    return path


def test_mask_source_is_validated():
    assert (CommonConfig().mask_source, CommonConfig().mask_label_name) == ("tiff_tree", None)
    with pytest.raises(ValueError, match="mask_source must be one of"):
        CommonConfig(mask_source="labels")


def test_cell_mode_reads_the_only_label_image(tmp_path):
    _image(tmp_path, labels={"cells": _labels((16, 16), [(1, 2, 2), (2, 9, 9)])})

    (leaf,) = run_patching_condition_repeat_tree(tmp_path, config=_cell_config())
    assert [(p.label_id, p.mask_file) for p in leaf.result.patches] == [
        (1, "a.ome.zarr/labels/cells"),
        (2, "a.ome.zarr/labels/cells"),
    ]


def test_the_label_image_is_chosen_by_name(tmp_path):
    _image(tmp_path, labels={"cells": _labels((16, 16), [(1, 2, 2)]), "nuclei": _labels((16, 16), [(7, 9, 9)])})

    (leaf,) = run_patching_condition_repeat_tree(tmp_path, config=_cell_config(mask_label_name="nuclei"))
    assert [(p.label_id, p.mask_file) for p in leaf.result.patches] == [(7, "a.ome.zarr/labels/nuclei")]

    with pytest.raises(ValueError, match="several label images \\(cells, nuclei\\)"):
        run_patching_condition_repeat_tree(tmp_path, config=_cell_config())
    with pytest.raises(ValueError, match="no label image 'membranes'"):
        run_patching_condition_repeat_tree(tmp_path, config=_cell_config(mask_label_name="membranes"))


def test_label_planes_follow_or_are_shared_by_the_image_planes(tmp_path):
    per_plane = np.stack([_labels((16, 16), [(z + 1, 2 + 4 * z, 2 + 4 * z)]) for z in range(3)])
    _image(tmp_path, shape=(3, 16, 16), axes="zyx", labels={"cells": per_plane})
    config = _cell_config(image_axes="ZYX", mask_axes="ZYX", z_policy="all")
    (leaf,) = run_patching_condition_repeat_tree(tmp_path, config=config)
    assert [(p.z0, p.label_id) for p in leaf.result.patches] == [(0, 1), (1, 2), (2, 3)]

    shared_root = tmp_path / "shared"
    _image(shared_root, shape=(3, 16, 16), axes="zyx", labels={"cells": per_plane[:1]})
    (leaf,) = run_patching_condition_repeat_tree(shared_root, config=config)
    assert [(p.z0, p.label_id) for p in leaf.result.patches] == [(0, 1), (1, 1), (2, 1)]


def test_a_label_image_that_does_not_fit_is_an_error(tmp_path):
    _image(tmp_path, labels={"cells": _labels((12, 16), [(1, 2, 2)])})
    with pytest.raises(ValueError, match="its Y axis has length 12, the image's 16"):
        run_patching_condition_repeat_tree(tmp_path, config=_cell_config())

    channels = tmp_path / "channels"
    _image(channels, shape=(2, 16, 16), axes="cyx", labels={"cells": _labels((2, 16, 16), [(1, 2, 2)])})
    with pytest.raises(ValueError, match="its channel axis has length 2"):
        run_patching_condition_repeat_tree(channels, config=_cell_config(image_axes="CYX"))

    floats = tmp_path / "floats"
    _image(floats, labels={"cells": _labels((16, 16), [(1, 2, 2)]).astype(np.float32)})
    with pytest.raises(ValueError, match="hold integers"):
        run_patching_condition_repeat_tree(floats, config=_cell_config())


def test_ngff_labels_needs_ome_zarr_images_and_no_mask_folder(tmp_path):
    _image(tmp_path, labels={"cells": _labels((16, 16), [(1, 2, 2)])})
    write_array(tmp_path / "control" / "b.tif", np.ones((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="1 image\\(s\\) are not OME-Zarr: b.tif"):
        run_patching_condition_repeat_tree(tmp_path, config=_cell_config())

    with pytest.raises(ValueError, match="remove mask_root_path"):
        run_patching_condition_repeat_tree(tmp_path, config=_cell_config(), mask_root_path=tmp_path)
    with pytest.raises(ValueError, match="remove mask_folder_path"):
        run_patching_pipeline(tmp_path / "control", config=_cell_config(), mask_folder_path=tmp_path)


def test_random_mode_samples_inside_the_labels(tmp_path):
    _image(tmp_path, shape=(32, 32), labels={"cells": _labels((32, 32), [(1, 20, 20)])})
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches = 3, 5
    config.common.mask_source = "ngff_labels"

    (leaf,) = run_patching_condition_repeat_tree(tmp_path, config=config)
    assert leaf.result.patches
    assert {p.mask_file for p in leaf.result.patches} == {"a.ome.zarr/labels/cells"}
    assert all(17 <= p.y0 <= 23 and 17 <= p.x0 <= 23 for p in leaf.result.patches)
