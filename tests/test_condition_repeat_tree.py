import csv

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.io import (
    SINGLE_REPEAT_LABEL,
    discover_condition_repeat_folders,
    write_array,
)
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from tests.helpers import make_synthetic_cell_image_and_labels


def _write_image(path, shape=(16, 16)):
    write_array(path, np.ones(shape, dtype=np.float32))


# ---------------------------------------------------------------------------
# discover_condition_repeat_folders
# ---------------------------------------------------------------------------


def test_discover_explicit_repeat_subfolders(tmp_path):
    root = tmp_path / "raw_data"
    _write_image(root / "treated" / "2026_01_01" / "a.npy")
    _write_image(root / "treated" / "2026_01_02" / "b.npy")
    _write_image(root / "control" / "2026_01_01" / "c.npy")

    leaves = discover_condition_repeat_folders(root)
    found = {(leaf.condition, leaf.repeat) for leaf in leaves}
    assert found == {
        ("control", "2026_01_01"),
        ("treated", "2026_01_01"),
        ("treated", "2026_01_02"),
    }
    for leaf in leaves:
        assert leaf.mask_folder is None


def test_discover_single_repeat_flat_condition_folder(tmp_path):
    root = tmp_path / "raw_data"
    _write_image(root / "control" / "c.npy")

    leaves = discover_condition_repeat_folders(root)
    assert len(leaves) == 1
    leaf = leaves[0]
    assert leaf.condition == "control"
    assert leaf.repeat == SINGLE_REPEAT_LABEL
    assert leaf.image_folder == root / "control"


def test_discover_mixed_conditions_explicit_and_single_repeat(tmp_path):
    root = tmp_path / "raw_data"
    _write_image(root / "treated" / "2026_01_01" / "a.npy")
    _write_image(root / "control" / "c.npy")

    leaves = discover_condition_repeat_folders(root)
    found = {(leaf.condition, leaf.repeat) for leaf in leaves}
    assert found == {("treated", "2026_01_01"), ("control", SINGLE_REPEAT_LABEL)}


def test_discover_mixing_loose_images_and_subfolders_raises(tmp_path):
    root = tmp_path / "raw_data"
    _write_image(root / "treated" / "loose.npy")
    _write_image(root / "treated" / "2026_01_01" / "a.npy")

    with pytest.raises(ValueError, match="mixes image files"):
        discover_condition_repeat_folders(root)


def test_discover_mirrors_mask_root_for_both_layouts(tmp_path):
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"
    _write_image(image_root / "treated" / "2026_01_01" / "a.npy")
    _write_image(image_root / "control" / "c.npy")
    mask_root.mkdir(parents=True)

    leaves = discover_condition_repeat_folders(image_root, mask_root_path=mask_root)
    by_condition = {leaf.condition: leaf for leaf in leaves}
    assert by_condition["treated"].mask_folder == mask_root / "treated" / "2026_01_01"
    assert by_condition["control"].mask_folder == mask_root / "control"


def test_discover_skips_empty_condition_folder(tmp_path):
    root = tmp_path / "raw_data"
    (root / "empty").mkdir(parents=True)
    _write_image(root / "control" / "c.npy")

    leaves = discover_condition_repeat_folders(root)
    assert [leaf.condition for leaf in leaves] == ["control"]


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        discover_condition_repeat_folders(tmp_path / "missing")


# ---------------------------------------------------------------------------
# run_patching_condition_repeat_tree
# ---------------------------------------------------------------------------


def test_tree_whole_fov_stamps_condition_repeat_and_mirrors_output(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "treated" / "2026_01_01" / "a.npy")
    _write_image(image_root / "control" / "c.npy")  # single-repeat

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = run_patching_condition_repeat_tree(image_root, config=cfg)
    assert len(results) == 2

    by_condition = {r.condition: r for r in results}
    treated = by_condition["treated"]
    assert treated.repeat == "2026_01_01"
    assert treated.result.patches[0].condition == "treated"
    assert treated.result.patches[0].repeat == "2026_01_01"
    assert (tmp_path / "out" / "treated" / "2026_01_01" / "patches.csv").exists()

    control = by_condition["control"]
    assert control.repeat == SINGLE_REPEAT_LABEL
    assert control.result.patches[0].condition == "control"
    assert control.result.patches[0].repeat == SINGLE_REPEAT_LABEL
    assert (tmp_path / "out" / "control" / SINGLE_REPEAT_LABEL / "patches.csv").exists()

    # original config passed in must not be mutated
    assert cfg.common.condition is None
    assert cfg.common.output_dir == str(tmp_path / "out")


def test_tree_cell_mode_explicit_and_single_repeat_with_mirrored_masks(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=5)
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"

    write_array(image_root / "treated" / "2026_01_01" / "img.npy", image)
    write_array(mask_root / "treated" / "2026_01_01" / "img_mask.npy", labels)
    write_array(image_root / "control" / "img.npy", image)
    write_array(mask_root / "control" / "img_mask.npy", labels)

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 24
    cfg.cell.exclude_border = True

    results = run_patching_condition_repeat_tree(image_root, config=cfg, mask_root_path=mask_root)
    assert len(results) == 2
    for r in results:
        assert len(r.result.patches) > 0
        for patch in r.result.patches:
            assert patch.condition == r.condition
            assert patch.repeat == r.repeat
            assert patch.image_file == "img.npy"
            assert patch.mask_file == "img_mask.npy"


def test_tree_cell_mode_missing_mask_folder_raises_per_leaf(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=6)
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"

    write_array(image_root / "treated" / "img.npy", image)
    # no matching mask folder written under mask_root for "treated"
    mask_root.mkdir(parents=True)

    cfg = CellPatcherConfig(mode="cell")
    with pytest.raises(ValueError, match="mask_folder_path is required"):
        run_patching_condition_repeat_tree(image_root, config=cfg, mask_root_path=mask_root)


def test_tree_grid_mode_ignores_mask_root_path(tmp_path):
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"
    _write_image(image_root / "control" / "c.npy")
    # mask_root exists but has nothing relevant -- grid mode must never touch it
    mask_root.mkdir(parents=True)

    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    results = run_patching_condition_repeat_tree(image_root, config=cfg, mask_root_path=mask_root)
    assert len(results) == 1
    assert len(results[0].result.patches) > 0


def test_tree_random_mode_mask_optional_per_leaf(tmp_path):
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"
    write_array(image_root / "treated" / "img.npy", np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    sampling_mask = np.zeros((64, 64), dtype=bool)
    sampling_mask[16:48, 16:48] = True
    write_array(mask_root / "treated" / "img_mask.npy", sampling_mask)
    write_array(image_root / "control" / "img.npy", np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    # no mask for "control" -- random mode must fall back to masks=None for that leaf

    cfg = CellPatcherConfig(mode="random")
    cfg.random.n_patches = 3
    cfg.random.patch_size = 8

    results = run_patching_condition_repeat_tree(image_root, config=cfg, mask_root_path=mask_root)
    by_condition = {r.condition: r for r in results}
    assert len(by_condition["treated"].result.patches) == 3
    assert len(by_condition["control"].result.patches) == 3


def test_tree_no_condition_folders_raises(tmp_path):
    root = tmp_path / "raw_data"
    root.mkdir()
    cfg = CellPatcherConfig(mode="whole_fov")
    with pytest.raises(ValueError, match="No condition subfolders"):
        run_patching_condition_repeat_tree(root, config=cfg)


def test_tree_aggregated_patch_counts_match_per_leaf_csv_rows(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "a" / "2026_01_01" / "x.npy")
    _write_image(image_root / "a" / "2026_01_02" / "y.npy")
    _write_image(image_root / "b" / "z.npy")

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = run_patching_condition_repeat_tree(image_root, config=cfg)
    assert len(results) == 3
    for r in results:
        csv_path = tmp_path / "out" / r.condition / r.repeat / "patches.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(r.result.patches)
        assert all(row["condition"] == r.condition for row in rows)
        assert all(row["repeat"] == r.repeat for row in rows)
