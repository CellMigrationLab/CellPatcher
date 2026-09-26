import csv

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.io import (
    SINGLE_CONDITION_LABEL,
    SINGLE_REPEAT_LABEL,
    SINGLE_TECHNICAL_REPEAT_LABEL,
    discover_condition_repeat_folders,
    write_array,
)
from cellpatcher.orchestration import load_config, run_patching_condition_repeat_tree
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
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


def test_discover_flat_root_is_single_condition_and_repeat(tmp_path):
    image_root = tmp_path / "experiment"
    mask_root = tmp_path / "masks"
    _write_image(image_root / "a.npy")
    _write_image(image_root / "b.npy")
    mask_root.mkdir()

    leaves = discover_condition_repeat_folders(image_root, mask_root_path=mask_root)
    assert len(leaves) == 1
    leaf = leaves[0]
    assert (leaf.condition, leaf.repeat) == (SINGLE_CONDITION_LABEL, SINGLE_REPEAT_LABEL)
    assert leaf.image_folder == image_root
    assert leaf.mask_folder == mask_root


def test_discover_root_mixing_loose_images_and_subfolders_raises(tmp_path):
    root = tmp_path / "experiment"
    _write_image(root / "loose.npy")
    _write_image(root / "control" / "c.npy")

    with pytest.raises(ValueError, match="mixes image files"):
        discover_condition_repeat_folders(root)


def test_discover_ignores_hidden_subfolders(tmp_path):
    flat_root = tmp_path / "flat_experiment"
    _write_image(flat_root / "a.npy")
    _write_image(flat_root / ".ipynb_checkpoints" / "a-checkpoint.npy")
    leaves = discover_condition_repeat_folders(flat_root)
    assert [(leaf.condition, leaf.repeat) for leaf in leaves] == [(SINGLE_CONDITION_LABEL, SINGLE_REPEAT_LABEL)]

    condition_root = tmp_path / "condition_experiment"
    _write_image(condition_root / "control" / "c.npy")
    _write_image(condition_root / "control" / ".ipynb_checkpoints" / "c-checkpoint.npy")
    _write_image(condition_root / ".ipynb_checkpoints" / "control" / "c.npy")
    leaves = discover_condition_repeat_folders(condition_root)
    assert [(leaf.condition, leaf.repeat) for leaf in leaves] == [("control", SINGLE_REPEAT_LABEL)]


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


def test_tree_flat_root_stamps_single_condition_and_repeat(tmp_path):
    image_root = tmp_path / "experiment"
    _write_image(image_root / "a.npy")

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = run_patching_condition_repeat_tree(image_root, config=cfg)
    assert len(results) == 1
    patch = results[0].result.patches[0]
    assert (patch.condition, patch.repeat) == (SINGLE_CONDITION_LABEL, SINGLE_REPEAT_LABEL)
    assert (tmp_path / "out" / SINGLE_CONDITION_LABEL / SINGLE_REPEAT_LABEL / "patches.csv").exists()


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


def _combined_tree_config(tmp_path):
    cfg = CellPatcherConfig(mode="grid")
    cfg.grid.tile_size = 8
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")
    return cfg


def test_tree_combine_metadata_writes_single_csv_and_config_at_output_root(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "treated" / "2026_01_01" / "a.npy")
    _write_image(image_root / "control" / "c.npy")
    out = tmp_path / "out"

    results = run_patching_condition_repeat_tree(image_root, config=_combined_tree_config(tmp_path), combine_metadata=True)

    assert (out / "patches.csv").exists()
    assert (out / RUN_CONFIG_FILENAME).exists()
    assert not list(out.rglob("*/patches.csv"))
    assert not list(out.rglob(f"*/{RUN_CONFIG_FILENAME}"))

    with (out / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(len(r.result.patches) for r in results) == 8
    assert {(row["condition"], row["repeat"]) for row in rows} == {
        ("treated", "2026_01_01"),
        ("control", SINGLE_REPEAT_LABEL),
    }
    for row in rows:
        # patch_file stays relative to the CSV, now through the leaf folder.
        assert row["patch_file"].startswith(f"{row['condition']}/{row['repeat']}/")
        assert (out / row["patch_file"]).exists()
        assert row["config_file"] == RUN_CONFIG_FILENAME
    assert all(r.result.metadata_file == str(out / "patches.csv") for r in results)

    restored = load_config(out / RUN_CONFIG_FILENAME)
    assert restored.grid.tile_size == 8
    assert restored.common.output_dir == str(out)


def test_tree_combine_metadata_requires_export(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "control" / "c.npy")
    with pytest.raises(ValueError, match="combine_metadata=True requires"):
        run_patching_condition_repeat_tree(image_root, config=CellPatcherConfig(mode="grid"), combine_metadata=True)


def test_tree_progress_callback_reports_each_leaf_then_completion(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "a" / "x.npy")
    _write_image(image_root / "b" / "y.npy")
    calls = []

    run_patching_condition_repeat_tree(
        image_root,
        config=CellPatcherConfig(mode="whole_fov"),
        progress_callback=lambda done, total, leaf: calls.append((done, total, leaf.condition if leaf else None)),
    )
    assert calls == [(0, 2, "a"), (1, 2, "b"), (2, 2, None)]


def test_save_run_config_false_skips_run_config_json(tmp_path):
    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.export.save_run_config = False
    cfg.common.output_dir = str(tmp_path)

    result = run_patching(images=[np.ones((8, 8), dtype=np.float32)], config=cfg)
    assert not (tmp_path / RUN_CONFIG_FILENAME).exists()
    assert (tmp_path / "patches.csv").exists()
    assert result.patches[0].config_file is None


# ---------------------------------------------------------------------------
# technical repeats: <condition>/<repeat>/<technical_repeat>/
# ---------------------------------------------------------------------------


def test_discover_technical_repeat_subfolders_inside_a_repeat(tmp_path):
    root = tmp_path / "raw_data"
    _write_image(root / "treated" / "2026_01_01" / "B02" / "a.npy")
    _write_image(root / "treated" / "2026_01_01" / "B03" / "b.npy")
    _write_image(root / "treated" / "2026_01_02" / "c.npy")  # a repeat without wells
    _write_image(root / "control" / "d.npy")  # a condition without repeats

    leaves = discover_condition_repeat_folders(root, mask_root_path=tmp_path)
    found = [(leaf.condition, leaf.repeat, leaf.technical_repeat) for leaf in leaves]
    assert found == [
        ("control", SINGLE_REPEAT_LABEL, SINGLE_TECHNICAL_REPEAT_LABEL),
        ("treated", "2026_01_01", "B02"),
        ("treated", "2026_01_01", "B03"),
        ("treated", "2026_01_02", SINGLE_TECHNICAL_REPEAT_LABEL),
    ]
    by_well = {leaf.technical_repeat: leaf for leaf in leaves if leaf.repeat == "2026_01_01"}
    assert by_well["B02"].image_folder == root / "treated" / "2026_01_01" / "B02"
    assert by_well["B02"].mask_folder == tmp_path / "treated" / "2026_01_01" / "B02"
    assert by_well["B02"].relative_folder.as_posix() == "treated/2026_01_01/B02"
    # Without a technical level the output path keeps the two-level layout.
    single = [leaf for leaf in leaves if leaf.repeat == "2026_01_02"][0]
    assert single.relative_folder.as_posix() == "treated/2026_01_02"


def test_discover_repeat_mixing_images_and_technical_subfolders_raises(tmp_path):
    root = tmp_path / "raw_data"
    _write_image(root / "treated" / "2026_01_01" / "loose.npy")
    _write_image(root / "treated" / "2026_01_01" / "B02" / "a.npy")
    with pytest.raises(ValueError, match="Repeat folder .* mixes image files"):
        discover_condition_repeat_folders(root)


def test_tree_stamps_technical_repeat_and_mirrors_it_in_the_output(tmp_path):
    image_root = tmp_path / "raw_data"
    for well in ("B02", "B03"):
        _write_image(image_root / "treated" / "2026_01_01" / well / "a.npy")
    _write_image(image_root / "control" / "2026_01_01" / "a.npy")

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = run_patching_condition_repeat_tree(image_root, config=cfg)
    stamped = {
        (r.condition, r.repeat, r.technical_repeat): {p.technical_repeat for p in r.result.patches}
        for r in results
    }
    assert stamped == {
        ("control", "2026_01_01", SINGLE_TECHNICAL_REPEAT_LABEL): {SINGLE_TECHNICAL_REPEAT_LABEL},
        ("treated", "2026_01_01", "B02"): {"B02"},
        ("treated", "2026_01_01", "B03"): {"B03"},
    }
    out = tmp_path / "out"
    # Same image name in two wells: each leaf keeps its own folder, no overwrite.
    assert (out / "treated" / "2026_01_01" / "B02" / "patches.csv").exists()
    assert (out / "treated" / "2026_01_01" / "B03" / "patches.csv").exists()
    assert (out / "control" / "2026_01_01" / "patches.csv").exists()
    assert cfg.common.technical_repeat is None  # the base config is not mutated


def test_tree_combined_csv_keeps_technical_repeats_apart(tmp_path):
    image_root = tmp_path / "raw_data"
    for well in ("B02", "B03"):
        _write_image(image_root / "treated" / "2026_01_01" / well / "a.npy")
    _write_image(image_root / "control" / "c.npy")
    out = tmp_path / "out"

    results = run_patching_condition_repeat_tree(
        image_root, config=_combined_tree_config(tmp_path), combine_metadata=True
    )
    with (out / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(len(r.result.patches) for r in results)
    assert {(row["condition"], row["repeat"], row["technical_repeat"]) for row in rows} == {
        ("treated", "2026_01_01", "B02"),
        ("treated", "2026_01_01", "B03"),
        ("control", SINGLE_REPEAT_LABEL, SINGLE_TECHNICAL_REPEAT_LABEL),
    }
    # Both wells hold a.npy, so patch ids repeat; the folder keeps the files distinct.
    assert len({row["patch_file"] for row in rows}) == len(rows)
    for row in rows:
        assert (out / row["patch_file"]).exists()
        if row["technical_repeat"] != SINGLE_TECHNICAL_REPEAT_LABEL:
            assert row["patch_file"].startswith(
                f"{row['condition']}/{row['repeat']}/{row['technical_repeat']}/"
            )


def test_tree_cell_mode_finds_masks_in_mirrored_technical_repeat_folders(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=5)
    image_root, mask_root = tmp_path / "raw_data", tmp_path / "masks"
    for well in ("B02", "B03"):
        write_array(image_root / "treated" / "2026_01_01" / well / "img.npy", image)
        write_array(mask_root / "treated" / "2026_01_01" / well / "img_mask.npy", labels)

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 16
    results = run_patching_condition_repeat_tree(image_root, config=cfg, mask_root_path=mask_root)
    assert [r.technical_repeat for r in results] == ["B02", "B03"]
    for r in results:
        assert r.result.patches
        assert {p.mask_file for p in r.result.patches} == {"img_mask.npy"}
        assert {p.technical_repeat for p in r.result.patches} == {r.technical_repeat}
