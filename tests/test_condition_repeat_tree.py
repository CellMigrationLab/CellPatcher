import csv

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.io import (
    SINGLE_BIOLOGICAL_REPEAT_LABEL,
    SINGLE_CONDITION_LABEL,
    SINGLE_TECHNICAL_REPEAT_LABEL,
    write_array,
)
from cellpatcher.orchestration import load_config, run_patching_condition_repeat_tree
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from tests.helpers import make_synthetic_cell_image_and_labels, named
from tests.helpers import preprocessed_tree as _tree


def _write_image(path, shape=(16, 16)):
    write_array(path, np.ones(shape, dtype=np.float32))


# ---------------------------------------------------------------------------
# run_patching_condition_repeat_tree
# ---------------------------------------------------------------------------


def test_tree_whole_fov_stamps_condition_repeat_and_mirrors_output(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "treated" / "2026_01_01" / "a.tif")
    _write_image(image_root / "control" / "c.tif")  # single-repeat

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = _tree(image_root, config=cfg)
    assert len(results) == 2

    by_condition = {r.condition: r for r in results}
    treated = by_condition["treated"]
    assert treated.biological_repeat == "2026_01_01"
    assert treated.result.patches[0].condition == "treated"
    assert treated.result.patches[0].biological_repeat == "2026_01_01"
    assert (tmp_path / "out" / "treated" / "2026_01_01" / "patches.csv").exists()

    control = by_condition["control"]
    assert control.biological_repeat == SINGLE_BIOLOGICAL_REPEAT_LABEL
    assert control.result.patches[0].condition == "control"
    assert control.result.patches[0].biological_repeat == SINGLE_BIOLOGICAL_REPEAT_LABEL
    assert (tmp_path / "out" / "control" / SINGLE_BIOLOGICAL_REPEAT_LABEL / "patches.csv").exists()

    # original config passed in must not be mutated
    assert cfg.common.condition == SINGLE_CONDITION_LABEL
    assert cfg.common.output_dir == str(tmp_path / "out")


def test_tree_flat_root_stamps_single_condition_and_repeat(tmp_path):
    image_root = tmp_path / "experiment"
    _write_image(image_root / "a.tif")

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = _tree(image_root, config=cfg)
    assert len(results) == 1
    patch = results[0].result.patches[0]
    assert (patch.condition, patch.biological_repeat) == (SINGLE_CONDITION_LABEL, SINGLE_BIOLOGICAL_REPEAT_LABEL)
    assert (tmp_path / "out" / SINGLE_CONDITION_LABEL / SINGLE_BIOLOGICAL_REPEAT_LABEL / "patches.csv").exists()


def test_tree_cell_mode_explicit_and_single_repeat_with_mirrored_masks(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=5)
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"

    write_array(image_root / "treated" / "2026_01_01" / "img.tif", image)
    write_array(mask_root / "treated" / "2026_01_01" / "img_mask.tif", labels)
    write_array(image_root / "control" / "img.tif", image)
    write_array(mask_root / "control" / "img_mask.tif", labels)

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 24
    cfg.cell.exclude_border = True

    results = _tree(image_root, config=cfg, mask_root_path=mask_root)
    assert len(results) == 2
    for r in results:
        assert len(r.result.patches) > 0
        for patch in r.result.patches:
            assert patch.condition == r.condition
            assert patch.biological_repeat == r.biological_repeat
            assert patch.image_file == "img.tif"
            assert patch.mask_file == "img_mask.tif"


def test_tree_cell_mode_mask_folder_with_several_masks_per_image(tmp_path):
    # Each mask folder holds every compartment's mask of each image (JUMP
    # style); config.matching.mask_type names the one to use (issue #45:
    # the preprocessing brick's mask types).
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=5)
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"
    for leaf in ("treated/2026_01_01", "control/2026_01_01"):
        write_array(image_root / leaf / "img.tif", image)
        for suffix in ("_cells", "_nuclei", "_mito", "_nucleoli"):
            write_array(mask_root / leaf / f"img{suffix}.tif", labels)

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 24
    with pytest.raises(ValueError, match="4 mask types .cells, mito, nuclei, nucleoli.; set matching.mask_type"):
        _tree(image_root, config=cfg, mask_root_path=mask_root)

    cfg.matching.mask_type = "cells"
    cfg.export.enabled = True
    cfg.common.output_dir = str(tmp_path / "out")
    results = _tree(image_root, config=cfg, mask_root_path=mask_root)
    assert len(results) == 2
    assert {patch.mask_file for r in results for patch in r.result.patches} == {"img_cells.tif"}
    assert {patch.object_set for r in results for patch in r.result.patches} == {"cells"}
    # The setting is recorded with the run.
    restored = load_config(tmp_path / "out" / "control" / "2026_01_01" / RUN_CONFIG_FILENAME)
    assert restored.matching.mask_type == "cells"


def test_tree_cell_mode_missing_mask_folder_raises_per_leaf(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=6)
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"

    write_array(image_root / "treated" / "img.tif", image)
    # no matching mask folder written under mask_root for "treated"
    mask_root.mkdir(parents=True)

    cfg = CellPatcherConfig(mode="cell")
    with pytest.raises(ValueError, match="reads masks, but the preprocessing run .* recorded none"):
        _tree(image_root, config=cfg, mask_root_path=mask_root)


def test_tree_grid_mode_ignores_mask_root_path(tmp_path):
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"
    _write_image(image_root / "control" / "c.tif")
    # mask_root exists but has nothing relevant -- grid mode must never touch it
    mask_root.mkdir(parents=True)

    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    results = _tree(image_root, config=cfg, mask_root_path=mask_root)
    assert len(results) == 1
    assert len(results[0].result.patches) > 0


def test_tree_random_mode_mask_root_needs_a_mask_for_every_image(tmp_path):
    # Issue #45: the whole tree is paired at once with the preprocessing
    # brick's rule, so an image without a mask is refused, never sampled
    # without one; random mode without a mask root still samples everywhere.
    image_root = tmp_path / "raw_data"
    mask_root = tmp_path / "masks"
    write_array(image_root / "treated" / "img.tif", np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    sampling_mask = np.zeros((64, 64), dtype=np.uint8)
    sampling_mask[16:48, 16:48] = 1  # a sampling region: a binary mask, and described so
    write_array(mask_root / "treated" / "img_mask.tif", sampling_mask)
    write_array(image_root / "control" / "img.tif", np.arange(64 * 64, dtype=np.float32).reshape(64, 64))

    cfg = CellPatcherConfig(mode="random")
    cfg.random.n_patches = 3
    cfg.random.patch_size = 8
    with pytest.raises(ValueError, match=r"(?s)present in 1 of 2 acquisitions.*control/img.tif.*A_SCOPE_INCOMPLETE"):
        _tree(image_root, config=cfg, mask_root_path=mask_root, representations={"mask": "binary"})

    write_array(mask_root / "control" / "img_mask.tif", np.ones((64, 64), dtype=np.uint8))
    results = _tree(image_root, config=cfg, mask_root_path=mask_root, representations={"mask": "binary"})
    assert {r.condition: len(r.result.patches) for r in results} == {"control": 3, "treated": 3}

    results = _tree(image_root, config=cfg)
    assert {r.condition: len(r.result.patches) for r in results} == {"control": 3, "treated": 3}


def test_tree_aggregated_patch_counts_match_per_leaf_csv_rows(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "a" / "2026_01_01" / "x.tif")
    _write_image(image_root / "a" / "2026_01_02" / "y.tif")
    _write_image(image_root / "b" / "z.tif")

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = _tree(image_root, config=cfg)
    assert len(results) == 3
    for r in results:
        csv_path = tmp_path / "out" / r.condition / r.biological_repeat / "patches.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(r.result.patches)
        assert all(row["condition"] == r.condition for row in rows)
        assert all(row["biological_repeat"] == r.biological_repeat for row in rows)


def _combined_tree_config(tmp_path):
    cfg = CellPatcherConfig(mode="grid")
    cfg.grid.tile_size = 8
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")
    return cfg


def test_tree_combine_metadata_writes_single_csv_and_config_at_output_root(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "treated" / "2026_01_01" / "a.tif")
    _write_image(image_root / "control" / "c.tif")
    out = tmp_path / "out"

    results = _tree(image_root, config=_combined_tree_config(tmp_path), combine_metadata=True)

    assert (out / "patches.csv").exists()
    assert (out / RUN_CONFIG_FILENAME).exists()
    assert not list(out.rglob("*/patches.csv"))
    assert not list(out.rglob(f"*/{RUN_CONFIG_FILENAME}"))

    with (out / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(len(r.result.patches) for r in results) == 8
    assert {(row["condition"], row["biological_repeat"]) for row in rows} == {
        ("treated", "2026_01_01"),
        ("control", SINGLE_BIOLOGICAL_REPEAT_LABEL),
    }
    for row in rows:
        # patch_file stays relative to the CSV, now through the leaf folder.
        assert row["patch_file"].startswith(f"{row['condition']}/{row['biological_repeat']}/")
        assert (out / row["patch_file"]).exists()
        assert row["config_file"] == RUN_CONFIG_FILENAME
    assert all(r.result.metadata_file == str(out / "patches.csv") for r in results)

    restored = load_config(out / RUN_CONFIG_FILENAME)
    assert restored.grid.tile_size == 8
    assert restored.common.output_dir == str(out)


def test_tree_combine_metadata_requires_export(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "control" / "c.tif")
    with pytest.raises(ValueError, match="combine_metadata=True requires"):
        run_patching_condition_repeat_tree(image_root, config=CellPatcherConfig(mode="grid"), combine_metadata=True)


def test_tree_progress_callback_reports_each_leaf_then_completion(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "a" / "x.tif")
    _write_image(image_root / "b" / "y.tif")
    calls = []

    _tree(
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

    result = run_patching(images=[np.ones((8, 8), dtype=np.float32)], config=cfg,
        image_metadata=named([np.ones((8, 8), dtype=np.float32)]))
    assert not (tmp_path / RUN_CONFIG_FILENAME).exists()
    assert (tmp_path / "patches.csv").exists()
    assert result.patches[0].config_file is None


# ---------------------------------------------------------------------------
# technical repeats: <condition>/<repeat>/<technical_repeat>/
# ---------------------------------------------------------------------------


def test_tree_stamps_technical_repeat_and_mirrors_it_in_the_output(tmp_path):
    image_root = tmp_path / "raw_data"
    for well in ("B02", "B03"):
        _write_image(image_root / "treated" / "2026_01_01" / well / "a.tif")
    _write_image(image_root / "control" / "2026_01_01" / "a.tif")

    cfg = CellPatcherConfig(mode="whole_fov")
    cfg.export.enabled = True
    cfg.export.image_format = "npy"
    cfg.common.output_dir = str(tmp_path / "out")

    results = _tree(image_root, config=cfg)
    stamped = {
        (r.condition, r.biological_repeat, r.technical_repeat): {p.technical_repeat for p in r.result.patches}
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
    assert cfg.common.technical_repeat == SINGLE_TECHNICAL_REPEAT_LABEL  # the base config is not mutated


def test_tree_combined_csv_keeps_technical_repeats_apart(tmp_path):
    image_root = tmp_path / "raw_data"
    for well in ("B02", "B03"):
        _write_image(image_root / "treated" / "2026_01_01" / well / "a.tif")
    _write_image(image_root / "control" / "c.tif")
    out = tmp_path / "out"

    results = _tree(
        image_root, config=_combined_tree_config(tmp_path), combine_metadata=True
    )
    with (out / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(len(r.result.patches) for r in results)
    assert {(row["condition"], row["biological_repeat"], row["technical_repeat"]) for row in rows} == {
        ("treated", "2026_01_01", "B02"),
        ("treated", "2026_01_01", "B03"),
        ("control", SINGLE_BIOLOGICAL_REPEAT_LABEL, SINGLE_TECHNICAL_REPEAT_LABEL),
    }
    # Both wells hold a.npy, so patch ids repeat; the folder keeps the files distinct.
    assert len({row["patch_file"] for row in rows}) == len(rows)
    for row in rows:
        assert (out / row["patch_file"]).exists()
        if row["technical_repeat"] != SINGLE_TECHNICAL_REPEAT_LABEL:
            assert row["patch_file"].startswith(
                f"{row['condition']}/{row['biological_repeat']}/{row['technical_repeat']}/"
            )


def test_tree_cell_mode_finds_masks_in_mirrored_technical_repeat_folders(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels(seed=5)
    image_root, mask_root = tmp_path / "raw_data", tmp_path / "masks"
    for well in ("B02", "B03"):
        write_array(image_root / "treated" / "2026_01_01" / well / "img.tif", image)
        write_array(mask_root / "treated" / "2026_01_01" / well / "img_mask.tif", labels)

    cfg = CellPatcherConfig(mode="cell")
    cfg.cell.patch_size = 16
    results = _tree(image_root, config=cfg, mask_root_path=mask_root)
    assert [r.technical_repeat for r in results] == ["B02", "B03"]
    for r in results:
        assert r.result.patches
        assert {p.mask_file for p in r.result.patches} == {"img_mask.tif"}
        assert {p.technical_repeat for p in r.result.patches} == {r.technical_repeat}


def test_tree_runs_wells_inside_a_biological_repeat_folder_named_single(tmp_path):
    image_root = tmp_path / "raw_data"
    _write_image(image_root / "ctrl" / "single" / "wellA" / "img1.tif")
    _write_image(image_root / "ctrl" / "single" / "wellB" / "img2.tif")

    results = _tree(image_root, config=CellPatcherConfig(mode="whole_fov"))
    stamped = {
        (r.condition, r.biological_repeat, r.technical_repeat): {p.image_file for p in r.result.patches}
        for r in results
    }
    assert stamped == {
        ("ctrl", "single", "wellA"): {"img1.tif"},
        ("ctrl", "single", "wellB"): {"img2.tif"},
    }
