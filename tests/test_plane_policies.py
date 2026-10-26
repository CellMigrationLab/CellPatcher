"""``common.z_policy``/``t_policy``: Z and T axes turned into 2D planes before patching."""

import csv
import json

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, GridConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from tests.helpers import named


def _stack(shape=(3, 16, 16)):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def _grid_config(image_axes="ZYX", **common):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = image_axes
    config.common.extract_arrays = True
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _cell_config(image_axes="ZYX", mask_axes="ZYX", **common):
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 6
    config.cell.exclude_border = False
    config.common.image_axes = image_axes
    config.common.mask_axes = mask_axes
    config.common.extract_arrays = True
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def test_keep_is_the_default_and_keeps_every_plane_in_each_patch():
    assert (CommonConfig().z_policy, CommonConfig().t_policy) == ("keep", "keep")
    result = run_patching(images=_stack(), config=_grid_config(), image_metadata=[{"image_file": "s.tif"}])
    assert [p.patch_id for p in result.patches][:2] == [
        "single/single/single/s/grid-y0-x0-h8-w8", "single/single/single/s/grid-y0-x8-h8-w8"
    ]
    assert {p.z0 for p in result.patches} == {None}
    assert result.patch_arrays[0].shape == (3, 8, 8)


def test_all_gives_one_patch_per_plane_with_its_z():
    stack = _stack()
    result = run_patching(images=stack, config=_grid_config(z_policy="all"), image_metadata=[{"position_id": "P"}])

    assert len(result.patches) == 3 * 4
    assert result.patches[0].patch_id == "P/z000/grid-y0-x0-h8-w8"
    assert result.patches[-1].patch_id == "P/z002/grid-y8-x8-h8-w8"
    assert {p.image_id for p in result.patches} == {"image_0000"}
    for patch, array in zip(result.patches, result.patch_arrays):
        assert (patch.z1 - patch.z0) == 1
        np.testing.assert_array_equal(array, stack[patch.z0, patch.y0:patch.y1, patch.x0:patch.x1])


def test_t_and_z_split_together_in_t_then_z_order():
    stack = _stack((2, 3, 8, 8))
    result = run_patching(
        images=stack, config=_grid_config("TZYX", z_policy="all", t_policy="all"), image_metadata=[{"position_id": "P"}]
    )

    assert [p.patch_id for p in result.patches] == [
        f"P/t{t:04d}/z{z:03d}/grid-y0-x0-h8-w8" for t in range(2) for z in range(3)
    ]
    assert [(p.t, p.z0) for p in result.patches] == [(t, z) for t in range(2) for z in range(3)]
    np.testing.assert_array_equal(result.patch_arrays[4], stack[1, 1])


def test_first_and_select_choose_one_plane_without_a_suffix():
    stack = _stack()
    first = run_patching(images=stack, config=_grid_config(z_policy="first"), image_metadata=named(stack))
    assert {(p.z0, p.patch_id.endswith("_z000")) for p in first.patches} == {(0, False)}

    selected = run_patching(images=stack, config=_grid_config(z_policy="select", z_index=2),
        image_metadata=named(stack))
    assert {p.z0 for p in selected.patches} == {2}
    np.testing.assert_array_equal(selected.patch_arrays[0], stack[2, :8, :8])


def test_projection_has_no_plane_index():
    stack = _stack()
    result = run_patching(images=stack, config=_grid_config(z_policy="max"), image_metadata=named(stack))
    assert {p.z0 for p in result.patches} == {None}
    np.testing.assert_array_equal(result.patch_arrays[0], stack.max(axis=0)[:8, :8])


def test_images_without_the_axis_are_unchanged():
    image = np.ones((16, 16), dtype=np.float32)
    keep = run_patching(images=image, config=_grid_config("YX"), image_metadata=named(image))
    split = run_patching(images=image, config=_grid_config("YX", z_policy="all", t_policy="all"),
        image_metadata=named(image))
    assert [p.patch_id for p in split.patches] == [p.patch_id for p in keep.patches]
    assert {p.z0 for p in split.patches} == {None}


def test_invalid_policies_and_indices_are_rejected():
    with pytest.raises(ValueError, match="z_policy must be one of"):
        CommonConfig(z_policy="median")
    with pytest.raises(ValueError, match="t_policy must be one of"):
        CommonConfig(t_policy="max")
    with pytest.raises(ValueError, match="needs z_index"):
        CommonConfig(z_policy="select")
    with pytest.raises(ValueError, match="outside the images"):
        run_patching(images=_stack(), config=_grid_config(z_policy="select", z_index=3), image_metadata=named(_stack()))
    # Set after construction: still checked when the run starts.
    config = _grid_config()
    config.common.z_policy = "median"
    with pytest.raises(ValueError, match="z_policy must be one of"):
        run_patching(images=_stack(), config=config, image_metadata=named(_stack()))


def test_image_metadata_z_conflicts_with_a_z_policy():
    with pytest.raises(ValueError, match="use one or the other"):
        run_patching(
            images=_stack(), config=_grid_config(z_policy="all"), image_metadata=[{"z": 1, "position_id": "P"}]
        )


def test_provenance_from_image_metadata_reaches_every_plane():
    metadata = [{"image_file": "a.tif", "source_relative_path": "control/a.tif"}]
    result = run_patching(images=_stack(), config=_grid_config(z_policy="all"), image_metadata=metadata)
    assert {(p.image_file, p.source_relative_path) for p in result.patches} == {("a.tif", "control/a.tif")}


def test_split_channels_follow_the_plane_suffix_and_share_one_position():
    stack = _stack((2, 3, 8, 8))  # C, Z, Y, X
    config = _grid_config("CZYX", z_policy="all")
    config.extraction.split_channels = True
    result = run_patching(images=stack, config=config, image_metadata=[{"image_file": "s.tif"}])

    assert [p.patch_id for p in result.patches] == [
        f"single/single/single/s/z{z:03d}/c{c}/grid-y0-x0-h8-w8" for z in range(3) for c in range(2)
    ]
    assert {p.position_id for p in result.patches} == {"single/single/single/s"}
    np.testing.assert_array_equal(result.patch_arrays[3], stack[1, 1])


def test_random_mode_is_deterministic_per_plane():
    config = CellPatcherConfig(mode="random")
    config.random.patch_size, config.random.n_patches = 4, 3
    config.common.image_axes, config.common.z_policy = "ZYX", "all"
    first = run_patching(images=_stack(), config=config, image_metadata=named(_stack()))
    second = run_patching(images=_stack(), config=config, image_metadata=named(_stack()))
    assert [(p.patch_id, p.y0, p.x0) for p in first.patches] == [(p.patch_id, p.y0, p.x0) for p in second.patches]
    assert len({p.patch_id for p in first.patches}) == 9


# ---------------------------------------------------------------------------
# Masks follow the image's planes or are shared
# ---------------------------------------------------------------------------


def _labels_per_plane(n_planes=3):
    """One square object per plane, at a different place in each plane."""
    labels = np.zeros((n_planes, 16, 16), dtype=np.int32)
    for z in range(n_planes):
        labels[z, 2 + 4 * z:5 + 4 * z, 2 + 4 * z:5 + 4 * z] = z + 1
    return labels


def test_cell_masks_with_the_image_z_length_follow_its_planes():
    result = run_patching(
        images=_stack(), masks=_labels_per_plane(), config=_cell_config(z_policy="all"),
        image_metadata=[{"position_id": "P"}],
    )
    assert [(p.z0, p.label_id) for p in result.patches] == [(0, 1), (1, 2), (2, 3)]
    assert [p.patch_id for p in result.patches] == ["P/mask/z000/label1", "P/mask/z001/label2", "P/mask/z002/label3"]


def test_a_2d_or_length_one_mask_is_shared_by_every_plane():
    mask = _labels_per_plane()[0]
    shared = run_patching(images=_stack(), masks=mask, config=_cell_config(mask_axes="YX", z_policy="all"),
        image_metadata=named(_stack()))
    assert [(p.z0, p.label_id) for p in shared.patches] == [(0, 1), (1, 1), (2, 1)]

    single = run_patching(images=_stack(), masks=mask[None], config=_cell_config(z_policy="all"),
        image_metadata=named(_stack()))
    assert [(p.z0, p.label_id) for p in single.patches] == [(0, 1), (1, 1), (2, 1)]


def test_mask_z_length_must_match_or_be_one():
    with pytest.raises(ValueError, match="must have the image's Z length"):
        run_patching(images=_stack(), masks=_labels_per_plane(2), config=_cell_config(z_policy="all"),
            image_metadata=named(_stack()))


def test_a_projection_cannot_use_masks_that_follow_the_planes():
    with pytest.raises(ValueError, match="projects the images' Z planes"):
        run_patching(images=_stack(), masks=_labels_per_plane(), config=_cell_config(z_policy="max"),
            image_metadata=named(_stack()))


def test_apply_mask_uses_the_matching_label_plane():
    config = _cell_config(z_policy="all")
    config.extraction.apply_mask = True
    result = run_patching(images=_stack() + 1, masks=_labels_per_plane(), config=config,
        image_metadata=named(_stack() + 1))
    for patch, array in zip(result.patches, result.patch_arrays):
        # Only the patch's own object survives; it is in its own plane.
        assert np.count_nonzero(array) == 9


# ---------------------------------------------------------------------------
# Folder runs and export
# ---------------------------------------------------------------------------


def test_folder_run_exports_one_file_and_row_per_plane(tmp_path):
    # .npy: tifffile would store a three-plane float stack as RGB.
    for name in ("a.npy", "b.npy"):
        write_array(tmp_path / "images" / name, _stack())
    config = _grid_config(z_policy="all")
    config.export.enabled = True
    config.common.output_dir = str(tmp_path / "out")

    result = run_patching_pipeline(tmp_path / "images", config=config)

    assert len(result.exported_files) == 2 * 3 * 4
    with open(tmp_path / "out" / "patches.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len({row["patch_file"] for row in rows}) == len(rows)
    assert {(row["image_file"], row["z0"]) for row in rows} == {
        (name, str(z)) for name in ("a.npy", "b.npy") for z in range(3)
    }
    assert rows[0]["patch_id"] == "single/single/single/a/z000/grid-y0-x0-h8-w8"
    assert rows[0]["patch_file"] == "single_single_single_a_z000_grid-y0-x0-h8-w8.tif"
    saved = json.loads((tmp_path / "out" / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["common"]["z_policy"] == "all"
