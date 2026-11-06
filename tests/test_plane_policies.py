"""``common.z_policy``/``t_policy``: Z and T axes turned into 2D planes before patching."""

import csv
import json

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, GridConfig
from cellpatcher.errors import ZStackNotSupportedError
from cellpatcher.orchestration import (
    run_patching_condition_repeat_tree,
    run_patching_pipeline,
)
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from tests.helpers import (
    file_run_config,
    named,
    preprocess,
    preprocessed_folder,
    write_tiff,
)


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
    assert result.patches[0].patch_id == "single/single/single/P/z000/grid-y0-x0-h8-w8"
    assert result.patches[-1].patch_id == "single/single/single/P/z002/grid-y8-x8-h8-w8"
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
        f"single/single/single/P/t{t:04d}/z{z:03d}/grid-y0-x0-h8-w8" for t in range(2) for z in range(3)
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


@pytest.mark.parametrize("policy", ["max", "mean", "sum"])
def test_a_projection_of_one_plane_is_accepted_and_changes_nothing(policy):
    """Owner decision on #86: a projection that combines nothing (no Z axis, or Z=1) is the plane itself."""
    one = _stack((1, 16, 16))
    kept = run_patching(images=one, config=_grid_config(), image_metadata=named(one))
    result = run_patching(images=one, config=_grid_config(z_policy=policy), image_metadata=named(one))
    assert {p.z0 for p in result.patches} == {None}
    assert [p.patch_id for p in result.patches] == [p.patch_id for p in kept.patches]
    assert result.patch_arrays[0].dtype == one.dtype
    np.testing.assert_array_equal(result.patch_arrays[0], one[0, :8, :8])
    flat = _stack((16, 16))
    flat_result = run_patching(images=flat, config=_grid_config("YX", z_policy=policy), image_metadata=named(flat))
    np.testing.assert_array_equal(flat_result.patch_arrays[0], flat[:8, :8])
    assert {p.z0 for p in flat_result.patches} == {None}


@pytest.mark.parametrize("policy", ["max", "mean", "sum"])
def test_a_projection_of_depth_planes_is_refused_by_name(policy):
    """Issue #83: a projection that would combine Z planes belongs to preprocessing."""
    stack = _stack()
    message = "would combine 3 Z planes.*recorded projection step in morphobricks-preprocessing"
    with pytest.raises(ZStackNotSupportedError, match=message) as caught:
        run_patching(images=stack, config=_grid_config(z_policy=policy), image_metadata=named(stack))
    assert (caught.value.axis, caught.value.length) == ("Z", 3)
    with pytest.raises(ZStackNotSupportedError, match="would combine 3 Z planes"):
        run_patching(images=_stack(), masks=_labels_per_plane(), config=_cell_config(z_policy=policy),
            image_metadata=named(_stack()))  # formerly "projects the images' Z planes" for masks following the planes


def test_a_z_projection_refusal_is_a_value_error_and_t_has_no_projection():
    assert issubclass(ZStackNotSupportedError, ValueError)
    with pytest.raises(ValueError, match="t_policy must be one of"):
        CommonConfig(t_policy="max")  # unchanged: T never had projections


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
    assert {p.position_id for p in result.patches} == {"s"}
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
    assert [p.patch_id for p in result.patches] == ["single/single/single/P/mask/z000/label1", "single/single/single/P/mask/z001/label2",
        "single/single/single/P/mask/z002/label3"]


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
    # The header names the axes: a three-plane stack without them could be read as RGB.
    for name in ("a.tif", "b.tif"):
        write_tiff(tmp_path / "images" / name, _stack(), "TYX")
    config = _grid_config("YX", t_policy="all")  # a file run reads the roles the run recorded (CP-2a)
    config.export.enabled = True
    config.common.output_dir = str(tmp_path / "out")

    result = preprocessed_folder(tmp_path / "images", config=config, axes="TYX")

    assert len(result.exported_files) == 2 * 3 * 4
    with open(tmp_path / "out" / "patches.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len({row["patch_file"] for row in rows}) == len(rows)
    assert {(row["image_file"], row["t"]) for row in rows} == {
        (name, str(t)) for name in ("a.tif", "b.tif") for t in range(3)
    }
    assert rows[0]["patch_id"] == "single/single/single/a/t0000/grid-y0-x0-h8-w8"
    assert rows[0]["patch_file"] == "single_single_single_a_t0000_grid-y0-x0-h8-w8.tif"
    saved = json.loads((tmp_path / "out" / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["common"]["t_policy"] == "all"


def test_a_mask_whose_header_disagrees_with_its_image_has_no_clean_run(tmp_path):
    # A ZYX mask (its ImageJ header names Z) for a TYX image would have its depth slices read
    # as time points. Since CP-2a masks are read in the roles the preprocessing run recorded,
    # and the run refuses this pair, so no file run reads it. (Preprocessing v0.4.0 refuses the
    # Z-stack mask itself.)
    image_folder, mask_folder = tmp_path / "exp", tmp_path / "masks"
    write_tiff(image_folder / "a.tif", np.random.default_rng(0).integers(0, 100, (3, 32, 32), dtype=np.uint16), "TYX")
    mask = np.zeros((3, 32, 32), dtype=np.uint16)
    mask[:, 5:15, 5:15] = 1
    write_tiff(mask_folder / "a_cells.tif", mask, "ZYX")
    refusal = r"does not have the time points \(T\) and depth slices \(Z\) of its image"
    with pytest.raises(AssertionError, match=refusal):
        preprocess(image_folder, "TYX", masks=mask_folder)


def test_mask_axes_that_agree_with_the_mask_header_are_accepted(tmp_path):
    image_folder, mask_folder = tmp_path / "exp", tmp_path / "masks"
    write_tiff(image_folder / "a.tif", np.random.default_rng(0).integers(0, 100, (3, 32, 32), dtype=np.uint16), "TYX")
    mask = np.zeros((3, 32, 32), dtype=np.uint16)
    mask[:, 5:15, 5:15] = 1
    write_tiff(mask_folder / "a_cells.tif", mask, "TYX")
    config = _cell_config(image_axes="YX", mask_axes="YX", t_policy="all")

    result = preprocessed_folder(image_folder, config=config, mask_folder_path=mask_folder, axes="TYX")
    assert [p.t for p in result.patches] == [0, 1, 2]


def test_a_file_run_refuses_declared_image_axes(tmp_path):
    # A TYX image (its ImageJ header names T) declared ZYX would have its time points read as
    # depth slices; declared CYX, as channels. Since CP-2a a file run declares nothing: it reads
    # the roles the preprocessing run recorded and refuses the setting (migration message).
    write_tiff(tmp_path / "exp" / "a.tif", _stack(), "TYX")
    for declared, policy in (("ZYX", {"z_policy": "all"}), ("CYX", {})):
        config = file_run_config(_grid_config(image_axes=declared, **policy), tmp_path / "exp", "TYX")
        with pytest.raises(ValueError, match=rf"common.image_axes='{declared}': each image's \(and mask's\) axes"):
            run_patching_pipeline(str(tmp_path / "exp"), config=config)


def test_a_run_mixing_stacks_and_single_planes_patches_both(tmp_path):
    # Each image is read in its own roles (audit A-M1): the 2-D image has no T, so its patches
    # have no plane index, as when it is patched alone. Before, one layout per run (CP-2a) gave it
    # t=0 because the other image has three planes.
    root = tmp_path / "raw"
    write_tiff(root / "a_cond" / "r1" / "a.tif", _stack(), "TYX")
    write_tiff(root / "b_cond" / "r1" / "b.tif", _stack((1, 16, 16)), "TYX")
    config = file_run_config(_grid_config("YX", t_policy="all"), root, "TYX", missing_as_singleton=True)
    leaves = run_patching_condition_repeat_tree(str(root), config=config)
    assert [sorted({p.t for p in leaf.result.patches}, key=str) for leaf in leaves] == [[0, 1, 2], [None]]


def test_a_plain_stack_without_axis_metadata_keeps_its_declared_axes(tmp_path):
    # tifffile calls a plain stack's axes IYX/QYX: no axis besides Y and X is named, so the
    # declared axes are used as before.
    import tifffile

    (tmp_path / "exp").mkdir()
    tifffile.imwrite(tmp_path / "exp" / "a.tif", _stack().astype(np.uint16), photometric="minisblack", metadata=None)
    assert tifffile.TiffFile(tmp_path / "exp" / "a.tif").series[0].axes == "IYX"
    config = file_run_config(_grid_config("YX", t_policy="all"), tmp_path / "exp", "TYX", stored="IYX")
    result = run_patching_pipeline(str(tmp_path / "exp"), config=config)
    assert sorted({p.t for p in result.patches}) == [0, 1, 2]


def test_lazy_delivery_of_a_file_reads_its_planes_through_the_run(tmp_path):
    # CP-2b: a path given to patch_array was read by CellPatcher (common.image_axes, its own header
    # check). It is refused; the run's patches are read plane by plane through the run.
    from cellpatcher import iter_run_patch_arrays
    from cellpatcher.delivery import patch_array

    write_tiff(tmp_path / "exp" / "a.tif", _stack(), "TYX")
    config = _grid_config("YX", t_policy="all")
    result = preprocessed_folder(tmp_path / "exp", config=config, axes="TYX")
    with pytest.raises(TypeError, match="iter_run_patch_arrays"):
        patch_array(result.patches[0], tmp_path / "exp" / "a.tif", _grid_config(image_axes="TYX", t_policy="all"))
    lazy = [array for _, array in iter_run_patch_arrays(result.patches, config)]
    assert len(lazy) == len(result.patch_arrays) > 3
    assert all(np.array_equal(lazy_array, eager) for lazy_array, eager in zip(lazy, result.patch_arrays))


def _tcyx_run_with_tyx_masks(tmp_path):
    image_folder, mask_folder = tmp_path / "exp", tmp_path / "masks"
    image = np.random.default_rng(0).integers(0, 100, (3, 2, 32, 32), dtype=np.uint16)
    write_tiff(image_folder / "a.tif", image, "TCYX")
    mask = np.zeros((3, 32, 32), dtype=np.uint16)
    mask[:, 5:15, 5:15] = 1
    write_tiff(mask_folder / "a_cells.tif", mask, "TYX")
    return image_folder, mask_folder


def test_a_file_run_with_t_masks_and_t_policy_keep_names_t_policy_not_mask_axes(tmp_path):
    # Cross-brick test F7 (2026-11-03): the message named mask_channel and common.mask_axes,
    # which a file run refuses; the real options are z_policy/t_policy or 2-D masks. (Converted to
    # T: preprocessing v0.4.0 refuses Z>1, so the Z form of this message cannot be reached.)
    image_folder, mask_folder = _tcyx_run_with_tyx_masks(tmp_path)
    config = _cell_config(image_axes="YX", mask_axes="YX")  # t_policy is the default "keep"
    with pytest.raises(ValueError, match=r"axes 'TYX'.* have T planes.*Set common.t_policy") as caught:
        preprocessed_folder(image_folder, config=config, mask_folder_path=mask_folder, axes="TCYX")
    assert "mask_axes" not in str(caught.value)
    assert "preprocessing run's masks" in str(caught.value)


@pytest.mark.parametrize("policy", ["all", "first"])
def test_the_t_policy_the_message_names_patches_that_run(tmp_path, policy):
    image_folder, mask_folder = _tcyx_run_with_tyx_masks(tmp_path)
    config = _cell_config(image_axes="YX", mask_axes="YX", t_policy=policy)
    result = preprocessed_folder(image_folder, config=config, mask_folder_path=mask_folder, axes="TCYX")
    assert len(result.patches) == (3 if policy == "all" else 1)


def test_in_memory_t_and_z_masks_with_keep_name_both_policies():
    masks = [np.ones((2, 3, 16, 16), dtype=np.int32)]
    config = _cell_config(image_axes="TZYX", mask_axes="TZYX")
    with pytest.raises(ValueError, match=r"have T and Z planes.*common.t_policy and common.z_policy"):
        run_patching(images=[np.zeros((2, 3, 16, 16), np.float32)], masks=masks, config=config,
                     image_metadata=named([None]))
