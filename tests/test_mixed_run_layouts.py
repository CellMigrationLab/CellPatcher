"""Each image of a file run is read in its own roles, whatever else is in the run (audit A-M1).

Owner decision of 2026-11-03: an axis is in an image's plane choices and ``patch_id`` only if
*that image* has the role, the rule MorphoFeature's run planes follow (``run_config._planes``).
Before, ``run_layout`` gave every image of a run Z (or T) when any image had more than one plane
along it: a YX image patched beside a ZYX image got ``z000`` in its ids, other random rectangles
with ``seed_scope="plane"``, and ``t_policy="select"`` refused the run (audit probes p1 and p3).
"""

import importlib.util
import json

import numpy as np
import pytest

from cellpatcher import iter_exported_patches, iter_run_patch_arrays
from cellpatcher.config import CellPatcherConfig
from cellpatcher.orchestration import run_patching_pipeline
from tests.helpers import file_run_config, write_ngff, write_tiff

needs_zarr = pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None, reason="needs the 'zarr' extra"
)

_RNG = np.random.default_rng(0)
_FLAT = _RNG.integers(0, 1000, (32, 32)).astype(np.uint16)
_STACK = _RNG.integers(0, 1000, (3, 32, 32)).astype(np.uint16)


def _experiment(root, *, with_stack):
    """The probe's experiment: ``ctrl/rep1/a`` (YX), and with *with_stack* ``ctrl/rep1/b`` (TYX)."""
    write_tiff(root / "ctrl" / "rep1" / "a.tif", _FLAT, "YX")
    if with_stack:
        write_tiff(root / "ctrl" / "rep1" / "b.tif", _STACK, "TYX")
    return root


def _config(mode="grid", **common):
    config = CellPatcherConfig(mode=mode)
    config.grid.tile_size = 16
    config.random.patch_size, config.random.n_patches, config.random.seed = 8, 3, 7
    config.common.extract_arrays = True
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _run(tmp_path, name, config, *, with_stack):
    """Patch the probe's experiment alone (YX only) or mixed (YX beside TYX); return the result."""
    root = _experiment(tmp_path / name / "exp", with_stack=with_stack)
    axes = "TYX" if with_stack else "YX"
    file_run_config(config, root, axes, missing_as_singleton=True)
    return run_patching_pipeline(root, config=config)


def _of(result, position):
    """``(patch_id, rectangle, z0, t)`` of every patch of one image, with its arrays."""
    rows = [
        ((p.patch_id, (p.y0, p.y1, p.x0, p.x1), p.z0, p.t), array)
        for p, array in zip(result.patches, result.patch_arrays, strict=True)
        if p.position_id == position
    ]
    return [row for row, _ in rows], [array for _, array in rows]


@pytest.mark.parametrize("policy", ["keep", "all", "first"])
def test_a_2d_image_gets_the_same_patches_alone_and_beside_a_stack(tmp_path, policy):
    # Probe p1: in the mixed run the 2-D image's ids were ctrl/rep1/single/a/t0000/grid-... .
    alone = _run(tmp_path, "alone", _config(t_policy=policy), with_stack=False)
    mixed = _run(tmp_path, "mixed", _config(t_policy=policy), with_stack=True)
    (alone_rows, alone_arrays), (mixed_rows, mixed_arrays) = _of(alone, "a"), _of(mixed, "a")
    assert mixed_rows == alone_rows
    assert [row[0] for row in alone_rows] == [
        f"ctrl/rep1/single/a/grid-y{y}-x{x}-h16-w16" for y in (0, 16) for x in (0, 16)
    ]
    assert all(array.shape == (16, 16) for array in mixed_arrays)
    for mixed_array, alone_array in zip(mixed_arrays, alone_arrays, strict=True):
        np.testing.assert_array_equal(mixed_array, alone_array)
    stack_ids = sorted(row[0] for row in _of(mixed, "b")[0])
    if policy == "keep":  # a kept axis never enters patch_id
        assert stack_ids[0] == "ctrl/rep1/single/b/grid-y0-x0-h16-w16"
    else:
        assert stack_ids[0] == "ctrl/rep1/single/b/t0000/grid-y0-x0-h16-w16"


def test_random_rectangles_of_a_plane_seed_do_not_depend_on_the_other_images(tmp_path):
    # seed_scope="plane" seeds each plane from its identity (t, z): t=0 given to the 2-D image in
    # a mixed run drew other rectangles than the image alone.
    configs = {}
    for name in ("alone", "mixed"):
        configs[name] = _config("random", t_policy="all")
        configs[name].random.seed_scope = "plane"
    alone = _run(tmp_path, "alone", configs["alone"], with_stack=False)
    mixed = _run(tmp_path, "mixed", configs["mixed"], with_stack=True)
    assert _of(mixed, "a")[0] == _of(alone, "a")[0]
    assert len(_of(alone, "a")[0]) == 3
    assert all(row[3] is None and "/t0" not in row[0] for row in _of(mixed, "a")[0])


def test_t_select_applies_only_to_the_images_with_t(tmp_path):
    # Probe p3: t_policy="select" with t_index=2 refused the mixed run ("t_index=2 is outside the
    # images, which have 1 planes along T"), as the 2-D image was read with T of length 1.
    result = _run(tmp_path, "mixed", _config(t_policy="select", t_index=2), with_stack=True)
    assert sorted({p.patch_id.rsplit("/", 1)[0] for p in result.patches}) == [
        "ctrl/rep1/single/a",
        "ctrl/rep1/single/b/t0002",
    ]
    (array,) = [a for p, a in zip(result.patches, result.patch_arrays, strict=True) if p.patch_id.endswith(
        "b/t0002/grid-y0-x0-h16-w16"
    )]
    np.testing.assert_array_equal(array, _STACK[2, :16, :16])


def test_a_t_index_outside_the_stack_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="t_index=3 is outside the images, which have 3 planes along T"):
        _run(tmp_path, "mixed", _config(t_policy="select", t_index=3), with_stack=True)


def test_raw_export_and_lazy_delivery_of_a_mixed_run_give_the_eager_arrays(tmp_path):
    # run_config.json records each image's layout when they differ, so a patch exported with raw
    # values is normalized on load in its own image's axes.
    config = _config(t_policy="all")
    config.normalization.enabled, config.normalization.scope = True, "image"
    config.normalization.export_values = "raw"
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    result = _run(tmp_path, "mixed", config, with_stack=True)
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert saved["preprocessing"]["layout"] == {
        "image": "TYX",
        "mask": "YX",
        "images": {
            "ctrl/rep1/a.tif": {"image": "YX", "mask": "YX"},
            "ctrl/rep1/b.tif": {"image": "TYX", "mask": "YX"},
        },
    }
    eager = dict(zip((p.patch_id for p in result.patches), result.patch_arrays, strict=True))
    loaded = {row["patch_id"]: array for row, array in iter_exported_patches(tmp_path / "out" / "patches.csv")}
    lazy = {patch.patch_id: array for patch, array in iter_run_patch_arrays(result.patches, config)}
    assert loaded.keys() == lazy.keys() == eager.keys()
    for patch_id, array in eager.items():
        np.testing.assert_allclose(loaded[patch_id], array, rtol=1e-6)
        np.testing.assert_array_equal(lazy[patch_id], array)


def test_a_run_of_one_layout_records_no_per_image_layouts(tmp_path):
    config = _config(t_policy="all")
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    _run(tmp_path, "alone", config, with_stack=False)
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert saved["preprocessing"]["layout"] == {"image": "YX", "mask": "YX"}


@needs_zarr
def test_ome_zarr_length_one_t_and_z_roles_name_their_plane_as_morphofeature_does(tmp_path):
    # The rule is per role: an OME-Zarr image records T and Z of length 1, so a policy that
    # chooses planes names t0000/z000 in its ids (MorphoFeature's planes are (t=0, z=0) too). A
    # kept axis of length 1 stays out of the arrays and ids, as before.
    root = tmp_path / "exp"
    data = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(1, 2, 1, 16, 16)
    write_ngff(root / "a.ome.zarr", data, "tczyx")
    run = file_run_config(_config(), root, "TCZYX")
    fov = _fov(run.common.preprocessing_run)
    assert (tuple(fov.roles), fov.t_count, fov.z_count) == (("T", "C", "Z", "Y", "X"), 1, 1)
    for policies, prefix, shape in (
        ({}, "single/single/single/a", (2, 16, 16)),
        ({"z_policy": "all", "t_policy": "all"}, "single/single/single/a/t0000/z000", (2, 16, 16)),
        ({"z_policy": "first"}, "single/single/single/a/z000", (2, 16, 16)),
    ):
        config = _config(**policies)
        config.common.preprocessing_run = run.common.preprocessing_run
        result = run_patching_pipeline(root, config=config)
        assert {p.patch_id.rsplit("/", 1)[0] for p in result.patches} == {prefix}
        assert {array.shape for array in result.patch_arrays} == {shape[:1] + (16, 16)}


def _fov(output):
    from morphobricks_preprocessing.run_inputs import RunInputs

    (fov,) = RunInputs.open(output).fovs()
    return fov


def _morphofeature_planes(fov):
    """MorphoFeature's ``run_config._planes`` (MorphoFeature 4f25a79): None for an axis the image lacks."""
    times = range(fov.t_count) if "T" in fov.roles else [None]
    depths = range(fov.z_count) if "Z" in fov.roles else [None]
    return [(t, z) for t in times for z in depths]


def _morphofeature_cut(fov, t, z, mode):
    """The ids MorphoFeature's ``patch_sets.cut_patch_set`` (4f25a79) gets for one plane.

    It calls ``run_patching`` on a YX plane with the FOV key's labels in ``CommonConfig`` and
    ``position_id``, ``t`` and ``z`` (those not None) in ``image_metadata``. Mirrored here so the
    check runs in every job (MorphoFeature is private, so CI cannot install it); the same check
    against the real ``cut_patch_set`` gave equal ids in the PR #77 review.
    """
    from cellpatcher import CommonConfig, GridConfig, RandomConfig, run_patching

    condition, biological_repeat, technical_repeat, position_id = fov.key
    config = CellPatcherConfig(mode=mode)
    config.common = CommonConfig(
        image_axes="YX", mask_axes="YX", condition=condition,
        biological_repeat=biological_repeat, technical_repeat=technical_repeat,
    )
    config.grid = GridConfig(tile_size=16)
    config.random = RandomConfig(patch_size=8, n_patches=3, seed=7, seed_scope="plane")
    metadata = {"position_id": position_id, "t": t, "z": z}
    result = run_patching(
        images=[np.broadcast_to(np.uint8(0), (32, 32))],
        config=config,
        image_metadata=[{key: value for key, value in metadata.items() if value is not None}],
    )
    return [patch.patch_id for patch in result.patches]


@pytest.mark.parametrize("mode", ["grid", "random"])
def test_ids_equal_morphofeature_cut_patch_set_ids(tmp_path, mode):
    # MorphoFeature cuts each plane of its run (t and z only for an axis the image has) with
    # cut_patch_set; a CellPatcher file run of the same images gives the same ids.
    from morphobricks_preprocessing.run_inputs import RunInputs

    config = _config(mode, t_policy="all")
    config.random.seed_scope = "plane"
    result = _run(tmp_path, "mixed", config, with_stack=True)
    expected = [
        patch_id
        for fov in RunInputs.open(config.common.preprocessing_run).fovs()
        for t, z in _morphofeature_planes(fov)
        for patch_id in _morphofeature_cut(fov, t, z, mode)
    ]
    assert len(expected) == (16 if mode == "grid" else 12)
    assert sorted(p.patch_id for p in result.patches) == sorted(expected)


def test_lazy_delivery_masks_each_patch_in_its_own_images_mask_layout(tmp_path):
    # A 2-D mask beside per-plane masks of a stack: lazy delivery selects each patch's mask plane
    # in its own image's mask layout (YX for the 2-D image), not the run's union (TYX).
    root = _experiment(tmp_path / "exp", with_stack=True)
    flat_labels = np.zeros((32, 32), np.uint16)
    flat_labels[2:10, 3:12], flat_labels[15:25, 14:28] = 1, 2
    stack_labels = np.stack([np.roll(flat_labels, t, axis=0) for t in range(3)])
    write_tiff(tmp_path / "masks" / "cells" / "ctrl" / "rep1" / "a.tif", flat_labels, "YX")
    write_tiff(tmp_path / "masks" / "cells" / "ctrl" / "rep1" / "b.tif", stack_labels, "TYX")
    config = _config("cell", t_policy="all")
    config.cell.patch_size, config.cell.exclude_border = 12, False
    config.extraction.apply_mask = True
    file_run_config(config, root, "TYX", missing_as_singleton=True, masks=tmp_path / "masks")
    result = run_patching_pipeline(root, config=config, mask_folder_path=tmp_path / "masks")
    assert {p.position_id for p in result.patches} == {"a", "b"}
    lazy = list(iter_run_patch_arrays(result.patches, config))
    assert [patch.patch_id for patch, _ in lazy] == [p.patch_id for p in result.patches]
    for (_, array), eager in zip(lazy, result.patch_arrays, strict=True):
        np.testing.assert_array_equal(array, eager)


def test_a_patch_of_no_recorded_image_is_refused_not_given_the_union_layout(tmp_path):
    # With a layout per image recorded, a row whose source_relative_path names none of them has
    # no known axes: the run's union (TYX) would be a guess for a 2-D image's patch.
    from cellpatcher.run_config_file import delivery_config

    config = _config(t_policy="all")
    config.normalization.enabled, config.normalization.scope = True, "image"
    config.normalization.export_values = "raw"
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    _run(tmp_path, "mixed", config, with_stack=True)
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert delivery_config(saved, "run_config.json", "ctrl/rep1/a.tif").common.image_axes == "YX"
    for source in (None, "ctrl/rep1/c.tif"):
        with pytest.raises(ValueError, match="records each image's layout"):
            delivery_config(saved, "run_config.json", source)
