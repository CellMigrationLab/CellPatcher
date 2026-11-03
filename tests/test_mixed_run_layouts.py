"""Each image of a file run is read in its own roles, whatever else is in the run (audit A-M1).

Owner decision of 2026-11-03: an axis is in an image's plane choices and ``patch_id`` only if
*that image* has the role, the rule MorphoFeature's run planes follow (``run_config._planes``).
Before, ``run_layout`` gave every image of a run Z (or T) when any image had more than one plane
along it: a YX image patched beside a ZYX image got ``z000`` in its ids, other random rectangles
with ``seed_scope="plane"``, and ``z_policy="select"`` refused the run (audit probes p1 and p3).
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
    """The probe's experiment: ``ctrl/rep1/a`` (YX), and with *with_stack* ``ctrl/rep1/b`` (ZYX)."""
    write_tiff(root / "ctrl" / "rep1" / "a.tif", _FLAT, "YX")
    if with_stack:
        write_tiff(root / "ctrl" / "rep1" / "b.tif", _STACK, "ZYX")
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
    """Patch the probe's experiment alone (YX only) or mixed (YX beside ZYX); return the result."""
    root = _experiment(tmp_path / name / "exp", with_stack=with_stack)
    axes = "ZYX" if with_stack else "YX"
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
    # Probe p1: in the mixed run the 2-D image's ids were ctrl/rep1/single/a/z000/grid-... .
    alone = _run(tmp_path, "alone", _config(z_policy=policy), with_stack=False)
    mixed = _run(tmp_path, "mixed", _config(z_policy=policy), with_stack=True)
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
        assert stack_ids[0] == "ctrl/rep1/single/b/z000/grid-y0-x0-h16-w16"


def test_random_rectangles_of_a_plane_seed_do_not_depend_on_the_other_images(tmp_path):
    # seed_scope="plane" seeds each plane from its identity (t, z): z=0 given to the 2-D image in
    # a mixed run drew other rectangles than the image alone.
    configs = {}
    for name in ("alone", "mixed"):
        configs[name] = _config("random", z_policy="all")
        configs[name].random.seed_scope = "plane"
    alone = _run(tmp_path, "alone", configs["alone"], with_stack=False)
    mixed = _run(tmp_path, "mixed", configs["mixed"], with_stack=True)
    assert _of(mixed, "a")[0] == _of(alone, "a")[0]
    assert len(_of(alone, "a")[0]) == 3
    assert all(row[2] is None and "/z" not in row[0] for row in _of(mixed, "a")[0])


def test_z_select_applies_only_to_the_images_with_z(tmp_path):
    # Probe p3: z_policy="select" with z_index=2 refused the mixed run ("z_index=2 is outside the
    # images, which have 1 planes along Z"), as the 2-D image was read with Z of length 1.
    result = _run(tmp_path, "mixed", _config(z_policy="select", z_index=2), with_stack=True)
    assert sorted({p.patch_id.rsplit("/", 1)[0] for p in result.patches}) == [
        "ctrl/rep1/single/a",
        "ctrl/rep1/single/b/z002",
    ]
    (array,) = [a for p, a in zip(result.patches, result.patch_arrays, strict=True) if p.patch_id.endswith(
        "b/z002/grid-y0-x0-h16-w16"
    )]
    np.testing.assert_array_equal(array, _STACK[2, :16, :16])


def test_a_z_index_outside_the_stack_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="z_index=3 is outside the images, which have 3 planes along Z"):
        _run(tmp_path, "mixed", _config(z_policy="select", z_index=3), with_stack=True)


def test_raw_export_and_lazy_delivery_of_a_mixed_run_give_the_eager_arrays(tmp_path):
    # run_config.json records each image's layout when they differ, so a patch exported with raw
    # values is normalized on load in its own image's axes.
    config = _config(z_policy="all")
    config.normalization.enabled, config.normalization.scope = True, "image"
    config.normalization.export_values = "raw"
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    result = _run(tmp_path, "mixed", config, with_stack=True)
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert saved["preprocessing"]["layout"] == {
        "image": "ZYX",
        "mask": "YX",
        "images": {
            "ctrl/rep1/a.tif": {"image": "YX", "mask": "YX"},
            "ctrl/rep1/b.tif": {"image": "ZYX", "mask": "YX"},
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
    config = _config(z_policy="all")
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


@pytest.mark.skipif(importlib.util.find_spec("morphofeature") is None, reason="needs MorphoFeature (parity check)")
@pytest.mark.parametrize("mode", ["grid", "random"])
def test_ids_equal_morphofeature_cut_patch_set_ids(tmp_path, mode):
    # MorphoFeature cuts each plane of its run (t and z only for an axis the image has) with
    # cut_patch_set; a CellPatcher file run of the same images gives the same ids.
    from morphobricks_preprocessing.run_inputs import RunInputs
    from morphofeature.channels import ChannelCatalogue
    from morphofeature.feature_requests import FovInput
    from morphofeature.identity import FovKey
    from morphofeature.patch_sets import PatchSet, cut_patch_set
    from morphofeature.run_config import _planes

    config = _config(mode, z_policy="all")
    config.random.seed_scope = "plane"
    result = _run(tmp_path, "mixed", config, with_stack=True)
    patch_set = PatchSet(name="set", mode=mode, side=16 if mode == "grid" else 8)
    if mode == "random":
        patch_set = PatchSet(name="set", mode=mode, side=8, n=3, seed=7, seed_scope="plane")
    expected = []
    for fov in RunInputs.open(config.common.preprocessing_run).fovs():
        for t, z in _planes(fov):
            plane = FovInput(
                FovKey(*fov.key), np.zeros((1, 32, 32), np.uint16), ChannelCatalogue({"c0": 0}), t=t, z=z
            )
            expected += [descriptor.patch_id for descriptor in cut_patch_set(patch_set, plane)["set"]]
    assert sorted(p.patch_id for p in result.patches) == sorted(expected)
