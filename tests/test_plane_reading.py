"""Folder runs that split planes read one plane at a time (docs/normalization_contract.md §3).

With the ``preprocessing`` extra, only the chosen planes of a TIFF or
OME-Zarr image are decoded, with the preprocessing brick's ``read_plane``;
the patches equal those of ``run_patching`` on the whole arrays.
"""

import importlib.util
import sys

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import run_patching
from tests.helpers import write_ngff

needs_preprocessing = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs the 'preprocessing' extra"
)
needs_zarr = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None or importlib.util.find_spec("zarr") is None,
    reason="needs the 'preprocessing' extra with zarr",
)


def _stack(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


def _write_tiff(path, data, axes):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})


def _config(image_axes, mode="grid", **common):
    config = CellPatcherConfig(mode=mode, grid=GridConfig(tile_size=8))
    config.common.image_axes = image_axes
    config.common.extract_arrays = True
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _same_result(folder_result, memory_result):
    assert [(p.patch_id, p.z0, p.t, p.y0, p.x0) for p in folder_result.patches] == [
        (p.patch_id, p.z0, p.t, p.y0, p.x0) for p in memory_result.patches
    ]
    for folder_array, memory_array in zip(folder_result.patch_arrays, memory_result.patch_arrays):
        np.testing.assert_array_equal(folder_array, memory_array)


@pytest.fixture
def read_planes(monkeypatch):
    """Record every plane the preprocessing reader decodes, as ``(z, c, t)``."""
    import morphobricks_preprocessing as mbp

    calls = []
    original = mbp.read_plane

    def spy(path, mapping, *, z=None, c=None, t=None, **kwargs):
        calls.append((z, c, t))
        return original(path, mapping, z=z, c=c, t=t, **kwargs)

    monkeypatch.setattr(mbp, "read_plane", spy)
    return calls


@needs_preprocessing
@pytest.mark.parametrize("policies", [{"z_policy": "all"}, {"z_policy": "select", "z_index": 2}])
def test_tiff_planes_equal_the_in_memory_run(tmp_path, read_planes, policies):
    stacks = [_stack((3, 2, 16, 16), seed) for seed in range(2)]  # Z, C, Y, X
    for name, stack in zip(("a.tif", "b.tif"), stacks):
        _write_tiff(tmp_path / name, stack, "ZCYX")
    config = _config("ZCYX", **policies)

    folder = run_patching_pipeline(tmp_path, config=config)
    memory = run_patching(images=stacks, config=config, image_metadata=[{"image_file": n} for n in ("a.tif", "b.tif")])
    _same_result(folder, memory)
    # Only the chosen planes were decoded, one channel at a time.
    planes = 3 if policies["z_policy"] == "all" else 1
    assert len(read_planes) == 2 * planes * 2


@needs_preprocessing
def test_t_and_z_split_from_a_tiff_with_a_kept_channel(tmp_path, read_planes):
    stack = _stack((2, 3, 2, 8, 8))  # T, Z, C, Y, X
    _write_tiff(tmp_path / "a.tif", stack, "TZCYX")
    config = _config("TZCYX", z_policy="all", t_policy="first")

    folder = run_patching_pipeline(tmp_path, config=config)
    _same_result(folder, run_patching(images=stack, config=config, image_metadata=[{"image_file": "a.tif"}]))
    assert sorted(read_planes) == [(z, c, 0) for z in range(3) for c in range(2)]


@needs_preprocessing
def test_random_seeds_continue_across_images(tmp_path, read_planes):
    stacks = [_stack((2, 32, 32), seed) for seed in range(3)]
    for index, stack in enumerate(stacks):
        _write_tiff(tmp_path / f"{index}.tif", stack, "ZYX")
    config = _config("ZYX", mode="random", z_policy="all")
    config.random.patch_size, config.random.n_patches = 4, 2

    folder = run_patching_pipeline(tmp_path, config=config)
    memory = run_patching(images=stacks, config=config, image_metadata=[{"image_file": f"{i}.tif"} for i in range(3)])
    _same_result(folder, memory)


@needs_preprocessing
def test_keep_and_projections_read_whole_images(tmp_path, read_planes):
    stack = _stack((3, 16, 16))
    _write_tiff(tmp_path / "a.tif", stack, "ZYX")
    for policy in ("keep", "max"):
        config = _config("ZYX", z_policy=policy)
        _same_result(
            run_patching_pipeline(tmp_path, config=config),
            run_patching(images=stack, config=config, image_metadata=[{"image_file": "a.tif"}]),
        )
    assert read_planes == []


@needs_preprocessing
def test_a_plane_over_the_memory_limit_is_an_error(tmp_path, monkeypatch):
    import morphobricks_preprocessing as mbp

    def too_large(*args, **kwargs):
        raise mbp.PlaneMemoryError("Reading one plane of a.tif whole needs about 2 GiB of working memory.")

    _write_tiff(tmp_path / "a.tif", _stack((3, 16, 16)), "ZYX")
    monkeypatch.setattr(mbp, "read_plane", too_large)
    with pytest.raises(ValueError, match="needs each plane whole in memory.*2 GiB"):
        run_patching_pipeline(tmp_path, config=_config("ZYX", z_policy="all"))


def test_without_the_extra_planes_come_from_the_whole_image(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "morphobricks_preprocessing", None)
    stack = _stack((3, 16, 16))
    np.save(tmp_path / "a.npy", stack)
    config = _config("ZYX", z_policy="all")
    _same_result(
        run_patching_pipeline(tmp_path, config=config),
        run_patching(images=stack, config=config, image_metadata=[{"image_file": "a.npy"}]),
    )


@needs_zarr
def test_ome_zarr_planes_are_read_in_the_configured_axis_order(tmp_path, read_planes):
    data = _stack((2, 3, 16, 16))  # stored C, Z, Y, X
    write_ngff(tmp_path / "a.ome.zarr", data, "czyx")
    config = _config("ZCYX", z_policy="all")

    folder = run_patching_pipeline(tmp_path, config=config)
    as_zcyx = np.transpose(data, (1, 0, 2, 3))
    _same_result(folder, run_patching(images=as_zcyx, config=config, image_metadata=[{"image_file": "a.ome.zarr"}]))
    assert len(read_planes) == 3 * 2
