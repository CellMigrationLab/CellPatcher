"""Folder runs that split planes read one plane at a time (docs/normalization_contract.md §3).

Since CP-2a a file run reads each image through the preprocessing run with
``RunInputs.read_plane``, one ``(z, t)`` plane (every channel) per call: with
plane policies only the chosen planes are read, and the patches equal those of
``run_patching`` on the whole arrays.
"""

import copy
import importlib.util

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import run_patching
from tests.helpers import file_run_config, write_ngff

needs_preprocessing = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs morphobricks-preprocessing (a core dependency)"
)
needs_zarr = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None or importlib.util.find_spec("zarr") is None,
    reason="needs the 'zarr' extra",
)


def _stack(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


def _write_tiff(path, data, axes):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})


def _config(image_axes, mode="grid", **common):
    """An in-memory config in *image_axes*, the layout a file run reads the same images in."""
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


def _folder(folder, config, axes):
    """Preprocess *folder*, then run the folder run with the file-run form of *config* (CP-2a)."""
    file_config = copy.deepcopy(config)
    file_config.common.image_axes = "YX"
    return run_patching_pipeline(folder, config=file_run_config(file_config, folder, axes))


@pytest.fixture
def read_planes(monkeypatch):
    """Record every plane a file run reads through ``RunInputs.read_plane``, as ``(z, t)``."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    calls = []
    original = RunInputs.read_plane

    def spy(self, fov, z=0, t=0, **kwargs):
        calls.append((z, t))
        return original(self, fov, z, t, **kwargs)

    monkeypatch.setattr(RunInputs, "read_plane", spy)
    return calls


@needs_preprocessing
@pytest.mark.parametrize("policies", [{"t_policy": "all"}, {"t_policy": "select", "t_index": 2}])
def test_tiff_planes_equal_the_in_memory_run(tmp_path, read_planes, policies):
    stacks = [_stack((3, 2, 16, 16), seed) for seed in range(2)]  # T, C, Y, X
    for name, stack in zip(("a.tif", "b.tif"), stacks):
        _write_tiff(tmp_path / "exp" / name, stack, "TCYX")
    config = _config("TCYX", **policies)

    folder = _folder(tmp_path / "exp", config, "TCYX")
    memory = run_patching(images=stacks, config=config, image_metadata=[{"image_file": n} for n in ("a.tif", "b.tif")])
    _same_result(folder, memory)
    # Only the chosen planes were read, every channel at once.
    planes = 3 if policies["t_policy"] == "all" else 1
    assert len(read_planes) == 2 * planes


@needs_preprocessing
def test_t_split_from_a_tiff_with_a_kept_channel(tmp_path, read_planes):
    # Was T and Z split together (TZCYX). Preprocessing v0.4.0 refuses Z>1 and a stored Z of length 1 is
    # not an axis of the file, so only the T split (here "first") remains for a kept channel.
    stack = _stack((2, 2, 8, 8))  # T, C, Y, X
    _write_tiff(tmp_path / "exp" / "a.tif", stack, "TCYX")
    config = _config("TCYX", t_policy="first")

    folder = _folder(tmp_path / "exp", config, "TCYX")
    _same_result(folder, run_patching(images=stack, config=config, image_metadata=[{"image_file": "a.tif"}]))
    assert sorted(read_planes) == [(0, 0)]


@needs_preprocessing
def test_random_planes_match_between_folder_and_memory_runs(tmp_path, read_planes):
    stacks = [_stack((2, 32, 32), seed) for seed in range(3)]
    for index, stack in enumerate(stacks):
        _write_tiff(tmp_path / "exp" / f"{index}.tif", stack, "TYX")
    config = _config("TYX", mode="random", t_policy="all")
    config.random.patch_size, config.random.n_patches = 4, 2

    folder = _folder(tmp_path / "exp", config, "TYX")
    memory = run_patching(images=stacks, config=config, image_metadata=[{"image_file": f"{i}.tif"} for i in range(3)])
    _same_result(folder, memory)


@needs_preprocessing
def test_keep_reads_every_plane(tmp_path, read_planes):
    # Was "keep and projections" over Z; Z>1 is refused by preprocessing v0.4.0 and T has no projection
    # policy, so "keep" over T is the case that remains.
    stack = _stack((3, 16, 16))
    _write_tiff(tmp_path / "exp" / "a.tif", stack, "TYX")
    config = _config("TYX", t_policy="keep")
    _same_result(
        _folder(tmp_path / "exp", config, "TYX"),
        run_patching(images=stack, config=config, image_metadata=[{"image_file": "a.tif"}]),
    )
    assert read_planes == [(0, t) for t in range(3)]


@needs_preprocessing
def test_a_plane_over_the_memory_limit_is_an_error(tmp_path, monkeypatch):
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.run_inputs import RunInputs

    def too_large(*args, **kwargs):
        raise mbp.PlaneMemoryError("Reading one plane of a.tif whole needs about 2 GiB of working memory.")

    _write_tiff(tmp_path / "exp" / "a.tif", _stack((3, 16, 16)), "TYX")
    config = file_run_config(_config("YX", t_policy="all"), tmp_path / "exp", "TYX")
    monkeypatch.setattr(RunInputs, "read_plane", too_large)
    with pytest.raises(ValueError, match="needs each plane whole in memory.*2 GiB"):
        run_patching_pipeline(tmp_path / "exp", config=config)


@needs_zarr
def test_ome_zarr_planes_are_read_in_the_canonical_axis_order(tmp_path, read_planes):
    data = _stack((2, 3, 16, 16))  # stored C, T, Y, X; read T, C, Y, X by role (CP-2a)
    write_ngff(tmp_path / "exp" / "a.ome.zarr", data, "ctyx")
    config = _config("TCYX", t_policy="all")

    folder = _folder(tmp_path / "exp", config, "CTYX")
    as_tcyx = np.transpose(data, (1, 0, 2, 3))
    _same_result(folder, run_patching(images=as_tcyx, config=config, image_metadata=[{"image_file": "a.ome.zarr"}]))
    assert len(read_planes) == 3


def _fov(**counts):
    """The parts of a ``Fov`` that ``fov_layout`` and ``run_layout`` read."""
    from types import SimpleNamespace

    values = {"relative_path": "a.tif", "roles": ("Z", "C"), "t_count": 1, "z_count": 1, "c_count": 2, **counts}
    return SimpleNamespace(**values)


def test_a_file_run_image_with_depth_planes_is_refused_by_name():
    """Issue #83: preprocessing v0.4.0 refuses Z>1; the file-run layout is the second defence."""
    from cellpatcher.config import CommonConfig
    from cellpatcher.errors import ZStackNotSupportedError
    from cellpatcher.run_reading import fov_layout

    message = r"a\.tif has 3 Z planes.*2-D planes only.*recorded projection step in morphobricks-preprocessing"
    with pytest.raises(ZStackNotSupportedError, match=message) as caught:
        fov_layout(_fov(z_count=3), CommonConfig())
    assert (caught.value.axis, caught.value.length) == ("Z", 3)


def test_a_file_run_image_with_one_depth_plane_is_still_read():
    from cellpatcher.config import CommonConfig
    from cellpatcher.run_reading import fov_layout, run_layout

    assert fov_layout(_fov(), CommonConfig()) == "CYX"  # a kept Z of length 1 is not an axis
    assert fov_layout(_fov(), CommonConfig(z_policy="all")) == "ZCYX"  # a plane policy still names it (z000)
    assert run_layout([_fov(t_count=2, roles=("T", "Z", "C"))]) == "TCYX"
