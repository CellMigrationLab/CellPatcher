"""Exported TIFF patches are written as planes, RGB only for an rgb unit, with their channel IDs (issue #129).

Before, ``save_patch_array`` called ``tifffile.imwrite`` without ``photometric``, so tifffile guessed it from
the shape: a 3- or 4-channel fluorescence patch (``CYX``), or any patch 3 or 4 pixels wide along its last
axis, became an RGB (or RGBA) TIFF, and tifffile warned that this default will change. The image type now
comes from what the run declares (``Fov.image_type``, ``image_metadata["image_type"]``,
``normalization.image_type``), never from the shape, and the pixels read back are unchanged.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.delivery import iter_exported_patches
from cellpatcher.export import export_patches, read_patch_table, save_patch_array, tiff_layout
from cellpatcher.io import read_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import run_patching
from cellpatcher.types import PatchRegion

tifffile = pytest.importorskip("tifffile")

needs_preprocessing = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs morphobricks-preprocessing"
)

FLUORESCENCE_IDS = ("DNA", "ACTIN", "PHASE", "TUBULIN")


def _tags(path):
    """The photometric, number of pages, samples per pixel, series shape and channel IDs of a TIFF."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        metadata = tif.shaped_metadata[0] if tif.shaped_metadata else {}
        return page.photometric.name, len(tif.pages), page.samplesperpixel, tif.series[0].shape, metadata


def _patch(patch_id, axes, channel_ids=None, image_type=None):
    return PatchRegion(patch_id=patch_id, y1=8, x1=8, output_axes=axes, channel_ids=channel_ids, image_type=image_type)


def _export_one(tmp_path, patch, array):
    """Export one patch without warnings, and return its file."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # tifffile's DeprecationWarning about the photometric default among them
        [path] = export_patches([patch], [array], tmp_path, save_metadata=False)
    return path


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_a_fluorescence_patch_is_written_as_planes_with_its_channel_ids(tmp_path, channels, dtype):
    rng = np.random.default_rng(channels)
    array = (rng.random((channels, 8, 8)) * 200).astype(dtype)
    ids = FLUORESCENCE_IDS[:channels]
    path = _export_one(tmp_path, _patch("p", "CYX", ids, "channels"), array)

    photometric, pages, samples, shape, metadata = _tags(path)
    assert photometric == "MINISBLACK" and samples == 1 and pages == channels  # one plane per channel
    assert shape == array.shape and metadata["channel_ids"] == list(ids)
    back = read_array(path)
    assert back.dtype == array.dtype and back.shape == array.shape and back.tobytes() == array.tobytes()


def test_a_patch_of_no_declared_type_is_written_as_planes_whatever_its_shape(tmp_path):
    # tifffile's guess took a last axis of 3 or 4 as colour samples, so a 2-channel patch 3 pixels wide
    # became an RGB image of 2 pages; a YX patch has no channel IDs to record.
    for name, array, axes in (("narrow", np.arange(48, dtype=np.uint16).reshape(2, 8, 3), "CYX"),
                              ("plane", np.arange(64, dtype=np.float32).reshape(8, 8), "YX")):
        path = _export_one(tmp_path, _patch(name, axes), array)
        photometric, _, samples, shape, metadata = _tags(path)
        assert (photometric, samples, shape) == ("MINISBLACK", 1, array.shape), name
        assert "channel_ids" not in metadata
        assert read_array(path).tobytes() == array.tobytes()


@pytest.mark.parametrize(("axes", "shape", "image_type"), [
    ("YXC", (8, 8, 3), None), ("YXC", (8, 8, 4), "channels"), ("ZYXC", (2, 8, 8, 3), None),
    ("YXZ", (8, 8, 5), "rgb"),  # a split channel of an rgb unit, its Z axis kept last
])
def test_a_patch_with_an_axis_after_x_is_one_page_per_yx_image(tmp_path, axes, shape, image_type):
    # Review of #129: tifffile pages an array by its last two axes, so a YXC fluorescence patch written as
    # plain minisblack became Y pages of X-by-C strips; each page is now one Y/X image, the last axis its samples.
    array = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = _export_one(tmp_path, _patch("p", axes, None, image_type), array)
    photometric, pages, samples, series, _ = _tags(path)
    assert (photometric, pages, samples, series) == ("MINISBLACK", int(np.prod(shape[:-3])), shape[-1], shape)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].shape == shape[-3:]  # Y, X, samples
    back = read_array(path)
    assert back.dtype == array.dtype and back.shape == array.shape and back.tobytes() == array.tobytes()


@pytest.mark.parametrize(("axes", "planarconfig"), [("CYX", "SEPARATE"), ("YXC", "CONTIG")])
def test_a_patch_of_an_rgb_unit_is_written_as_rgb(tmp_path, axes, planarconfig):
    image = np.random.default_rng(1).integers(0, 255, (3, 8, 8), dtype=np.uint8)
    array = image if axes == "CYX" else np.moveaxis(image, 0, -1)
    path = _export_one(tmp_path, _patch("rgb", axes, ("R", "G", "B"), "rgb"), array)

    photometric, pages, samples, shape, metadata = _tags(path)
    assert (photometric, pages, samples, shape) == ("RGB", 1, 3, array.shape)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].planarconfig.name == planarconfig
    assert metadata["channel_ids"] == ["R", "G", "B"]
    assert read_array(path).tobytes() == array.tobytes()


def test_one_channel_plane_of_an_rgb_unit_is_written_as_a_plane(tmp_path):
    assert tiff_layout(_patch("g", "YX", ("G",), "rgb"), np.zeros((8, 8))) == ("minisblack", None)


@pytest.mark.parametrize(("axes", "shape", "message"), [
    ("CYX", (4, 8, 8), "holds 4 channels, not 3"),
    ("YCX", (8, 3, 8), "C just before Y and X or last"),
    (None, (3, 8, 8), "no axes recorded"),
])
def test_an_rgb_patch_that_cannot_be_an_rgb_tiff_is_refused_before_any_file(tmp_path, axes, shape, message):
    patches = [_patch("ok", "CYX", None, "channels"), _patch("bad", axes, None, "rgb")]
    with pytest.raises(ValueError, match=message):
        export_patches(patches, [np.zeros((3, 8, 8)), np.zeros(shape)], tmp_path, save_metadata=False)
    assert not list(tmp_path.iterdir())


def test_npy_patches_are_unchanged(tmp_path):
    array = np.arange(3 * 64, dtype=np.float32).reshape(3, 8, 8)
    [path] = export_patches([_patch("p", "CYX", FLUORESCENCE_IDS[:3], "channels")], [array], tmp_path,
                            image_format="npy", save_metadata=False)
    assert np.load(path).tobytes() == array.tobytes()


def test_save_patch_array_refuses_another_photometric(tmp_path):
    with pytest.raises(ValueError, match="photometric must be one of"):
        save_patch_array(np.zeros((3, 8, 8)), tmp_path / "p.tif", photometric="cmyk")
    assert not (tmp_path / "p.tif").exists()


# ---------------------------------------------------------------------------
# In-memory runs: image_metadata["image_type"] or normalization.image_type
# ---------------------------------------------------------------------------


def _grid(tmp_path, axes="CYX", **normalization):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = axes
    config.common.output_dir = str(tmp_path / "out")
    config.common.extract_arrays = True
    config.export.enabled = True
    if normalization:
        config.normalization = NormalizationConfig(enabled=True, scope="patch", **normalization)
    return config


def test_an_in_memory_three_channel_run_writes_planes_and_reads_back_identical(tmp_path):
    image = np.random.default_rng(2).integers(0, 4000, (3, 16, 16), dtype=np.uint16)
    meta = [{"image_file": "img.tif", "channel_ids": ("DNA", "ACTIN", "PHASE")}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = run_patching([image], config=_grid(tmp_path), image_metadata=meta)

    assert {patch.image_type for patch in result.patches} == {None}  # nothing declares a type
    for patch, array in zip(result.patches, result.patch_arrays, strict=True):
        path = tmp_path / "out" / patch.patch_file
        photometric, pages, _, _, metadata = _tags(path)
        assert (photometric, pages, metadata["channel_ids"]) == ("MINISBLACK", 3, ["DNA", "ACTIN", "PHASE"])
        assert read_array(path).tobytes() == np.asarray(array).tobytes()
    # The readers (read_patch_table, iter_exported_patches and so ExportedPatchTorchDataset) give the arrays back.
    read, _ = read_patch_table(tmp_path / "out" / "patches.csv")
    assert [patch.patch_id for patch in read] == [patch.patch_id for patch in result.patches]
    delivered = list(iter_exported_patches(tmp_path / "out" / "patches.csv"))
    assert len(delivered) == len(result.patch_arrays)
    for (_, array), expected in zip(delivered, result.patch_arrays, strict=True):
        assert array.dtype == expected.dtype and array.tobytes() == expected.tobytes()


def test_an_in_memory_image_declared_rgb_is_written_as_rgb(tmp_path):
    image = np.random.default_rng(3).integers(0, 255, (3, 16, 16), dtype=np.uint8)
    meta = [{"image_file": "img.tif", "channel_ids": ("R", "G", "B"), "image_type": "rgb"}]
    result = run_patching([image], config=_grid(tmp_path), image_metadata=meta)
    assert {patch.image_type for patch in result.patches} == {"rgb"}
    for patch, array in zip(result.patches, result.patch_arrays, strict=True):
        path = tmp_path / "out" / patch.patch_file
        assert _tags(path)[:3] == ("RGB", 1, 3)
        assert read_array(path).tobytes() == np.asarray(array).tobytes()


@pytest.mark.filterwarnings("ignore::cellpatcher.normalization.NormalizationWarning")
def test_normalization_image_type_rgb_declares_the_image_rgb(tmp_path):
    image = np.random.default_rng(4).integers(0, 200, (3, 16, 16), dtype=np.uint8)
    result = run_patching([image], config=_grid(tmp_path, image_type="rgb"), image_metadata=[{"image_file": "a.tif"}])
    assert {patch.image_type for patch in result.patches} == {"rgb"}
    for patch, array in zip(result.patches, result.patch_arrays, strict=True):
        path = tmp_path / "out" / patch.patch_file
        assert _tags(path)[:3] == ("RGB", 1, 3)
        assert read_array(path).tobytes() == np.asarray(array).tobytes()


def test_an_unknown_or_contradicting_image_type_is_refused(tmp_path):
    image = np.zeros((3, 16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="image_type='fluorescence' for image 0; it is one of"):
        run_patching([image], config=_grid(tmp_path), image_metadata=[{"image_file": "a.tif",
                                                                        "image_type": "fluorescence"}])
    with pytest.raises(ValueError, match="normalization.image_type='rgb'"):
        run_patching([image], config=_grid(tmp_path, image_type="rgb"),
                     image_metadata=[{"image_file": "a.tif", "image_type": "channels"}])
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# File runs: the unit's declared image type (Fov.image_type)
# ---------------------------------------------------------------------------


def _file_grid(tmp_path, run):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.preprocessing_run = str(run.output)
    config.common.output_dir = str(tmp_path / "patches")
    config.export.enabled = True
    return config


def _exported(results):
    return [(Path(path), patch) for leaf in results
            for path, patch in zip(leaf.result.exported_files, leaf.result.patches, strict=True)]


@needs_preprocessing
def test_a_file_run_of_rgb_units_writes_rgb_without_normalization(tmp_path):
    # Spec v28 declares two rgb units: the run's Fov.image_type, not a normalization setting, decides.
    from morphobricks_preprocessing.testing_described import make_described_run

    from tests.test_described_ids import _fixture

    run = make_described_run(tmp_path / "v28", _fixture("v28"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        results = run_patching_condition_repeat_tree(run.experiment, config=_file_grid(tmp_path, run))
    exported = _exported(results)
    assert exported
    for path, patch in exported:
        assert patch.image_type == "rgb" and patch.output_axes == "CYX"
        photometric, pages, samples, _, metadata = _tags(path)
        assert (photometric, pages, samples) == ("RGB", 1, 3)
        assert metadata["channel_ids"] == list(patch.channel_ids)
        source = tifffile.imread(run.experiment / patch.source_relative_path)  # Y, X, S as stored
        crop = np.moveaxis(source, -1, 0)[:, patch.y0:patch.y1, patch.x0:patch.x1]
        assert read_array(path).tobytes() == np.ascontiguousarray(crop).tobytes()


@needs_preprocessing
def test_a_file_run_of_channel_units_writes_planes_with_the_channel_ids(tmp_path):
    from morphobricks_preprocessing.testing_described import make_described_run

    from tests.test_described_ids import _fixture

    run = make_described_run(tmp_path / "v07", _fixture("v07"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        results = run_patching_condition_repeat_tree(run.experiment, config=_file_grid(tmp_path, run))
    exported = _exported(results)
    assert exported
    for path, patch in exported:
        assert patch.image_type == "channels" and patch.channel_ids == ("DNA", "ACTIN")
        photometric, pages, samples, _, metadata = _tags(path)
        assert (photometric, pages, samples, metadata["channel_ids"]) == ("MINISBLACK", 2, 1, ["DNA", "ACTIN"])
        source = tifffile.imread(run.experiment / patch.source_relative_path)  # C, Y, X
        crop = source[:, patch.y0:patch.y1, patch.x0:patch.x1]
        assert read_array(path).tobytes() == np.ascontiguousarray(crop).tobytes()
