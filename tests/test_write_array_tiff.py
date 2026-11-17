"""``write_array`` writes a TIFF with an explicit photometric, laid out from the axes it is given (issue #131).

Before, ``cellpatcher.io.write_array`` called ``tifffile.imwrite`` without ``photometric``, so tifffile guessed
it from the shape: an array with 3 or 4 along its first or last axis became an RGB(A) TIFF, with a
``DeprecationWarning``. As for exported patches (#129/#130), a TIFF is now written as planes
(``minisblack``) unless the caller declares RGB, one page per Y/X image, and the layout comes from the
caller's ``axes``, never from the shape.
"""

import warnings

import numpy as np
import pytest

from cellpatcher.io import read_array, write_array

tifffile = pytest.importorskip("tifffile")


def _write(path, array, **kwargs):
    """Write *array* with every warning an error (tifffile's DeprecationWarning about the photometric among them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_array(path, array, **kwargs)


def _tags(path):
    """The photometric, number of pages, samples per pixel and planar configuration of a TIFF's first page."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        planarconfig = tifffile.PLANARCONFIG(page.planarconfig).name
        return page.photometric.name, len(tif.pages), page.samplesperpixel, planarconfig


def _assert_round_trip(path, array):
    """The array read back has the written array's shape, dtype and bytes."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_array(path)
    assert back.shape == array.shape
    assert back.dtype == array.dtype
    assert back.tobytes() == array.tobytes()


def _array(shape, dtype):
    rng = np.random.default_rng(len(shape) + shape[0])
    return (rng.random(shape) * 200).astype(dtype)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_cyx_channels_are_written_as_planes_one_page_each(tmp_path, channels, dtype):
    array = _array((channels, 8, 8), dtype)
    path = tmp_path / "a.tif"
    _write(path, array, axes="CYX")
    assert _tags(path) == ("MINISBLACK", channels, 1, "CONTIG")
    _assert_round_trip(path, array)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_yxc_channels_are_one_page_with_the_channels_as_samples(tmp_path, channels, dtype):
    array = _array((8, 8, channels), dtype)
    path = tmp_path / "a.tif"
    _write(path, array, axes="YXC")
    assert _tags(path) == ("MINISBLACK", 1, channels, "CONTIG")
    _assert_round_trip(path, array)


def test_a_stack_before_y_and_x_is_one_page_per_y_x_image(tmp_path):
    array = _array((3, 4, 8, 8), np.uint16)
    path = tmp_path / "a.tif"
    _write(path, array, axes="ZCYX")
    assert _tags(path) == ("MINISBLACK", 12, 1, "CONTIG")
    _assert_round_trip(path, array)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (8, 3), (3, 8)])
def test_one_image_3_or_4_pixels_wide_is_one_plane(tmp_path, shape):
    array = _array(shape, np.uint8)
    path = tmp_path / "a.tif"
    _write(path, array, axes="YX")
    assert _tags(path) == ("MINISBLACK", 1, 1, "CONTIG")
    _assert_round_trip(path, array)


@pytest.mark.parametrize(
    ("axes", "shape", "planarconfig"), [("CYX", (3, 8, 8), "SEPARATE"), ("YXC", (8, 8, 3), "CONTIG")]
)
def test_an_rgb_array_declared_rgb_is_one_rgb_page(tmp_path, axes, shape, planarconfig):
    array = _array(shape, np.uint8)
    path = tmp_path / "a.tif"
    _write(path, array, axes=axes, photometric="rgb")
    assert _tags(path) == ("RGB", 1, 3, planarconfig)
    _assert_round_trip(path, array)


def test_a_tiff_without_axes_is_refused_naming_the_fix(tmp_path):
    path = tmp_path / "a.tif"
    with pytest.raises(ValueError, match=r"issue #131\): pass axes=.*axes='YX'.*photometric='rgb'"):
        write_array(path, np.zeros((3, 8, 8), np.uint8))
    assert not path.exists()


@pytest.mark.parametrize(
    ("axes", "shape", "photometric", "match"),
    [
        ("YX", (3, 8, 8), "minisblack", "must match ndim"),
        ("YXZC", (8, 8, 2, 3), "minisblack", "one page per Y/X image"),
        ("XY", (8, 8), "minisblack", "one page per Y/X image"),
        ("CYX", (4, 8, 8), "rgb", "An RGB TIFF needs three channels"),
        ("YX", (8, 8), "rgb", "An RGB TIFF needs three channels"),
        ("CZYX", (3, 2, 8, 8), "rgb", "An RGB TIFF needs three channels"),
        ("YX", (8, 8), "rgba", "photometric must be one of"),
    ],
)
def test_layouts_that_are_not_one_page_per_y_x_image_are_refused(tmp_path, axes, shape, photometric, match):
    path = tmp_path / "a.tif"
    with pytest.raises(ValueError, match=match):
        write_array(path, np.zeros(shape, np.uint8), axes=axes, photometric=photometric)
    assert not path.exists()


def test_npy_needs_no_axes(tmp_path):
    array = _array((3, 8, 8), np.uint16)
    write_array(tmp_path / "a.npy", array)
    _assert_round_trip(tmp_path / "a.npy", array)
