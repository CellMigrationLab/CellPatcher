"""Loading the preprocessing outputs and the checks of docs/normalization_contract.md §5.

The fixtures are simulated experiments whose ``plane_histograms.csv`` and
``run_manifest.json`` are written by the real ``morphobricks_preprocessing``
package. Tests needing it skip when the ``preprocessing`` extra is not installed.
"""

import csv
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.statistics import (
    INSTALL_COMMAND,
    PreprocessingCheckError,
    PreprocessingUnavailableError,
    SourcePlane,
    check_images,
    check_planes,
    load_plane_statistics,
    require_preprocessing,
)

needs_preprocessing = pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None,
    reason="needs the 'preprocessing' extra",
)

# Image paths of the simulated experiment: a condition without repeat folders,
# and two technical repeats holding the same file name.
STACK_IMAGES = ("control/img1.tif", "treated/bio1/wellA/field.tif", "treated/bio1/wellB/field.tif")
STACK_LABELS = {
    "control/img1.tif": ("control", "single", "single"),
    "treated/bio1/wellA/field.tif": ("treated", "bio1", "wellA"),
    "treated/bio1/wellB/field.tif": ("treated", "bio1", "wellB"),
}
Z_PLANES = 3


def _write_tiff(path, array, axes):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})


def _preprocess(root, output, axes):
    """Run the preprocessing brick on ``root``; return the output folder."""
    import morphobricks_preprocessing as mbp

    result = mbp.run_experiment(root, mbp.AxisMapping(axes, axes), output_folder=output)
    assert result.written, [str(issue) for issue in result.issues]
    return output


@pytest.fixture
def stack_experiment(tmp_path):
    """Three uint16 Z-stacks (ZYX) and their preprocessing outputs."""
    root = tmp_path / "experiment"
    rng = np.random.default_rng(0)
    for relative_path in STACK_IMAGES:
        _write_tiff(root / relative_path, rng.integers(1, 4000, (Z_PLANES, 20, 24), dtype=np.uint16), "ZYX")
    output = _preprocess(root, tmp_path / "preprocessing", "ZYX")
    return root, output


def _read_stack(root, relative_path):
    import tifffile

    return tifffile.imread(root / relative_path)


def _stack_planes(root, relative_paths=STACK_IMAGES):
    for relative_path in relative_paths:
        stack = _read_stack(root, relative_path)
        for z in range(stack.shape[0]):
            plane = SourcePlane(relative_path, *STACK_LABELS[relative_path], z=z, axis_lengths={"Z": stack.shape[0]})
            yield plane, stack[z]


def _codes(error_info):
    return [issue.code for issue in error_info.value.issues]


# ---------------------------------------------------------------------------
# The extra
# ---------------------------------------------------------------------------


def test_missing_extra_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "morphobricks_preprocessing", None)

    with pytest.raises(PreprocessingUnavailableError, match="not installed") as error_info:
        require_preprocessing()
    assert INSTALL_COMMAND in str(error_info.value)
    with pytest.raises(PreprocessingUnavailableError):
        load_plane_statistics("plane_histograms.csv")


def test_module_imports_without_the_extra():
    # cellpatcher.statistics is imported at the top of this file, which also
    # runs when the extra is missing; importing cellpatcher must never need it.
    import cellpatcher

    assert cellpatcher.load_plane_statistics is load_plane_statistics
    assert cellpatcher.check_images is check_images
    assert cellpatcher.check_planes is check_planes
    assert cellpatcher.SourcePlane is SourcePlane


# ---------------------------------------------------------------------------
# load_plane_statistics
# ---------------------------------------------------------------------------


@needs_preprocessing
def test_load_reads_rows_and_finds_the_manifest_next_to_the_table(stack_experiment):
    import morphobricks_normalization as mbn
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.manifest import file_sha256

    _, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)

    assert len(statistics.records) == len(STACK_IMAGES) * Z_PLANES
    assert statistics.source_relative_paths == frozenset(STACK_IMAGES)
    assert statistics.manifest_path == output / mbp.MANIFEST_FILENAME
    assert statistics.manifest is not None
    assert statistics.csv_sha256 == file_sha256(output / mbn.OUTPUT_FILENAME)
    assert len(statistics.records_for_image("control/img1.tif")) == Z_PLANES


@needs_preprocessing
def test_load_without_a_manifest_skips_it(stack_experiment, tmp_path):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(output / mbn.OUTPUT_FILENAME, alone / mbn.OUTPUT_FILENAME)

    statistics = load_plane_statistics(alone / mbn.OUTPUT_FILENAME)
    assert statistics.manifest is None
    assert statistics.manifest_path is None


@needs_preprocessing
def test_load_with_an_explicit_manifest(stack_experiment, tmp_path):
    import morphobricks_normalization as mbn
    import morphobricks_preprocessing as mbp

    _, output = stack_experiment
    elsewhere = tmp_path / "elsewhere.json"
    shutil.copy(output / mbp.MANIFEST_FILENAME, elsewhere)

    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME, manifest=elsewhere)
    assert statistics.manifest_path == elsewhere


@needs_preprocessing
def test_load_missing_files_are_errors(stack_experiment, tmp_path):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_plane_statistics(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="run manifest"):
        load_plane_statistics(output / mbn.OUTPUT_FILENAME, manifest=tmp_path / "missing.json")


@needs_preprocessing
def test_load_detects_a_table_changed_after_the_manifest(stack_experiment):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    csv_path = output / mbn.OUTPUT_FILENAME
    records = mbn.read_plane_histograms(csv_path)
    mbn.write_plane_histograms(records[:-1], csv_path)

    with pytest.raises(PreprocessingCheckError, match="changed or replaced") as error_info:
        load_plane_statistics(csv_path)
    assert _codes(error_info) == ["plane_histograms_changed"]


@needs_preprocessing
def test_load_rejects_another_table_version(stack_experiment):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    csv_path = output / mbn.OUTPUT_FILENAME
    # plane_id holds commas, so the rows are edited with the csv module.
    with open(csv_path, newline="", encoding="utf-8") as f:
        header, *rows = list(csv.reader(f))
    for row in rows:
        row[header.index("schema_version")] = "1"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])

    with pytest.raises(PreprocessingCheckError, match="schema version") as error_info:
        load_plane_statistics(csv_path, manifest=None)
    assert _codes(error_info) == ["unreadable_plane_histograms"]


@needs_preprocessing
def test_load_rejects_another_manifest_version(stack_experiment):
    import morphobricks_normalization as mbn
    import morphobricks_preprocessing as mbp

    _, output = stack_experiment
    manifest_path = output / mbp.MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["manifest_version"] = "2"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PreprocessingCheckError, match="not supported") as error_info:
        load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    assert _codes(error_info) == ["unreadable_manifest"]


# ---------------------------------------------------------------------------
# check_images
# ---------------------------------------------------------------------------


@needs_preprocessing
def test_check_images_passes_for_the_preprocessed_images(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    check_images(statistics, {path: "ZYX" for path in STACK_IMAGES}, experiment_root=root)


@needs_preprocessing
def test_check_images_reports_all_images_without_rows_in_one_message(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    images = {path: "ZYX" for path in (*STACK_IMAGES, "control/new.tif", "treated/bio1/wellA/later.tif")}

    with pytest.raises(PreprocessingCheckError) as error_info:
        check_images(statistics, images, experiment_root=root)
    (issue,) = error_info.value.issues
    assert issue.code == "images_without_rows"
    assert issue.paths == ("control/new.tif", "treated/bio1/wellA/later.tif")
    assert "2 images found by CellPatcher have no rows" in str(error_info.value)


@needs_preprocessing
def test_check_images_detects_a_changed_image_with_the_manifest(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    image_path = root / "control" / "img1.tif"
    stat = image_path.stat()
    os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    images = {path: "ZYX" for path in STACK_IMAGES}

    with pytest.raises(PreprocessingCheckError) as error_info:
        check_images(load_plane_statistics(output / mbn.OUTPUT_FILENAME), images, experiment_root=root)
    (issue,) = error_info.value.issues
    assert (issue.code, issue.paths) == ("images_changed", ("control/img1.tif",))

    # Without the manifest there is no size or time to compare.
    alone = output.parent / "alone"
    alone.mkdir()
    shutil.copy(output / mbn.OUTPUT_FILENAME, alone / mbn.OUTPUT_FILENAME)
    check_images(load_plane_statistics(alone / mbn.OUTPUT_FILENAME), images)


@needs_preprocessing
def test_check_images_detects_other_axis_roles(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    images = {path: "ZYX" for path in STACK_IMAGES}
    images["control/img1.tif"] = "CYX"

    with pytest.raises(PreprocessingCheckError) as error_info:
        check_images(statistics, images, experiment_root=root)
    (issue,) = error_info.value.issues
    assert issue.code == "axis_mapping_mismatch"
    assert issue.paths == ("control/img1.tif (CellPatcher CYX, preprocessing ZYX)",)


@needs_preprocessing
def test_check_images_needs_the_root_with_a_manifest(stack_experiment):
    import morphobricks_normalization as mbn

    _, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    with pytest.raises(ValueError, match="experiment_root"):
        check_images(statistics, {"control/img1.tif": "ZYX"})


# ---------------------------------------------------------------------------
# check_planes
# ---------------------------------------------------------------------------


@needs_preprocessing
def test_check_planes_links_every_z_plane_to_its_row(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)

    records = check_planes(statistics, _stack_planes(root))
    assert [record.plane_id for record in records] == [
        f"{path}::z={z},c=0,t=0" for path in STACK_IMAGES for z in range(Z_PLANES)
    ]


@needs_preprocessing
def test_check_planes_detects_changed_pixels_without_a_manifest(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME, manifest=None)
    path = "treated/bio1/wellB/field.tif"
    stack = _read_stack(root, path)
    stack[1, 0, 0] = 65000  # a new maximum in plane z=1
    plane = SourcePlane(path, *STACK_LABELS[path], z=1, axis_lengths={"Z": Z_PLANES})

    with pytest.raises(PreprocessingCheckError, match="the images changed") as error_info:
        check_planes(statistics, [(plane, stack[1])])
    (issue,) = error_info.value.issues
    assert (issue.code, issue.paths) == ("pixel_values_changed", (f"{path}::z=1,c=0,t=0",))


@needs_preprocessing
def test_check_planes_compares_dtype_size_and_labels(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    path = "control/img1.tif"
    stack = _read_stack(root, path)

    def plane(z, condition="control"):
        return SourcePlane(path, condition, "single", "single", z=z, axis_lengths={"Z": Z_PLANES})

    with pytest.raises(PreprocessingCheckError) as error_info:
        check_planes(statistics, [
            (plane(0), stack[0].astype(np.float32)),
            (plane(1), stack[1, :10]),
            (plane(2, condition="treated"), stack[2]),
        ])
    issues = {issue.code: issue.paths for issue in error_info.value.issues}
    assert issues == {
        "label_mismatch": (f"{path}::z=2,c=0,t=0 (CellPatcher treated/single/single, table control/single/single)",),
        "dtype_mismatch": (f"{path}::z=0,c=0,t=0 (CellPatcher float32, table uint16)",),
        "pixel_count_mismatch": (f"{path}::z=1,c=0,t=0 (CellPatcher 240 pixels, table 480)",),
    }


@needs_preprocessing
def test_check_planes_reports_planes_without_rows(stack_experiment):
    import morphobricks_normalization as mbn

    root, output = stack_experiment
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)
    stack = _read_stack(root, "control/img1.tif")
    plane = SourcePlane("control/img2.tif", "control", "single", "single", z=0, axis_lengths={"Z": Z_PLANES})

    with pytest.raises(PreprocessingCheckError, match="1 plane has no row") as error_info:
        check_planes(statistics, [(plane, stack[0])])
    assert _codes(error_info) == ["missing_plane_rows"]


# ---------------------------------------------------------------------------
# Plane identity (contract §3)
# ---------------------------------------------------------------------------


@needs_preprocessing
def test_plane_id_always_has_all_three_coordinates():
    # Absent or length-1 axes are coordinate 0, given as None or 0.
    assert SourcePlane("a.tif", "c", "b", "t").plane_id == "a.tif::z=0,c=0,t=0"
    assert SourcePlane("a.tif", "c", "b", "t", z=0, axis_lengths={"Z": 1}).plane_id == "a.tif::z=0,c=0,t=0"

    stack_plane = SourcePlane("a.tif", "c", "b", "t", z=2, c=1, axis_lengths={"Z": 3, "C": 2})
    assert stack_plane.plane_id == "a.tif::z=2,c=1,t=0"


def test_source_plane_reads_missing_labels_as_single():
    plane = SourcePlane("a.tif", None, None, "B02")
    assert plane.labels == ("single", "single", "B02")


def test_source_plane_needs_a_coordinate_for_longer_axes():
    with pytest.raises(ValueError, match="z coordinate is needed"):
        SourcePlane("a.tif", "c", "b", "t", axis_lengths={"Z": 3})
    with pytest.raises(ValueError, match="outside the image"):
        SourcePlane("a.tif", "c", "b", "t", c=2, axis_lengths={"C": 2})
    with pytest.raises(ValueError, match="only name the axes"):
        SourcePlane("a.tif", "c", "b", "t", axis_lengths={"Y": 1})


@needs_preprocessing
def test_two_dimensional_images_match_with_or_without_a_z_axis(tmp_path):
    import morphobricks_normalization as mbn

    root = tmp_path / "experiment"
    image = np.arange(20 * 24, dtype=np.uint16).reshape(20, 24)
    _write_tiff(root / "control" / "a.tif", image, "YX")
    output = _preprocess(root, tmp_path / "preprocessing", "YX")
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)

    # CellPatcher may read the image as YX or with a length-1 Z axis; the
    # table has the image without a Z axis. Both are the same plane.
    as_2d = SourcePlane("control/a.tif", "control", "single", "single")
    with_z = SourcePlane("control/a.tif", "control", "single", "single", z=0, axis_lengths={"Z": 1})
    records = check_planes(statistics, [(as_2d, image), (with_z, image)])
    assert [record.plane_id for record in records] == ["control/a.tif::z=0,c=0,t=0"] * 2


# ---------------------------------------------------------------------------
# Both bricks on one three-level tree
# ---------------------------------------------------------------------------


@needs_preprocessing
def test_cellpatcher_patches_link_to_the_preprocessing_rows(tmp_path):
    import morphobricks_normalization as mbn
    import tifffile

    root = tmp_path / "experiment"
    rng = np.random.default_rng(1)
    # Labels such as NA and 001 must survive both CSVs as text.
    for relative_path in ("NA/img.tif", "treated/001/wellA/field.tif", "treated/001/wellB/field.tif"):
        _write_tiff(root / relative_path, rng.integers(0, 255, (16, 16), dtype=np.uint8), "YX")
    output = _preprocess(root, tmp_path / "preprocessing", "YX")
    statistics = load_plane_statistics(output / mbn.OUTPUT_FILENAME)

    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    patches = [patch for leaf in run_patching_condition_repeat_tree(root, config=config) for patch in leaf.result.patches]
    paths = {patch.source_relative_path for patch in patches}
    assert paths == {"NA/img.tif", "treated/001/wellA/field.tif", "treated/001/wellB/field.tif"}

    check_images(statistics, {path: "YX" for path in paths}, experiment_root=root)
    planes = []
    for patch in patches:
        plane = SourcePlane(patch.source_relative_path, patch.condition, patch.biological_repeat, patch.technical_repeat)
        planes.append((plane, tifffile.imread(root / patch.source_relative_path)))
    records = check_planes(statistics, planes)

    by_path = {record.source_relative_path: record for record in records}
    for patch in patches:
        record = by_path[patch.source_relative_path]
        assert (record.condition, record.biological_repeat, record.technical_repeat) == (
            patch.condition, patch.biological_repeat, patch.technical_repeat,
        )
    assert {record.plane_id for record in records} == {f"{path}::z=0,c=0,t=0" for path in paths}
