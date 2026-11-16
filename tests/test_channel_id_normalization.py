"""Normalization by canonical channel ID (W3: normalization v0.5.0, Spec ADR 0003 HV2).

The runs of described images are in ``tests/test_channel_ids.py`` (``v04``: one ID at a different C index in
each condition); here, the rules that name each channel plane of an image.
"""

import warnings

import numpy as np
import pytest

from cellpatcher.normalization import NormalizationWarning
from cellpatcher.statistics import PreprocessingCheckError


def test_an_image_whose_channel_ids_are_not_its_rows_is_refused():
    # W3: at a CSV scope each C index's ID comes from the plane_histograms.csv row (its channels[c]); an image
    # whose own IDs (image_metadata "channel_ids", a file run's Fov.channel_ids) say otherwise is refused, never
    # normalized with another channel's parameters.
    import morphobricks_normalization as mbn

    from cellpatcher.normalization import _check_row_channel_ids
    from cellpatcher.statistics import ChannelRecord

    histogram = mbn.compute_plane_histogram(np.arange(16, dtype=np.uint8).reshape(4, 4))
    record = mbn.PlaneRecord(
        "control/a.tif", "control", "bio1", "single", (histogram, histogram), position_id="a", channels=("C0", "C1")
    )
    rows = [ChannelRecord(record, 0), ChannelRecord(record, 1)]
    meta = {"source_relative_path": record.source_relative_path}
    _check_row_channel_ids({**meta, "channel_ids": record.channels}, rows)  # the same IDs: accepted
    # No IDs of its own: two channels are refused, never matched to the rows by position (v0.7.1, CP2; v0.7.0
    # accepted them); a single channel's one row names it.
    with pytest.raises(PreprocessingCheckError, match="no image_metadata") as raised:
        _check_row_channel_ids(meta, rows)
    assert [issue.code for issue in raised.value.issues] == ["channel_ids_required"]
    _check_row_channel_ids({**meta, "c": 1}, rows[1:])  # one plane named by its explicit C index
    with pytest.raises(PreprocessingCheckError, match="no image_metadata") as raised:
        _check_row_channel_ids(meta, rows[1:])  # one plane of a 2-channel image, its C index not given
    assert [issue.code for issue in raised.value.issues] == ["channel_ids_required"]
    with pytest.raises(PreprocessingCheckError, match="no image_metadata") as raised:
        _check_row_channel_ids({**meta, "c": 0}, rows[1:])  # a given C index that is not the row's
    assert [issue.code for issue in raised.value.issues] == ["channel_ids_required"]
    with pytest.raises(PreprocessingCheckError, match="Channel C0 of .* is 'C1' for the image but 'C0'") as raised:
        _check_row_channel_ids({**meta, "channel_ids": record.channels[::-1]}, rows)
    assert [issue.code for issue in raised.value.issues] == ["channel_id_mismatch"]


def test_in_memory_channels_are_named_by_position_and_file_runs_by_their_ids():
    from cellpatcher.normalization import image_channel_ids

    assert image_channel_ids({}, 2, True, False) == ("C0", "C1")
    assert image_channel_ids({}, 3, True, True) == ("R", "G", "B")
    assert image_channel_ids({"c": 2}, 1, False, False) == ("C2",)
    assert image_channel_ids({"channel_ids": ("ACTIN", "DNA")}, 2, True, False) == ("ACTIN", "DNA")
    assert image_channel_ids({"channel_ids": ("ACTIN", "DNA"), "c": 1}, 1, False, False) == ("DNA",)
    with pytest.raises(ValueError, match="names 2 channels; the image has 3"):
        image_channel_ids({"channel_ids": ("ACTIN", "DNA")}, 3, True, False)


@pytest.mark.parametrize("ids", [("DNA", "DNA"), ("DNA", "my channel")])
def test_repeated_or_malformed_channel_ids_are_refused_not_discarded_as_zero_spread(ids):
    # Review of #104: at the patch scope a repeated or malformed ID reached the brick's fit, whose ValueError was
    # caught as "zero intensity spread", so every patch was discarded with a misleading warning. Refused instead.
    from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
    from cellpatcher.normalization import image_channel_ids
    from cellpatcher.pipeline import run_patching

    with pytest.raises(ValueError, match='image_metadata\\["channel_ids"\\]'):
        image_channel_ids({"channel_ids": ids}, 2, True, False)
    with pytest.raises(ValueError, match='image_metadata\\["channel_ids"\\]'):
        image_channel_ids({"channel_ids": ids, "c": 0}, 1, False, False)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.extract_arrays = "CYX", True
    config.normalization = NormalizationConfig(enabled=True, method="minmax", scope="patch")
    image = np.random.default_rng(0).integers(0, 100, (2, 16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match='image_metadata\\["channel_ids"\\]'):
        run_patching(images=image, config=config, image_metadata=[{"position_id": "a", "channel_ids": ids}])


def test_rows_of_earlier_tables_are_refused_never_converted():
    """W3 (normalization v0.5.0, Spec ADR 0002 AR10): C-index rows are refused; ID rows read.

    The ';'-separated columns of earlier tables are unknown columns, refused by read_patch_table (CP-5).
    """
    import morphobricks_normalization as mbn

    from cellpatcher.normalization import parameters_from_row

    settings = {"method": "minmax", "percentile": [1.0, 99.8]}
    old_row = {
        "patch_id": "p", "c": "", "normalization_method": "min-max", "normalization_scope": "image",
        "normalization_group": "a.tif", "normalization_exact": "true",
    }
    c_index_row = {**old_row, "Normalization_Offset_C0": "1.5", "Normalization_Scale_C0": "3.0"}
    with pytest.raises(ValueError, match="normalization_parameters_version"):
        parameters_from_row(c_index_row, settings)
    parameters = mbn.Parameters("min-max", "image", "a.tif", True, (1.5, 2.0), (3.0, 4.0), ("ACTIN", "DNA"))
    new_row = {**old_row, **parameters.to_columns()}
    assert parameters_from_row(new_row, settings) == parameters
    assert parameters_from_row({"patch_id": "p", "normalization_method": ""}, settings) is None


# --- CP2 (v0.7.1): at a CSV scope a multichannel image is matched to its rows by channel ID only ---------------


@pytest.fixture
def two_channel_run(tmp_path):
    """One CYX image of two channels with the same extrema (0 and 255) and different distributions.

    ``check_planes`` compares a plane's minimum, maximum and pixel count with its row, so the two channels
    swapped pass it: only their channel IDs can tell them apart. The run's description names them ``C0``
    and ``C1`` (:func:`tests.helpers.describe_experiment`).
    """
    from tests.helpers import preprocess, write_tiff

    rng = np.random.default_rng(7)
    image = np.stack([rng.integers(0, 80, (32, 32)), rng.integers(170, 256, (32, 32))]).astype(np.uint8)
    image[:, 0, 0], image[:, 0, 1] = 0, 255
    root = tmp_path / "experiment"
    write_tiff(root / "ctrl" / "rep1" / "a.tif", image, "CYX")
    run = preprocess(root, "CYX", output=tmp_path / "run")
    return root, image, run


def _csv_scope_config(run):
    from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig

    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16))
    config.common.image_axes, config.common.extract_arrays = "CYX", True
    config.common.condition, config.common.biological_repeat = "ctrl", "rep1"
    config.normalization = NormalizationConfig(
        enabled=True, method="percentile", scope="image", plane_histograms=str(run / "plane_histograms.csv")
    )
    return config


def _no_fit(monkeypatch):
    """Make every CSV-scope fit fail, so a refusal is proven to come before any fit."""
    from cellpatcher.normalization import Normalizer

    def fit(self, key):
        raise AssertionError("a group was fitted before the refusal")

    monkeypatch.setattr(Normalizer, "_group_parameters", fit)


def _in_memory(image, run, **meta):
    from cellpatcher.pipeline import run_patching

    metadata = [{"source_relative_path": "ctrl/rep1/a.tif", "position_id": "a", **meta}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NormalizationWarning)
        return run_patching(images=image, config=_csv_scope_config(run), image_metadata=metadata)


def test_cp2_a_multichannel_image_without_channel_ids_is_refused_at_a_csv_scope_before_any_fit(
    two_channel_run, monkeypatch
):
    """v0.7.0 matched its C indices to the rows by position: the swapped channels took each other's parameters."""
    _, image, run = two_channel_run
    permuted = image[::-1].copy()
    _no_fit(monkeypatch)
    for given in (permuted, image):
        with pytest.raises(PreprocessingCheckError, match="image_metadata\\[\"channel_ids\"\\]") as raised:
            _in_memory(given, run)
        assert [issue.code for issue in raised.value.issues] == ["channel_ids_required"]
        assert "position" in raised.value.issues[0].message
        assert "channel_ids" in raised.value.issues[0].action


def test_cp2_the_right_channel_ids_are_accepted_and_give_the_file_runs_parameters(two_channel_run):
    from cellpatcher.orchestration import run_patching_condition_repeat_tree

    root, image, run = two_channel_run
    result = _in_memory(image, run, channel_ids=("C0", "C1"))
    config = _csv_scope_config(run)
    config.common.image_axes, config.common.preprocessing_run = "YX", str(run)
    config.common.condition = config.common.biological_repeat = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NormalizationWarning)
        (leaf,) = run_patching_condition_repeat_tree(root, config=config)
    in_memory = {p.patch_id: p.normalization_parameters for p in result.patches}
    file_run = {p.patch_id: p.normalization_parameters for p in leaf.result.patches}
    assert in_memory and in_memory == file_run  # a file run passes its Fov.channel_ids: unaffected
    (parameters,) = set(in_memory.values())
    assert parameters.channels == ("C0", "C1") and parameters.offset[0] != parameters.offset[1]


def test_cp2_wrong_channel_ids_are_refused_before_any_fit(two_channel_run, monkeypatch):
    _, image, run = two_channel_run
    _no_fit(monkeypatch)
    # The recorded image named in the wrong order, and the swapped image named truly: either way an ID is not its
    # row's at that C index (an in-memory image is given in its recorded C order, its IDs checked).
    for given in (image, image[::-1].copy()):
        with pytest.raises(PreprocessingCheckError) as raised:
            _in_memory(given, run, channel_ids=("C1", "C0"))
        assert [issue.code for issue in raised.value.issues] == ["channel_id_mismatch"]


def test_cp2_a_single_channel_needs_no_id_and_a_given_one_is_checked(two_channel_run, monkeypatch):
    from cellpatcher.pipeline import run_patching

    _, image, run = two_channel_run
    config = _csv_scope_config(run)
    config.common.image_axes = "YX"
    metadata = {"source_relative_path": "ctrl/rep1/a.tif", "position_id": "a", "c": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NormalizationWarning)
        result = run_patching(images=image[1], config=config, image_metadata=[metadata])
        assert {p.normalization_parameters.channels for p in result.patches} == {("C1",)}
        ids = {**metadata, "channel_ids": ("C0", "C1")}
        named = run_patching(images=image[1], config=config, image_metadata=[ids])
        assert [p.normalization_parameters for p in named.patches] == [
            p.normalization_parameters for p in result.patches
        ]
        _no_fit(monkeypatch)
        with pytest.raises(PreprocessingCheckError) as raised:
            run_patching(images=image[1], config=config, image_metadata=[{**metadata, "channel_ids": ("C1", "C0")}])
    assert [issue.code for issue in raised.value.issues] == ["channel_id_mismatch"]


@pytest.mark.parametrize(
    ("label", "plane", "axes", "meta"),
    [
        ("a YX plane without c", 1, "YX", {}),
        ("a one-channel CYX image without c", 1, "CYX", {}),
        ("a one-channel CYX image whose c is not its C index", 1, "CYX", {"c": 1}),
    ],
)
def test_cp2_one_plane_of_a_multichannel_image_without_ids_or_its_c_is_refused_before_any_fit(
    two_channel_run, monkeypatch, label, plane, axes, meta
):
    """Review of v0.7.1 CP2: one channel plane of a 2-channel image was matched to the row C0 by position.

    Without ``channel_ids`` and without its ``c`` (or with a C axis of one channel, whose ``c`` the CSV
    lookup ignores), the plane of channel C1 took channel C0's parameters: its extrema and pixel count equal
    C0's, all ``check_planes`` compares. It is refused (``channel_ids_required``) before any fit.
    """
    from cellpatcher.pipeline import run_patching

    _, image, run = two_channel_run
    config = _csv_scope_config(run)
    config.common.image_axes = axes
    pixels = image[plane] if axes == "YX" else image[plane : plane + 1]
    metadata = {"source_relative_path": "ctrl/rep1/a.tif", "position_id": "a", **meta}
    _no_fit(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NormalizationWarning)
        with pytest.raises(PreprocessingCheckError) as raised:
            run_patching(images=pixels, config=config, image_metadata=[metadata])
    assert [issue.code for issue in raised.value.issues] == ["channel_ids_required"], label
