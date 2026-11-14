"""Normalization by canonical channel ID (W3: normalization v0.5.0, Spec ADR 0003 HV2).

The runs of described images are in ``tests/test_channel_ids.py`` (``v04``: one ID at a different C index in
each condition); here, the rules that name each channel plane of an image.
"""

import numpy as np
import pytest

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
    _check_row_channel_ids(meta, rows)  # no IDs of its own (an in-memory image): the rows name them
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
