"""Images of different channel sets in one normalized run (roadmap G1; Spec fixtures v05, v06, v17).

Before G1 a normalized run refused images with different numbers of channels (``Spec Stage 3, gate
G3``). Now:

- channels are matched by ID, never by position, and are never padded or dropped;
- parameters are pooled and applied per channel ID, and every patch keeps its image's own C order
  (``normalization_channels``, the patch's channel axis);
- a channel only some planes of a fitted group have stays refused (``A_SCOPE_INCOMPLETE``) unless
  ``normalization.partial_channels="planes_with_channel"``, the normalization brick's explicit
  opt-in, which ``run_config.json`` records.

The fixtures' images are written by preprocessing's ``make_described_run`` with one intensity range per
channel ID (the **swap detector**: a channel read at another C index lands outside its range). Every
normalized patch value of channel ``X`` must equal the value of a fit over only the planes that have
``X`` (the run of only that channel's images), applied to the image's own ``X`` plane.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellpatcher.config import CellPatcherConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import run_patching

VALID = Path(__file__).parent / "spec_fixtures" / "fixtures" / "valid"
FIXTURES = {
    "v05": "v05_variable_channel_counts.toml",
    "v06": "v06_technical_repeats_different_layouts.toml",
    "v17": "v17_partial_channel_presence.toml",
}
BASE = {"DNA": 1000, "ACTIN": 5000, "PAXILLIN": 9000, "PHASE": 13000}
SPREAD = 2000  # each channel's values lie in [BASE, BASE + SPREAD): disjoint ranges


def _write_image(path, unit, seed):
    """Write the unit's channels in its order, each in its own intensity range (uint16, CYX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixels = np.stack([
        BASE[channel] + rng.integers(0, SPREAD, (16, 16)) for channel in unit.channels
    ]).astype(np.uint16)
    tifffile.imwrite(path, pixels, photometric="minisblack", metadata={"axes": "CYX"})


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import morphobricks_preprocessing.testing_described as described

    original = described._write_image
    described._write_image = _write_image
    try:
        folder = tmp_path_factory.mktemp("g1")
        return {
            name: described.make_described_run(folder / name, (VALID / toml).read_bytes())
            for name, toml in FIXTURES.items()
        }
    finally:
        described._write_image = original


def _config(run, scope, output=None, *, partial=None, mode="grid", **sections):
    config = CellPatcherConfig(mode=mode)
    config.common.preprocessing_run = str(run.output)
    config.common.extract_arrays = True
    config.grid.tile_size = 8
    config.random.patch_size, config.random.n_patches, config.random.seed = 8, 2, 1
    config.normalization.enabled, config.normalization.scope = True, scope
    config.normalization.method = "minmax"
    if partial is not None:
        config.normalization.partial_channels = partial
    if output is not None:
        config.common.output_dir, config.export.enabled = str(output), True
    for name, value in sections.items():
        section, attribute = name.split("__")
        setattr(getattr(config, section), attribute, value)
    return config


def _patches(run, config):
    leaves = run_patching_condition_repeat_tree(run.experiment, config=config,
                                                combine_metadata=config.export.enabled)
    return [(patch, array) for leaf in leaves for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays,
                                                                       strict=True)]


@functools.cache
def _opened(output):
    """Return the run's images by path and its plane statistics, opened once per run."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    from cellpatcher.run_open import statistics_of

    inputs = RunInputs.open(output)
    return {fov.relative_path: fov for fov in inputs.fovs()}, statistics_of(inputs)


def _fovs(run):
    return _opened(run.output)[0]


@functools.cache
def _fitted(output, scope, path, channel):
    """Return the parameters of *channel* fitted over only the planes of *path*'s group that have it."""
    import morphobricks_normalization as mbn

    from cellpatcher.normalization import repeat_groups

    statistics = _opened(output)[1]
    settings = mbn.NormalizationSettings(method="minmax", scope=scope, partial_channels="planes_with_channel")
    groups = repeat_groups(statistics) if scope == "biological_repeat" else None
    index, _ = mbn.scope_groups(statistics.records, settings, groups)
    row = next(r for r in statistics.records if r.source_relative_path == path)
    with_channel = [r for r in index[mbn.group_key(row, settings, groups)] if r.has_channel(channel)]
    parameters = mbn.fit(with_channel, settings).select((channel,))
    return mbn.NormalizationParameters(parameters.method, parameters.offset[0], parameters.scale[0])


def _expected(run, scope, patch, channel):
    """Return channel *channel*'s normalized values of *patch* from a fit over only the planes with it."""
    stack = tifffile.imread(run.experiment / patch.source_relative_path)
    c = _fovs(run)[patch.source_relative_path].channel_ids.index(channel)
    raw = stack[c, patch.y0:patch.y1, patch.x0:patch.x1]
    assert BASE[channel] <= raw.min() and raw.max() < BASE[channel] + SPREAD  # the swap detector, on the source
    return _fitted(run.output, scope, patch.source_relative_path, channel).apply(raw, dtype=np.float32)


def _check_every_value(run, scope, patches):
    fovs = _fovs(run)
    counts = set()
    for patch, array in patches:
        order = fovs[patch.source_relative_path].channel_ids
        assert patch.normalization_channels == order  # the image's own C order, every channel kept
        assert array.shape[0] == len(order)
        counts.add(len(order))
        for c, channel in enumerate(order):
            np.testing.assert_array_equal(array[c], _expected(run, scope, patch, channel))
    return counts


@pytest.mark.parametrize("name", ["v05", "v06", "v17"])
def test_image_scope_patches_images_of_different_channel_sets_by_channel_id(runs, name):
    # The old refusal ("different numbers of channels") becomes this acceptance.
    run = runs[name]
    patches = _patches(run, _config(run, "image"))
    counts = _check_every_value(run, "image", patches)
    assert len(counts) > 1 or name == "v06"  # v05, v17: 2 and 3 channels; v06: 3 and 4, other orders
    if name == "v06":
        orders = {patch.normalization_channels for patch, _ in patches}
        assert ("DNA", "ACTIN", "PAXILLIN") in orders and ("ACTIN", "DNA", "PAXILLIN") in orders


@pytest.mark.parametrize("name", ["v05", "v06", "v17"])
def test_patch_scope_patches_images_of_different_channel_sets(runs, name):
    run = runs[name]
    fovs = _fovs(run)
    for patch, array in _patches(run, _config(run, "patch")):
        order = fovs[patch.source_relative_path].channel_ids
        assert patch.normalization_channels == order and array.shape[0] == len(order)
        assert set(patch.normalization_parameters.channels) == set(order)  # stored by ID


def _scope_incomplete(run, tmp_path):
    with pytest.raises(ValueError) as raised:
        _patches(run, _config(run, "biological_repeat", tmp_path / "refused"))
    assert "A_SCOPE_INCOMPLETE" in str(raised.value) and "PHASE" in str(raised.value)
    assert not (tmp_path / "refused").exists()  # refused when the run is opened, before any read or write


@pytest.mark.parametrize("name", ["v05", "v06", "v17"])
def test_a_channel_some_planes_of_a_group_lack_is_refused_unless_opted_in(runs, name, tmp_path):
    run = runs[name]
    _scope_incomplete(run, tmp_path)
    output = tmp_path / "opted"
    patches = _patches(run, _config(run, "biological_repeat", output, partial="planes_with_channel"))
    _check_every_value(run, "biological_repeat", patches)
    saved = json.loads((output / "run_config.json").read_text("utf-8"))
    assert saved["normalization"]["partial_channels"] == "planes_with_channel"
    assert saved["normalization_settings"]["partial_channels"] == "planes_with_channel"  # recorded


def test_the_opt_in_is_a_choice_of_two(runs):
    from cellpatcher.config import NormalizationConfig

    with pytest.raises(ValueError, match="partial_channels must be one of"):
        NormalizationConfig(partial_channels="pad")
    assert NormalizationConfig().partial_channels == "refuse"


def test_split_channels_by_id_across_channel_sets(runs):
    # One patch per channel ID: an ID every image has is patched in every image, at its own C index.
    run = runs["v06"]
    patches = _patches(run, _config(run, "image", extraction__split_channels=["DNA"]))
    for patch, array in patches:
        assert patch.channel_id == "DNA" and patch.normalization_channels == ("DNA",)
        np.testing.assert_array_equal(array[0] if array.ndim == 3 else array, _expected(run, "image", patch, "DNA"))


def test_images_with_different_channel_counts_are_normalized_by_channel_id():
    # Roadmap G1 (was refused: "different numbers of channels"): each image keeps its channels and its
    # own C order, every channel normalized with its own parameters, nothing padded or dropped.
    two, three = np.ones((2, 16, 16), np.uint8), np.ones((3, 16, 16), np.uint8)
    two[:, 0, 0] = three[:, 0, 0] = 9
    two[1, 1, 1], three[2, 2, 2] = 50, 90
    config = CellPatcherConfig(mode="whole_fov")
    config.common.image_axes, config.common.extract_arrays = "CYX", True
    config.normalization.enabled, config.normalization.scope, config.normalization.method = True, "patch", "minmax"
    config.normalization.patch_region = "patch"
    metadata = [{"position_id": "a", "channel_ids": ("DNA", "ACTIN")},
                {"position_id": "b", "channel_ids": ("ACTIN", "PHASE", "DNA")}]
    result = run_patching(images=[two, three], config=config, image_metadata=metadata)
    by_position = {patch.position_id: (patch, array) for patch, array in zip(result.patches, result.patch_arrays,
                                                                              strict=True)}
    for position, image, order in (("a", two, ("DNA", "ACTIN")), ("b", three, ("ACTIN", "PHASE", "DNA"))):
        patch, array = by_position[position]
        assert patch.normalization_channels == order and array.shape[0] == len(order)
        for c in range(len(order)):  # min-max of each channel's own pixels: 0 at its minimum, 1 at its maximum
            assert array[c].min() == 0.0 and array[c].max() == 1.0
            np.testing.assert_array_equal(array[c] == 1.0, image[c] == image[c].max())
