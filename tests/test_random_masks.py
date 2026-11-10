"""A sampling mask changes random rectangles only when it excludes draws (matrix finding F8).

Random mode draws ``(y0, x0)`` from one seeded stream that depends on the seed, the FOV key
and plane, the plane's shape and the patch size only (``cellpatcher.strategies.random``). With
an explicit ``min_mask_fraction=0`` a mask excludes nothing, so the rectangles are those of a
run without it; with no fraction (the default) a draw needs one mask pixel, as before F8; with
a positive fraction the patches are the stream's draws that reach the
fraction, in order. MorphoFeature cuts random sets plane by plane with ``run_patching``
(``patch_sets.cut_patch_set``, MorphoFeature f167a39): the last tests replay that call and
compare it with a CellPatcher file run, with and without a masks folder.
"""

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, RandomConfig
from cellpatcher.errors import SamplingExhaustedError
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import run_patching
from cellpatcher.strategies.random import ZERO_FRACTION_WARNING, RandomPatchStrategy
from tests.helpers import preprocess, write_tiff

SHAPE = (48, 64)


def _masks():
    """Masks of every kind: empty, sparse, a block, full, and noise."""
    rng = np.random.default_rng(1)
    block = np.zeros(SHAPE, dtype=bool)
    block[5:30, 20:50] = True
    sparse = np.zeros(SHAPE, dtype=np.uint16)
    sparse[40, 3] = 7
    return {
        "empty": np.zeros(SHAPE, dtype=bool),
        "sparse": sparse,
        "block": block,
        "full": np.ones(SHAPE, dtype=bool),
        "noise": rng.random(SHAPE) < 0.3,
    }


def _rectangles(regions):
    return [(r.patch_id, r.y0, r.y1, r.x0, r.x1) for r in regions]


def _draw(config, labels=None):
    return RandomPatchStrategy(config).generate_regions(image_shape=SHAPE, labels=labels)


def _stream(config, count):
    """The first *count* draws of the stream: the same seed, shape and size without a mask."""
    return _draw(RandomConfig(patch_size=config.patch_size, n_patches=count, seed=config.seed))


# --- min_mask_fraction == 0: the mask excludes nothing --------------------------------------------


@pytest.mark.parametrize("kind", sorted(_masks()))
@pytest.mark.parametrize("seed", [0, 7, 2**40 + 3])
def test_a_mask_at_fraction_zero_gives_the_rectangles_without_it(kind, seed):
    config = RandomConfig(patch_size=8, n_patches=25, seed=seed, min_mask_fraction=0)
    with pytest.warns(UserWarning, match="excludes no rectangle"):
        masked = _draw(config, _masks()[kind])
    assert _rectangles(masked) == _rectangles(_draw(config))


@pytest.mark.parametrize("scope", ["plane_channel", "plane", "position"])
def test_run_patching_at_fraction_zero_is_the_same_with_and_without_masks(scope):
    images = [np.zeros(SHAPE, dtype=np.uint16)] * 2
    metadata = [{"position_id": "a", "z": 1}, {"position_id": "b", "t": 2}]
    random = RandomConfig(patch_size=8, n_patches=9, seed=4, seed_scope=scope, min_mask_fraction=0)
    config = CellPatcherConfig(mode="random", random=random)
    without = run_patching(images=images, config=config, image_metadata=metadata).patches
    with pytest.warns(UserWarning, match="excludes no rectangle"):
        masked = run_patching(images=images, masks=[_masks()["block"]] * 2, config=config, image_metadata=metadata)
    assert _rectangles(masked.patches) == _rectangles(without)


def test_an_empty_mask_at_fraction_zero_is_not_exhausted():
    """Only a filtering mask can run out of attempts."""
    config = RandomConfig(patch_size=8, n_patches=3, seed=0, min_mask_fraction=0)
    with pytest.warns(UserWarning):
        assert len(_draw(config, _masks()["empty"])) == 3
    with pytest.raises(SamplingExhaustedError):
        _draw(RandomConfig(patch_size=8, n_patches=3, seed=0, min_mask_fraction=0.1), _masks()["empty"])


def test_a_mask_at_fraction_zero_is_still_checked():
    config = RandomConfig(patch_size=8, n_patches=3, seed=0, min_mask_fraction=0)
    with pytest.raises(ValueError, match="must match image YX shape"):
        _draw(config, np.ones((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="must match image YX shape"):
        _draw(RandomConfig(patch_size=8, n_patches=3, seed=0), np.ones((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="must be 2D"):
        _draw(config, np.ones((2, *SHAPE), dtype=bool))


def test_no_mask_gives_no_warning(recwarn):
    _draw(RandomConfig(patch_size=8, n_patches=3, seed=0, min_mask_fraction=0))
    _draw(RandomConfig(patch_size=8, n_patches=3, seed=0), _masks()["block"])  # the default filters
    assert not [w for w in recwarn if str(w.message) == ZERO_FRACTION_WARNING]


# --- min_mask_fraction > 0: the stream's draws that reach the fraction ----------------------------


@pytest.mark.parametrize("kind", ["block", "noise", "full"])
@pytest.mark.parametrize("fraction", [0.1, 0.5, 1.0])
def test_a_positive_fraction_keeps_the_streams_draws_that_reach_it_in_order(kind, fraction):
    mask = _masks()[kind].astype(bool)
    config = RandomConfig(patch_size=8, n_patches=12, seed=5, min_mask_fraction=fraction)
    try:
        kept = _draw(config, mask)
    except SamplingExhaustedError:
        assert kind == "noise" and fraction >= 0.5  # 30 % noise rarely fills a whole patch
        return
    stream = _stream(config, 1000)
    expected = [r for r in stream if mask[r.y0:r.y1, r.x0:r.x1].mean() >= fraction][:12]
    assert _rectangles(kept) == _rectangles(expected)
    assert _rectangles(_draw(config, mask)) == _rectangles(kept)  # deterministic
    assert all(mask[r.y0:r.y1, r.x0:r.x1].mean() >= fraction for r in kept)


def test_one_pixel_fraction_is_the_former_any_overlap_rule():
    """Migration (F8): 1 / patch_size**2 keeps exactly the draws with at least one mask pixel."""
    mask = _masks()["sparse"]
    config = RandomConfig(patch_size=8, n_patches=6, seed=3, min_mask_fraction=1 / 8**2)
    stream = _stream(config, 1000)
    expected = [r for r in stream if np.any(mask[r.y0:r.y1, r.x0:r.x1])][:6]
    assert _rectangles(_draw(config, mask)) == _rectangles(expected)


@pytest.mark.parametrize("kind", ["sparse", "block", "noise", "empty"])
def test_no_fraction_keeps_the_former_any_overlap_rule(kind):
    """The default (None) keeps exactly the draws with a mask pixel, as main did before F8."""
    mask = _masks()[kind].astype(bool)
    config = RandomConfig(patch_size=8, n_patches=6, seed=3)
    assert config.min_mask_fraction is None
    if kind == "empty":
        with pytest.raises(SamplingExhaustedError):
            _draw(config, mask)
        return
    expected = [r for r in _stream(config, 1000) if mask[r.y0:r.y1, r.x0:r.x1].any()][:6]
    assert _rectangles(_draw(config, mask)) == _rectangles(expected)
    one_pixel = RandomConfig(patch_size=8, n_patches=6, seed=3, min_mask_fraction=1 / 8**2)
    assert _rectangles(_draw(one_pixel, mask)) == _rectangles(expected)


def test_a_positive_fraction_differs_from_a_run_without_mask():
    config = RandomConfig(patch_size=8, n_patches=12, seed=5, min_mask_fraction=0.5)
    assert _rectangles(_draw(config, _masks()["block"])) != _rectangles(_stream(config, 12))


# --- parity with MorphoFeature's cut_patch_set -----------------------------------------------------


def _experiment(tmp_path):
    """Two YX images of ctrl/rep1 and a masks folder holding two binary mask types (a block, noise)."""
    rng = np.random.default_rng(2)
    root, masks = tmp_path / "experiment", tmp_path / "masks"
    for name in ("a", "b"):
        write_tiff(root / "ctrl" / "rep1" / f"{name}.tif", rng.integers(0, 900, SHAPE).astype(np.uint16), "YX")
        for kind, mask in (("tissue", _masks()["block"]), ("nuclei", _masks()["noise"])):
            write_tiff(masks / kind / "ctrl" / "rep1" / f"{name}.tif", mask.astype(np.uint8), "YX")
    binary = {"tissue": "binary", "nuclei": "binary"}  # 0/1 masks: a region and many separate objects
    return root, masks, preprocess(root, masks=masks, output=tmp_path / "run", representations=binary)


def _file_run(root, run, random, masks=None):
    """``(patch_id, bounds)`` of a CellPatcher file run, with the tissue masks when *masks* is given."""
    config = CellPatcherConfig(mode="random", random=random)
    config.common.preprocessing_run = str(run)
    config.matching.mask_type = "tissue" if masks is not None else None
    result = run_patching_pipeline(root, config=config, mask_folder_path=masks)
    return sorted((p.patch_id, (p.y0, p.y1, p.x0, p.x1)) for p in result.patches)


def _cut_patch_set(random, sampling_mask=None):
    """MorphoFeature's call (patch_sets.cut_patch_set and _config at f167a39), one plane per FOV."""
    found = []
    for name in ("a", "b"):
        config = CellPatcherConfig(mode="random", random=random)
        config.common = CommonConfig(
            image_axes="YX", mask_axes="YX", condition="ctrl", biological_repeat="rep1", technical_repeat="single"
        )
        masks = None if sampling_mask is None else [np.asarray(sampling_mask) != 0]
        result = run_patching(
            images=[np.broadcast_to(np.uint8(0), SHAPE)], masks=masks, config=config,
            image_metadata=[{"position_id": name}], fraction_masks=[{}],
        )
        found += [(p.patch_id, (p.y0, p.y1, p.x0, p.x1)) for p in result.patches]
    return sorted(found)


@pytest.mark.filterwarnings("ignore:random mode. min_mask_fraction is 0")
@pytest.mark.parametrize("scope", ["position", "plane"])
def test_morphofeature_cuts_the_file_runs_random_patches_with_and_without_masks(tmp_path, scope):
    root, masks, run = _experiment(tmp_path)
    zero = RandomConfig(patch_size=8, n_patches=10, seed=7, seed_scope=scope, min_mask_fraction=0)
    expected = _cut_patch_set(zero)
    assert len(expected) == 20 and all(i.startswith("ctrl/rep1/single/") for i, _ in expected)
    assert _file_run(root, run, zero) == expected
    assert _file_run(root, run, zero, masks) == expected  # a masks folder changes nothing at 0
    assert _cut_patch_set(zero, _masks()["block"]) == expected

    default = RandomConfig(patch_size=8, n_patches=10, seed=7, seed_scope=scope)
    assert _file_run(root, run, default, masks) == _cut_patch_set(default, _masks()["block"]) != expected
    half = RandomConfig(patch_size=8, n_patches=10, seed=7, seed_scope=scope, min_mask_fraction=0.5)
    filtered = _cut_patch_set(half, _masks()["block"])
    assert _file_run(root, run, half, masks) == filtered != expected


@pytest.mark.parametrize("value", [False, True, np.True_, np.False_, "0", "0.5", np.nan, np.inf, -np.inf, -0.1, 1.5])
def test_min_mask_fraction_refuses_booleans_strings_and_non_finite_values(value):
    """``True`` is not 1, ``False`` is not an explicit 0, and a string or NaN is not a fraction."""
    with pytest.raises(ValueError, match="min_mask_fraction"):
        RandomConfig(min_mask_fraction=value)


@pytest.mark.parametrize("value", [None, 0, 0.0, 1, 0.5, np.float64(0.25), np.int64(1)])
def test_min_mask_fraction_accepts_none_and_numbers_in_zero_to_one(value):
    assert RandomConfig(min_mask_fraction=value).min_mask_fraction is value
