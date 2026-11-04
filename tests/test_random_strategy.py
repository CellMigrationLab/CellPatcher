import numpy as np
import pytest

from cellpatcher.config import RandomConfig
from cellpatcher.strategies.random import RandomPatchStrategy


def test_random_strategy_returns_n_patches_and_size_and_bounds():
    s = RandomPatchStrategy(RandomConfig(patch_size=16, n_patches=10, seed=0))
    patches = s.generate_regions(image_shape=(64, 64), image_id="img")
    assert len(patches) == 10
    for p in patches:
        assert p.height == 16 and p.width == 16
        assert 0 <= p.y0 < p.y1 <= 64
        assert 0 <= p.x0 < p.x1 <= 64


def test_random_strategy_deterministic_with_same_seed():
    cfg = RandomConfig(patch_size=8, n_patches=5, seed=123)
    p1 = RandomPatchStrategy(cfg).generate_regions(image_shape=(32, 32))
    p2 = RandomPatchStrategy(cfg).generate_regions(image_shape=(32, 32))
    assert [(r.y0, r.x0) for r in p1] == [(r.y0, r.x0) for r in p2]


def test_random_strategy_different_seed_changes_output():
    p1 = RandomPatchStrategy(RandomConfig(patch_size=8, n_patches=5, seed=1)).generate_regions(image_shape=(32, 32))
    p2 = RandomPatchStrategy(RandomConfig(patch_size=8, n_patches=5, seed=2)).generate_regions(image_shape=(32, 32))
    assert [(r.y0, r.x0) for r in p1] != [(r.y0, r.x0) for r in p2]


def test_random_strategy_seed_none_returns_valid_patches():
    cfg = RandomConfig(patch_size=4, n_patches=3, seed=None)
    p1 = RandomPatchStrategy(cfg).generate_regions(image_shape=(20, 20))
    p2 = RandomPatchStrategy(cfg).generate_regions(image_shape=(20, 20))
    assert len(p1) == 3
    assert len(p2) == 3
    for p in p1 + p2:
        assert 0 <= p.y0 < p.y1 <= 20
        assert 0 <= p.x0 < p.x1 <= 20


def test_random_strategy_patch_size_larger_than_image_raises():
    with pytest.raises(ValueError):
        RandomPatchStrategy(RandomConfig(patch_size=40, n_patches=3)).generate_regions(image_shape=(20, 20))


def test_random_strategy_mask_restricts_sampling():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:15, 10:15] = True
    s = RandomPatchStrategy(RandomConfig(patch_size=6, n_patches=5, seed=0))
    patches = s.generate_regions(image_shape=(30, 30), labels=mask)
    for p in patches:
        assert np.any(mask[p.y0:p.y1, p.x0:p.x1])


def test_random_strategy_mask_shape_mismatch_raises():
    s = RandomPatchStrategy(RandomConfig(patch_size=6, n_patches=5, seed=0))
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(30, 30), labels=np.zeros((20, 20), dtype=bool))


def test_random_strategy_impossible_mask_raises():
    mask = np.zeros((20, 20), dtype=bool)
    s = RandomPatchStrategy(RandomConfig(patch_size=5, n_patches=3, seed=0))
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(20, 20), labels=mask)


def test_random_strategy_patch_ids_deterministic():
    s = RandomPatchStrategy(RandomConfig(patch_size=8, n_patches=3, seed=0))
    patches = s.generate_regions(image_shape=(32, 32), image_id="img")
    # The last level of patch_id: each draw's own rectangle (cellpatcher.ids).
    assert [p.patch_id for p in patches] == [f"random-y{p.y0}-x{p.x0}-h8-w8" for p in patches]
    assert [p.patch_id for p in patches] == [
        p.patch_id for p in s.generate_regions(image_shape=(32, 32), image_id="other")
    ]


def test_random_strategy_touches_border_when_on_edge():
    s = RandomPatchStrategy(RandomConfig(patch_size=10, n_patches=50, seed=42))
    patches = s.generate_regions(image_shape=(20, 20))
    assert any(p.touches_border for p in patches)


def test_random_strategy_sampling_with_replacement_allows_duplicates():
    s = RandomPatchStrategy(RandomConfig(patch_size=4, n_patches=3, seed=0))
    patches = s.generate_regions(image_shape=(4, 4))
    assert len(patches) == 3
    assert all((p.y0, p.y1, p.x0, p.x1) == (0, 4, 0, 4) for p in patches)


# ---------------------------------------------------------------------------
# Additional strategy edge cases
# ---------------------------------------------------------------------------

def test_random_strategy_non_2d_mask_raises():
    s = RandomPatchStrategy(RandomConfig(patch_size=4, n_patches=2, seed=0))
    with pytest.raises(ValueError):
        s.generate_regions(image_shape=(20, 20), labels=np.zeros((2, 20, 20), dtype=bool))
