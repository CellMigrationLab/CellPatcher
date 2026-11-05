"""Strict re-check of the scientific settings at the point a run uses them (pure; no I/O).

The config dataclasses are mutable and documented as edited in place
(``cfg.random.min_mask_fraction = 0.5``), so a check in ``__post_init__`` alone is bypassed by
an assignment after construction. The checks here are shared by ``__post_init__`` and by every
route that executes a run, and use the helpers of :mod:`cellpatcher.fractions` (never ``float()``
or ``bool()``, which turn ``True`` into 1.0 and ``"0.5"`` into 0.5).
"""

from __future__ import annotations

from cellpatcher.fractions import check_unit_fraction


def check_random_settings(random: object) -> None:
    """Refuse a ``RandomConfig`` whose ``min_mask_fraction`` is not ``None`` or a number in [0, 1]."""
    check_unit_fraction(random.min_mask_fraction, "min_mask_fraction")


def check_run_settings(config: object) -> None:
    """Re-check the scientific settings of a ``CellPatcherConfig`` before a run uses them.

    Called by ``run_patching`` and by the folder and tree routes (``open_file_run``) before any file is read, so a
    setting edited after construction is refused as at construction.
    """
    check_random_settings(config.random)

