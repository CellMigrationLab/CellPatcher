"""The masks an in-memory ``run_patching`` call can take with ``cell.binary_mask_policy`` (layer: pure).

In memory only a **boolean** mask is binary. An integer mask is instance labels, used as stored
(D47: an instances mask holding 0 and 1 is one object), exactly as a file run's mask declared
``instances``. A policy is therefore refused with an integer in-memory mask, as the file route
refuses it on an ``instances`` mask (Spec ADR 0002 AR5 and AR11 (4), ADR 0001 AQ9): there is
nothing to convert, and the policy is never ignored.

A file run reads its mask through the resolver, which labels a ``binary`` mask with the recorded
policy before patching (:mod:`cellpatcher.run_request`). Its per-image ``run_patching`` calls, and
the notebook's object scan of a file run, run inside :func:`resolved_masks`, so its
already-labelled integer masks are not refused. The in-memory object scan
(:func:`cellpatcher.scan_cell_objects`, :func:`cellpatcher.scan_cell_patch_size`) refuses as
``run_patching`` does.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cellpatcher import CellPatcherConfig

_RESOLVED: ContextVar[bool] = ContextVar("cellpatcher_resolved_masks", default=False)


@contextmanager
def resolved_masks() -> Iterator[None]:
    """Mark the ``run_patching`` calls inside the block as a file run's, on masks the resolver selected."""
    token = _RESOLVED.set(True)
    try:
        yield
    finally:
        _RESOLVED.reset(token)


def check_mask_policy(config: CellPatcherConfig, masks: Sequence, patch_region: str) -> None:
    """Refuse a ``cell.binary_mask_policy`` the cell-mode *masks* of a ``run_patching`` call cannot take.

    *patch_region* is ``normalization.patch_region`` resolved for the mode
    (:func:`cellpatcher.normalization.resolved_patch_region`).

    Raises
    ------
    ValueError
        For the policy with an in-memory integer mask (an instances mask: nothing to convert), and
        for masking by the labels the policy computes (:func:`_refuse_masking_computed_labels`).
    """
    if config.cell.binary_mask_policy is None:
        return
    for index, mask in enumerate(masks):
        refuse_policy_on_instances(config.cell.binary_mask_policy, mask, f"image {index}")
    _refuse_masking_computed_labels(config, masks, patch_region)


def _mask_dtype(mask: object) -> np.dtype:
    """Return the pixel type of *mask* without reading a lazy array's pixels."""
    dtype = getattr(mask, "dtype", None)
    return np.asarray(mask).dtype if dtype is None else np.dtype(dtype)


def refuse_policy_on_instances(policy: str | None, mask: object, where: str) -> None:
    """Refuse *policy* with an integer in-memory *mask*: an instances mask is used as stored (D47).

    *where* names the mask's image in the message (``"image 0"``). Nothing is refused without a
    policy, nor inside :func:`resolved_masks` (a file run's masks, labelled by the resolver).
    ``run_patching`` (:func:`check_mask_policy`) and the object scan
    (:func:`cellpatcher.scan_cell_objects`) share this rule.

    Raises
    ------
    ValueError
        For a policy with an integer mask, naming the fix.
    """
    if policy is None or _RESOLVED.get():
        return
    dtype = _mask_dtype(mask)
    if np.issubdtype(dtype, np.integer):
        raise ValueError(
            f"cell.binary_mask_policy={policy!r} converts a binary mask to objects, but the mask of "
            f"{where} is integer ({dtype}): an in-memory integer mask is instance labels, used as "
            "stored (D47: a mask holding 0 and 1 is one object), so there is nothing to convert. Pass a "
            "boolean mask (mask != 0) to label it by connected components, or leave "
            "cell.binary_mask_policy None to use the integer labels as stored."
        )


def _refuse_masking_computed_labels(config: CellPatcherConfig, masks: Sequence, region: str) -> None:
    """Refuse using a binary mask's values as the labels connected components computed.

    Masking (``extraction.apply_mask``) and patch-scope normalization over
    ``patch_region="mask"``/``"bbox"`` select the pixels whose mask value
    equals the patch's ``label_id``. With
    ``cell.binary_mask_policy="connected_components"`` those labels are
    computed, not the binary mask's 0/1 values: every object but the first
    would be blanked or reported as having no pixels, and the first would
    take in every touching neighbour (issue #44).
    """
    if config.cell.binary_mask_policy is None:
        return
    if config.extraction.apply_mask:
        refusal = (
            "extraction.apply_mask=True cannot mask by the labels that "
            "cell.binary_mask_policy='connected_components' computes for a binary mask: the mask holds "
            "0/1, not those labels. Label the mask first (integer instance labels), or turn apply_mask off."
        )
    elif config.normalization.enabled and config.normalization.scope == "patch" and region in ("mask", "bbox"):
        refusal = (
            f"normalization.scope='patch' with patch_region={region!r} cannot select each object's pixels by "
            "the labels that cell.binary_mask_policy='connected_components' computes for a binary mask: the "
            "mask holds 0/1, not those labels. Label the mask first (integer instance labels), or use "
            "normalization.patch_region='patch' or another scope."
        )
    else:
        return
    for mask in masks:
        dtype = _mask_dtype(mask)
        if dtype.kind == "b" or np.issubdtype(dtype, np.floating):
            raise ValueError(refusal)
