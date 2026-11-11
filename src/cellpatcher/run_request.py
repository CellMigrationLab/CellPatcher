"""What a file run asks of its preprocessing run's resolver, and the cell-mode object set (layer: orchestration).

Spec ADR 0002 (resolver contract v1): CellPatcher does not interpret the experiment description
itself. A file run builds one ``Request`` and reads through the ``FovSelection``s:

- the channel IDs ``extraction.split_channels`` names, resolved to their C index **in each image**
  (CP-3);
- in ``mode="cell"``, the object set by its **mask ID** (``matching.mask_id``):
  ``MaskRequest(mask_id, "objects", cell.binary_mask_policy)`` (CP-2). An ``instances`` mask is
  read as stored (D47: an ``instances`` mask holding 0 and 1 is one object); a ``binary`` mask is
  an object set only with the explicit per-request policy, and is then labelled plane by plane
  with the one shared labeller, :func:`cellpatcher.label_objects` (AR5). Without the policy the
  resolver refuses it (``A_MASK_NOT_INSTANCES``).

Every refusal of the resolver (``A_SCOPE_INCOMPLETE``, ``A_MASK_UNCHECKED``,
``A_MASK_NOT_INSTANCES``, ...) is raised with its code before any pixel is read; no subset is taken.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cellpatcher.object_geometry import CONNECTIVITIES, label_objects

if TYPE_CHECKING:
    from morphobricks_preprocessing.run_inputs import RunInputs
    from morphobricks_preprocessing.run_inputs_request import MaskSelection, Request


@dataclass(frozen=True)
class ObjectMask:
    """The cell-mode object set as the resolver selected it.

    Attributes
    ----------
    mask_id
        The description's mask ID.
    source_type
        The mask type the run recorded its files under (what ``RunInputs.read_mask`` takes).
    representation
        ``instances`` (read as stored) or ``binary`` (labelled with *conversion*).
    conversion
        The recorded ``binary_mask_policy`` (``connected_components``), or None.
    connectivity
        The connectivity the conversion labels with (``cell.binary_mask_connectivity``); None
        without a conversion.
    """

    mask_id: str
    source_type: str
    representation: str
    conversion: str | None = None
    connectivity: int | None = None

    def objects(self, mask: np.ndarray) -> np.ndarray:
        """Return the object labels of a mask read in a ``...YX`` layout: as stored, or labelled per plane."""
        if self.conversion is None:
            return mask
        array = np.asarray(mask)
        planes = array.reshape((-1, *array.shape[-2:]))
        labelled = [label_objects(plane, self.conversion, self.connectivity) for plane in planes]
        return np.stack(labelled).reshape(array.shape)


@dataclass(frozen=True)
class ResolvedRun:
    """What the resolver selected for a file run.

    Attributes
    ----------
    channel_indices
        For ``extraction.split_channels`` naming channel IDs: each image's C indices of those IDs,
        keyed by its path (CP-3); empty otherwise.
    object_mask
        The cell-mode object set (CP-2); None outside cell mode.
    issues
        Every refusal of the resolver as a preprocessing issue with its code (nothing is selected
        then); the caller raises them (``PreprocessingCheckError``) before any read.
    """

    channel_indices: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    object_mask: ObjectMask | None = None
    issues: tuple = ()


def resolve_run(
    inputs: RunInputs, ids: tuple[str, ...], mask_id: str | None, policy: str | None, connectivity: int
) -> ResolvedRun:
    """Resolve the run's channel IDs and (with *mask_id*, cell mode) its object set, before any read.

    *ids* are the channel IDs ``extraction.split_channels`` names (may be empty); *policy* and
    *connectivity* are ``cell.binary_mask_policy`` and ``cell.binary_mask_connectivity``.
    CellPatcher declares RGB support (P24). Nothing is asked when there are no IDs and *mask_id*
    is None.

    Every refusal of the resolver comes back in :attr:`ResolvedRun.issues`, each with its code:
    ``A_SCOPE_INCOMPLETE`` for a channel or mask some image lacks (an undeclared ID is present in 0
    of n), ``A_MASK_UNCHECKED`` for a mask file the mask step has not checked,
    ``A_MASK_NOT_INSTANCES`` for a ``binary`` mask without ``cell.binary_mask_policy``. Nothing is
    selected then.

    Raises
    ------
    ValueError
        For ``cell.binary_mask_policy`` on a mask that is not declared ``binary`` (nothing to convert).
    """
    if not ids and mask_id is None:
        return ResolvedRun()
    try:
        resolved = inputs.resolve_request(_request(ids, mask_id, policy))
    except ValueError as exc:  # a malformed request: the policy on a mask that is not binary
        if policy is None or mask_id is None:
            raise
        raise ValueError(
            f"cell.binary_mask_policy={policy!r} converts a binary mask to objects, but "
            f"{exc}. Leave cell.binary_mask_policy None for an instances mask (it is used as stored)."
        ) from None
    if resolved.refused:
        return ResolvedRun(issues=tuple(_refusal_issue(refusal, ids) for refusal in resolved.refusals))
    indices = {}
    if ids:
        indices = {
            selection.relative_path: tuple(sorted(selection.index_of(channel) for channel in ids))
            for selection in resolved.selections
        }
    if mask_id is None:
        return ResolvedRun(indices)
    return ResolvedRun(indices, _object_mask([s.masks[0] for s in resolved.selections], connectivity))


def _request(ids: tuple[str, ...], mask_id: str | None, policy: str | None) -> Request:
    """Return the run's ``Request``: the channel IDs and, in cell mode, the object set with its policy."""
    from morphobricks_preprocessing import Capabilities, MaskRequest, Request

    masks = () if mask_id is None else (MaskRequest(mask_id, "objects", policy),)
    return Request(channel_ids=tuple(ids), masks=masks, capabilities=Capabilities(rgb_support=True))


def _refusal_issue(refusal: object, ids: tuple[str, ...]) -> object:
    """Return one resolver refusal as a preprocessing issue with its code, naming the setting behind it."""
    from morphobricks_preprocessing.validation import error

    if refusal.subject in ids:
        setting = f"extraction.split_channels names {refusal.subject!r}"
        hint = (
            "Name channel IDs every image of the run has (the run's description lists them), or select and "
            "record an eligible subset in preprocessing."
        )
    else:
        setting = f"The cell-mode object set (matching.mask_id) {refusal.subject!r}"
        hint = _MASK_HINTS.get(refusal.code, "See the message; the run's description declares its masks.")
    return error(refusal.code, f"{setting}: {refusal.message}", list(refusal.files), hint)


_MASK_HINTS = {
    "A_MASK_NOT_INSTANCES": (
        "A binary mask has no object identity. Set cell.binary_mask_policy='connected_components' to label its "
        "connected components for this run (recorded in run_config.json), or use an instances mask."
    ),
    "A_MASK_UNCHECKED": "Run preprocessing's mask step on the run's output folder, then patch again.",
    "A_SCOPE_INCOMPLETE": "Name a mask ID of the run's description that every image has, or complete the masks.",
}


def _object_mask(selections: list[MaskSelection], connectivity: int) -> ObjectMask:
    """Return the object set the resolver selected (the same declared mask for every image)."""
    first = selections[0]
    if any((s.mask_id, s.source_type, s.representation, s.conversion) != (
        first.mask_id, first.source_type, first.representation, first.conversion
    ) for s in selections):
        raise ValueError(f"The resolver selected the mask {first.mask_id!r} differently for different images.")
    if first.conversion is None:
        return ObjectMask(first.mask_id, first.source_type, first.representation)
    warnings.warn(
        f"The binary mask {first.mask_id!r} is labelled by {CONNECTIVITIES[connectivity]} components "
        f"(cell.binary_mask_policy={first.conversion!r}): its label_id values are computed, not taken from the mask.",
        UserWarning,
        stacklevel=4,
    )
    return ObjectMask(first.mask_id, first.source_type, first.representation, first.conversion, connectivity)
