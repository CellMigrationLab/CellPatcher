"""What a file run asks of its preprocessing run's resolver, and the cell-mode object set (layer: orchestration).

Spec ADR 0002 (the shared resolver) and ADR 0005 (one policy for binary masks, resolver
contract version 3): CellPatcher does not interpret the experiment description itself. A file
run builds one ``Request`` and reads through the ``FovSelection``s:

- the channel IDs ``extraction.split_channels`` names, which every image must have (``A_SCOPE_INCOMPLETE``
  otherwise; CP-3): each is then patched at its C index **in each image**, by the image's ``Fov.channel_ids``;
- in ``mode="cell"``, the object set by its **mask ID** (``matching.mask_id``):
  ``MaskRequest(mask_id, "objects", cell.binary_mask_policy)`` (CP-2). Its planes are read with
  preprocessing's ``RunInputs.read_objects``, the one helper that applies a recorded conversion
  (BM2, BM6): an ``instances`` mask as stored (D47: an ``instances`` mask holding 0 and 1 is one
  object); a ``binary`` mask, only with the explicit per-request policy, as its 8-connected
  components (BM1). CellPatcher labels nothing. Without the policy the resolver refuses a
  ``binary`` mask (``A_MASK_NOT_INSTANCES``); a policy on a Z-stack is ``A_VOLUME_NOT_SUPPORTED``.
  A ``parent_labels`` (grouped) mask is never an object set (``A_MASK_NOT_INSTANCES``), and a
  policy on a mask that is not ``binary`` is the resolver's own ``ValueError``, raised unchanged;
- in ``mode="random"``, the sampling mask by its mask ID as an **area**:
  ``MaskRequest(mask_id, "region")``. CellPatcher does not declare ``parent_labels_support``
  (it does not read groups), so a ``parent_labels`` mask comes back with the recorded marker
  ``conversion = "parent_labels_as_area"`` (every non-zero pixel inside, groups not kept), which
  is shown (:data:`PARENT_LABELS_AS_AREA_MESSAGE`). An ``instances`` or ``binary`` region is read
  as stored and taken as ``mask != 0`` by normalization's ``check_region`` (BM3, BM7).

The resolver's version is checked right after it answers, before anything else
(:func:`check_resolver_version`, BM10): this CellPatcher supports exactly
:data:`SUPPORTED_RESOLVER_VERSIONS`. Its ``Provenance.mask_conversions`` records are copied as is
into ``run_config.json`` under ``mask_conversions`` (BM2). Every refusal of the resolver
(``A_SCOPE_INCOMPLETE``, ``A_MASK_UNCHECKED``, ``A_MASK_NOT_INSTANCES``, ``A_VOLUME_NOT_SUPPORTED``, ...)
is raised with its code and its message unchanged before any pixel is read; no subset is taken.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from morphobricks_preprocessing.run_inputs import RunInputs
    from morphobricks_preprocessing.run_inputs_request import FovSelection, MaskSelection, Request

SUPPORTED_RESOLVER_VERSIONS = frozenset({3})
"""The resolver contract versions this CellPatcher reads (Spec ADR 0005 BM10): exactly 3, never ">= 3"."""

CONNECTED_COMPONENTS_MESSAGE = (
    "{mask_id!r}: the binary mask is converted to objects by connected_components (8-connected), as the "
    "preprocessing run's resolver recorded (run_config.json, mask_conversions); its label_id values are "
    "computed by RunInputs.read_objects, not taken from the mask."
)
"""How a ``binary`` object set read with the recorded conversion is shown, with its mask ID."""


@dataclass(frozen=True)
class ObjectMask:
    """The cell-mode object set as the resolver selected it.

    Attributes
    ----------
    mask_id
        The description's mask ID.
    source_type
        The mask type the run recorded its files under.
    representation
        ``instances`` (read as stored) or ``binary`` (read with *conversion*).
    conversion
        The recorded ``binary_mask_policy`` (``connected_components``), or None.
    selections
        Each image's ``FovSelection``, keyed by its relative path: what
        ``RunInputs.read_objects(selection, mask_id, t, z)`` reads.
    """

    mask_id: str
    source_type: str
    representation: str
    conversion: str | None = None
    selections: Mapping[str, FovSelection] = field(default_factory=dict, compare=False, repr=False)


PARENT_LABELS_AS_AREA = "parent_labels_as_area"
"""The resolver's marker for a ``parent_labels`` region read as an area (preprocessing ``PARENT_LABELS_AS_AREA``)."""

PARENT_LABELS_AS_AREA_MESSAGE = "{mask_id!r}: parent_labels used as an area; grouping not kept"
"""How a ``parent_labels`` region read as an area is shown (experiment-v1 §4.5b), with its mask ID."""


@dataclass(frozen=True)
class RegionMask:
    """The random-mode sampling mask as the resolver selected it, as an area.

    Attributes
    ----------
    mask_id
        The description's mask ID.
    representation
        Its declared representation (``instances``, ``binary`` or ``parent_labels``).
    conversion
        The resolver's recorded marker: ``"parent_labels_as_area"`` for a ``parent_labels`` mask
        (CellPatcher does not declare ``parent_labels_support``), else None.
    """

    mask_id: str
    representation: str
    conversion: str | None = None


@dataclass(frozen=True)
class ResolvedRun:
    """What the resolver selected for a file run.

    Attributes
    ----------
    object_mask
        The cell-mode object set (CP-2); None outside cell mode.
    region_mask
        The random-mode sampling mask (W3); None when no region was asked.
    issues
        Every refusal of the resolver as a preprocessing issue with its code and message (nothing is
        selected then); the caller raises them (``PreprocessingCheckError``) before any read.
    mask_conversions
        The resolver's ``Provenance.mask_conversions`` as records (``mask_id``, ``conversion``,
        ``connectivity``, ``labeller``), in its order: what ``run_config.json`` copies (BM2).
    """

    object_mask: ObjectMask | None = None
    issues: tuple = ()
    region_mask: RegionMask | None = None
    mask_conversions: tuple[dict, ...] = ()


def resolve_run(
    inputs: RunInputs,
    ids: tuple[str, ...],
    mask_id: str | None,
    policy: str | None,
    *,
    region_id: str | None = None,
) -> ResolvedRun:
    """Resolve the run's channel IDs, (with *mask_id*, cell mode) its object set and (with *region_id*) its area.

    *ids* are the channel IDs ``extraction.split_channels`` names (may be empty); *policy* is
    ``cell.binary_mask_policy``, the request field passed to the resolver as is; *region_id* is
    the random-mode sampling mask's ID, asked in the role ``region``. CellPatcher declares RGB
    support (P24) and not ``parent_labels_support``. Nothing is asked when there are no IDs and
    no mask.

    The resolver's version is checked first (:func:`check_resolver_version`). Every refusal of
    the resolver comes back in :attr:`ResolvedRun.issues`, each with its code and message:
    ``A_SCOPE_INCOMPLETE`` for a channel or mask some image lacks (an undeclared ID is present in
    0 of n), ``A_MASK_UNCHECKED`` for a mask file the mask step has not checked,
    ``A_MASK_NOT_INSTANCES`` for a ``binary`` mask without ``cell.binary_mask_policy`` or a
    ``parent_labels`` mask as objects, ``A_VOLUME_NOT_SUPPORTED`` for a conversion on a Z-stack.
    Nothing is selected then.

    Raises
    ------
    morphobricks_normalization.contracts.ResolverVersionError
        ``RESOLVER_VERSION_UNSUPPORTED``, before any read, for a resolver whose contract version
        is not in :data:`SUPPORTED_RESOLVER_VERSIONS`.
    ValueError
        The resolver's own, unchanged, for a malformed request (``cell.binary_mask_policy`` on a
        mask that is not declared ``binary``).
    """
    if not ids and mask_id is None and region_id is None:
        return ResolvedRun()
    resolved = inputs.resolve_request(_request(ids, mask_id, policy, region_id))
    check_resolver_version(resolved.provenance.resolver_version)
    conversions = tuple(conversion.as_record() for conversion in resolved.provenance.mask_conversions)
    if resolved.refused:
        issues = tuple(_refusal_issue(refusal, ids, region_id) for refusal in resolved.refusals)
        return ResolvedRun(issues=issues, mask_conversions=conversions)
    object_mask = None
    if mask_id is not None:
        object_mask = _object_mask(resolved.selections, mask_id)
    region_mask = None
    if region_id is not None:
        region_mask = _region_mask([_selected(s, region_id, "region") for s in resolved.selections])
    return ResolvedRun(object_mask, region_mask=region_mask, mask_conversions=conversions)


def check_resolver_version(found: int) -> None:
    """Refuse a resolver whose contract version this CellPatcher does not support (Spec ADR 0005 BM10).

    Called immediately after ``resolve_request`` returns, before any pixel read, output or record.

    Raises
    ------
    morphobricks_normalization.contracts.ResolverVersionError
        ``RESOLVER_VERSION_UNSUPPORTED`` from normalization's one constructor, unchanged, when
        *found* is not in :data:`SUPPORTED_RESOLVER_VERSIONS`.
    """
    if found not in SUPPORTED_RESOLVER_VERSIONS:
        from morphobricks_normalization.contracts import refuse_resolver_version

        raise refuse_resolver_version(found, SUPPORTED_RESOLVER_VERSIONS)


def _selected(selection: FovSelection, mask_id: str, role: str) -> MaskSelection:
    """Return the ``(mask_id, role)`` mask selection of one image."""
    for mask in selection.masks:
        if (mask.mask_id, mask.role) == (mask_id, role):
            return mask
    raise ValueError(f"The resolver selected no {role!r} use of the mask {mask_id!r} for {selection.relative_path}.")


def _request(ids: tuple[str, ...], mask_id: str | None, policy: str | None, region_id: str | None) -> Request:
    """Return the run's ``Request``: the channel IDs, in cell mode the object set with its policy, the region."""
    from morphobricks_preprocessing import Capabilities, MaskRequest, Request

    masks = () if mask_id is None else (MaskRequest(mask_id, "objects", policy),)
    if region_id is not None:
        masks = (*masks, MaskRequest(region_id, "region"))
    capabilities = Capabilities(rgb_support=True, parent_labels_support=False)
    return Request(channel_ids=tuple(ids), masks=masks, capabilities=capabilities)


def _refusal_issue(refusal: object, ids: tuple[str, ...], region_id: str | None = None) -> object:
    """Return one resolver refusal as a preprocessing issue: its code and message unchanged, the setting in the action.

    Spec ADR 0005 (construction and propagation): the resolver's message is passed on as is;
    what CellPatcher adds (the setting behind it and what to do) is the issue's suggested action.
    """
    from morphobricks_preprocessing.validation import error

    if refusal.subject in ids:
        setting = f"extraction.split_channels names {refusal.subject!r}"
        hint = (
            "Name channel IDs every image of the run has (the run's description lists them), or select and "
            "record an eligible subset in preprocessing."
        )
    else:
        setting = (
            f"The random-mode sampling mask (matching.mask_type) {refusal.subject!r}" if refusal.subject == region_id
            else f"The cell-mode object set (matching.mask_id) {refusal.subject!r}"
        )
        hint = _MASK_HINTS.get(refusal.code, "See the message; the run's description declares its masks.")
        if refusal.code == "A_MASK_NOT_INSTANCES" and "parent_labels" in refusal.message:
            hint = _PARENT_LABELS_HINT
    return error(refusal.code, refusal.message, list(refusal.files), f"{setting}. {hint}")


_MASK_HINTS = {
    "A_MASK_NOT_INSTANCES": (
        "A binary mask has no object identity. Set cell.binary_mask_policy='connected_components' to have the "
        "preprocessing run convert it to its 8-connected components for this run (recorded in run_config.json "
        "under mask_conversions), or use an instances mask."
    ),
    "A_VOLUME_NOT_SUPPORTED": (
        "A binary mask is converted plane by plane in 2-D only. Project the stack in morphobricks-preprocessing, "
        "or convert it to an instances mask there."
    ),
    "A_MASK_UNCHECKED": "Run preprocessing's mask step on the run's output folder, then patch again.",
    "A_SCOPE_INCOMPLETE": "Name a mask ID of the run's description that every image has, or complete the masks.",
}

_PARENT_LABELS_HINT = (
    "A parent_labels (grouped) mask is never an object set and is never converted (no binary_mask_policy applies). "
    "Name an instances mask of the run's description as the cell-mode objects (matching.mask_id)."
)


def _region_mask(selections: list[MaskSelection]) -> RegionMask:
    """Return the area the resolver selected (the same declared mask for every image), showing a recorded marker."""
    first = selections[0]
    if any((s.mask_id, s.representation, s.conversion) != (first.mask_id, first.representation, first.conversion)
           for s in selections):
        raise ValueError(f"The resolver selected the mask {first.mask_id!r} differently for different images.")
    if first.conversion not in (None, PARENT_LABELS_AS_AREA):
        raise ValueError(f"The resolver selected the area {first.mask_id!r} with conversion {first.conversion!r}, "
                         "which this CellPatcher does not apply.")
    if first.conversion is not None:
        warnings.warn(PARENT_LABELS_AS_AREA_MESSAGE.format(mask_id=first.mask_id), UserWarning, stacklevel=4)
    return RegionMask(first.mask_id, first.representation, first.conversion)


def _object_mask(selections: tuple[FovSelection, ...], mask_id: str) -> ObjectMask:
    """Return the object set the resolver selected (the same declared mask for every image), showing a conversion."""
    chosen = [_selected(s, mask_id, "objects") for s in selections]
    first = chosen[0]
    if any((s.mask_id, s.source_type, s.representation, s.conversion) != (
        first.mask_id, first.source_type, first.representation, first.conversion
    ) for s in chosen):
        raise ValueError(f"The resolver selected the mask {first.mask_id!r} differently for different images.")
    if first.conversion is not None:
        warnings.warn(CONNECTED_COMPONENTS_MESSAGE.format(mask_id=first.mask_id), UserWarning, stacklevel=4)
    by_path = {selection.relative_path: selection for selection in selections}
    return ObjectMask(first.mask_id, first.source_type, first.representation, first.conversion, by_path)
