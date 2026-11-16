"""The object sets an in-memory call takes (layer: pure).

Spec ADR 0005 BM4: in memory (``run_patching``, ``scan_cell_objects``, ``scan_cell_patch_size``)
an object set is a plain NumPy array of integer instance labels, used as stored (D47: an
instances mask holding 0 and 1 is one object). A boolean, float or other array, and anything
that is not a plain NumPy array (a list, None, a ``numpy.ma.MaskedArray``, a memory map), is
refused with ``A_MASK_NOT_INSTANCES`` by preprocessing's ``masks.check_object_set``, the one
constructor of that message. Nothing in memory converts or labels a mask, and
``cell.binary_mask_policy`` is refused there: it is only the request field of a described file
run, passed to the preprocessing run's resolver, whose ``RunInputs.read_objects`` applies the
recorded conversion (BM2, BM6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellpatcher.config import CellPatcherConfig

IN_MEMORY_POLICY_MESSAGE = (
    "cell.binary_mask_policy={policy!r} is set, but {where} takes its object sets in memory, where "
    "nothing is converted (Spec ADR 0005 BM4): binary_mask_policy is only the request field of a "
    "described file run (common.preprocessing_run), where the preprocessing run's resolver records the "
    "conversion and RunInputs.read_objects applies it. Pass integer instance labels with "
    "cell.binary_mask_policy None, or run the files through a described preprocessing run with the policy."
)
"""The refusal of ``cell.binary_mask_policy`` in an in-memory call, with the policy and the call."""


def refuse_in_memory_policy(policy: str | None, where: str) -> None:
    """Refuse ``cell.binary_mask_policy`` in the in-memory call *where* (``"run_patching"``).

    Raises
    ------
    ValueError
        When *policy* is not None (:data:`IN_MEMORY_POLICY_MESSAGE`).
    """
    if policy is not None:
        raise ValueError(IN_MEMORY_POLICY_MESSAGE.format(policy=policy, where=where))


def check_in_memory_objects(config: CellPatcherConfig, masks: object) -> None:
    """Refuse ``run_patching``'s ``cell.binary_mask_policy`` and every cell-mode mask that is not an integer array.

    *masks* are the call's ``masks`` argument (one mask, or a list or tuple of one per image); only
    ``mode="cell"`` reads them as object sets, and None is left to the call's own refusal. Each is
    named ``the mask of image_<index>`` in the message, as the cell strategy names it.

    Raises
    ------
    ValueError
        For ``cell.binary_mask_policy`` (:func:`refuse_in_memory_policy`).
    morphobricks_preprocessing.mask_rules.MaskNotInstancesError
        ``A_MASK_NOT_INSTANCES`` from ``check_object_set``, its message unchanged, before any patch.
    """
    from morphobricks_preprocessing.masks import check_object_set

    refuse_in_memory_policy(config.cell.binary_mask_policy, "run_patching")
    if config.mode != "cell" or masks is None:
        return
    for index, mask in enumerate(masks if isinstance(masks, (list, tuple)) else [masks]):
        check_object_set(mask, name=f"the mask of image_{index:04d}")
