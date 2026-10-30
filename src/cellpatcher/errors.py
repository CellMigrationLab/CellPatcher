"""Typed refusals that callers can catch per image or per plane.

CellPatcher refuses rather than guesses. Most refusals are configuration or
input-layout errors that should stop a run. The ones defined here are
refusals about the *data of one image plane* (a sparse sampling mask, labels
CellPatcher cannot store), which a caller such as MorphoFeature may want to
record as a refused plane instead of aborting the whole run.

Every class derives from :class:`CellPatcherRefusal`, which derives from
``ValueError``, so existing ``except ValueError`` handlers keep working.
"""


class CellPatcherRefusal(ValueError):
    """Base class of CellPatcher's typed, per-plane data refusals.

    Catch this to turn a refusal into a per-plane record; never treat a
    caught refusal as a measurement (no fill, no zero, no fallback).
    """


class SamplingExhaustedError(CellPatcherRefusal):
    """Random sampling ran out of attempts before collecting ``n_patches`` regions.

    Raised by :class:`cellpatcher.RandomPatchStrategy` when a sampling mask
    (with ``min_mask_fraction``) is too sparse. Attributes: ``requested``,
    ``collected`` and ``attempts``.
    """

    def __init__(
        self,
        message: str,
        *,
        requested: int | None = None,
        collected: int | None = None,
        attempts: int | None = None,
    ) -> None:
        # Keyword defaults keep the error picklable (BaseException.__reduce__
        # rebuilds it from the message, then restores these attributes), so
        # it crosses multiprocessing workers intact.
        super().__init__(message)
        self.requested = requested
        self.collected = collected
        self.attempts = attempts


class LabelRangeError(CellPatcherRefusal):
    """A mask holds a label CellPatcher cannot store as an int32 instance label.

    Raised by :class:`cellpatcher.CellSegmentationStrategy` (and so by
    ``scan_cell_objects``) for negative labels or labels above the int32
    maximum. The mask is refused, never wrapped, clipped or relabelled.
    Attribute: ``label`` (the offending minimum or maximum value).
    """

    def __init__(self, message: str, *, label: int | None = None) -> None:
        super().__init__(message)
        self.label = label
