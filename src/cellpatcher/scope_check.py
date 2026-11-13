"""Refuse a normalization group a channel of which only some planes have, from the table alone (layer: pure).

Issue #107: at a CSV scope (``biological_repeat`` or ``image``) the normalization brick's ``fit`` refuses a
group whose planes do not all have every channel some plane of it has (Spec ``A_SCOPE_INCOMPLETE``): a channel
is never fitted over part of a group nor filled in. Whether a group is complete is decided by the rows of
``plane_histograms.csv``, which exist before CellPatcher reads any image, so a folder or tree run checks every
group of the planes it patches when the run is opened (:func:`cellpatcher.file_runs.open_file_run`), before any
read or write, as the run's other consumers (the resolver, MorphoFeature, MorphoRelate, fiberfeatures) do.

Nothing is fitted here: a group with no finite pixels or zero spread is still refused by the patching, and only
when a kept patch uses it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import morphobricks_normalization as mbn

_LISTED = 5
"""The planes named per channel, then how many more (as the brick's own refusal)."""


def refuse_incomplete_scopes(
    records: Iterable[mbn.PlaneRecord],
    settings: mbn.NormalizationSettings,
    repeat_groups: Mapping[tuple[str, str], str] | None,
    planes: Iterable[tuple[str, int | None, int | None]],
) -> None:
    """Refuse the groups of *planes* with a channel only some of their planes have (``A_SCOPE_INCOMPLETE``).

    Parameters
    ----------
    records
        Every row of the run's ``plane_histograms.csv``.
    settings
        The brick's settings of the run (its scope and ``per_time_point`` choose the groups).
    repeat_groups
        At the ``biological_repeat`` scope, the repeat group of every condition and biological repeat, from the
        run's description; None otherwise.
    planes
        The planes the run patches, as ``(source_relative_path, z, t)``; None for an axis the image does not have.

    Raises
    ------
    ValueError
        Naming every such group and channel, with how many of the group's planes have it and which lack it.
    """
    if settings.image_type == "rgb":  # R, G and B are pooled together: the brick refuses no partial channel
        return
    records = tuple(records)
    index, _ = mbn.scope_groups(records, settings, repeat_groups)  # its issues are warned by the patching
    by_image = defaultdict(list)
    for row in records:
        by_image[row.source_relative_path].append(row)
    keys = {
        mbn.group_key(row, settings, repeat_groups)
        for path, z, t in planes
        for row in by_image[path]
        if z in (None, row.z) and t in (None, row.t)
    }
    failures = []
    for key in sorted(keys):
        rows = index[key]
        for channel in sorted({channel for row in rows for channel in row.channels}):
            lacking = [row.plane_id for row in rows if not row.has_channel(channel)]
            if lacking:
                failures.append(
                    f"{mbn.group_name(key)}: A_SCOPE_INCOMPLETE: channel {channel} is present in "
                    f"{len(rows) - len(lacking)} of {len(rows)} planes (absent from {', '.join(lacking[:_LISTED])}"
                    + (f" and {len(lacking) - _LISTED} more)" if len(lacking) > _LISTED else ")")
                )
    if failures:
        raise ValueError(
            f"Normalization parameters cannot be computed for {len(failures)} group channel(s):\n"
            + "\n".join(f"  - {failure}" for failure in failures)
            + "\nA channel is never fitted over part of a group nor filled in: choose a scope whose groups have it "
            "in every plane (the image scope), or leave the channel out (contract §7)."
        )
