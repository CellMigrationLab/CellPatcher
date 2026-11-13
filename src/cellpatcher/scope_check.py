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
        :func:`group_failures` with the brick's ``A_SCOPE_INCOMPLETE`` refusal of every such group, the message
        the patching would give.
    """
    # The RGB exemption and the channel-presence test below copy the brick's private _record_items and
    # _lacking_planes, until it offers a public check-only helper (morphobricks-normalization#96). The refusal
    # itself is the brick's: fit raises ChannelScopeIncompleteError on a group's rows before pooling any pixel.
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
        channels = {channel for row in rows for channel in row.channels}
        if all(row.has_channel(channel) for row in rows for channel in channels):
            continue  # complete: never fitted here, so no-pixel and zero-spread refusals stay the patching's
        try:
            mbn.fit(rows, settings)
        except mbn.ChannelScopeIncompleteError as exc:
            failures.append(f"{mbn.group_name(key)}: {exc}")  # as Normalizer._group_parameters words it
    if failures:
        raise group_failures(failures)


def group_failures(failures: Iterable[str]) -> ValueError:
    """Return the refusal of the groups whose parameters cannot be computed, one ``"<group>: <why>"`` each (§7).

    The one wording of this refusal, whether the run is opened (:func:`refuse_incomplete_scopes`) or patched
    (:class:`cellpatcher.normalization.Normalizer`).
    """
    failures = sorted(set(failures))
    return ValueError(
        f"Normalization parameters cannot be computed for {len(failures)} group(s):\n"
        + "\n".join(f"  - {failure}" for failure in failures)
        + "\nChoose a larger scope or another method (contract §7)."
    )
