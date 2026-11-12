"""One patch's normalization by channel ID: the parameters, the patch's C order, applying them (layer: pure).

Normalization v0.5.0 (Spec ADR 0003 HV2) names every channel by its canonical channel ID: the brick's
``Parameters`` hold the IDs in sorted order and ``Parameters.apply`` takes the ID of each entry along the
channel axis. A patch keeps its own C order (``normalization_channels``, a ``patches.csv`` column), so
each channel is normalized with the parameters of its ID, never by its position. Moved out of
:mod:`cellpatcher.normalization`, which re-exports these names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from morphobricks_normalization.channels import check_channel_ids
from morphobricks_normalization.rgb import RGB_CHANNELS

from cellpatcher.axes import normalize_axes

if TYPE_CHECKING:
    from cellpatcher.types import PatchRegion

NORMALIZATION_CHANNELS_COLUMN = "normalization_channels"
"""The ``patches.csv`` column of a patch's channel IDs in its C order, ``;``-separated (the parameters' are sorted)."""


def channel_order_from_row(row: Mapping) -> tuple[str, ...] | None:
    """Return a ``patches.csv`` row's channel IDs in its C order (``normalization_channels``); None if blank."""
    text = row.get(NORMALIZATION_CHANNELS_COLUMN)
    if text in (None, ""):
        return None
    return tuple(str(text).split(";"))


def refuse_earlier_parameter_row(row: Mapping) -> bool:
    """Return whether a ``patches.csv`` row has parameter columns, refusing those of an earlier table (no conversion).

    Rows written before normalization v0.5.0 (CellPatcher 0.4.0 or earlier) are refused with what to do, never
    converted (Spec ADR 0002 AR10): the ``;``-separated ``normalization_offset``/``normalization_scale``, and
    ``Normalization_Offset_C<i>``/``Normalization_Scale_C<i>`` without ``normalization_parameters_version``.

    Raises
    ------
    ValueError
        For a row of an earlier table.
    """
    prefixes = ("Normalization_Offset_", "Normalization_Scale_")
    pairs = any(str(name).startswith(prefixes) and value not in (None, "") for name, value in row.items())
    if not pairs and row.get("normalization_offset"):
        raise ValueError(
            "The row has the ;-separated normalization_offset/normalization_scale columns of a patches.csv written "
            "before the per-channel columns; such rows are not read (no conversion). Run CellPatcher again."
        )
    if pairs and row.get("normalization_parameters_version") in (None, ""):
        raise ValueError(
            "The row has Normalization_Offset_/Normalization_Scale_ columns without normalization_parameters_version: "
            "a patches.csv written before normalization v0.5.0 (CellPatcher 0.4.0 or earlier), whose columns name "
            "channels by C index. Such rows are not read (no conversion): run CellPatcher again on a current "
            "preprocessing run."
        )
    return pairs


@dataclass(frozen=True)
class PatchNormalization:
    """The normalization of one patch: the brick's ``Parameters``, one offset and scale per channel.

    Attributes
    ----------
    parameters
        ``morphobricks_normalization.Parameters``: the method name (e.g.
        ``"percentile 1-99.8"``), ``normalization.scope``, the group key
        (contract §7; None for the patch scope until :func:`finish_patch_groups`
        knows the final ``patch_id``), exactness, and the offsets and scales
        of the image channels the patch holds, by channel ID in sorted order.
    channels
        The patch's channel IDs in its own C order (each entry along its C axis), or
        None when not known (then the parameters must hold one channel).
    """

    parameters: object
    channels: tuple[str, ...] | None = None

    @property
    def method(self) -> str:
        """Return the method name of the parameters (e.g. ``"percentile 1-99.8"``)."""
        return self.parameters.method

    @property
    def scope(self) -> str:
        """Return the scope of the parameters."""
        return self.parameters.scope

    @property
    def group(self) -> str | None:
        """Return the group of the parameters (``normalization_group``)."""
        return self.parameters.group

    @property
    def exact(self) -> bool:
        """Return whether the parameters are exact (not a histogram approximation)."""
        return self.parameters.exact

    @classmethod
    def from_patch(cls, patch: object) -> PatchNormalization | None:
        """Return the normalization recorded on a patch (``normalization_parameters``), or None when there is none."""
        parameters = getattr(patch, "normalization_parameters", None)
        return None if parameters is None else cls(parameters, getattr(patch, "normalization_channels", None))

    def channel_order(self, count: int) -> tuple[str, ...]:
        """Return the channel ID of each of *count* entries along a C axis, refusing a count that does not fit."""
        order = self.channels
        if order is None:
            if self.parameters.channel_count != 1 or count != 1:
                raise ValueError(
                    f"The normalization parameters have channel(s) {', '.join(self.parameters.channels)}, but the "
                    "patch's channel order (normalization_channels) is not recorded, so no channel can be matched "
                    "to its parameters by ID."
                )
            return tuple(self.parameters.channels)
        if len(order) != count:
            raise ValueError(
                f"The patch has {count} channels but {len(order)} channel IDs ({', '.join(order)}) of normalization "
                "parameters."
            )
        return order

    def unapply(self, array: np.ndarray, axes: str) -> np.ndarray:
        """Map normalized values back to raw units per channel, as float32: ``values * scale + offset``.

        For RGB parameters the three channels share one offset and scale, so
        this is the same mapping for each.
        """
        array = np.asarray(array)
        axes = normalize_axes(axes, array.ndim)
        parameters = self.parameters
        offsets, scales = parameters.offset, parameters.scale
        if "C" not in axes:
            position = parameters.channels.index(self._single_channel())
            return (array.astype(np.float64) * scales[position] + offsets[position]).astype(np.float32)
        channel_axis = axes.index("C")
        out = np.empty(array.shape, dtype=np.float32)
        for channel, channel_id in enumerate(self.channel_order(array.shape[channel_axis])):
            position = parameters.channels.index(channel_id)
            index = (slice(None),) * channel_axis + (channel,)
            out[index] = array[index].astype(np.float64) * scales[position] + offsets[position]
        return out

    def _single_channel(self) -> str:
        """Return the ID of the one channel of an array without a C axis: the patch's first (its only, once split)."""
        if self.channels is not None:
            return self.channels[0]
        return self.channel_order(1)[0]

    def stamp(self, patch: PatchRegion) -> None:
        """Record the normalization on the patch: the ``normalization_*`` fields and ``normalization_parameters``."""
        patch.normalization_method = self.method
        patch.normalization_scope = self.scope
        patch.normalization_group = self.group
        patch.normalization_exact = self.exact
        patch.normalization_parameters = self.parameters
        patch.normalization_channels = self.channels

    def apply(self, array: np.ndarray, axes: str, clip: tuple[float, float] | None = None) -> np.ndarray:
        """Normalize *array* (with *axes*) per channel into a new float32 array.

        Computed by the brick's ``Parameters.apply``. *clip*
        (``config.normalization.clip``) then limits the float32 values; with
        the bounds rounded to float32 too, this equals clipping before
        rounding, because rounding keeps the order of values.
        """
        array = np.asarray(array)
        axes = normalize_axes(axes, array.ndim)
        parameters = self.parameters
        if "C" not in axes:
            channel = self._single_channel()
            out = parameters.select([channel]).apply(array, channels=[channel])
        else:
            channel_axis = axes.index("C")
            order = self.channel_order(array.shape[channel_axis])
            if len(order) != parameters.channel_count:
                raise ValueError(
                    f"The patch has {len(order)} channels but {parameters.channel_count} sets of "
                    "normalization parameters."
                )
            out = parameters.apply(array, channel_axis=channel_axis, channels=order)
        if clip is not None:
            np.clip(out, np.float32(clip[0]), np.float32(clip[1]), out=out)
        return out


def split_channel_normalization(patch: PatchRegion, channel: int) -> None:
    """Keep only the parameters of C index *channel* on a split channel patch (``Parameters.select``, N46).

    The C index is mapped to its channel ID by the patch's own order (``normalization_channels``), never by position
    in the parameters.
    """
    if patch.normalization_parameters is None:
        return
    normalization = PatchNormalization.from_patch(patch)
    order = normalization.channels if normalization.channels is not None else normalization.channel_order(1)
    channel_id = order[channel]
    patch.normalization_parameters = patch.normalization_parameters.select([channel_id])
    patch.normalization_channels = (channel_id,)


def image_channel_ids(meta: Mapping, count: int, has_channels: bool, rgb: bool) -> tuple[str, ...]:
    """Return the canonical ID of each of an image's *count* channel planes, in its C order (ADR 0003 HV2).

    ``meta["channel_ids"]`` (a file run: ``Fov.channel_ids``) names every C index of the image. Without it (an
    in-memory image, which has no description) the channels are named by position, ``C<i>``, or ``R``, ``G``,
    ``B`` for ``"rgb"`` (the samples of an RGB unit, experiment-v1 §4.4b); these names hold within the image only.
    An image without a C axis is the one channel ``meta["c"]`` (0 when not given).

    Raises
    ------
    ValueError
        If ``channel_ids`` does not name every channel of the image, has no ID for its ``c``, repeats an ID or
        holds a text that is not a channel ID (refused here, before any fit, so that no patch is discarded for it).
    """
    ids = meta.get("channel_ids")
    where = meta.get("source_relative_path") or meta.get("image_file") or "an image"
    if ids is not None:
        ids = check_channel_ids(ids, f'{where}: image_metadata["channel_ids"]')
    if has_channels:
        if ids is None:
            if rgb and count == len(RGB_CHANNELS):
                return RGB_CHANNELS
            return tuple(f"C{index}" for index in range(count))
        if len(ids) != count:
            raise ValueError(
                f'{where}: image_metadata["channel_ids"] names {len(ids)} channels; the image has {count}.'
            )
        return tuple(ids)
    index = int(meta.get("c") or 0)
    if ids is None:
        return (f"C{index}",)
    if not 0 <= index < len(ids):
        raise ValueError(f'{where}: image_metadata["channel_ids"] names {len(ids)} channels, none for c={index}.')
    return (ids[index],)
