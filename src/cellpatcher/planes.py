"""Turn the Z and T axes of an image into the 2D planes CellPatcher patches.

``CommonConfig.z_policy`` and ``CommonConfig.t_policy`` choose, per axis:

- ``"keep"`` (default): every patch keeps all planes along the axis, as before
  these options existed;
- ``"all"``: one patch per plane;
- ``"first"``, or ``"select"`` with ``z_index``/``t_index``: one plane;
- Z only: ``"max"``, ``"mean"`` or ``"sum"`` project the stack.

A mask follows the image's planes along an axis it has with the same length,
and is shared by every plane when it lacks the axis or has length 1 (the rule
of ``docs/normalization_contract.md`` §4 for OME-NGFF labels, applied to every
mask).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from cellpatcher.axes import axis_index, normalize_axes
from cellpatcher.preprocess import project_axis, select_axis_index

Z_POLICIES = ("keep", "all", "first", "select", "max", "mean", "sum")
T_POLICIES = ("keep", "all", "first", "select")
_PROJECTIONS = ("max", "mean", "sum")


def validate_plane_policies(common) -> None:
    """Check ``z_policy``/``z_index`` and ``t_policy``/``t_index`` of a ``CommonConfig``.

    Raises
    ------
    ValueError
        If a policy is unknown, or ``"select"`` has no index.
    """
    for axis, policies in (("z", Z_POLICIES), ("t", T_POLICIES)):
        policy = getattr(common, f"{axis}_policy")
        index = getattr(common, f"{axis}_index")
        if policy not in policies:
            raise ValueError(f"{axis}_policy must be one of {list(policies)}; got {policy!r}.")
        if policy == "select" and index is None:
            raise ValueError(f"{axis}_policy='select' needs {axis}_index.")
        if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
            raise ValueError(f"{axis}_index must be a non-negative int or None; got {index!r}.")


def splits_planes(common) -> bool:
    """True when the policies turn a Z or T axis into planes (not ``"keep"`` for both)."""
    return common.z_policy != "keep" or common.t_policy != "keep"


@dataclass(frozen=True)
class ImagePlane:
    """One plane of an image (and its mask) chosen by the plane policies.

    Attributes
    ----------
    image, image_axes
        The plane's pixels and axes: the image's axes without the Z and T
        axes the policies resolved.
    mask, mask_axes
        The matching mask plane, or None when there is no mask.
    z, t
        The plane's Z and T index, or None when the axis is absent, kept, or
        projected.
    suffix
        Appended to the patch ids of this plane when an axis is split
        (``"all"``), e.g. ``"_t001_z002"``; empty otherwise.
    """

    image: np.ndarray
    image_axes: str
    mask: np.ndarray | None
    mask_axes: str | None
    z: int | None
    t: int | None
    suffix: str


def iter_image_planes(image, image_axes, mask, mask_axes, common) -> Iterator[ImagePlane]:
    """Yield the planes of one image chosen by ``common.z_policy`` and ``common.t_policy``.

    Planes come in T-then-Z order. An image without a Z or T axis is not
    changed along it, whatever the policy.

    Raises
    ------
    ValueError
        If an index is outside the image, a mask's Z or T length is neither
        the image's nor 1, or a Z projection meets a mask that follows the
        image's Z planes.
    """
    validate_plane_policies(common)
    image = np.asarray(image)
    image_axes = normalize_axes(image_axes, image.ndim)
    if mask is not None:
        mask = np.asarray(mask)
        mask_axes = normalize_axes(mask_axes, mask.ndim)

    choices = []
    for axis in ("T", "Z"):
        policy = getattr(common, f"{axis.lower()}_policy")
        index = getattr(common, f"{axis.lower()}_index")
        choices.append(_axis_choices(image, image_axes, mask, mask_axes, axis, policy, index))

    for (t, t_label), (z, z_label) in itertools.product(*choices):
        plane, plane_axes = image, image_axes
        plane_mask, plane_mask_axes = mask, mask_axes
        for axis, value in (("T", t), ("Z", z)):
            plane, plane_axes, plane_mask, plane_mask_axes = _resolve(
                plane, plane_axes, plane_mask, plane_mask_axes, axis, value
            )
        suffix = "".join(label for label in (t_label, z_label) if label)
        yield ImagePlane(
            image=plane,
            image_axes=plane_axes,
            mask=plane_mask,
            mask_axes=plane_mask_axes,
            z=z if isinstance(z, int) else None,
            t=t if isinstance(t, int) else None,
            suffix=suffix,
        )


def _axis_choices(image, image_axes, mask, mask_axes, axis, policy, index) -> list[tuple]:
    """``(value, id label)`` per plane along ``axis``: an index, a projection name, or None (unchanged)."""
    position = axis_index(image_axes, axis)
    if position is None or policy == "keep":
        return [(None, "")]
    length = int(image.shape[position])
    if mask is not None and axis_index(mask_axes, axis) is not None:
        mask_length = int(mask.shape[axis_index(mask_axes, axis)])
        if mask_length not in (length, 1):
            raise ValueError(
                f"The masks have {mask_length} planes along {axis} but the images have {length}; a mask "
                f"must have the image's {axis} length (one label plane per image plane) or length 1 "
                "(shared by all planes)."
            )
        if policy in _PROJECTIONS and mask_length == length and length > 1:
            raise ValueError(
                f"z_policy={policy!r} projects the images' Z planes, but the masks have one label plane "
                "per Z plane. Use masks without a Z axis, or z_policy='all', 'first' or 'select'."
            )
    if policy in _PROJECTIONS:
        return [(policy, "")]
    if policy == "all":
        return [(i, f"_{axis.lower()}{i:03d}") for i in range(length)]
    selected = 0 if policy == "first" else index
    if not 0 <= selected < length:
        raise ValueError(
            f"{axis.lower()}_index={selected} is outside the images, which have {length} planes along {axis}."
        )
    return [(selected, "")]


def _resolve(image, image_axes, mask, mask_axes, axis, value):
    """Apply one axis choice to the image and its mask."""
    if value is None:
        return image, image_axes, mask, mask_axes
    if isinstance(value, str):
        image, image_axes = project_axis(image, image_axes, axis, value)
    else:
        image, image_axes = select_axis_index(image, image_axes, axis, value)
    if mask is not None and axis_index(mask_axes, axis) is not None:
        # A length-1 mask axis is shared; otherwise it follows the image.
        mask_length = int(mask.shape[axis_index(mask_axes, axis)])
        mask_value = 0 if mask_length == 1 or isinstance(value, str) else value
        mask, mask_axes = select_axis_index(mask, mask_axes, axis, mask_value)
    return image, image_axes, mask, mask_axes
