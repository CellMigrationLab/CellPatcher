"""Turn the Z and T axes of an image into the 2D planes CellPatcher patches.

``CommonConfig.z_policy`` and ``CommonConfig.t_policy`` choose, per axis:

- ``"keep"`` (default): every patch keeps all planes along the axis, as before
  these options existed;
- ``"all"``: one patch per plane;
- ``"first"``, or ``"select"`` with ``z_index``/``t_index``: one plane;
- Z only: ``"max"``, ``"mean"`` or ``"sum"`` accept an image with no Z axis or one Z plane and return
  that plane unchanged; with more than one Z plane they raise
  :class:`cellpatcher.errors.ZStackNotSupportedError`: CellPatcher combines no planes, a Z projection
  is a recorded step of morphobricks-preprocessing (owner decision of 5 November 2026, #83).

A mask follows the image's planes along an axis it has with the same length,
and is shared by every plane when it lacks the axis or has length 1 (the rule
of ``docs/normalization_contract.md`` §4 for OME-NGFF labels, applied to every
mask). Images have at most one Z plane, in memory as in a file run (preprocessing v0.4.0 for
files, :func:`cellpatcher.axes.refuse_z_stack` for arrays): more raise
:class:`cellpatcher.errors.ZStackNotSupportedError` under every ``z_policy``.

Images in memory are split with :func:`iter_image_planes`; the images of a file run are read
plane by plane through the preprocessing run (:mod:`cellpatcher.run_reading`).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from cellpatcher.axes import axis_index, normalize_axes, refuse_z_stack
from cellpatcher.preprocess import project_axis, select_axis_index, z_projection_refusal

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
        The plane's Z and T index, or None when the axis is absent or kept.
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


@dataclass(frozen=True)
class PlaneChoice:
    """One plane chosen by the policies, independent of the pixels.

    ``t`` and ``z`` are an index, a projection name (``"max"``, ``"mean"``, ``"sum"``; Z only, and
    only over an axis of one plane), or None when the axis is absent or kept.
    """

    t: int | str | None
    z: int | str | None
    suffix: str

    def value(self, axis: str) -> int | str | None:
        """The choice along ``"T"`` or ``"Z"``; None for any other axis, which is never chosen."""
        return {"T": self.t, "Z": self.z}.get(axis)

    def index(self, axis: str) -> int | None:
        """The plane index along ``"T"`` or ``"Z"``; None when kept, absent or projected."""
        value = self.value(axis)
        return value if isinstance(value, int) else None


def plane_choices(image_shape, image_axes, mask_shape, mask_axes, common, *, single_planes=False) -> list[PlaneChoice]:
    """The planes of an image of *image_shape* chosen by the policies, in T-then-Z order.

    With *single_planes* (normalization is enabled), every Z or T axis longer
    than 1 must end as one index per plane (``docs/normalization_contract.md``
    §3): ``"keep"`` is refused.

    Raises
    ------
    ValueError
        If an index is outside the image, a mask's Z or T length is neither
        the image's nor 1, or *single_planes* refuses a policy.
    ZStackNotSupportedError
        If the image or a mask has more than one Z plane, whatever ``z_policy`` says (issue #83).
    """
    validate_plane_policies(common)
    image_axes = normalize_axes(image_axes, len(image_shape))
    refuse_z_stack(image_shape, image_axes)
    if mask_shape is not None:
        mask_axes = normalize_axes(mask_axes, len(mask_shape))
        refuse_z_stack(mask_shape, mask_axes, "The mask")
    choices = []
    for axis in ("T", "Z"):
        policy = getattr(common, f"{axis.lower()}_policy")
        index = getattr(common, f"{axis.lower()}_index")
        choices.append(
            _axis_choices(image_shape, image_axes, mask_shape, mask_axes, axis, policy, index, single_planes)
        )
    return [
        PlaneChoice(t=t, z=z, suffix=t_label + z_label)
        for (t, t_label), (z, z_label) in itertools.product(*choices)
    ]


def iter_image_planes(image, image_axes, mask, mask_axes, common, *, single_planes=False) -> Iterator[ImagePlane]:
    """Yield the planes of one in-memory image chosen by ``common.z_policy`` and ``common.t_policy``.

    Planes come in T-then-Z order. An image without a Z or T axis is not
    changed along it, whatever the policy. Raises like :func:`plane_choices`.
    """
    image = np.asarray(image)
    image_axes = normalize_axes(image_axes, image.ndim)
    if mask is not None:
        mask = np.asarray(mask)
        mask_axes = normalize_axes(mask_axes, mask.ndim)
    mask_shape = None if mask is None else mask.shape
    for choice in plane_choices(image.shape, image_axes, mask_shape, mask_axes, common, single_planes=single_planes):
        plane, plane_axes = apply_plane_choice(image, image_axes, choice)
        plane_mask, plane_mask_axes = select_mask_plane(mask, mask_axes, choice)
        yield ImagePlane(
            plane, plane_axes, plane_mask, plane_mask_axes, choice.index("Z"), choice.index("T"), choice.suffix
        )


def apply_plane_choice(image, image_axes, choice: PlaneChoice):
    """The plane of an in-memory image for *choice*: ``(pixels, axes)``."""
    plane, plane_axes = np.asarray(image), normalize_axes(image_axes, np.ndim(image))
    for axis in ("T", "Z"):
        value = choice.value(axis)
        if axis not in plane_axes or value is None:
            continue
        if isinstance(value, str):
            plane, plane_axes = project_axis(plane, plane_axes, axis, value)
        else:
            plane, plane_axes = select_axis_index(plane, plane_axes, axis, value)
    return plane, plane_axes


def patch_plane_choice(patch, image_axes, common) -> PlaneChoice:
    """The plane a patch was cut from: its ``t``/``z0``, or the policy's projection (of one plane).

    Used to rebuild a patch from its source image (``cellpatcher.delivery``).
    """
    values = {}
    for axis, index in (("T", patch.t), ("Z", patch.z0)):
        policy = getattr(common, f"{axis.lower()}_policy")
        if axis not in normalize_axes(image_axes) or policy == "keep":
            values[axis] = None
        elif policy in _PROJECTIONS:
            values[axis] = policy
        else:
            values[axis] = index
    return PlaneChoice(t=values["T"], z=values["Z"], suffix="")


def select_mask_plane(mask, mask_axes, choice: PlaneChoice):
    """The mask plane for *choice*: a mask axis with the image's length follows it, length 1 is shared."""
    if mask is None:
        return None, mask_axes
    for axis in ("T", "Z"):
        value = choice.value(axis)
        position = axis_index(mask_axes, axis)
        if value is None or position is None:
            continue
        mask_value = 0 if mask.shape[position] == 1 or isinstance(value, str) else value
        mask, mask_axes = select_axis_index(mask, mask_axes, axis, mask_value)
    return mask, mask_axes


def _axis_choices(image_shape, image_axes, mask_shape, mask_axes, axis, policy, index, single_planes) -> list[tuple]:
    """``(value, id label)`` per plane along ``axis``: an index, a projection name, or None (unchanged)."""
    position = axis_index(image_axes, axis)
    if position is None:
        return [(None, "")]
    length = int(image_shape[position])
    if policy in _PROJECTIONS and length > 1:
        raise z_projection_refusal(policy, length)
    if single_planes and length > 1 and policy == "keep":
        raise ValueError(
            f"Normalization needs one plane per patch, but {axis.lower()}_policy={policy!r} keeps all "
            f"{length} planes along {axis} (the preprocessing statistics describe single planes). Set "
            f"common.{axis.lower()}_policy to 'all', 'first' or 'select'."
        )
    if policy == "keep":
        return [(None, "")]
    if mask_shape is not None and axis_index(mask_axes, axis) is not None:
        mask_length = int(mask_shape[axis_index(mask_axes, axis)])
        if mask_length not in (length, 1):
            raise ValueError(
                f"The masks have {mask_length} planes along {axis} but the images have {length}; a mask "
                f"must have the image's {axis} length (one label plane per image plane) or length 1 "
                "(shared by all planes)."
            )
    if policy in _PROJECTIONS:
        return [(policy, "")]  # one plane: the projection returns it unchanged
    if policy == "all":
        return [(i, f"_{axis.lower()}{i:03d}") for i in range(length)]
    selected = 0 if policy == "first" else index
    if not 0 <= selected < length:
        raise ValueError(
            f"{axis.lower()}_index={selected} is outside the images, which have {length} planes along {axis}."
        )
    return [(selected, "")]
