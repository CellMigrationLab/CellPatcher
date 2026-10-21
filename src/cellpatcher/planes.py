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

Images in memory are split with :func:`iter_image_planes`; image files with
:func:`iter_file_planes`, which reads only the chosen planes when it can.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class PlaneChoice:
    """One plane chosen by the policies, independent of the pixels.

    ``t`` and ``z`` are an index, a projection name (``"max"``, ``"mean"``,
    ``"sum"``; Z only), or None when the axis is absent or kept.
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


def plane_choices(image_shape, image_axes, mask_shape, mask_axes, common) -> list[PlaneChoice]:
    """The planes of an image of *image_shape* chosen by the policies, in T-then-Z order.

    Raises
    ------
    ValueError
        If an index is outside the image, a mask's Z or T length is neither
        the image's nor 1, or a Z projection meets a mask that follows the
        image's Z planes.
    """
    validate_plane_policies(common)
    image_axes = normalize_axes(image_axes, len(image_shape))
    if mask_shape is not None:
        mask_axes = normalize_axes(mask_axes, len(mask_shape))
    choices = []
    for axis in ("T", "Z"):
        policy = getattr(common, f"{axis.lower()}_policy")
        index = getattr(common, f"{axis.lower()}_index")
        choices.append(_axis_choices(image_shape, image_axes, mask_shape, mask_axes, axis, policy, index))
    return [
        PlaneChoice(t=t, z=z, suffix=t_label + z_label)
        for (t, t_label), (z, z_label) in itertools.product(*choices)
    ]


def iter_image_planes(image, image_axes, mask, mask_axes, common) -> Iterator[ImagePlane]:
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
    for choice in plane_choices(image.shape, image_axes, mask_shape, mask_axes, common):
        plane, plane_axes = image, image_axes
        for axis in ("T", "Z"):
            value = choice.value(axis)
            if isinstance(value, str):
                plane, plane_axes = project_axis(plane, plane_axes, axis, value)
            elif value is not None:
                plane, plane_axes = select_axis_index(plane, plane_axes, axis, value)
        plane_mask, plane_mask_axes = select_mask_plane(mask, mask_axes, choice)
        yield ImagePlane(
            plane, plane_axes, plane_mask, plane_mask_axes, choice.index("Z"), choice.index("T"), choice.suffix
        )


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


def iter_file_planes(path, image_axes, mask, mask_axes, common) -> Iterator[ImagePlane]:
    """Yield the planes of an image file chosen by the policies, reading only those planes when possible.

    With the ``preprocessing`` extra, a TIFF or OME-Zarr image whose planes
    are split or selected (``"all"``, ``"first"``, ``"select"``) is read one
    2D plane at a time with the preprocessing brick's ``read_plane``
    (``docs/normalization_contract.md`` §3): only the chosen planes are
    decoded, and a plane that cannot be read whole within the brick's memory
    limit is an error. Other images, ``"keep"`` and projections are read
    whole with :func:`cellpatcher.io.read_image`, as before. Either way the
    planes equal those of :func:`iter_image_planes` on the whole image.

    Raises
    ------
    ValueError
        As :func:`plane_choices`, or if a plane needs more memory than the limit.
    """
    from cellpatcher.io import read_image

    mbp = _plane_reader(path, common)
    if mbp is None:
        yield from iter_image_planes(read_image(path, image_axes), image_axes, mask, mask_axes, common)
        return

    path = Path(path).expanduser()
    target = normalize_axes(image_axes)
    mapping, shape, dtype = _file_layout(mbp, path, target)
    if mask is not None:
        mask = np.asarray(mask)
        mask_axes = normalize_axes(mask_axes, mask.ndim)
    mask_shape = None if mask is None else mask.shape
    for choice in plane_choices(shape, target, mask_shape, mask_axes, common):
        plane_axes = "".join(axis for axis in target if choice.value(axis) is None)
        pixels = np.empty([shape[target.index(axis)] for axis in plane_axes], dtype=dtype)
        # Axes kept in the plane (channels, or a "keep" axis) are read one
        # 2D plane at a time and stacked back in place.
        stacked = [axis for axis in plane_axes if axis not in "YX"]
        y_before_x = plane_axes.index("Y") < plane_axes.index("X")
        for position in np.ndindex(*(pixels.shape[plane_axes.index(axis)] for axis in stacked)):
            coordinates = {"T": choice.index("T"), "Z": choice.index("Z"), "C": None}
            coordinates.update(zip(stacked, position))
            plane = _read_plane(mbp, path, mapping, coordinates)
            where = tuple(position[stacked.index(axis)] if axis in stacked else slice(None) for axis in plane_axes)
            pixels[where] = plane if y_before_x else plane.T
        plane_mask, plane_mask_axes = select_mask_plane(mask, mask_axes, choice)
        yield ImagePlane(
            pixels, plane_axes, plane_mask, plane_mask_axes, choice.index("Z"), choice.index("T"), choice.suffix
        )


def _plane_reader(path, common):
    """The preprocessing package when *path* can be read plane by plane under *common*, else None."""
    from cellpatcher.io import is_ome_zarr
    from cellpatcher.statistics import PreprocessingUnavailableError, require_preprocessing

    policies = (common.z_policy, common.t_policy)
    if not splits_planes(common) or any(policy in _PROJECTIONS for policy in policies):
        return None
    if not (is_ome_zarr(path) or Path(path).suffix.lower() in (".tif", ".tiff")):
        return None
    try:
        return require_preprocessing()
    except PreprocessingUnavailableError:
        return None


def _file_layout(mbp, path: Path, target: str):
    """The preprocessing ``AxisMapping`` of a file, and its shape and dtype in *target* order."""
    meta = mbp.inspect_image(path, path.name)
    if not meta.readable:
        raise ValueError(f"{path.name} cannot be read: {meta.error}")
    if meta.kind == "ome-zarr":
        # OME-Zarr axes come from the metadata and are matched by role, as in
        # io.conform_axes: an axis the configured axes lack must have length 1.
        for axis, length in zip(meta.axes, meta.shape):
            if axis not in target and length != 1:
                raise ValueError(
                    f"{path.name} has an axis {axis!r} of length {length} that the axes {target!r} do not "
                    f"have; its axes are {meta.axes!r}. Set common.image_axes to include every axis "
                    "longer than 1."
                )
        lengths = dict(zip(meta.axes, meta.shape))
        roles = tuple(axis if axis in target else None for axis in meta.axes)
        shape = tuple(lengths.get(axis, 1) for axis in target)
    else:
        # A TIFF's dimensions are those tifffile.imread returns, which
        # common.image_axes names in order.
        normalize_axes(target, len(meta.shape))
        roles = tuple(target)
        shape = tuple(meta.shape)
    return mbp.AxisMapping(meta.axes, roles), shape, np.dtype(meta.dtype)


def _read_plane(mbp, path: Path, mapping, coordinates) -> np.ndarray:
    """Read one (Y, X) plane, with None for roles the file does not have."""
    present = set(mapping.roles)
    wanted = {axis.lower(): (value if axis in present else None) for axis, value in coordinates.items()}
    try:
        return mbp.read_plane(path, mapping, **wanted).pixels
    except mbp.PlaneMemoryError as exc:
        raise ValueError(
            f"CellPatcher needs each plane whole in memory to patch it, and a plane of {path.name} is too "
            f"large: {exc}"
        ) from exc


def _axis_choices(image_shape, image_axes, mask_shape, mask_axes, axis, policy, index) -> list[tuple]:
    """``(value, id label)`` per plane along ``axis``: an index, a projection name, or None (unchanged)."""
    position = axis_index(image_axes, axis)
    if position is None or policy == "keep":
        return [(None, "")]
    length = int(image_shape[position])
    if mask_shape is not None and axis_index(mask_axes, axis) is not None:
        mask_length = int(mask_shape[axis_index(mask_axes, axis)])
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
