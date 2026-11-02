"""Read the images and masks of a file run through preprocessing's ``RunInputs`` (layer: io).

Design v0.4 §6 (CP-2a): a folder or tree run reads every image with ``RunInputs.read_plane``
and every mask with ``RunInputs.read_mask``, as the preprocessing run recorded them. Nothing
here names an axis of a file, pairs a mask or reads a calibration: an image's axes are the
roles the run recorded for it (:attr:`Fov.roles`), and a mask's are those the run recorded
for that mask (its ``roles``, and ``shared`` for a stated 2-D mask serving every plane).

The arrays handed to the patching core are in one canonical layout, built from those roles:
the run's T, Z and C axes in that order (:func:`run_layout`), then Y and X, as
``RunInputs.read_image`` orders them. Nothing is read transposed or reordered
other than by role.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

from cellpatcher.planes import (
    ImagePlane,
    PlaneChoice,
    apply_plane_choice,
    plane_choices,
    select_mask_plane,
)

if TYPE_CHECKING:
    from morphobricks_preprocessing.run_inputs import Fov, RunInputs

    from cellpatcher.config import CommonConfig

_STACK_ROLES = "TZC"
_FOLDER = "folder"  # morphobricks_preprocessing.masks.FOLDER, the source of a masks-folder TIFF


def run_layout(fovs: Sequence[Fov]) -> str:
    """Return the canonical layout of every image array of a run: T, Z, C in that order, then ``YX``.

    T or Z is in it when an image has more than one plane along it, C when an image has a
    channel role. An image without one of these axes has length 1 along it in its array (its
    one plane), never a guessed plane.
    """
    counts = {"T": lambda fov: fov.t_count, "Z": lambda fov: fov.z_count, "C": lambda fov: 2}
    return "".join(
        role for role in _STACK_ROLES if any(role in fov.roles and counts[role](fov) > 1 for fov in fovs)
    ) + "YX"


def _plane_shape(fov: Fov, layout: str) -> tuple[int, ...]:
    """Return an image's T, Z and C lengths in *layout*, from the run's records, with 0 for Y and X.

    Only the T and Z lengths are used, to choose the planes; Y and X are not read here.
    """
    lengths = {"T": fov.t_count, "Z": fov.z_count, "C": fov.c_count, "Y": 0, "X": 0}
    return tuple(lengths[axis] for axis in layout)


def read_fov(inputs: RunInputs, fov: Fov, layout: str, choice: PlaneChoice | None = None) -> tuple[np.ndarray, str]:
    """Read an image, or the planes *choice* names, through ``RunInputs.read_plane``.

    Parameters
    ----------
    inputs, fov
        The opened run and one of its images.
    layout
        The run's canonical layout (:func:`run_layout`).
    choice
        The T and Z planes to read (an index, a projection over every plane, or None for every
        plane); None reads the whole image.

    Returns
    -------
    tuple
        The pixels, as stored, and their axes: *layout* without the axes *choice* resolved to
        one plane (a projection is computed after reading every plane along its axis).

    Raises
    ------
    morphobricks_preprocessing.run_inputs.RunInputsError
        If the image changed since the run.
    """
    choice = choice or PlaneChoice(t=None, z=None, suffix="")
    counts = {"T": fov.t_count, "Z": fov.z_count}
    indices = {}
    for axis in ("T", "Z"):
        value = choice.value(axis) if axis in layout else None
        indices[axis] = [value] if isinstance(value, int) else list(range(counts[axis] if axis in layout else 1))
    stack = np.stack([np.stack([_read_plane(inputs, fov, z, t) for z in indices["Z"]]) for t in indices["T"]])
    kept = "".join(
        axis for axis in "TZCYX" if axis in "YX" or (axis in layout and not isinstance(choice.value(axis), int))
    )
    pixels = stack.reshape([size for axis, size in zip("TZCYX", stack.shape) if axis in kept])
    projections = PlaneChoice(
        t=choice.t if isinstance(choice.t, str) else None, z=choice.z if isinstance(choice.z, str) else None, suffix=""
    )
    return apply_plane_choice(pixels, kept, projections)


def _read_plane(inputs: RunInputs, fov: Fov, z: int, t: int) -> np.ndarray:
    """Read one ``(C, Y, X)`` plane, refusing a plane too large for the brick's memory limit."""
    from morphobricks_preprocessing import PlaneMemoryError

    try:
        return inputs.read_plane(fov, z=z, t=t)
    except PlaneMemoryError as exc:
        raise ValueError(
            f"CellPatcher needs each plane whole in memory to patch it, and a plane of {fov.relative_path} is too "
            f"large: {exc}"
        ) from exc


def mask_record(inputs: RunInputs, fov: Fov, mask_type: str) -> Mapping | None:
    """Return the run's record of an image's mask of *mask_type*, or None when it recorded none."""
    for record in (inputs.manifest.masks or {}).get("files", ()):
        if record["image"] == fov.relative_path and record["type"] == mask_type:
            return record
    return None


def mask_layout(record: Mapping) -> str:
    """Return the canonical layout of a recorded mask: ``YX`` when shared, else its T and Z roles, then ``YX``."""
    if record.get("shared"):
        return "YX"
    return "".join(role for role in "TZ" if role in str(record["roles"])) + "YX"


def read_fov_mask(inputs: RunInputs, fov: Fov, record: Mapping, layout: str) -> np.ndarray:
    """Read an image's recorded mask through ``RunInputs.read_mask``, in the run's mask *layout*.

    A stated shared 2-D mask is read once; a mask with depth slices or time points is read
    plane by plane, the brick checking each plane's labels. An axis of *layout* the mask does
    not have is added with length 1 (one plane, shared by the image's planes along it).

    Raises
    ------
    morphobricks_preprocessing.run_inputs.RunInputsError
        If the mask changed since the run.
    ValueError
        For a label outside 0 to the int32 limit (the brick's ``check_labels``).
    """
    own = mask_layout(record)
    roles, shape = str(record["roles"]), list(record["shape"])
    ts = range(shape[roles.index("T")]) if "T" in own else [None]
    zs = range(shape[roles.index("Z")]) if "Z" in own else [None]
    stack = np.stack([np.stack([inputs.read_mask(fov, record["type"], z=z, t=t) for z in zs]) for t in ts])
    return stack.reshape([size for axis, size in zip("TZYX", stack.shape) if axis in layout])


def run_mask_layout(records: Sequence[Mapping]) -> str:
    """Return the canonical layout of every mask array of a run: the T and Z any mask has, then ``YX``."""
    return "".join(role for role in "TZ" if any(role in mask_layout(record) for record in records)) + "YX"


def mask_file_name(record: Mapping) -> str:
    """Return a mask's ``mask_file``: its file name, or ``<image>/labels/<name>`` for a label image."""
    path = PurePosixPath(str(record["path"]))
    if record["source"] == _FOLDER:
        return path.name
    return "/".join(path.parts[-3:])


def iter_fov_planes(
    inputs: RunInputs,
    fov: Fov,
    layout: str,
    mask: np.ndarray | None,
    mask_axes: str,
    common: CommonConfig,
    *,
    single_planes: bool = False,
) -> Iterator[ImagePlane]:
    """Yield the planes of an image chosen by the plane policies, reading only those planes.

    The planes and their mask planes equal those :func:`cellpatcher.planes.iter_image_planes`
    gives for the whole image. Raises like :func:`cellpatcher.planes.plane_choices`.
    """
    mask_shape = None if mask is None else mask.shape
    shape = _plane_shape(fov, layout)
    for choice in plane_choices(shape, layout, mask_shape, mask_axes, common, single_planes=single_planes):
        pixels, axes = read_fov(inputs, fov, layout, choice)
        plane_mask, plane_mask_axes = select_mask_plane(mask, mask_axes, choice)
        yield ImagePlane(pixels, axes, plane_mask, plane_mask_axes, choice.index("Z"), choice.index("T"), choice.suffix)
