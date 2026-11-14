"""A CellPatcher export loaded as PyTorch tensors, for the PyTorch notebook (layer: orchestration).

The notebook (:mod:`cellpatcher.notebook_torch`) only shows the result. The patches are read
with :class:`cellpatcher.torch_dataset.ExportedPatchTorchDataset` from the export's
``patches.csv``, with the array axes every row records (``output_axes``), and kept as exported:
every channel (``channel_policy="keep"``: a channel is never repeated or dropped to fit a model)
and the values unscaled (``scale_policy="none"``: normalization is CellPatcher's
``config.normalization``).
Files exported with raw values are normalized on load with each row's parameters, as everywhere
else. The first patches are stacked into one batch with PyTorch's ``default_collate``, as a
``DataLoader`` does. Patches of different sizes are refused, never
resized: a batch needs one size, which the export chooses. The batch is float32, the type
models take: the patches are read in their exported type (``dtype_policy="keep"``) and converted
only when float32 holds every value exactly (always for 8- and 16-bit integers and for normalized
patches, which are float32); otherwise they are refused, never rounded. The exported type is
recorded with the batch. Needs the ``torch`` extra.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from cellpatcher.torch_dataset import PatchTensorConfig
    from cellpatcher.types import PatchRegion

PATCH_TABLE = "patches.csv"
"""The table of an export (``ExportConfig.metadata_filename``'s default, the notebook's export)."""
FLOAT32_EXACT_INTEGERS = 2**24
"""The largest integer magnitude below which float32 holds every integer exactly."""


@dataclass(frozen=True)
class TorchBatch:
    """The first patches of an export, as one batch of tensors.

    Attributes
    ----------
    table
        The export's ``patches.csv``.
    axes
        The axes of its patch arrays (its ``output_axes`` column).
    n_patches
        The patches it lists.
    images
        The batch: ``(patches, channels, height, width)``, float32, as
        :class:`~cellpatcher.torch_dataset.ExportedPatchTorchDataset` gives each patch.
    exported_dtype
        The type of the patches as exported (and normalized on load), before the exact
        conversion to float32.
    patch_ids
        Each batch patch's ``patch_id``.
    first_channels
        Each batch patch's first channel, as a NumPy array, for display.
    """

    table: Path
    axes: str
    n_patches: int
    images: torch.Tensor
    exported_dtype: str
    patch_ids: tuple[str, ...]
    first_channels: tuple[np.ndarray, ...]

    @property
    def value_range(self) -> tuple[float, float]:
        """Return the smallest and largest value of the batch."""
        return float(self.images.min()), float(self.images.max())


def _tensor_config() -> PatchTensorConfig:
    """Return the tensor settings that keep the patches as exported: every channel, the values unscaled."""
    from cellpatcher.torch_dataset import PatchTensorConfig

    return PatchTensorConfig(channel_policy="keep", dtype_policy="keep", scale_policy="none")


def _exact_float32(values: np.ndarray) -> np.ndarray:
    """Return *values* as float32, refusing values float32 cannot hold exactly (never rounded)."""
    converted = values.astype(np.float32)
    if np.issubdtype(values.dtype, np.integer):
        exact = values.size == 0 or max(-int(values.min()), int(values.max())) <= FLOAT32_EXACT_INTEGERS
    elif np.issubdtype(values.dtype, np.floating):
        exact = np.array_equal(converted.astype(values.dtype), values, equal_nan=True)
    else:
        exact = False
    if not exact:
        raise ValueError(
            f"The patches are {values.dtype} with values float32 cannot hold exactly; they are not rounded. "
            "Load them with ExportedPatchTorchDataset and PatchTensorConfig(dtype_policy='keep'), or export "
            "them normalized (float32)."
        )
    return converted


def _array_axes(table: Path, patches: Sequence[PatchRegion]) -> str:
    """Return the axes of the export's arrays, which every row's ``output_axes`` names; never guessed."""
    axes = sorted({patch.output_axes or "" for patch in patches})
    if axes == [""]:
        raise ValueError(
            f"{table} has no output_axes column: export the patches again with this CellPatcher, so the axes "
            "of their arrays are recorded."
        )
    if len(axes) > 1:
        raise ValueError(
            f"The patches of {table} have arrays with different axes ({', '.join(a or 'none' for a in axes)}); "
            "load them separately, or export them with one layout."
        )
    return axes[0]


def load_exported_batch(output_folder: str | os.PathLike, batch_size: int = 8) -> TorchBatch:
    """Load the first *batch_size* patches of a CellPatcher export as one batch of tensors.

    Parameters
    ----------
    output_folder
        The export's folder, holding ``patches.csv`` (the CellPatcher notebook's output folder).
    batch_size
        The most patches in the batch.

    Returns
    -------
    TorchBatch

    Raises
    ------
    ValueError
        For a folder without ``patches.csv``, an export without patches, a batch size below 1,
        patches of different sizes, or values float32 cannot hold exactly.
    FileNotFoundError
        For a patch file the table names but that is missing.
    """
    import torch
    from torch.utils.data import default_collate

    from cellpatcher.export import read_patch_table
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    if batch_size < 1:
        raise ValueError(f"The batch needs at least one patch; got {batch_size}.")
    text = str(output_folder).strip()
    table = Path(text).expanduser() / PATCH_TABLE
    if not text or not table.is_file():
        raise ValueError(
            f"{text or 'No folder'} holds no {PATCH_TABLE}: give the output folder of a CellPatcher export "
            "(step 4 of the CellPatcher notebook)."
        )
    patches, _ = read_patch_table(table)
    if not patches:
        raise ValueError(f"{table} lists no patches.")
    axes = _array_axes(table, patches)
    dataset = ExportedPatchTorchDataset(table, axes=axes, tensor_config=_tensor_config())
    items = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    shapes = sorted({tuple(item["image"].shape) for item in items})
    if len(shapes) > 1:
        raise ValueError(
            f"The patches have different sizes ({', '.join(map(str, shapes))}); a batch needs one size. Export "
            "them with one size: a fixed patch size or 'Resize patches to' in step 2 of the CellPatcher notebook."
        )
    batch = default_collate(items)
    exported = batch["image"].numpy()
    images = torch.from_numpy(_exact_float32(exported))
    return TorchBatch(
        table=table,
        axes=axes,
        n_patches=len(dataset),
        images=images,
        exported_dtype=str(exported.dtype),
        patch_ids=tuple(batch["patch_id"]),
        first_channels=tuple(image[0].numpy() for image in images),
    )
