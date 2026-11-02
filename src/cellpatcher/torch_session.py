"""A CellPatcher export loaded as PyTorch tensors, for the PyTorch notebook (layer: orchestration).

The notebook (:mod:`cellpatcher.notebook_torch`) only shows the result. The patches are read
with :class:`cellpatcher.torch_dataset.ExportedPatchTorchDataset` from the export's
``patches.csv``, with the array axes every row records (``output_axes``), and kept as exported:
every channel (``channel_policy="keep"``: a channel is never repeated or dropped to fit a model)
and the values unscaled (``scale_policy="none"``: an integer patch is not divided by its type's
maximum, which is deprecated; normalization is CellPatcher's ``config.normalization``).
Files exported with raw values are normalized on load with each row's parameters, as everywhere
else. The first patches are stacked into one batch with PyTorch's ``default_collate``, as a
``DataLoader`` does. Patches of different sizes are refused, never
resized: a batch needs one size, which the export chooses. Needs the ``torch`` extra.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch

    from cellpatcher.torch_dataset import PatchTensorConfig
    from cellpatcher.types import PatchRegion

PATCH_TABLE = "patches.csv"
"""The table of an export (``ExportConfig.metadata_filename``'s default, the notebook's export)."""


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
        The batch: ``(patches, channels, height, width)``, as
        :class:`~cellpatcher.torch_dataset.ExportedPatchTorchDataset` gives each patch.
    patch_ids
        Each batch patch's ``patch_id``.
    first_channels
        Each batch patch's first channel, as a NumPy array, for display.
    """

    table: Path
    axes: str
    n_patches: int
    images: torch.Tensor
    patch_ids: tuple[str, ...]
    first_channels: tuple[np.ndarray, ...]

    @property
    def value_range(self) -> tuple[float, float]:
        """Return the smallest and largest value of the batch."""
        return float(self.images.min()), float(self.images.max())


def _tensor_config() -> PatchTensorConfig:
    """Return the tensor settings that keep the patches as exported: every channel, the values unscaled."""
    from cellpatcher.torch_dataset import PatchTensorConfig

    return PatchTensorConfig(channel_policy="keep", scale_policy="none")


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
        or patches of different sizes.
    FileNotFoundError
        For a patch file the table names but that is missing.
    """
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
    images = batch["image"]
    return TorchBatch(
        table=table,
        axes=axes,
        n_patches=len(dataset),
        images=images,
        patch_ids=tuple(batch["patch_id"]),
        first_channels=tuple(image[0].numpy() for image in images),
    )
