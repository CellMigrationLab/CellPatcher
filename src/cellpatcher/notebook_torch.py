"""The CellPatcher PyTorch notebook: an export's patches as a batch of tensors (layer: ui).

``notebooks/CellPatcher_Pytorch/CellPatcher_Pytorch.ipynb`` creates :class:`TorchNotebook`
and calls one step per cell. It reads the output folder of a CellPatcher export (the
CellPatcher notebook's step 4) with :func:`cellpatcher.torch_session.load_exported_batch` and
shows the batch a ``DataLoader`` would give. Needs the ``notebook`` and ``torch`` extras.
"""

from __future__ import annotations

import html

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from morphobricks_preprocessing import notebook_style as style

from cellpatcher import __version__
from cellpatcher.notebook_ui import guarded, path_row
from cellpatcher.torch_session import TorchBatch, load_exported_batch


def batch_html(batch: TorchBatch) -> str:
    """Return what the batch holds: the table, the number of patches and the tensor."""
    low, high = batch.value_range
    return style.status("ok", f"{len(batch.patch_ids)} of {batch.n_patches} patches loaded as one batch", [
        f"Table: {batch.table}; patch arrays {batch.axes}",
        f"Tensor: {tuple(batch.images.shape)} (patches, channels, height, width), {batch.images.dtype}",
        f"Values from {low:.6g} to {high:.6g}",
    ])


def draw_batch(batch: TorchBatch) -> None:
    """Draw the first channel of every patch of the batch, titled with its place in the batch."""
    figure, axes = plt.subplots(1, len(batch.first_channels), figsize=(2 * len(batch.first_channels), 2.2),
                                squeeze=False)
    for index, (axis, plane) in enumerate(zip(axes[0], batch.first_channels, strict=True)):
        axis.imshow(plane, cmap="gray")
        axis.set_title(f"batch[{index}]", fontsize=8)
        axis.axis("off")
    plt.show()
    plt.close(figure)


class TorchNotebook:
    """The PyTorch notebook's page: the export to load, and the batch it gives."""

    def __init__(self) -> None:
        self.output_folder = widgets.Text(description="CellPatcher output folder",
                                          placeholder="…/my_experiment_CellPatcher",
                                          style={"description_width": "200px"}, layout=widgets.Layout(width="80%"))
        self.batch_size = widgets.BoundedIntText(value=8, min=1, max=64, description="Patches in the batch",
                                                 style={"description_width": "200px"})
        self.load_button = style.button("Load the patches", style="primary", main=True)
        self.message = widgets.HTML()
        self.output = widgets.Output()
        self.load_button.on_click(lambda _: self.load())

    def welcome(self) -> None:
        """Show that the notebook is ready, with the package's version."""
        display(style.page("CellPatcher for PyTorch is ready", f"CellPatcher {html.escape(__version__)}", [
            widgets.HTML(style.status("ok", "Run the next cell.", [
                "Step 1: load the patches of a CellPatcher export as PyTorch tensors.",
            ])),
        ]))

    def patches_step(self) -> None:
        """Show step 1: the export's folder and the batch it gives."""
        display(style.page("Step 1: the patches as tensors", "An export of the CellPatcher notebook, read with "
                           "ExportedPatchTorchDataset.", [
            style.card("Export", [
                path_row(self.output_folder, self.message),
                style.help_text("The output folder of the CellPatcher notebook's step 4 (it holds "
                                "<b>patches.csv</b>, <b>images.csv</b> and <b>run_config.json</b>)."),
                self.batch_size, self.load_button, self.message, self.output,
            ]),
        ]))

    def load(self) -> TorchBatch | None:
        """Load the first patches of the export as one batch and draw them."""
        self.message.value = style.working("Loading the patches…")
        batch = guarded(self.message, lambda: load_exported_batch(style.clean_path(self.output_folder.value),
                                                                  self.batch_size.value))
        self.output.clear_output()
        if batch is not None:
            self.message.value = batch_html(batch)
            with self.output:
                draw_batch(batch)
        return batch
