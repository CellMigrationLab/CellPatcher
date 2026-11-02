"""The CellPatcher strategy examples notebook: each patching strategy on one image of a run (layer: ui).

``notebooks/CellPatcher_Strategy_Examples/CellPatcher_Strategy_Examples.ipynb`` creates
:class:`StrategyExamplesNotebook` and calls one step per cell: the preprocessing run (the main
notebook's step 1), then one page per strategy. Each example is the main notebook's preview
(:func:`cellpatcher.notebook_session.preview_image`) of the chosen image with that strategy and
its settings: the image is read through the preprocessing run's ``RunInputs``, patched in
memory with the file-run code, and nothing is written. The patches shown are therefore those
the CellPatcher notebook's export gives with the same settings. Needs the ``notebook`` extra.
"""

from __future__ import annotations

import html

import ipywidgets as widgets
from IPython.display import display
from morphobricks_preprocessing import notebook_style as style

from cellpatcher import __version__
from cellpatcher.notebook_session import Preview, preview_image
from cellpatcher.notebook_ui import CellPatcherNotebook, draw_preview, guarded

STRATEGIES = {
    "cell": ("Cell", "One patch per segmented object, centred on it; needs the run's masks."),
    "grid": ("Grid", "Regular tiles covering the image."),
    "random": ("Random", "Patches at random positions, the same ones for the same seed."),
    "whole_fov": ("Whole FOV", "One patch holding the whole image."),
}
"""Each strategy's name and what it gives, in the notebook's order."""


class StrategyExamplesNotebook(CellPatcherNotebook):
    """The examples' pages: the preprocessing run, then one example per patching strategy.

    The run, its check and the settings are the CellPatcher notebook's
    (:class:`~cellpatcher.notebook_ui.CellPatcherNotebook`); each strategy's page shows that
    strategy's settings only, and its button previews the chosen image with them.
    """

    def __init__(self) -> None:
        self.example_buttons = {
            mode: style.button(f"Show the {name} example", style="primary", main=True)
            for mode, (name, _) in STRATEGIES.items()
        }
        self.example_messages = {mode: widgets.HTML() for mode in STRATEGIES}
        self.example_outputs = {mode: widgets.Output() for mode in STRATEGIES}
        super().__init__()
        self.image_choice.description = "Example image"
        for mode, button in self.example_buttons.items():
            button.on_click(lambda _, mode=mode: self.show_example(mode))

    def _show_mode(self) -> None:
        """Show each strategy's settings on its own page, whatever strategy was shown last."""
        size = self.form.cell.size.value
        style.set_visible(self.form.cell.fixed, size == "fixed")
        style.set_visible(self.scan_button, size not in {"auto", "fixed"})

    def welcome(self) -> None:
        """Show that the examples are ready, with the package's version."""
        display(style.page("CellPatcher strategy examples are ready", f"CellPatcher {html.escape(__version__)}", [
            widgets.HTML(style.status("ok", "Run the next cells from top to bottom.", [
                "Step 1: the preprocessing run and the example image. Then one example per patching strategy.",
            ])),
        ]))

    def run_step(self) -> None:
        """Show step 1, the CellPatcher notebook's, and the image the examples patch."""
        super().run_step()
        display(style.card("Example image", [
            self.image_choice,
            style.help_text("Every example patches this image of the run (filled in once the run is checked)."),
        ]))

    def strategy_step(self, mode: str) -> None:
        """Show one strategy's page: what it gives, its settings and its button."""
        name, description = STRATEGIES[mode]
        controls = list(self.form.mode_widgets()[mode])
        if mode == "cell":
            controls = [self.form.mask_type, *controls, self.scan_button, self.settings_message]
        elif mode == "random":
            controls = [*controls, self.form.mask_type]
        display(style.page(f"{name} strategy", description, [
            style.card("Settings", controls or [style.help_text("This strategy has no settings.")]),
            style.card("Example", [self.example_buttons[mode], self.example_messages[mode],
                                   self.example_outputs[mode]]),
        ]))

    def show_example(self, mode: str) -> Preview | None:
        """Patch the example image with *mode* and its settings, in memory, and draw the patches."""
        message, output = self.example_messages[mode], self.example_outputs[mode]
        self.form.mode.value = mode
        message.value = style.working(f"Patching {self.image_choice.value or 'the image'}…")
        preview = guarded(message, lambda: preview_image(self.config(), self._summary(), self.image_choice.value))
        output.clear_output()
        if preview is not None:
            name = STRATEGIES[mode][0]
            patches = preview.result.patches
            message.value = style.status("ok", f"{name}: {len(patches)} patches in {preview.image}", [
                "Rectangles: where the patches are, on the first plane and channel. Below: the first patches.",
                "Nothing was written. Use these settings in the CellPatcher notebook to export the patches.",
            ])
            with output:
                draw_preview(preview)
        return preview
