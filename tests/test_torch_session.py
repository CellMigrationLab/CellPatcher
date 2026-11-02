"""A CellPatcher export as PyTorch tensors, for the PyTorch notebook (cellpatcher.torch_session, CP-3b).

The notebook reads the CellPatcher notebook's export with ExportedPatchTorchDataset, with the
arrays' axes from patches.csv (output_axes) and every channel as exported. Skipped without torch
(the torch extra).
"""

import numpy as np
import pytest
import tifffile

pytest.importorskip("torch", reason="the torch extra installs torch")

import torch

from cellpatcher import CellPatcherConfig
from cellpatcher.notebook_session import export_patches_of_run, summarize_run
from cellpatcher.torch_session import load_exported_batch


@pytest.fixture
def clean(tmp_path):
    from morphobricks_preprocessing.testing import make_clean_run

    return make_clean_run(tmp_path / "clean", axes="CYX", shape=(2, 32, 40), mask_types=("Cells",))


def _export(clean, folder, **normalization):
    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 16
    for name, value in normalization.items():
        setattr(config.normalization, name, value)
    results = export_patches_of_run(config, summarize_run(clean.output), folder)
    return [patch for leaf in results for patch in leaf.result.patches]


def test_the_batch_is_the_exported_patches(clean, tmp_path):
    patches = _export(clean, tmp_path / "out")
    batch = load_exported_batch(tmp_path / "out", batch_size=5)
    assert (batch.n_patches, batch.axes, tuple(batch.images.shape)) == (len(patches), "CYX", (5, 2, 16, 16))
    assert batch.images.dtype == torch.float32
    for index, patch_id in enumerate(batch.patch_ids):
        (patch,) = [p for p in patches if p.patch_id == patch_id]
        exported = tifffile.imread(tmp_path / "out" / patch.patch_file)
        assert np.array_equal(batch.images[index].numpy(), exported.astype(np.float32))  # every channel kept
        assert np.array_equal(batch.first_channels[index], exported[0].astype(np.float32))


def test_raw_files_are_normalized_on_load(clean, tmp_path):
    _export(clean, tmp_path / "normalized", enabled=True, scope="image")
    _export(clean, tmp_path / "raw", enabled=True, scope="image", export_values="raw")
    normalized = load_exported_batch(tmp_path / "normalized")
    raw = load_exported_batch(tmp_path / "raw")
    assert normalized.patch_ids == raw.patch_ids
    assert torch.equal(normalized.images, raw.images)


def test_patches_of_different_sizes_are_refused_never_resized(clean, tmp_path):
    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size, config.grid.include_partial, config.grid.partial_mode = 16, True, "clipped"
    export_patches_of_run(config, summarize_run(clean.output), tmp_path / "out")
    with pytest.raises(ValueError, match="different sizes"):
        load_exported_batch(tmp_path / "out")


@pytest.mark.parametrize(("folder", "size", "message"), [
    ("", 8, "No folder holds no patches.csv"),
    ("missing", 8, "holds no patches.csv"),
    ("out", 0, "at least one patch"),
])
def test_what_cannot_be_loaded_is_refused(clean, tmp_path, folder, size, message):
    _export(clean, tmp_path / "out")
    with pytest.raises(ValueError, match=message):
        load_exported_batch(str(tmp_path / folder) if folder else "  ", batch_size=size)


def test_the_notebook_page_loads_the_export(clean, tmp_path):
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from cellpatcher.notebook_torch import TorchNotebook

    _export(clean, tmp_path / "out")
    notebook = TorchNotebook()
    notebook.welcome()
    notebook.patches_step()
    notebook.output_folder.value = str(tmp_path / "out")
    notebook.batch_size.value = 4
    notebook.load_button.click()
    assert "4 of 16 patches loaded as one batch" in notebook.message.value
    assert "(4, 2, 16, 16)" in notebook.message.value and "patch arrays CYX" in notebook.message.value
    notebook.output_folder.value = str(tmp_path)
    assert notebook.load() is None
    assert "holds no patches.csv" in notebook.message.value
