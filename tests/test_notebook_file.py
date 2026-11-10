"""The CellPatcher notebook files: LabConstrictor's portability rules, their release pin, and that they run.

The rules follow LabConstrictor's notebook portability guide, as preprocessing's notebook does
(morphobricks-preprocessing PR #118): every code cell starts with ``# @title``; Colab-only code
(Drive, the widget manager, pip) runs inside ``if "google.colab" in sys.modules:``; no shell or
line magics and no ``/content`` path outside it; the GitHub install is pinned to one release tag;
``current_version`` and ``notebook_name`` as LabConstrictor's version check reads them. Every
code cell's source is hidden (LabConstrictor's code-hiding guide), and the cells only call the
package (AGENTS.md: no backend logic in notebooks). CP-3 wrote these rules for the main notebook;
CP-3b applies them to every notebook (the PyTorch and strategy-examples notebooks installed
``main`` with a ``GITHUB_TOKEN`` and computed in their cells), whose Step 0 is the main
notebook's, except for its name, extras and interface.
"""

import ast
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOKS = {
    "CellPatcher": ROOT / "notebooks" / "CellPatcher" / "CellPatcher.ipynb",
    "CellPatcher_Pytorch": ROOT / "notebooks" / "CellPatcher_Pytorch" / "CellPatcher_Pytorch.ipynb",
    "CellPatcher_Strategy_Examples": (
        ROOT / "notebooks" / "CellPatcher_Strategy_Examples" / "CellPatcher_Strategy_Examples.ipynb"
    ),
}
VERSIONS = ROOT / "notebooks" / "notebook_latest_versions.yaml"
HIDDEN = {
    "cellView": "form",
    "collapsed": False,
    "jupyter": {"source_hidden": True},
    "jupyterlabHideCode": {"locked": True},
}
STEP_CALLS = {
    "CellPatcher": [
        "notebook.run_step()",
        "notebook.settings_step()",
        "notebook.preview_step()",
        "notebook.export_step()",
    ],
    "CellPatcher_Pytorch": ["notebook.patches_step()"],
    "CellPatcher_Strategy_Examples": [
        "notebook.run_step()",
        'notebook.strategy_step("cell")',
        'notebook.strategy_step("grid")',
        'notebook.strategy_step("random")',
        'notebook.strategy_step("whole_fov")',
    ],
}
SETTINGS = {
    # notebook: (EXTRAS before Colab adds zarr, INTERFACE)
    "CellPatcher": ("notebook", "cellpatcher.notebook_ui:CellPatcherNotebook"),
    "CellPatcher_Pytorch": ("notebook,torch", "cellpatcher.notebook_torch:TorchNotebook"),
    "CellPatcher_Strategy_Examples": ("notebook", "cellpatcher.notebook_examples:StrategyExamplesNotebook"),
}
EVERY = pytest.mark.parametrize("name", list(NOTEBOOKS))


def _notebook(name="CellPatcher"):
    return json.loads(NOTEBOOKS[name].read_text(encoding="utf-8"))


def _code_cells(name="CellPatcher"):
    return ["".join(c["source"]) for c in _notebook(name)["cells"] if c["cell_type"] == "code"]


def _markdown(name="CellPatcher"):
    return "\n".join("".join(c["source"]) for c in _notebook(name)["cells"] if c["cell_type"] == "markdown")


def _setting(name, notebook="CellPatcher"):
    match = re.search(rf'^{name} = "([^"]*)"$', _code_cells(notebook)[0], re.M)
    assert match is not None, f"the setup cell sets no {name}"
    return match.group(1)


def _colab_guard_lines(source):
    """Return the line numbers inside the setup cell's top-level Colab guard."""
    guards = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "'google.colab' in sys.modules"
    ]
    assert len(guards) == 1, "the setup cell needs one `if 'google.colab' in sys.modules:` block"
    (guard,) = guards
    return set(range(guard.body[0].lineno, guard.body[-1].end_lineno + 1))


def _needs_torch(name):
    if "torch" in SETTINGS[name][0]:
        pytest.importorskip("torch", reason="the PyTorch notebook needs the torch extra")


@EVERY
def test_notebook_is_valid_and_has_no_outputs(name):
    notebook = _notebook(name)
    assert notebook["nbformat"] == 4
    assert "widgets" not in notebook["metadata"]  # no saved widget state (the old file held 0.8 MB)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)


@EVERY
def test_every_code_cell_is_titled_and_hidden(name):
    for cell in _notebook(name)["cells"]:
        if cell["cell_type"] == "code":
            first_line = "".join(cell["source"]).splitlines()[0]
            assert first_line.startswith("# @title Run to "), first_line
            assert cell["metadata"] == HIDDEN, first_line


@EVERY
def test_colab_only_code_is_guarded_and_no_magics_are_used(name):
    for number, source in enumerate(_code_cells(name)):
        assert not re.search(r"^\s*[!%]", source, re.M), "magics only run in IPython"
        assert "get_ipython" not in source
        guarded = _colab_guard_lines(source) if number == 0 else set()
        for line_number, line in enumerate(source.splitlines(), start=1):
            colab = "google.colab" in line and "sys.modules" not in line
            if colab or "/content" in line:
                assert line_number in guarded, f"Colab-only code outside the guard: {line!r}"


@EVERY
def test_colab_guard_mounts_drive_enables_the_widget_manager_and_installs_zarr(name):
    source = _code_cells(name)[0]
    guarded = "\n".join(
        line for number, line in enumerate(source.splitlines(), start=1) if number in _colab_guard_lines(source)
    )
    assert 'drive.mount("/content/drive")' in guarded
    assert "output.enable_custom_widget_manager()" in guarded
    assert 'EXTRAS += ",zarr"' in guarded
    # The old notebooks needed a GITHUB_TOKEN secret; the repository is public and the install pinned.
    for text in (source, *_code_cells(name)[1:], _markdown(name)):
        assert "GITHUB_TOKEN" not in text and "userdata" not in text and "x-access-token" not in text


@EVERY
def test_github_install_is_pinned_to_the_release_tag(name):
    tag = _setting("RELEASE_TAG", name)
    assert re.fullmatch(r"v\d+\.\d+\.\d+(rc\d+)?", tag), tag
    assert tag == _setting("RELEASE_TAG")  # every notebook installs the same release
    code = "\n".join(_code_cells(name))
    assert code.count("git+") == 1 and "git clone" not in code
    assert '@ git+{REPOSITORY}@{RELEASE_TAG}"' in _code_cells(name)[0]
    assert _setting("REPOSITORY", name) == "https://github.com/CellMigrationLab/CellPatcher.git"
    # The notebook's version is its release, and the Colab callout names the tag installed.
    assert _setting("current_version", name) == tag.removeprefix("v")
    assert f"**{tag}**" in _markdown(name)


@EVERY
def test_version_metadata_matches_labconstrictor_index_and_the_package(name):
    import cellpatcher

    notebook = NOTEBOOKS[name]
    assert _setting("notebook_name", name) == notebook.stem
    # LabConstrictor's sync_notebook_versions.py writes "<folder>": "<notebook>": "<version>".
    text = VERSIONS.read_text(encoding="utf-8")
    version = re.escape(_setting("current_version", name))
    assert re.search(rf'^"{notebook.parent.name}":\n  "{notebook.stem}": "{version}"$', text, re.M), text
    assert cellpatcher.__version__ == _setting("current_version", name)


@EVERY
def test_step_0_is_the_main_notebook_s(name):
    # One Step 0 for every notebook: only its title, name, extras and interface differ.
    def general(source):
        source = re.sub(r"^# @title Run to start the .*$", "# @title", source, count=1, flags=re.M)
        return re.sub(r'^(notebook_name|EXTRAS|INTERFACE) = "[^"]*"$', r'\1 = ""', source, flags=re.M)

    assert general(_code_cells(name)[0]) == general(_code_cells()[0])
    extras, interface = SETTINGS[name]
    assert (_setting("EXTRAS", name), _setting("INTERFACE", name)) == (extras, interface)


@EVERY
def test_notebook_keeps_processing_in_the_package(name):
    sources = [source.split("\n", 1)[-1].strip() for source in _code_cells(name)]
    # Step 0 must work before the package or third-party dependencies are installed.
    setup_imports = set()
    for node in ast.walk(ast.parse(sources[0])):
        if isinstance(node, ast.Import):
            setup_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            setup_imports.add(node.module.split(".")[0])
    assert setup_imports <= {"subprocess", "sys", "importlib", "pathlib", "platform", "google"}
    assert sources[1:] == STEP_CALLS[name]


def test_notebook_names_buttons_and_fields_that_exist():
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    from cellpatcher.notebook_ui import CellPatcherNotebook

    nb = CellPatcherNotebook()
    text = _markdown()
    for control in (nb.check_button, nb.scan_button, nb.preview_button, nb.export_button, nb.moved,
                    nb.parameters_button):
        assert f"**{control.description.strip()}**" in text
    for box in (nb.run_folder, nb.output_folder, nb.form.mask_type, nb.form.cell.size, nb.image_choice):
        assert f"**{box.description}**" in text


def test_examples_notebook_names_buttons_and_fields_that_exist():
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    from cellpatcher.notebook_examples import StrategyExamplesNotebook

    nb = StrategyExamplesNotebook()
    text = _markdown("CellPatcher_Strategy_Examples")
    for control in (nb.check_button, nb.scan_button, nb.moved, *nb.example_buttons.values()):
        assert f"**{control.description.strip()}**" in text
    form = nb.form
    for box in (nb.run_folder, nb.image_choice, form.mask_type, form.cell.size, form.cell.exclude_border,
                form.cell.apply_mask, form.cell.components, form.grid_size, form.grid_overlap, form.grid_partial,
                form.random_count, form.random_size, form.random_seed, form.random_masks):
        assert f"**{box.description}**" in text


def test_torch_notebook_names_buttons_and_fields_that_exist():
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    from cellpatcher.notebook_torch import TorchNotebook

    nb = TorchNotebook()
    text = _markdown("CellPatcher_Pytorch")
    assert f"**{nb.load_button.description}**" in text
    for box in (nb.output_folder, nb.batch_size):
        assert f"**{box.description}**" in text


@EVERY
def test_notebook_cells_run(capsys, name):
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    _needs_torch(name)
    module, _, interface = SETTINGS[name][1].partition(":")

    namespace: dict = {}
    for source in _code_cells(name):
        exec(source, namespace)
    assert type(namespace["notebook"]).__name__ == interface
    assert type(namespace["notebook"]).__module__ == module
    assert "Local Jupyter: Python" in capsys.readouterr().out


def _run_in_a_kernel(name, after=None):
    """Run a notebook as Jupyter runs it: a fresh kernel in its folder; *after* adds cells after some.

    On POSIX the kernel talks over IPC sockets in a short temporary folder: with TCP, another
    process on the machine (another test run) can take a port between its choice and its use,
    and the cell then waits for its timeout. The folder is not pytest's ``tmp_path``, whose path
    can exceed a Unix socket path's limit (104 bytes on macOS).
    """
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    pytest.importorskip("ipykernel", reason="the test extra installs ipykernel")
    nbclient = pytest.importorskip("nbclient", reason="the test extra installs nbclient")
    nbformat = pytest.importorskip("nbformat", reason="the test extra installs nbformat")
    _needs_torch(name)

    path = NOTEBOOKS[name]
    notebook = nbformat.read(path, as_version=4)
    if after:
        cells = []
        for cell in notebook.cells:
            cells.append(cell)
            for marker, source in after.items():
                if cell.cell_type == "code" and cell.source.split("\n", 1)[-1].strip() == marker:
                    cells.append(nbformat.v4.new_code_cell(source))
        notebook.cells = cells
    from jupyter_client.manager import KernelManager

    manager = KernelManager(kernel_name="python3")
    sockets = tempfile.mkdtemp(prefix="cpk-") if os.name == "posix" else None
    if sockets is not None:
        manager.transport, manager.ip = "ipc", os.path.join(sockets, "k")
        assert len(manager.ip) < 80, manager.ip  # room for "-<port>" within the socket path limit
    client = nbclient.NotebookClient(
        notebook, km=manager, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    )
    try:
        client.execute()
    finally:  # nbclient shuts down only the kernels of managers it made itself
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if sockets is not None:
            shutil.rmtree(sockets, ignore_errors=True)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [output for cell in code for output in cell.outputs]
    assert not [output for output in outputs if output.output_type == "error"]
    setup_text = "".join(o.get("text", "") for o in code[0].outputs)
    assert "Local Jupyter: Python" in setup_text
    assert "Using the local checkout" in setup_text
    for cell in code:  # Step 0's ready card and every step's page are widgets
        if cell.source.startswith("# @title"):
            assert any(
                "application/vnd.jupyter.widget-view+json" in output.get("data", {}) for output in cell.outputs
            ), cell.source.splitlines()[0]
    return notebook


def test_notebook_runs_in_a_kernel():
    _run_in_a_kernel("CellPatcher")


@pytest.fixture
def clean(tmp_path):
    from tests.helpers import make_clean_run

    return make_clean_run(tmp_path / "clean", axes="CYX", shape=(2, 32, 40), mask_types=("Cells",))


def test_examples_notebook_runs_every_example_in_a_kernel(clean, tmp_path):
    check = (
        f"notebook.run_folder.value = {str(clean.output)!r}\n"
        "assert notebook.check_run() is not None, notebook.run_message.value\n"
        "notebook.form.cell.size.value = 'fixed'\n"
        "notebook.form.cell.fixed.value = 12\n"
        "notebook.form.cell.exclude_border.value = False\n"
        "notebook.form.grid_size.value = 16\n"
        "notebook.form.random_size.value, notebook.form.random_count.value = 8, 3"
    )
    after = {"notebook.run_step()": check}
    for mode in ("cell", "grid", "random", "whole_fov"):
        after[f'notebook.strategy_step("{mode}")'] = (
            f"preview = notebook.show_example({mode!r})\n"
            f"assert preview is not None, notebook.example_messages[{mode!r}].value\n"
            "assert preview.result.patches and all(patch.mode == "
            f"{mode!r} for patch in preview.result.patches)"
        )
    _run_in_a_kernel("CellPatcher_Strategy_Examples", after)


def test_torch_notebook_loads_an_export_in_a_kernel(clean, tmp_path):
    pytest.importorskip("torch", reason="the PyTorch notebook needs the torch extra")
    from cellpatcher import CellPatcherConfig
    from cellpatcher.notebook_session import export_patches_of_run, summarize_run

    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 16
    export_patches_of_run(config, summarize_run(clean.output), tmp_path / "export")
    load = (
        f"notebook.output_folder.value = {str(tmp_path / 'export')!r}\n"
        "batch = notebook.load()\n"
        "assert batch is not None, notebook.message.value\n"
        "assert tuple(batch.images.shape)[0] == 8 and batch.n_patches == 16, batch"
    )
    _run_in_a_kernel("CellPatcher_Pytorch", {"notebook.patches_step()": load})


@EVERY
def test_step_1_sends_users_to_preprocessing_first(name):
    # Owner decision (2026-11-02): the notebooks do not embed the preprocessing steps; the step that
    # opens a run says plainly to run the preprocessing notebook first, with a published description
    # (Spec ADR 0002 AR10: an undescribed run is refused; the description is published in the
    # preprocessing notebook). The PyTorch notebook reads a CellPatcher export, not a run.
    if name == "CellPatcher_Pytorch":
        assert "CellPatcher notebook** (step 4)" in _markdown(name)
        return
    cells = ["".join(c["source"]) for c in _notebook(name)["cells"] if c["cell_type"] == "markdown"]
    (step_1,) = [cell for cell in cells if "Step 1" in cell.splitlines()[1]]
    assert "**Run preprocessing first.**" in step_1
    assert "**MorphoBricks preprocessing notebook** (version 0.5)" in step_1
    assert "**described**" in step_1 and "without a published description is refused" in step_1
    assert "version 0.4" not in _markdown(name)


@EVERY
def test_colab_callouts_keep_the_data_on_drive(name):
    text = _markdown(name)
    assert "**Colab users:**" in text and "**JupyterLab users:**" in text
    assert "/content/drive/MyDrive" in text
