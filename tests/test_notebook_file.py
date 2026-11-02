"""The CellPatcher notebook file: LabConstrictor's portability rules, its release pin, and that it runs (CP-3).

The rules follow LabConstrictor's notebook portability guide, as preprocessing's notebook does
(morphobricks-preprocessing PR #118): every code cell starts with ``# @title``; Colab-only code
(Drive, the widget manager, pip) runs inside ``if "google.colab" in sys.modules:``; no shell or
line magics and no ``/content`` path outside it; the GitHub install is pinned to one release tag;
``current_version`` and ``notebook_name`` as LabConstrictor's version check reads them. Every
code cell's source is hidden (LabConstrictor's code-hiding guide), and the cells only call the
package (AGENTS.md: no backend logic in notebooks).
"""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "CellPatcher" / "CellPatcher.ipynb"
VERSIONS = ROOT / "notebooks" / "notebook_latest_versions.yaml"
HIDDEN = {
    "cellView": "form",
    "collapsed": False,
    "jupyter": {"source_hidden": True},
    "jupyterlabHideCode": {"locked": True},
}
STEP_CALLS = [
    "notebook.run_step()",
    "notebook.settings_step()",
    "notebook.preview_step()",
    "notebook.export_step()",
]


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells():
    return ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"]


def _markdown():
    return "\n".join("".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "markdown")


def _setting(name):
    match = re.search(rf'^{name} = "([^"]*)"$', _code_cells()[0], re.M)
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


def test_notebook_is_valid_and_has_no_outputs():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert "widgets" not in notebook["metadata"]  # no saved widget state (the old file held 0.8 MB)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)


def test_every_code_cell_is_titled_and_hidden():
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            first_line = "".join(cell["source"]).splitlines()[0]
            assert first_line.startswith("# @title Run to "), first_line
            assert cell["metadata"] == HIDDEN, first_line


def test_colab_only_code_is_guarded_and_no_magics_are_used():
    for number, source in enumerate(_code_cells()):
        assert not re.search(r"^\s*[!%]", source, re.M), "magics only run in IPython"
        assert "get_ipython" not in source
        guarded = _colab_guard_lines(source) if number == 0 else set()
        for line_number, line in enumerate(source.splitlines(), start=1):
            colab = "google.colab" in line and "sys.modules" not in line
            if colab or "/content" in line:
                assert line_number in guarded, f"Colab-only code outside the guard: {line!r}"


def test_colab_guard_mounts_drive_enables_the_widget_manager_and_installs_zarr():
    source = _code_cells()[0]
    guarded = "\n".join(
        line for number, line in enumerate(source.splitlines(), start=1) if number in _colab_guard_lines(source)
    )
    assert 'drive.mount("/content/drive")' in guarded
    assert "output.enable_custom_widget_manager()" in guarded
    assert 'EXTRAS = "notebook,zarr"' in guarded
    # The old notebook needed a GITHUB_TOKEN secret; the repository is public and the install pinned.
    assert "GITHUB_TOKEN" not in source and "userdata" not in source


def test_github_install_is_pinned_to_the_release_tag():
    tag = _setting("RELEASE_TAG")
    assert re.fullmatch(r"v\d+\.\d+\.\d+(rc\d+)?", tag), tag
    setup = _code_cells()[0]
    assert setup.count("git+") == 1
    assert '@ git+{REPOSITORY}@{RELEASE_TAG}"' in setup
    assert _setting("REPOSITORY") == "https://github.com/CellMigrationLab/CellPatcher.git"
    # The notebook's version is its release, and the Colab callout names the tag installed.
    assert _setting("current_version") == tag.removeprefix("v")
    assert f"**{tag}**" in _markdown()


def test_version_metadata_matches_labconstrictor_index_and_the_package():
    import cellpatcher

    assert _setting("notebook_name") == NOTEBOOK.stem
    # LabConstrictor's sync_notebook_versions.py writes "<folder>": "<notebook>": "<version>".
    text = VERSIONS.read_text(encoding="utf-8")
    assert re.search(rf'^"{NOTEBOOK.stem}":\n  "{NOTEBOOK.stem}": "{re.escape(_setting("current_version"))}"$',
                     text, re.M), text
    assert cellpatcher.__version__ == _setting("current_version")


def test_notebook_keeps_processing_in_the_package():
    sources = [source.split("\n", 1)[-1].strip() for source in _code_cells()]
    # Step 0 must work before the package or third-party dependencies are installed.
    setup_imports = set()
    for node in ast.walk(ast.parse(sources[0])):
        if isinstance(node, ast.Import):
            setup_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            setup_imports.add(node.module.split(".")[0])
    assert setup_imports <= {"subprocess", "sys", "importlib", "pathlib", "platform", "google"}
    assert sources[1:] == STEP_CALLS


def test_notebook_names_buttons_and_fields_that_exist():
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    from cellpatcher.notebook_ui import CellPatcherNotebook

    nb = CellPatcherNotebook()
    text = _markdown()
    for control in (nb.check_button, nb.scan_button, nb.preview_button, nb.export_button, nb.moved):
        assert f"**{control.description.strip()}**" in text
    for box in (nb.run_folder, nb.output_folder, nb.form.mask_type, nb.form.cell.size, nb.image_choice):
        assert f"**{box.description}**" in text


def test_notebook_cells_run(capsys):
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    from cellpatcher.notebook_ui import CellPatcherNotebook

    namespace: dict = {}
    for source in _code_cells():
        exec(source, namespace)
    assert isinstance(namespace["notebook"], CellPatcherNotebook)
    assert "Local Jupyter: Python" in capsys.readouterr().out


def test_notebook_runs_in_a_kernel():
    # The whole file, as Jupyter runs it: a fresh kernel started in the notebook's folder.
    pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
    pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")
    pytest.importorskip("ipykernel", reason="the test extra installs ipykernel")
    nbclient = pytest.importorskip("nbclient", reason="the test extra installs nbclient")
    nbformat = pytest.importorskip("nbformat", reason="the test extra installs nbformat")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    )
    client.execute()
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [output for cell in code for output in cell.outputs]
    assert not [output for output in outputs if output.output_type == "error"]
    setup_text = "".join(o.get("text", "") for o in code[0].outputs)
    assert "Local Jupyter: Python" in setup_text
    assert "Using the local checkout" in setup_text
    for cell in code:  # Step 0's ready card and every step's page are widgets
        assert any(
            "application/vnd.jupyter.widget-view+json" in output.get("data", {}) for output in cell.outputs
        ), cell.source.splitlines()[0]


def test_step_1_sends_users_to_preprocessing_first():
    # Owner decision (2026-11-02): until preprocessing v0.4.0 is final the notebook does not embed
    # the preprocessing steps; Step 1 says plainly to run the preprocessing notebook or CLI first.
    cells = ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "markdown"]
    (step_1,) = [cell for cell in cells if "Step 1" in cell.splitlines()[1]]
    assert "**Run preprocessing first.**" in step_1
    assert "MorphoBricks preprocessing notebook" in step_1 and "morphobricks-preprocessing check" in step_1
