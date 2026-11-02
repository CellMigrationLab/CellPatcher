# Developer notes

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

## Architecture

CellPatcher is coordinate-first: strategies generate `PatchRegion` metadata, then downstream steps optionally filter, extract arrays, apply label masks, resize, and export. Keep mode logic in the package backend and keep notebooks as API consumers.

## Pipeline order

`generate regions → filter → extract → apply mask → resize → export`

## Preprocessing boundary

C/Z/T handling is explicit preprocessing (`prepare_*` helpers and iterators). Core patching modes remain 2D `Y/X` and do not perform true 3D patching or automatic model prep.

## Notebook hygiene

- The CellPatcher notebook only creates `cellpatcher.notebook_ui.CellPatcherNotebook` and calls
  one step per cell; its actions are `cellpatcher.notebook_session` (no widgets, tested
  without them) and its pages `cellpatcher.notebook_ui`/`notebook_form`. The strategy examples
  notebook is `cellpatcher.notebook_examples.StrategyExamplesNotebook` (the same actions, one
  page per strategy) and the PyTorch notebook `cellpatcher.notebook_torch.TorchNotebook`
  (its action is `cellpatcher.torch_session.load_exported_batch`). The before/after views of the
  preview are `cellpatcher.normalization_review` (pure; histograms from the normalization brick).
- Every notebook's first cell is the same (`tests/test_notebook_file.py` checks it), except its
  title, `notebook_name`, `EXTRAS` and `INTERFACE` (`"module:class"` of its page); change all
  three together.
- LabConstrictor portability (`.tools/docs/notebook_portability.md`): `# @title` first on every
  code cell; Colab-only code (Drive, the widget manager, pip through `subprocess`) only inside
  `if "google.colab" in sys.modules:`; no magics and no `/content` path outside it; the GitHub
  install pinned to `RELEASE_TAG`; `current_version`/`notebook_name` with the
  `notebooks/notebook_latest_versions.yaml` entry. `tests/test_notebook_file.py` checks these
  for every notebook and runs each in a local kernel (nbclient; the strategy examples and the
  PyTorch notebook with a test run or export filled in; the PyTorch notebook only with `torch`
  installed, as in the CI job `pytest-torch`).
- Before tagging a release: set `RELEASE_TAG` and `current_version` in every notebook's first
  cell and in `notebooks/notebook_latest_versions.yaml`, `version` in `setup.py` and
  `__version__` to the new version.
- **No installer for a release candidate** (owner decision of 2026-11-02): an `rc` tag
  (`v0.2.0rc1`) is installed with pip or on Colab only. The release workflow's
  `.tools/python/bump_version.py` refuses a pre-release version with that reason before it
  changes anything, so no installer is built; the installers are built at the final release
  (`v0.2.0`). `tests/test_installers.py` checks the refusal (a CellPatcher change to a template
  file, see "LabConstrictor template" below).
- **Manual installer test before the final `v0.2.0`** (owner decision of 2026-11-02): the
  `v0.2.0` installers are the first built with the synced template's scripts, and CI cannot build
  or run an installer. Before the final `v0.2.0` is announced, install the release's installer on
  **Windows, macOS and Linux** (a computer without CellPatcher) and check on each that:
  1. the installer finishes without an error (on failure, `<install folder>/menuinst_debug.log`
     says which step stopped);
  2. the CellPatcher shortcut launches JupyterLab;
  3. `import cellpatcher` works in the app's Python (`<install folder>/bin/python`, or
     `python.exe` on Windows), and `cellpatcher.__file__` is in that environment;
  4. the CellPatcher notebook (`Welcome.ipynb` and the notebooks it links) opens in the app;
  5. the bricks import in the app's Python: `import morphobricks_preprocessing,
     morphobricks_normalization, zarr` (the package is installed with its dependencies and the
     `zarr` extra, see "LabConstrictor template").
  Record the operating systems and versions tested in the release notes.
- Strip outputs before committing; avoid duplicating backend patching logic in notebooks.

## LabConstrictor template

The installers, the release and notebook workflows and `.tools/` come from the
[LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor) template. The repository is
at template version **0.1.19** (`.template_sync/manifest.yaml`; synced on 2026-11-02 from
LabConstrictor `139efea`).

**How to sync.** Use the template's own mechanism
(`.tools/docs/template_synchronization.md`): the **Sync with Template Repository** workflow
(`.github/workflows/sync_template.yml`, run by hand: **Actions > Sync with Template Repository
> Run workflow**; CellPatcher has turned off its weekly schedule) compares this repository's
`template_version` with the template's; when the template is newer it checks out the template's
`.template_sync/` and runs its migrations
(`python3 .template_sync/sync.py --from-version <ours> --to-version <theirs>`), then opens a pull
request. It needs the repository secret `LABCONSTRICTOR_SYNC_TOKEN` (a fine-grained token,
contents, pull requests and workflows: read and write); without it the workflow fails when an
update is available. The same can be done by hand in a clone: `git checkout <template>/main --
.template_sync`, then the `sync.py` command. Each migration edits only the files it names, and
the newer ones (0.1.16 and later) leave a file they do not recognise unchanged and say so in their
output; read that output and the pull request's diff. Then run the full gate (`tools/check.py` and
the four test environments below): `tests/test_installers.py` and `tests/test_ci_workflow.py`
fail when a sync drops one of the changes listed below.

(Before 0.1.19 the sync workflow checked out `.github/` and `.tools/` wholesale from the template
and pushed them to `main`, overwriting every change made here. That is why CellPatcher kept a
stale `bump_constructor.py`, and why its changes are listed here and guarded by tests.)

**CellPatcher's own files** (not in the template, so a sync leaves them alone):
`.github/workflows/cellpatcher_tests.yml` (the CI: the code-rules job, the extras matrix and the
CPU-torch job; named so that it does not meet the template's own `tests.yml`, which runs only in
the template repository), `.github/workflows/update_docs_figures.yml`,
`.tools/python/generate_docs_figures.py`.

**CellPatcher's changes to template files** (no exclude list exists in the template's mechanism, so
each is guarded by a test and proposed upstream in
[labconstrictor-upstream-proposals.md](labconstrictor-upstream-proposals.md)):

| File | Change | Why | Guard |
| --- | --- | --- | --- |
| `.tools/python/bump_version.py` | refuses `a`/`b`/`rc` versions before changing anything | no installer for a release candidate (owner decision of 2026-11-02) | `test_no_installer_is_built_for_a_release_candidate` |
| `app/bash_bat_scripts/post_install.sh`, `.bat` | `pip install "<app>[zarr]"`, with dependencies and with build isolation (the template passes `--no-deps --no-build-isolation`) | CellPatcher's dependencies, the MorphoBricks bricks from GitHub at pinned tags, are in `setup.py`, not in `requirements.txt`; the app reads OME-Zarr. pip builds the bricks from source, with their build backend (hatchling), which the app's environment does not have: without build isolation the install fails | `test_post_install_installs_the_package_with_its_dependencies_and_the_zarr_extra` |
| `.github/workflows/sync_template.yml` | no `schedule:` (the template's runs weekly); `workflow_dispatch` only | owner decision of 2026-11-02: template syncs are run by hand | `test_the_template_sync_runs_only_by_hand` |
| `app/menuinst/notebook_launcher.json` | `"icon": ""` on every platform | CellPatcher has no logo; migration 0.1.10 writes `app/logo/CellPatcher_logo.*`, files that do not exist | `test_the_launcher_names_no_icon_file_the_repository_does_not_have` |

`construct.yaml`, `environment.yaml`, `requirements.txt` and `requirements_gpu.txt` are written by
the template's scripts (`bump_constructor.py` at release, `merge_requirements.py` on notebook
changes); do not edit them by hand. `tests/test_installers.py` runs the synced
`bump_constructor.py` and checks that `construct.yaml` lists the package as it writes it, that the
package lands in `<app>/src/cellpatcher/` and that the installed app imports `cellpatcher`.

**Installer package layout (CP-3b, resolved by the sync).** The template's `bump_constructor.py`
before LabConstrictor `a76be61` (2026-07-06) mapped `src/cellpatcher/x.py` to
`CellPatcher/src/cellpatcher/cellpatcher/x.py`, where `find_packages(where="src")` found no
package. CP-3b made `setup.py` read either layout; the synced script maps it to
`CellPatcher/src/cellpatcher/x.py`, and the release workflow runs it before every build, so no
layout needs that workaround any more and `setup.py` reads `src/` only again.

**CI.** `cellpatcher_tests.yml` runs `tools/check.py --rules`, pytest with `.[test]`,
`.[test,zarr]` and `.[test,zarr,notebook]`, and `pytest-torch`: `.[test,zarr,notebook]` with CPU
`torch` pinned (`TORCH_VERSION`, from `https://download.pytorch.org/whl/cpu`), which fails if any
test is skipped, so the torch tests and the PyTorch notebook's kernel run do run. To raise torch,
change `TORCH_VERSION` and run the gate with that version locally
(`pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu`).

## Regenerate docs figures

```bash
PYTHONPATH=src python .tools/python/generate_docs_figures.py
```


## Validation and tests

Before a pull request:

```bash
python tools/check.py           # pytest, Ruff and the code rules
python tools/check.py --rules   # only Ruff and the code rules (CI)
```

The code rules are the MorphoBricks standards (design v0.4 §9): the shared
`tools/code_rules.py` (identical in every MorphoBricks repository, checked by its SHA-256),
configured in `pyproject.toml`. Existing violations, and Ruff's findings counted per file,
function and rule, are in `tools/code_rules_baseline.json`, which may only shrink
(`--update-baseline` lowers it; it never adds an entry), with a ratchet on `# noqa` counts. New
and moved code must comply.

## Demo fixture (for smoke tests and docs demos)

A small real TIFF fixture is bundled for smoke tests and notebook demos:

- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Use it for validation and examples, not for benchmarking or biological conclusions.

## Repository development notes

- Package code lives in `src/cellpatcher`.
- Notebooks should call package APIs and avoid duplicating backend patching logic.
- See `AGENTS.md` for contributor/agent guidance.
- See `notebooks/README.md` for notebook-specific guidance.
