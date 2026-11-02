# Proposals for the LabConstrictor template

> See also: [Developer notes, "LabConstrictor template"](developer_notes.md#labconstrictor-template)

CellPatcher synced the [LabConstrictor](https://github.com/CellMigrationLab/LabConstrictor)
template to version 0.1.19 (LabConstrictor `139efea`) on 2026-11-02 with the template's
`.template_sync` migrations, and had to change some template files again afterwards. These are
proposals for the template, written for its maintainers. **Nothing has been opened on
LabConstrictor**: the owner decides which of them to send, as issues or pull requests.

Each proposal names what CellPatcher does today, so that adopting it upstream lets CellPatcher drop
its change at the next sync.

## 1. Release candidates: no installer, said clearly (`bump_version.py`, `release_constructor.yml`)

**Today.** `.tools/python/bump_version.py` accepts only `X.Y.Z`; a release tagged `v0.2.0rc1`
stops the release workflow with `Version must look like X.Y.Z`, which reads like a bug. CellPatcher
publishes release candidates for pip and Colab only (no installer, owner decision of 2026-11-02),
and added a refusal before the `X.Y.Z` check that names the reason:

```python
if re.fullmatch(r"\d+\.\d+\.\d+(?:a|b|rc)\d+", args.new_version):
    sys.exit(f"{args.new_version} is a pre-release: no installer is built for it. ...")
```

**Proposal.**
- In `release_constructor.yml`, skip both jobs for a GitHub pre-release
  (`if: github.event_name != 'release' || !github.event.release.prerelease`), so a pre-release
  neither builds installers nor fails.
- In `bump_version.py`, recognise PEP 440 pre-releases (`a`, `b`, `rc`; also SemVer
  `-rc.1`) and refuse them with that reason before any file is changed, for releases not marked
  as pre-releases and for manual runs.
- Optionally, a setting (for example `build_prerelease_installers: false` in a template config
  file) for apps that do want installers for release candidates.

## 2. An extension point for an app's own CI jobs

**Today.** Before 0.1.19 the sync workflow overwrote `.github/` and `.tools/` wholesale, so an
app's own test workflow (CellPatcher's code-rules job and extras matrix in `tests.yml`) was
dropped at every sync. The migrations now write only the files they name, which solves most of
this, but nothing says which workflow names belong to the template; the template's own
`tests.yml` (template-only, guarded by `github.repository`) uses the most common name an app
would choose. CellPatcher moved its CI to `.github/workflows/cellpatcher_tests.yml`.

**Proposal.**
- Document the workflow files the template owns (`check_app_manifest.yml`,
  `release_constructor.yml`, `sync_template.yml`, `update_on_notebook_change.yml`,
  `update_on_src_change.yaml`) and promise that migrations never create, change or delete any
  other file in `.github/workflows/`.
- Rename the template's own `tests.yml` and `check_schema_drift.yml` (for example
  `template_tests.yml`) so they cannot meet an app's file of the same name.
- Optionally, a `workflow_call` hook in `release_constructor.yml` (for example a job that runs
  `.github/workflows/app_release_checks.yml` when it exists) so an app can gate a release on its
  own tests without editing a template workflow.

## 3. A list of files an app has changed (`.template_sync`)

**Today.** The migrations have no exclude or override list. The older migrations replace a whole
file from a resource copy (0.1.12 to 0.1.13 rewrote `bump_version.py` and dropped CellPatcher's
refusal without a word); the newer ones (0.1.16 and later) skip a file they do not recognise and
say so. CellPatcher records its changes in its developer notes and guards each with a test, which
fails the sync pull request if one is dropped.

**Proposal.**
- An app-owned file, for example `.template_sync/local_changes.yaml` (the bootstrap checkout,
  `git checkout upstream/main -- .template_sync`, keeps a file the template does not have), listing files the app has changed, each with a
  reason. A migration that would write a listed file leaves it unchanged and prints what it would
  have done (or writes `<file>.template` beside it) so the app can merge by hand.
- Make every migration that replaces a whole file check first that the file is a known template
  version (as 0.1.16 to 0.1.18 do), and skip it with a message otherwise.
- Recognise known historical template versions when deciding "customised": migration 0.1.18
  skipped CellPatcher's `update_on_notebook_change.yml` as customised although it was the
  template's own file (of March 2026, plus migration 0.1.1's edit); CellPatcher pinned its actions
  by hand as the message asked.

## 4. Installing the app's package with its dependencies and extras (`post_install`)

**Today.** `post_install.sh` and `post_install.bat` install the app's package with
`pip install --no-deps --no-build-isolation <app>`: everything must be in `requirements.txt`.
CellPatcher's dependencies are in `setup.py` (the MorphoBricks bricks, from GitHub at pinned tags), and it needs its `zarr` extra. CellPatcher changed the line to
`pip install --no-build-isolation "<app>[zarr]"` in both scripts.

**Proposal.** A setting for the package install, for example in a template config file or in
`labconstrictor-app.yaml`: the extras to install and whether to install the
package's dependencies (default: today's `--no-deps`). The scripts would read it instead of
being edited.

## 5. Smaller findings from the sync

- **Empty icons (migration 0.1.10).** An initialised launcher with `"icon": ""` (no logo uploaded)
  is rewritten to `app/logo/<name>_logo.ico/.png/.icns`, files that do not exist. Keep an empty
  icon, or use the derived path only when the file exists. CellPatcher restored `""`.
- **`requirements_gpu.txt` created empty (migration 0.1.5, `ensure_requirements_gpu`).** The
  same migration ships it in `construct.yaml`, and `post_install` installs it instead of
  `requirements.txt` on a computer with an NVIDIA GPU: until `merge_requirements.py` runs again
  (only on a notebook or requirements change) such an install gets none of the requirements. Run
  the merge in the migration, or copy `requirements.txt`. CellPatcher generated it with the synced
  `merge_requirements.py`.
- **Windows "Apps" version (`post_install.bat`).** `DISPLAY_VERSION` is rendered once from
  `construct.yaml` and `bump_version.py` updates only a `PKG_VERSION` line, which the current
  script does not have, so the Windows Apps list keeps showing the version of the sync (0.1.2 in
  CellPatcher). Bump `DISPLAY_VERSION` in `bump_version.py`, or read it from `construct.yaml`
  at install time as `LC_APP_VERSION` already is.
- **Partial updates of a changed script.** Migrations patch selected hunks of
  `bump_constructor.py`, so an app's copy differs from the template's current file (CellPatcher's
  has no `ensure_app_manifest_in_extra_files`). Shipping the whole script, rendered with the app's
  names, when the app's copy is a known template version would keep apps identical to the template.
- **Machine paths in generated files.** `merge_requirements.py` writes the absolute paths of its
  source files into the header of `requirements.txt`/`requirements_gpu.txt` (a runner path in
  CI, a home folder locally). Repository-relative paths would make the files reproducible.
