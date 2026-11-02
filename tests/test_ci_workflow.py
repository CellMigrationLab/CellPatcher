"""CellPatcher's CI workflow has the jobs a template sync must not drop.

The workflow is CellPatcher's own file (.github/workflows/cellpatcher_tests.yml), outside the
LabConstrictor template: the template's .template_sync migrations do not write it, and its own
tests.yml (template-only) has another name. These checks read the file as text, so they run without
PyYAML in every test environment.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "cellpatcher_tests.yml"


def _job(name):
    """Return the text of one top-level job of the workflow."""
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)", text, re.M | re.S)
    assert match is not None, f"no job {name!r} in {WORKFLOW.name}"
    return match.group(1)


def test_the_workflow_is_not_named_as_a_template_workflow():
    template_names = {
        "tests.yml", "check_app_manifest.yml", "check_schema_drift.yml", "release_constructor.yml",
        "sync_template.yml", "update_on_notebook_change.yml", "update_on_src_change.yaml",
    }
    assert WORKFLOW.name not in template_names
    assert not (WORKFLOWS / "tests.yml").exists()


def test_the_rules_job_runs_the_code_rules():
    job = _job("rules")
    assert "python tools/check.py --rules" in job
    assert 'ruff==0.17.3' in job


def test_the_pytest_job_runs_every_extras_combination():
    job = _job("pytest")
    assert "extras: ['test', 'test,zarr', 'test,zarr,notebook']" in job
    assert 'pip install -e ".[${{ matrix.extras }}]"' in job
    assert "python -m pytest" in job


def test_the_torch_job_runs_everything_with_pinned_cpu_torch_and_no_skips():
    job = _job("pytest-torch")
    version = re.search(r"TORCH_VERSION: '(\d+\.\d+\.\d+)'", job)
    assert version is not None, "torch must be pinned to an exact version"
    assert '"torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu' in job
    assert 'pip install -e ".[test,zarr,notebook]"' in job
    assert "python -m pytest -q -rs" in job
    assert "skipped" in job and "exit 1" in job


def test_the_template_sync_runs_only_by_hand():
    # CellPatcher's change to a template file (owner decision of 2026-11-02, docs/developer_notes.md):
    # the template's sync_template.yml runs weekly; CellPatcher's runs on workflow_dispatch only.
    text = (WORKFLOWS / "sync_template.yml").read_text(encoding="utf-8")
    triggers = re.search(r"^on:\n(.*?)(?=^\S)", text, re.M | re.S)
    assert triggers is not None, "sync_template.yml has no on: block"
    active = [line for line in triggers.group(1).splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert active == ["  workflow_dispatch:"], active
    assert not re.search(r"^\s*(schedule|- cron):", text, re.M)


def test_every_action_of_the_ci_is_pinned_by_commit_sha():
    # As the template's workflows are (LabConstrictor migration 0.1.18): a moved tag cannot change
    # what CI runs.
    uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", WORKFLOW.read_text(encoding="utf-8"), re.M)
    assert uses
    for action in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action
