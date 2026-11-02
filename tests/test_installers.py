"""What the installers ship agrees with the package (CP-3).

The LabConstrictor installer (construct.yaml) copies each source file it lists and installs the
package from setup.py, so a module missing from its list is missing from the installed app (the
CP-1 and CP-2 modules were). The release workflow regenerates the list
(.tools/python/bump_constructor.py); this test fails a pull request that leaves it stale.

Some of these tests guard CellPatcher's changes to files of the LabConstrictor template (the
release-candidate refusal, the package install in post_install, the launcher's icons): a template
sync that drops one fails here (docs/developer_notes.md, "LabConstrictor template").
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import cellpatcher

ROOT = Path(__file__).parents[1]


def test_the_installer_ships_every_module_of_the_package():
    construct = (ROOT / "construct.yaml").read_text(encoding="utf-8")
    listed = set(re.findall(r"^- (src/cellpatcher/\S+\.py):", construct, re.M))
    modules = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src" / "cellpatcher").rglob("*.py")}
    assert modules - listed == set()
    assert listed - modules == set()


def test_the_installer_ships_the_notebook_and_the_package_s_setup():
    text = (ROOT / "construct.yaml").read_text(encoding="utf-8")
    assert "- notebooks/CellPatcher/CellPatcher.ipynb:" in text
    assert "- setup.py: CellPatcher/setup.py" in text


def test_the_package_version_is_the_setup_version():
    match = re.search(r'^    version="([^"]+)",$', (ROOT / "setup.py").read_text(encoding="utf-8"), re.M)
    assert match is not None and match.group(1) == cellpatcher.__version__


def test_no_installer_is_built_for_a_release_candidate():
    # Owner decision (2026-11-02): rc releases are pip/Colab only; the release workflow's
    # bump_version.py refuses them, with that reason, before it changes any file. The
    # .tools folder is synced from the LabConstrictor template: this test fails if a sync drops it.
    construct = (ROOT / "construct.yaml").read_bytes()
    for version in ("0.2.0rc1", "0.2.0a1", "0.2.0b2"):
        result = subprocess.run(
            [sys.executable, str(ROOT / ".tools" / "python" / "bump_version.py"), version],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert result.returncode != 0
        assert "is a pre-release: no installer is built for it" in result.stderr
        assert "pip or on Google Colab only" in result.stderr
    assert (ROOT / "construct.yaml").read_bytes() == construct


def _src_and_setup(construct_text):
    """Return setup.py and the package's files as ``(source, destination)``, as an installer copies them."""
    pairs = re.findall(r"^- (\S+): (\S+)$", construct_text, re.M)
    return sorted((source, target) for source, target in pairs if source == "setup.py" or source.startswith("src/"))


def _release_script_construct(tmp_path):
    """Return construct.yaml as the release workflow writes it, by running the synced bump_constructor.py."""
    copy = tmp_path / "repository"
    for name in ("construct.yaml", "setup.py"):
        copy.mkdir(exist_ok=True)
        shutil.copy2(ROOT / name, copy / name)
    for folder in ("src", "notebooks", ".tools/meta"):
        shutil.copytree(ROOT / folder, copy / folder, ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run(
        [sys.executable, str(ROOT / ".tools" / "python" / "bump_constructor.py")],
        cwd=copy, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return (copy / "construct.yaml").read_text(encoding="utf-8")


def test_construct_yaml_lists_the_package_as_the_release_script_writes_it(tmp_path):
    # The release workflow regenerates construct.yaml's list with .tools/python/bump_constructor.py
    # (a template file) before it builds an installer; the committed list must be the same, so the
    # install test below tests what an installer ships. A template sync that changes the mapping
    # fails here.
    committed = (ROOT / "construct.yaml").read_text(encoding="utf-8")
    assert _src_and_setup(_release_script_construct(tmp_path)) == _src_and_setup(committed)


def test_the_installer_puts_the_package_where_setup_py_finds_it():
    # The template's bump_constructor.py before LabConstrictor a76be61 (2026-07-06) copied the
    # package to <app>/src/cellpatcher/cellpatcher/, where setup.py's find_packages(where="src")
    # found nothing: the installed app had no cellpatcher (CP-3 review). The template synced on
    # 2026-11-02 maps src/<path> to <app>/src/<path>, and setup.py reads src/ only.
    pairs = _src_and_setup((ROOT / "construct.yaml").read_text(encoding="utf-8"))
    assert ("setup.py", "CellPatcher/setup.py") in pairs
    assert all(target == f"CellPatcher/{source}" for source, target in pairs if source.startswith("src/"))


def test_the_installed_app_imports_cellpatcher(tmp_path):
    # The installer copies the files to the places construct.yaml names, then post_install runs
    # `pip install <prefix>/CellPatcher[zarr]`: the whole package must install from there.
    app = tmp_path / "prefix"
    for source, destination in _src_and_setup((ROOT / "construct.yaml").read_text(encoding="utf-8")):
        (app / destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, app / destination)
    site = tmp_path / "site"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", "--disable-pip-version-check",
         "--target", str(site), str(app / "CellPatcher")],
        capture_output=True, text=True, check=False,
    )
    assert install.returncode == 0, install.stderr
    installed = {path.relative_to(site).as_posix() for path in (site / "cellpatcher").rglob("*.py")}
    package = {path.relative_to(ROOT / "src").as_posix() for path in (ROOT / "src" / "cellpatcher").rglob("*.py")}
    assert installed == package
    probe = subprocess.run(
        [sys.executable, "-c", "import cellpatcher, cellpatcher.strategies.cell; print(cellpatcher.__file__)"],
        env={**os.environ, "PYTHONPATH": str(site)}, capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(probe.stdout.strip()).parent == site / "cellpatcher"


@pytest.mark.parametrize("script", ["post_install.sh", "post_install.bat"])
def test_post_install_installs_the_package_with_its_dependencies_and_the_zarr_extra(script):
    # CellPatcher's change to a template file (docs/developer_notes.md): the template installs the
    # app's package with --no-deps, but CellPatcher's dependencies (the MorphoBricks bricks, from
    # GitHub) are in setup.py, not in requirements.txt, and the app reads OME-Zarr (zarr extra).
    text = (ROOT / "app" / "bash_bat_scripts" / script).read_text(encoding="utf-8")
    installs = [line for line in text.splitlines() if "pip install" in line and "PROJECT_ROOT" in line
                and "requirements" not in line.lower()]
    assert len(installs) == 1, installs
    assert "--no-deps" not in installs[0]
    assert ("${PROJECT_ROOT}[zarr]" if script.endswith(".sh") else "%PROJECT_ROOT%[zarr]") in installs[0]


def test_the_launcher_names_no_icon_file_the_repository_does_not_have():
    # CellPatcher ships no logo: the template's migration 0.1.10 writes app/logo/<name>_logo.* for
    # an empty icon, a file this repository does not have; the launcher keeps "" (no icon).
    launcher = json.loads((ROOT / "app" / "menuinst" / "notebook_launcher.json").read_text(encoding="utf-8"))
    for platform in launcher["menu_items"][0]["platforms"].values():
        assert platform["icon"] == "" or (ROOT / platform["icon"]).is_file()
