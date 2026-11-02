"""What the installers ship agrees with the package (CP-3).

The LabConstrictor installer (construct.yaml) copies each source file it lists and installs the
package from setup.py, so a module missing from its list is missing from the installed app (the
CP-1 and CP-2 modules were). The release workflow regenerates the list
(.tools/python/bump_constructor.py); this test fails a pull request that leaves it stale.
"""

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


def _installer_files(layout):
    """Return setup.py and the package's files as ``(source, destination)``, as an installer copies them."""
    pairs = re.findall(r"^- (\S+): (\S+)$", (ROOT / "construct.yaml").read_text(encoding="utf-8"), re.M)
    pairs = [(source, target) for source, target in pairs if source == "setup.py" or source.startswith("src/")]
    if layout == "template":
        # LabConstrictor's bump_constructor.py since a76be61 (2026-07-06): src/<path> -> <app>/src/<path>.
        pairs = [(source, f"CellPatcher/{source}" if source.startswith("src/") else destination)
                 for source, destination in pairs]
    return pairs


@pytest.mark.parametrize("layout", ["construct.yaml", "template"])
def test_the_installed_app_imports_cellpatcher(tmp_path, layout):
    # The installer copies the files to the places construct.yaml names, then post_install runs
    # `pip install <prefix>/CellPatcher[zarr]`. construct.yaml, as this repository's (stale) copy of
    # the template writes it, puts the package in src/cellpatcher/cellpatcher/, where
    # find_packages(where="src") found nothing: the app had no cellpatcher (CP-3 review). The
    # current template puts it in src/cellpatcher/. Both must install the whole package.
    app = tmp_path / "prefix"
    for source, destination in _installer_files(layout):
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
