"""What the installers ship agrees with the package (CP-3).

The LabConstrictor installer (construct.yaml) copies each source file it lists and installs the
package from setup.py, so a module missing from its list is missing from the installed app (the
CP-1 and CP-2 modules were). The release workflow regenerates the list
(.tools/python/bump_constructor.py); this test fails a pull request that leaves it stale.
"""

import re
from pathlib import Path

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
