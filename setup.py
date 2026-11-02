from pathlib import Path

from setuptools import find_packages, setup

# Where the package's folder is. In this repository and in LabConstrictor installers built
# with the current template, it is src/cellpatcher. Installers built with the template synced
# into this repository before its fix (.tools/python/bump_constructor.py, fixed upstream in
# LabConstrictor a76be61) copy it to src/cellpatcher/cellpatcher, where find_packages(where="src")
# finds nothing and the installed app has no cellpatcher module; read it from there instead.
# tests/test_installers.py installs both layouts.
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = "src/cellpatcher" if (HERE / "src/cellpatcher/cellpatcher/__init__.py").is_file() else "src"

setup(
    name="cellpatcher",
    version="0.2.0rc1",
    package_dir={"": SOURCE_ROOT},
    packages=find_packages(where=SOURCE_ROOT),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "tifffile",
        "imageio",
        # The MorphoBricks preprocessing brick: image discovery and names
        # (position_id, D85), mask pairing and header checks, plane statistics,
        # manifests and the shared D86 rules (labels, identity text, source
        # checks). Pinned to its tag and raised deliberately; the ``zarr`` extra
        # below adds its OME-Zarr reader at the same tag.
        "morphobricks-preprocessing @ git+https://github.com/CellMigrationLab/"
        "morphobricks-preprocessing@v0.4.0rc2",
        # CellPatcher imports the normalization brick directly (plane histograms,
        # the plane table, pooling, normalization parameters, SINGLE), so it
        # declares it, at the tag the preprocessing tag above pins (owner
        # decision D-h; tests/test_dependency_pins.py checks that they agree).
        "morphobricks-normalization @ git+https://github.com/CellMigrationLab/"
        "morphobricks-normalization.git@v0.3.2",
    ],
    extras_require={
        "test": [
            "pytest",
            # tools/check.py and tests/test_code_rules.py: the code standards (design v0.4
            # §9). Pinned, because the baseline of existing findings is Ruff's at this version.
            "ruff==0.17.3",
            # tests/test_notebook_file.py runs the notebook in a local kernel (with the
            # notebook extra; skipped without it).
            "nbclient",
            "nbformat",
            "ipykernel",
        ],
        "notebook": [
            "ipywidgets",
            "matplotlib",
            "pillow",
            "tqdm",
            "joblib",
            "jl-hidecode",
        ],
        "torch": [
            "torch",
        ],
        # OME-Zarr images and OME-NGFF label masks: the preprocessing brick's
        # zarr reader, at the same tag as the core dependency above. The former
        # name of this extra, ``preprocessing``, is gone: the preprocessing brick
        # is a core dependency.
        "zarr": [
            "morphobricks-preprocessing[zarr] @ git+https://github.com/CellMigrationLab/"
            "morphobricks-preprocessing@v0.4.0rc2",
        ],
    },
)
