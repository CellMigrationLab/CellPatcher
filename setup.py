from setuptools import find_packages, setup

setup(
    name="cellpatcher",
    version="0.8.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
        # below adds its OME-Zarr reader at the same tag. v0.6.0 is Spec ADR 0005
        # steps 2 and 2b (resolver contract v3, RunInputs.read_objects, the
        # conversion helpers); v0.6.1 fixes the checked-fingerprint race; v0.7.0
        # adds RunStoppedError / RunReportUnreadableError for a stopped run (#322).
        "morphobricks-preprocessing @ git+https://github.com/CellMigrationLab/"
        "morphobricks-preprocessing@v0.7.0",
        # CellPatcher imports the normalization brick directly (plane histograms,
        # the plane table, pooling, normalization parameters, SINGLE), so it
        # declares it, at the tag the preprocessing tag above pins (owner
        # decision D-h; tests/test_dependency_pins.py checks that they agree). v0.7.0 added
        # regions.check_region and contracts.refuse_resolver_version (Spec ADR 0005);
        # v0.7.1 writes a signed zero as 0.0 (no API or message change).
        "morphobricks-normalization @ git+https://github.com/CellMigrationLab/"
        "morphobricks-normalization.git@v0.7.1",
    ],
    extras_require={
        "test": [
            "pytest",
            # tests/test_installers.py pip-installs the installer's layouts (a venv made
            # without pip, such as uv's, has none).
            "pip",
            # tests/test_installers.py runs the release script .tools/python/bump_constructor.py
            # (the installer's file list), which reads construct.yaml with PyYAML.
            "pyyaml",
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
            "morphobricks-preprocessing@v0.7.0",
        ],
    },
)
