from setuptools import find_packages, setup

setup(
    name="cellpatcher",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "tifffile",
        "imageio",
    ],
    extras_require={
        "test": [
            "pytest",
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
        # Plane statistics of the MorphoBricks preprocessing brick, used for
        # normalization (docs/normalization_contract.md §1). Pinned to its tag
        # and raised deliberately. Histograms, the plane table, pooling and
        # normalization parameters moved to the normalization brick, which
        # CellPatcher imports directly; the preprocessing tag brings it (v0.2.0
        # for v0.2.0), with no separate pin, so every MorphoBricks brick
        # installs one version (MorphoFeature decision D79).
        "preprocessing": [
            "morphobricks-preprocessing[zarr] @ git+https://github.com/CellMigrationLab/"
            "morphobricks-preprocessing@v0.2.0",
        ],
    },
)
