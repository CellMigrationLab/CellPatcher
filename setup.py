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
        # normalization (docs/normalization_contract.md §1). Pinned to a
        # commit and raised deliberately. Histograms, the plane table, pooling
        # and normalization parameters moved to the normalization brick, which
        # CellPatcher imports directly; its URL is the one the preprocessing
        # commit requires, so both resolve to the same install.
        "preprocessing": [
            "morphobricks-preprocessing[zarr] @ git+https://github.com/CellMigrationLab/"
            "morphobricks-preprocessing@112977665fc067726b6f3776c5d44dffd9b3430a",
            "morphobricks-normalization @ git+https://github.com/CellMigrationLab/"
            "morphobricks-normalization.git@v0.1.0",
        ],
    },
)
