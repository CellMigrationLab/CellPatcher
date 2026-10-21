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
        # commit and raised deliberately.
        "preprocessing": [
            "morphobricks-preprocessing[zarr] @ git+https://github.com/CellMigrationLab/"
            "morphobricks-preprocessing@52a0635f6c965b1d5a4bde40b5835272dffd1cca",
        ],
    },
)
