from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("torch")

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExportConfig, ExtractionConfig, GridConfig
from cellpatcher.extract import extract_patch
from cellpatcher.masking import apply_label_mask_to_patch
from cellpatcher.pipeline import run_patching
from cellpatcher.preprocess import iter_image_planes_for_patching
from cellpatcher.resize import resize_patch
from cellpatcher.torch_dataset import PatchTensorConfig, PatchTorchDataset, patch_array_to_tensor
from cellpatcher.types import PatchRegion


def test_tensor_config_validation():
    PatchTensorConfig()
    with pytest.raises(ValueError):
        PatchTensorConfig(channel_policy="bad")
    with pytest.raises(ValueError):
        PatchTensorConfig(dtype_policy="bad")
    with pytest.raises(ValueError):
        PatchTensorConfig(scale_policy="bad")
    with pytest.raises(ValueError):
        PatchTensorConfig(squeeze_singleton_zt="yes")
    with pytest.raises(ValueError):
        PatchTensorConfig(reject_non_singleton_zt=1)
    with pytest.raises(ValueError):
        PatchTensorConfig(return_metadata="false")


def test_patch_array_to_tensor_core_behaviors():
    t = patch_array_to_tensor(np.array([[0, 255]], dtype=np.uint8), axes="YX")
    assert tuple(t.shape) == (3, 1, 2)
    assert str(t.dtype).endswith("float32")
    assert float(t.max()) == pytest.approx(1.0)

    t = patch_array_to_tensor(np.ones((2, 3), dtype=np.uint8), axes="YX", config=PatchTensorConfig(channel_policy="keep"))
    assert tuple(t.shape) == (1, 2, 3)

    t = patch_array_to_tensor(np.ones((3, 4, 5), dtype=np.uint8), axes="CYX")
    assert tuple(t.shape) == (3, 4, 5)

    t = patch_array_to_tensor(np.ones((4, 5, 3), dtype=np.uint8), axes="YXC")
    assert tuple(t.shape) == (3, 4, 5)

    t = patch_array_to_tensor(np.ones((1, 4, 5), dtype=np.uint8), axes="CYX")
    assert tuple(t.shape) == (3, 4, 5)

    t = patch_array_to_tensor(np.ones((4, 5, 1), dtype=np.uint8), axes="YXC")
    assert tuple(t.shape) == (3, 4, 5)

    t = patch_array_to_tensor(np.ones((2, 4, 5), dtype=np.uint8), axes="CYX", config=PatchTensorConfig(channel_policy="keep"))
    assert tuple(t.shape) == (2, 4, 5)

    t = patch_array_to_tensor(np.ones((4, 5, 4), dtype=np.uint8), axes="YXC", config=PatchTensorConfig(channel_policy="keep"))
    assert tuple(t.shape) == (4, 4, 5)

    with pytest.raises(ValueError, match="channel_policy='keep'"):
        patch_array_to_tensor(np.ones((2, 4, 5), dtype=np.uint8), axes="CYX")

    t = patch_array_to_tensor(np.ones((1, 4, 5), dtype=np.uint16), axes="ZYX")
    assert tuple(t.shape) == (3, 4, 5)

    with pytest.raises(ValueError, match="prepare_image_for_patching"):
        patch_array_to_tensor(np.ones((2, 4, 5), dtype=np.uint16), axes="ZYX")

    t = patch_array_to_tensor(np.ones((1, 4, 5), dtype=np.uint16), axes="TYX")
    assert tuple(t.shape) == (3, 4, 5)

    with pytest.raises(ValueError, match="iter_image_planes_for_patching"):
        patch_array_to_tensor(np.ones((2, 4, 5), dtype=np.uint16), axes="TYX")

    t = patch_array_to_tensor(np.array([[0.5, 2.0]], dtype=np.float64), axes="YX")
    assert float(t.max()) == pytest.approx(2.0)

    t = patch_array_to_tensor(
        np.array([[1, 2]], dtype=np.uint16),
        axes="YX",
        config=PatchTensorConfig(dtype_policy="keep", channel_policy="keep"),
    )
    assert "uint16" in str(t.dtype)


def test_patch_array_to_tensor_scale_and_dtype_policies():
    arr = np.array([[0, 65535]], dtype=np.uint16)

    t_auto = patch_array_to_tensor(arr, axes="YX", config=PatchTensorConfig(channel_policy="keep"))
    assert str(t_auto.dtype).endswith("float32")
    assert float(t_auto.max()) == pytest.approx(1.0)

    t_none = patch_array_to_tensor(
        arr,
        axes="YX",
        config=PatchTensorConfig(channel_policy="keep", dtype_policy="float32", scale_policy="none"),
    )
    assert str(t_none.dtype).endswith("float32")
    assert float(t_none.max()) == pytest.approx(65535.0)

    # scale_policy is ignored when dtype_policy='keep'
    t_keep_none = patch_array_to_tensor(
        arr,
        axes="YX",
        config=PatchTensorConfig(channel_policy="keep", dtype_policy="keep", scale_policy="none"),
    )
    t_keep_auto = patch_array_to_tensor(
        arr,
        axes="YX",
        config=PatchTensorConfig(channel_policy="keep", dtype_policy="keep", scale_policy="auto"),
    )
    assert "uint16" in str(t_keep_none.dtype)
    np.testing.assert_array_equal(t_keep_none.numpy(), t_keep_auto.numpy())


def test_patch_array_to_tensor_invalid_axes_and_noncontiguous():
    with pytest.raises(ValueError):
        patch_array_to_tensor(np.ones((3, 4), dtype=np.uint8), axes="CX")

    with pytest.raises(ValueError):
        patch_array_to_tensor(np.ones((3, 4), dtype=np.uint8), axes="CYX")

    arr = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    arr_nc = np.transpose(arr, (0, 2, 1))  # non-contiguous CYX-like with changed spatial order
    assert not arr_nc.flags.c_contiguous
    t = patch_array_to_tensor(arr_nc, axes="CYX", config=PatchTensorConfig(channel_policy="keep"))
    assert tuple(t.shape) == (3, 5, 4)
    assert t.is_contiguous()


def test_patch_torch_dataset_core_behaviors():
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    patches = [PatchRegion(patch_id="p1", image_id="image_0000", y0=0, y1=4, x0=0, x1=4, label_id=1)]

    ds = PatchTorchDataset(images=image, patches=patches, image_axes="YX")
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample, dict)
    assert tuple(sample["image"].shape) == (3, 4, 4)
    assert sample["patch_id"] == "p1"

    ds_plain = PatchTorchDataset(
        images=image,
        patches=patches,
        image_axes="YX",
        tensor_config=PatchTensorConfig(return_metadata=False),
    )
    tensor = ds_plain[0]
    assert tuple(tensor.shape) == (3, 4, 4)

    ds_transform = PatchTorchDataset(
        images=image,
        patches=patches,
        image_axes="YX",
        transform=lambda x: x + 2,
        tensor_config=PatchTensorConfig(return_metadata=False),
    )
    assert float(ds_transform[0].min()) >= float(ds_plain[0].min()) + 2

    with pytest.raises(ValueError, match="Unknown image_id"):
        PatchTorchDataset(images=image, patches=[PatchRegion(patch_id="x", image_id="bad", y0=0, y1=2, x0=0, x1=2)])


def test_patch_torch_dataset_mask_resize_and_extra_metadata():
    image = np.ones((8, 8), dtype=np.uint8) * 10
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 3
    patch = PatchRegion(patch_id="p1", image_id="image_0000", y0=1, y1=7, x0=1, x1=7, label_id=3)

    extraction = ExtractionConfig(apply_mask=True, output_size=4)
    ds = PatchTorchDataset(
        images=image,
        patches=[patch],
        image_axes="YX",
        masks=mask,
        mask_axes="YX",
        extraction=extraction,
        extra_metadata_by_image_id={"image_0000": {"plane_id": "z000"}},
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (3, 4, 4)
    assert sample["extra_metadata"]["plane_id"] == "z000"

    with pytest.raises(ValueError, match="requires masks"):
        PatchTorchDataset(images=image, patches=[patch], image_axes="YX", extraction=ExtractionConfig(apply_mask=True))

    with pytest.raises(ValueError, match="patch.label_id"):
        PatchTorchDataset(
            images=image,
            patches=[PatchRegion(patch_id="p2", image_id="image_0000", y0=1, y1=3, x0=1, x1=3)],
            image_axes="YX",
            masks=mask,
            extraction=ExtractionConfig(apply_mask=True),
        )


def test_patch_torch_dataset_helper_parity_resize_only():
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    patch = PatchRegion(patch_id="p1", image_id="image_0000", y0=1, y1=7, x0=2, x1=8)
    extraction = ExtractionConfig(output_size=(4, 6), apply_mask=False)
    cfg = PatchTensorConfig(channel_policy="keep", return_metadata=False)

    ds = PatchTorchDataset(images=image, patches=[patch], image_axes="YX", extraction=extraction, tensor_config=cfg)
    actual = ds[0].numpy()

    expected = extract_patch(image, patch, axes="YX", pad_mode=extraction.pad_mode, constant_values=extraction.constant_values)
    expected = resize_patch(expected, axes="YX", output_size=extraction.output_size, order=extraction.resize_order)
    expected = patch_array_to_tensor(expected, axes="YX", config=cfg).numpy()

    np.testing.assert_allclose(actual, expected)


def test_patch_torch_dataset_helper_parity_mask_and_resize():
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[2:6, 2:6] = 5
    patch = PatchRegion(patch_id="p1", image_id="image_0000", y0=1, y1=7, x0=1, x1=7, label_id=5)

    extraction = ExtractionConfig(apply_mask=True, output_size=5, mask_fill_value=17)
    cfg = PatchTensorConfig(channel_policy="keep", return_metadata=False)
    ds = PatchTorchDataset(
        images=image,
        patches=[patch],
        image_axes="YX",
        masks=labels,
        mask_axes="YX",
        extraction=extraction,
        tensor_config=cfg,
    )
    actual = ds[0].numpy()

    expected = extract_patch(image, patch, axes="YX", pad_mode=extraction.pad_mode, constant_values=extraction.constant_values)
    label_patch = extract_patch(labels, patch, axes="YX", pad_mode=extraction.pad_mode, constant_values=0)
    expected = apply_label_mask_to_patch(expected, label_patch, label_id=patch.label_id, image_axes="YX", label_axes="YX", fill_value=17)
    expected = resize_patch(expected, axes="YX", output_size=5, order=extraction.resize_order)
    expected = patch_array_to_tensor(expected, axes="YX", config=cfg).numpy()

    np.testing.assert_allclose(actual, expected)


def test_patch_torch_dataset_lazy_init_sentinel(monkeypatch):
    calls = {"n": 0}

    def wrapped_extract(*args, **kwargs):
        calls["n"] += 1
        return extract_patch(*args, **kwargs)

    monkeypatch.setattr("cellpatcher.torch_dataset.extract_patch", wrapped_extract)

    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    patch = PatchRegion(patch_id="p1", image_id="image_0000", y0=0, y1=4, x0=0, x1=4)
    ds = PatchTorchDataset(images=image, patches=[patch], image_axes="YX")
    assert calls["n"] == 0
    assert len(ds) == 1
    _ = ds[0]
    assert calls["n"] >= 1


def test_patch_torch_dataset_multichannel_and_plane_lookup_bridge():
    image = np.ones((2, 8, 8), dtype=np.uint16)
    patch = PatchRegion(patch_id="p", image_id="image_0000", y0=0, y1=4, x0=0, x1=4)
    ds = PatchTorchDataset(
        images=image,
        patches=[patch],
        image_axes="CYX",
        tensor_config=PatchTensorConfig(channel_policy="keep", return_metadata=False),
    )
    assert tuple(ds[0].shape) == (2, 4, 4)

    image_yxc = np.ones((8, 8, 4), dtype=np.uint16)
    ds_yxc = PatchTorchDataset(
        images=image_yxc,
        patches=[patch],
        image_axes="YXC",
        tensor_config=PatchTensorConfig(channel_policy="keep", return_metadata=False),
    )
    assert tuple(ds_yxc[0].shape) == (4, 4, 4)

    planes = list(iter_image_planes_for_patching(np.zeros((2, 6, 6)), "ZYX", z_policy="all"))
    patches = [PatchRegion(patch_id="z0", image_id="image_0000", y0=0, y1=3, x0=0, x1=3)]
    lookup = {f"image_{i:04d}": {"plane_id": p.plane_id, **p.metadata} for i, p in enumerate(planes)}
    ds2 = PatchTorchDataset(images=[p.array for p in planes], patches=patches, image_axes=planes[0].axes, extra_metadata_by_image_id=lookup)
    assert ds2[0]["extra_metadata"]["plane_id"] == "z000"


def test_patch_torch_dataset_real_fixture_smoke():
    fixture = pytest.importorskip("tests.fixtures.helpers")
    image, labels = fixture.load_real_tiff_fixture()
    from cellpatcher.config import CellPatcherConfig, GridConfig

    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=128, overlap=0.0, include_partial=False))
    cfg.common.image_axes = "YX"
    result = run_patching(images=image, config=cfg)
    ds = PatchTorchDataset(images=image, patches=result.patches, image_axes="YX")
    assert len(ds) == len(result.patches)
    assert len(ds) > 0
    sample = ds[0]
    assert tuple(sample["image"].shape)[0] == 3
    assert str(sample["image"].dtype).endswith("float32")
    assert "patch_id" in sample
    assert "image_id" in sample
    assert "patch" in sample
    assert "label_id" in sample


def _write_csv(path, fieldnames, rows):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_exported_patch_torch_dataset_basic_npy_load(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    b = np.ones((4, 4), dtype=np.uint8)
    np.save(tmp_path / "p1.npy", a)
    np.save(tmp_path / "p2.npy", b)
    csv_path = tmp_path / "patches.csv"
    _write_csv(csv_path, ["patch_id", "image_id", "label_id"], [
        {"patch_id": "p1", "image_id": "image_0000", "label_id": "1"},
        {"patch_id": "p2", "image_id": "image_0001", "label_id": "2"},
    ])

    ds = ExportedPatchTorchDataset(csv_path)
    assert len(ds) == 2
    sample = ds[0]
    assert tuple(sample["image"].shape) == (3, 4, 4)
    assert sample["patch_id"] == "p1"
    assert sample["image_id"] == "image_0000"
    assert sample["label_id"] == "1"
    assert "patch" in sample and "path" in sample


def test_exported_patch_torch_dataset_return_tensor_transform_and_extra_metadata(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    np.save(tmp_path / "p1.npy", np.zeros((2, 2), dtype=np.uint8))
    csv_path = tmp_path / "patches.csv"
    _write_csv(csv_path, ["patch_id", "image_id"], [{"patch_id": "p1", "image_id": "image_0000"}])

    ds = ExportedPatchTorchDataset(
        csv_path,
        tensor_config=PatchTensorConfig(return_metadata=False),
        transform=lambda x: x + 1,
    )
    t = ds[0]
    assert tuple(t.shape) == (3, 2, 2)
    assert float(t.min()) == pytest.approx(1.0)

    ds_meta = ExportedPatchTorchDataset(
        csv_path,
        extra_metadata_by_image_id={"image_0000": {"plane_id": "z001"}},
    )
    assert ds_meta[0]["extra_metadata"]["plane_id"] == "z001"


def test_exported_patch_torch_dataset_filename_column_and_relative_path(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    sub = tmp_path / "nested"
    sub.mkdir()
    np.save(sub / "custom.npy", np.ones((3, 3), dtype=np.uint8))
    csv_path = tmp_path / "patches.csv"
    _write_csv(csv_path, ["patch_id", "filename"], [{"patch_id": "p1", "filename": "nested/custom.npy"}])

    ds = ExportedPatchTorchDataset(csv_path, filename_column="filename")
    sample = ds[0]
    assert sample["path"].endswith("nested/custom.npy")


def test_exported_patch_torch_dataset_error_cases(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    with pytest.raises(FileNotFoundError):
        ExportedPatchTorchDataset(tmp_path / "missing.csv")

    bad_csv = tmp_path / "bad.csv"
    _write_csv(bad_csv, ["image_id"], [{"image_id": "image_0000"}])
    with pytest.raises(ValueError, match="patch_id"):
        ExportedPatchTorchDataset(bad_csv)

    csv_missing_patch = tmp_path / "missing_patch.csv"
    _write_csv(csv_missing_patch, ["patch_id"], [{"patch_id": "not_here"}])
    with pytest.raises(FileNotFoundError):
        ExportedPatchTorchDataset(csv_missing_patch)[0]


def test_exported_patch_torch_dataset_ambiguous_and_prefer_format(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    arr = np.ones((3, 3), dtype=np.uint8)
    np.save(tmp_path / "p1.npy", arr)
    from cellpatcher.io import write_array

    write_array(tmp_path / "p1.tif", arr)

    csv_path = tmp_path / "patches.csv"
    _write_csv(csv_path, ["patch_id"], [{"patch_id": "p1"}])

    # Default prefer_format=None: ambiguous → raises ValueError
    with pytest.raises(ValueError, match="Multiple exported patch files"):
        ExportedPatchTorchDataset(csv_path)[0]

    # Explicit prefer_format resolves ambiguity
    ds = ExportedPatchTorchDataset(csv_path, prefer_format="npy")
    assert ds[0]["path"].endswith("p1.npy")

    ds_tif = ExportedPatchTorchDataset(csv_path, prefer_format="tif")
    assert ds_tif[0]["path"].endswith("p1.tif")


def test_exported_patch_torch_dataset_axes_conversion_yxc(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    np.save(tmp_path / "p1.npy", np.ones((4, 5, 3), dtype=np.uint8))
    csv_path = tmp_path / "patches.csv"
    _write_csv(csv_path, ["patch_id"], [{"patch_id": "p1"}])

    ds = ExportedPatchTorchDataset(
        csv_path,
        axes="YXC",
        tensor_config=PatchTensorConfig(channel_policy="keep"),
    )
    assert tuple(ds[0]["image"].shape) == (3, 4, 5)


def test_exported_patch_torch_dataset_integration_with_run_patching_export(tmp_path):
    from cellpatcher.config import CellPatcherConfig, CommonConfig, ExportConfig, GridConfig
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    cfg = CellPatcherConfig(mode="grid")
    cfg.common = CommonConfig(image_axes="YX", output_dir=str(tmp_path), extract_arrays=False)
    cfg.grid = GridConfig(tile_size=32, overlap=0.0, include_partial=False)
    cfg.export = ExportConfig(enabled=True, image_format="npy", save_metadata_csv=True)

    result = run_patching(images=image, config=cfg)
    ds = ExportedPatchTorchDataset(result.metadata_file, axes=cfg.common.image_axes)
    sample = ds[0]
    assert tuple(sample["image"].shape)[0] == 3


def test_real_tiff_fixture_exported_torch_dataset(tmp_path):
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset

    fixture = pytest.importorskip("tests.fixtures.helpers")
    image, _labels = fixture.load_real_tiff_fixture()

    cfg = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=128, overlap=0.0, include_partial=False))
    cfg.common = CommonConfig(image_axes="YX", output_dir=str(tmp_path), extract_arrays=False)
    cfg.export = ExportConfig(enabled=True, image_format="npy", save_metadata_csv=True)

    result = run_patching(images=image, config=cfg)
    assert result.metadata_file is not None

    dataset = ExportedPatchTorchDataset(result.metadata_file, axes=cfg.common.image_axes)
    assert len(dataset) == len(result.patches)
    assert len(dataset) > 0

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert tuple(sample["image"].shape)[0] == 3
    assert str(sample["image"].dtype).endswith("float32")
    assert sample["patch_id"]
    assert sample["image_id"]
    assert sample["path"]
    assert Path(sample["path"]).exists()
    assert sample["patch"]["patch_id"] == sample["patch_id"]


def test_real_tiff_fixture_cell_apply_mask_torch_dataset():
    fixture = pytest.importorskip("tests.fixtures.helpers")
    image, labels = fixture.load_real_tiff_fixture()

    cfg = CellPatcherConfig(mode="cell")
    cfg.common.image_axes = "YX"
    cfg.common.mask_axes = "YX"
    cfg.cell.patch_size = "auto"
    cfg.cell.exclude_border = True
    cfg.extraction.apply_mask = True
    cfg.extraction.output_size = 32

    result = run_patching(images=image, masks=labels, config=cfg)
    ds = PatchTorchDataset(
        images=image,
        patches=result.patches,
        image_axes=cfg.common.image_axes,
        masks=labels,
        mask_axes=cfg.common.mask_axes,
        extraction=cfg.extraction,
    )

    assert len(ds) == len(result.patches)
    assert len(ds) > 0
    sample = ds[0]
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert "label_id" in sample
