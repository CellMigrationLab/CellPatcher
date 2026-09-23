from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from cellpatcher.axes import axis_index, normalize_axes, require_spatial_axes
from cellpatcher.config import ExtractionConfig
from cellpatcher.extract import extract_patch
from cellpatcher.masking import apply_label_mask_to_patch
from cellpatcher.resize import resize_patch
from cellpatcher.io import read_array
from cellpatcher.export import sanitize_filename

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in runtime import checks
    raise ImportError("Install torch to use cellpatcher.torch_dataset.") from exc


@dataclass
class PatchTensorConfig:
    channel_policy: str = "repeat_to_rgb"
    dtype_policy: str = "float32"
    scale_policy: str = "auto"
    squeeze_singleton_zt: bool = True
    reject_non_singleton_zt: bool = True
    return_metadata: bool = True

    def __post_init__(self) -> None:
        if self.channel_policy not in {"repeat_to_rgb", "keep"}:
            raise ValueError("channel_policy must be one of: 'repeat_to_rgb', 'keep'.")
        if self.dtype_policy not in {"float32", "keep"}:
            raise ValueError("dtype_policy must be one of: 'float32', 'keep'.")
        if self.scale_policy not in {"auto", "none"}:
            raise ValueError("scale_policy must be one of: 'auto', 'none'.")
        for name in ("squeeze_singleton_zt", "reject_non_singleton_zt", "return_metadata"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool.")


def _squeeze_zt_axes(array: np.ndarray, axes: str, config: PatchTensorConfig) -> tuple[np.ndarray, str]:
    # Torch tensors built here are always 2D-spatial (+ optional channel);
    # T/Z axes of size 1 are just dropped, while a real (non-singleton) T/Z
    # axis means the array wasn't reduced to a single plane upstream and
    # this function refuses to guess which plane to use.
    out = array
    out_axes = axes
    for axis_name in ("T", "Z"):
        if axis_name in out_axes:
            idx = axis_index(out_axes, axis_name)
            size = out.shape[idx]
            if size != 1 and config.reject_non_singleton_zt:
                raise ValueError(
                    "Non-singleton T/Z axes are not supported in patch_array_to_tensor. "
                    "Use prepare_image_for_patching, iter_image_planes_for_patching, "
                    "or iter_cell_planes_for_patching before building Torch datasets."
                )
            if size == 1 and config.squeeze_singleton_zt:
                out = np.take(out, indices=0, axis=idx)
                out_axes = out_axes.replace(axis_name, "")
    return out, out_axes


def patch_array_to_tensor(array, *, axes, config=None):
    cfg = config if config is not None else PatchTensorConfig()
    arr = np.asarray(array)
    ax = normalize_axes(axes, arr.ndim)
    arr, ax = _squeeze_zt_axes(arr, ax, cfg)

    y_idx, x_idx = require_spatial_axes(ax)
    c_idx = axis_index(ax, "C") if "C" in ax else None

    # Torch's convention is channel-first (C, H, W); get there from
    # whatever axis order the array is actually in.
    if c_idx is None:
        # No channel axis at all: add one of size 1.
        chw = arr[np.newaxis, ...]
    else:
        if c_idx == 0 and y_idx == 1 and x_idx == 2 and ax == "CYX":
            chw = arr
        elif ax == "YXC":
            chw = np.transpose(arr, (2, 0, 1))
        else:
            chw = np.moveaxis(arr, c_idx, 0)

    if chw.ndim != 3:
        raise ValueError("patch_array_to_tensor expects a 2D spatial patch with optional channel axis.")

    channels = int(chw.shape[0])
    if cfg.channel_policy == "repeat_to_rgb":
        # Grayscale (1-channel) is broadcast to 3 channels for models that
        # expect RGB input; already-3-channel data passes through unchanged.
        if channels == 1:
            chw = np.repeat(chw, 3, axis=0)
        elif channels != 3:
            raise ValueError(
                f"channel_policy='repeat_to_rgb' requires 1 or 3 channels, got {channels}. "
                "Use channel_policy='keep' for other channel counts."
            )

    if cfg.dtype_policy == "float32":
        chw = chw.astype(np.float32, copy=False)
        # Integer source data is assumed to span its dtype's full range, so
        # dividing by that max rescales it into [0, 1] like a normalized
        # float image; float source data is left as-is.
        if cfg.scale_policy == "auto" and np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            if info.max > 0:
                chw = chw / float(info.max)
    # torch.from_numpy requires a C-contiguous array; the axis moves above
    # can leave a non-contiguous view.
    chw = np.ascontiguousarray(chw)
    tensor = torch.from_numpy(chw)
    return tensor


class PatchTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        patches,
        *,
        image_axes="YX",
        masks=None,
        mask_axes="YX",
        extraction=None,
        tensor_config=None,
        transform=None,
        extra_metadata_by_image_id=None,
    ):
        if isinstance(images, (list, tuple)):
            self._images = list(images)
        else:
            self._images = [images]
        if len(self._images) == 0:
            raise ValueError("images must contain at least one image.")

        self._patches = list(patches)
        self._image_axes = image_axes
        self._mask_axes = mask_axes
        self._extraction = extraction if extraction is not None else ExtractionConfig()
        self._tensor_config = tensor_config if tensor_config is not None else PatchTensorConfig()
        self._transform = transform
        self._extra_metadata = extra_metadata_by_image_id or {}

        self._image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(self._images)}

        self._masks = None
        if masks is not None:
            if isinstance(masks, (list, tuple)):
                self._masks = list(masks)
            else:
                self._masks = [masks]
            if len(self._masks) != len(self._images):
                raise ValueError("When masks are provided, number of masks must match number of images.")
            self._mask_lookup = {f"image_{idx:04d}": mask for idx, mask in enumerate(self._masks)}
        else:
            self._mask_lookup = {}

        # Fail fast at construction time (not lazily in __getitem__) if any
        # patch references an image that wasn't provided.
        for patch in self._patches:
            if patch.image_id is None or patch.image_id not in self._image_lookup:
                raise ValueError(f"Unknown image_id for dataset patch: {patch.image_id}")

        if self._extraction.apply_mask:
            if self._masks is None:
                raise ValueError("extraction.apply_mask=True requires masks to be provided.")
            for patch in self._patches:
                if patch.label_id is None:
                    raise ValueError("extraction.apply_mask=True requires every patch.label_id to be set.")

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, index):
        # Extraction/masking/resizing happen lazily per __getitem__ call
        # (rather than eagerly for the whole dataset up front), so this
        # mirrors _maybe_extract_arrays' three steps but on a single patch
        # at load time instead of the whole batch at once.
        patch = self._patches[index]
        image = self._image_lookup[patch.image_id]

        array = extract_patch(
            image,
            patch,
            axes=self._image_axes,
            pad_mode=self._extraction.pad_mode,
            constant_values=self._extraction.constant_values,
        )

        if self._extraction.apply_mask:
            label_patch = extract_patch(
                self._mask_lookup[patch.image_id],
                patch,
                axes=self._mask_axes,
                pad_mode=self._extraction.pad_mode,
                constant_values=0,
            )
            array = apply_label_mask_to_patch(
                array,
                label_patch,
                label_id=patch.label_id,
                image_axes=self._image_axes,
                label_axes=self._mask_axes,
                fill_value=self._extraction.mask_fill_value,
            )

        if self._extraction.output_size is not None:
            array = resize_patch(
                array,
                axes=self._image_axes,
                output_size=self._extraction.output_size,
                order=self._extraction.resize_order,
            )

        tensor = patch_array_to_tensor(array, axes=self._image_axes, config=self._tensor_config)
        if self._transform is not None:
            tensor = self._transform(tensor)

        if not self._tensor_config.return_metadata:
            return tensor

        return {
            "image": tensor,
            "patch_id": patch.patch_id,
            "image_id": patch.image_id,
            "label_id": patch.label_id,
            "patch": patch.to_dict(),
            "extra_metadata": self._extra_metadata.get(patch.image_id, {}),
        }


def _load_exported_patch(path):
    in_path = Path(path)
    suffix = in_path.suffix.lower()
    if suffix == ".npy":
        return np.load(in_path)
    return read_array(in_path)


class ExportedPatchTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_file,
        *,
        root_dir=None,
        axes="YX",
        tensor_config=None,
        transform=None,
        extra_metadata_by_image_id=None,
        filename_column=None,
        prefer_format=None,
    ):
        self._metadata_file = Path(metadata_file)
        if not self._metadata_file.exists():
            raise FileNotFoundError(f"metadata_file not found: {self._metadata_file}")

        self._root_dir = Path(root_dir) if root_dir is not None else self._metadata_file.parent
        self._axes = axes
        self._tensor_config = tensor_config if tensor_config is not None else PatchTensorConfig()
        self._transform = transform
        self._extra_metadata = extra_metadata_by_image_id or {}
        self._filename_column = filename_column

        if prefer_format is not None:
            fmt = str(prefer_format).lower()
            if fmt not in {"npy", "tif", "tiff"}:
                raise ValueError("prefer_format must be one of: 'npy', 'tif', 'tiff'.")
            self._prefer_format = fmt
        else:
            self._prefer_format = None

        # Rows are read into memory once at construction time (rather than
        # streamed per __getitem__ call) so len()/random access both work
        # without re-reading the CSV.
        with self._metadata_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "patch_id" not in fieldnames:
                raise ValueError("metadata_file must contain a 'patch_id' column.")
            self._rows = list(reader)

    def __len__(self):
        return len(self._rows)

    def _resolve_row_path(self, row: dict) -> Path:
        patch_id = row.get("patch_id")
        if not patch_id:
            raise ValueError("Each metadata row must contain a non-empty patch_id.")

        if self._filename_column is not None:
            if self._filename_column not in row:
                raise ValueError(f"filename_column '{self._filename_column}' is not present in metadata row.")
            filename_value = row.get(self._filename_column)
            if filename_value is None or str(filename_value).strip() == "":
                raise ValueError(f"filename_column '{self._filename_column}' is empty for patch_id '{patch_id}'.")
            out_path = Path(filename_value)
            if not out_path.is_absolute():
                out_path = self._root_dir / out_path
            if not out_path.exists():
                raise FileNotFoundError(f"Exported patch file not found for patch_id '{patch_id}': {out_path}")
            return out_path

        # A `patch_file` column (written by export_patches on every current
        # export) records the exact exported filename. Prefer it over
        # guessing from patch_id, unless the caller asked for a specific
        # resolution strategy via prefer_format.
        if self._prefer_format is None:
            patch_file_value = row.get("patch_file")
            if patch_file_value and str(patch_file_value).strip() != "":
                out_path = Path(patch_file_value)
                if not out_path.is_absolute():
                    out_path = self._root_dir / out_path
                if not out_path.exists():
                    raise FileNotFoundError(f"Exported patch file not found for patch_id '{patch_id}': {out_path}")
                return out_path

        # Last resort: neither filename_column nor patch_file resolved a
        # path, so guess the export filename from patch_id across every
        # supported extension (prefer_format, if set, is tried first).
        stem = sanitize_filename(patch_id)
        all_formats = ["npy", "tif", "tiff"]
        ordered = (
            [self._prefer_format] + [ext for ext in all_formats if ext != self._prefer_format]
            if self._prefer_format is not None
            else all_formats
        )
        candidates = [self._root_dir / f"{stem}.{ext}" for ext in ordered]
        existing = [p for p in candidates if p.exists()]

        if len(existing) == 1:
            return existing[0]
        if len(existing) == 0:
            raise FileNotFoundError(
                f"No exported patch file found for patch_id '{patch_id}' in {self._root_dir}. "
                f"Tried formats: {ordered}"
            )
        # Multiple formats exist: resolve using prefer_format if set
        if self._prefer_format is not None:
            return existing[0]  # candidates already ordered with prefer_format first
        raise ValueError(
            f"Multiple exported patch files found for patch_id '{patch_id}': {', '.join(str(p) for p in existing)}. "
            "Set prefer_format or provide filename_column."
        )

    def __getitem__(self, index):
        row = self._rows[index]
        path = self._resolve_row_path(row)
        array = _load_exported_patch(path)

        tensor = patch_array_to_tensor(array, axes=self._axes, config=self._tensor_config)
        if self._transform is not None:
            tensor = self._transform(tensor)

        if not self._tensor_config.return_metadata:
            return tensor

        image_id = row.get("image_id")
        return {
            "image": tensor,
            "patch_id": row.get("patch_id"),
            "image_id": image_id,
            "label_id": row.get("label_id"),
            "patch": row,
            "extra_metadata": self._extra_metadata.get(image_id, {}),
            "path": str(path),
        }
