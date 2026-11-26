"""Original MDPI JPEG input with frozen scan/window assignments.

The reference HDF5 is used only to prepare metadata. Its resized ``frames``
datasets are never read. Labels remain the user's interpolated trajectory,
not measured probe poses; units are nominal millimetres.
"""

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA = "mdpi_original_dualtrack_v1"
SPLITS = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROJECT = PROJECT_ROOT.parent / "Sensorless_Small_DS"
DEFAULT_SOURCE_ROOT = SOURCE_PROJECT / "data/MDPI Compiled Data"
DEFAULT_REFERENCE = SOURCE_PROJECT / "data/MDPI_interpolated_v4/dataset.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/mdpi_original_dualtrack_v1"
DEFAULT_PREPROCESSING = {
    "local_crop_xywh": [387, 247, 256, 256],
    "global_roi_xywh": [0, 0, 1030, 750],
    "global_size_xy": [224, 224],
    "normalization": "uint8_div_255",
    "global_interpolation": "INTER_AREA",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source paths must be relative and confined to source_root")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Source path escapes source_root: " + str(relative))
    return result


def _image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot decode original JPEG: " + str(path))
    return image


def _preprocessing(spec, size_xy):
    required = {"local_crop_xywh", "global_roi_xywh", "global_size_xy", "normalization"}
    if not required.issubset(spec):
        raise ValueError("Incomplete preprocessing specification")
    if spec["normalization"] != "uint8_div_255" or spec.get("global_interpolation", "INTER_AREA") != "INTER_AREA":
        raise ValueError("Unsupported normalization/interpolation")
    if spec["global_size_xy"] != [224, 224]:
        raise ValueError("Checkpoint expects 224x224 global input")
    for key in ("local_crop_xywh", "global_roi_xywh"):
        box = spec[key]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Crop coordinates must be four integers")
        x, y, width, height = box
        if min(x, y) < 0 or min(width, height) < 1 or x + width > size_xy[0] or y + height > size_xy[1]:
            raise ValueError("Crop is outside original image: " + key)
    if spec["local_crop_xywh"][2:] != [256, 256]:
        raise ValueError("Checkpoint local spatial attention expects a native 256x256 crop")


def _geometry(spec, size_xy, width_mm):
    """Nominal full-image coordinates, recording crop and pixel-centre resize."""
    source = np.eye(4)
    spacing = width_mm / size_xy[0]
    source[0, 0] = source[1, 1] = spacing
    source[0, 3] = -(size_xy[0] - 1) * spacing / 2
    source[1, 3] = -(size_xy[1] - 1) * spacing / 2
    transforms = {}
    for prefix, key, out_size in (
        ("local", "local_crop_xywh", [256, 256]),
        ("global", "global_roi_xywh", [224, 224]),
    ):
        x, y, width, height = spec[key]
        sx, sy = width / out_size[0], height / out_size[1]
        mapping = np.eye(4)
        mapping[0, 0], mapping[1, 1] = sx, sy
        mapping[0, 3], mapping[1, 3] = x + (sx - 1) / 2, y + (sy - 1) / 2
        transforms[prefix + "_pixel_to_source_pixel"] = mapping.tolist()
        transforms[prefix + "_pixel_to_nominal_image_mm"] = (source @ mapping).tolist()
    return {"source_size_xy": size_xy, "source_width_nominal_mm": width_mm,
            "source_isotropy_is_assumed": True,
            "label_origin": "original full-image canonical frame; fixed R=I",
            **transforms}


def _frozen_windows(eligible, source_indices, length=10, max_gap=3):
    selected = np.flatnonzero(eligible)
    segments = np.split(selected, np.flatnonzero(np.diff(np.asarray(source_indices)[selected]) > max_gap) + 1)
    return [segment[start:start + length].tolist() for segment in segments
            for start in range(0, len(segment) - length + 1, length)]


def validate_manifest(manifest):
    """Validate all split metadata; this never opens source images or volumes."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported original-input dataset schema")
    if manifest.get("coordinate_units") != "nominal_mm_interpolated_reference":
        raise ValueError("Unsupported coordinate units")
    if manifest.get("window_length") != 10 or manifest.get("max_source_gap") != 3:
        raise ValueError("Require frozen ten-frame windows and maximum gap three")
    scans, windows = manifest.get("scans", {}), manifest.get("windows", {})
    if not scans or set(windows) != set(SPLITS):
        raise ValueError("Require scans and all three split window lists")
    ownership = {key: {} for key in ("source_scan", "split_group_id", "decoded_sha256", "volume_sha256")}
    for scan_id, row in scans.items():
        split = row["split"]
        if row["scan_id"] != scan_id or split not in SPLITS:
            raise ValueError("Invalid scan identity/split")
        n = len(row["source_indices"])
        indices = np.asarray(row["source_indices"])
        valid = np.asarray(row["eligible"])
        positions = np.asarray(row["positions_yz"], dtype=np.float64)
        if (n < 10 or indices.dtype.kind not in "iu" or not np.array_equal(indices, np.arange(n))
                or valid.shape != (n,) or valid.dtype != np.bool_
                or positions.shape != (n, 2) or not np.isfinite(positions).all()):
            raise ValueError("Invalid frame indices, eligibility or interpolated labels")
        for name in ("image_paths", "jpg_sha256", "decoded_sha256", "eligibility_reasons"):
            if len(row[name]) != n:
                raise ValueError("Per-frame manifest arrays differ in length")
        expected_paths = [(Path(row["source_scan"]) / "Images" / (str(i) + ".jpg")).as_posix() for i in range(n)]
        if row["image_paths"] != expected_paths:
            raise ValueError("Image paths do not preserve numeric JPG/coordinate association")
        for path in [row["source_scan"], row["source_trajectory"], row["source_volume"], *row["image_paths"]]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Unsafe source path in manifest")
        _preprocessing(manifest["preprocessing"], row["image_size_xy"])
        for kind, values in (
            ("source_scan", [row["source_scan"]]),
            ("split_group_id", [row["split_group_id"]]),
            ("decoded_sha256", row["decoded_sha256"]),
            ("volume_sha256", [row["source_volume_sha256"]]),
        ):
            for value in values:
                if not value:
                    raise ValueError("Missing source identity/hash")
                previous = ownership[kind].setdefault(value, split)
                if previous != split:
                    raise ValueError("Source dependency crosses splits: " + kind)
    actual = defaultdict(list)
    for split in SPLITS:
        if not windows[split]:
            raise ValueError("Empty split: " + split)
        for window in windows[split]:
            scan_id, frames = window["scan_id"], window["frame_indices"]
            if scan_id not in scans or scans[scan_id]["split"] != split:
                raise ValueError("Window belongs to a different scan split")
            if len(frames) != 10 or any(type(i) is not int for i in frames):
                raise ValueError("Windows must contain ten integer frame rows")
            actual[scan_id].append(frames)
    for scan_id, row in scans.items():
        expected = _frozen_windows(row["eligible"], row["source_indices"])
        if actual[scan_id] != expected:
            raise ValueError("Window rows differ from frozen nonoverlap policy: " + scan_id)
    return manifest


def prepare_dataset(source_root=DEFAULT_SOURCE_ROOT, reference_h5=DEFAULT_REFERENCE,
                    output_dir=DEFAULT_OUTPUT, preprocessing=None, progress=None):
    """Validate original bytes/labels and publish a new metadata-only dataset."""
    source_root, reference_h5, output_dir = [Path(p).resolve() for p in (source_root, reference_h5, output_dir)]
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite prepared dataset: " + str(output_dir))
    spec = copy.deepcopy(DEFAULT_PREPROCESSING if preprocessing is None else preprocessing)
    manifest = {"schema": SCHEMA, "source_root": str(source_root),
                "reference_metadata_h5": str(reference_h5), "reference_metadata_sha256": file_sha256(reference_h5),
                "coordinate_units": "nominal_mm_interpolated_reference",
                "label_source": "original coords_interpolated.npy; column i corresponds to JPG i",
                "source_to_world_mapping": "t=(0,-reference_Y,reference_X); R=I; unmeasured tx fixed zero",
                "probe_pose_ground_truth": False,
                "split_scope": "same-corpus held-out scans; acquisition dates shared across splits",
                "window_length": 10, "max_source_gap": 3, "preprocessing": spec,
                "scans": {}, "windows": {split: [] for split in SPLITS}}
    with h5py.File(reference_h5, "r") as ref:
        if _text(ref.attrs.get("schema_version", "")) != "mdpi_interpolated_v4":
            raise ValueError("Reference metadata must use mdpi_interpolated_v4 schema")
        config = json.loads(_text(ref.attrs["config_json"]))
        for scan_id, group in ref["scans"].items():
            source_scan = _text(group.attrs["original_source_scan"])
            folder = _safe_path(source_root, source_scan)
            indices = group["source_frame_indices"][:]
            n = len(indices)
            if not np.array_equal(indices, np.arange(n)):
                raise ValueError("Reference original frame numbering changed")
            candidates = list((folder / "Images").glob("*.jpg"))
            if {p.name for p in candidates} != {str(i) + ".jpg" for i in range(n)}:
                raise ValueError("Original JPG IDs must be exactly 0..N-1: " + source_scan)
            paths = [(Path(source_scan) / "Images" / (str(i) + ".jpg")).as_posix() for i in range(n)]
            jpg_hashes = [_text(v) for v in group["original_jpg_sha256"][:]]
            decoded_hashes = [_text(v) for v in group["decoded_jpg_sha256"][:]]
            constant = []
            size_xy = config["source_size_xy"]
            _preprocessing(spec, size_xy)
            for i, relative in enumerate(paths):
                path = _safe_path(source_root, relative)
                if file_sha256(path) != jpg_hashes[i]:
                    raise ValueError("Original JPG hash mismatch: " + str(path))
                image = _image(path)
                if list(image.shape[::-1]) != size_xy:
                    raise ValueError("Unexpected original JPEG dimensions: " + str(path))
                if hashlib.sha256(image.tobytes()).hexdigest() != decoded_hashes[i]:
                    raise ValueError("Decoded original JPEG hash mismatch: " + str(path))
                constant.append(bool(image.min() == image.max()))
            frequencies = Counter(decoded_hashes)
            reasons = ["duplicate_decoded_image" if frequencies[h] > 1 else "constant_image" if c else ""
                       for h, c in zip(decoded_hashes, constant)]
            eligible = np.asarray([not value for value in reasons], dtype=bool)
            if not np.array_equal(eligible, group["geometry_valid"][:]):
                raise ValueError("Original eligibility differs from frozen metadata")
            if reasons != [_text(v) for v in group["geometry_reason"][:]]:
                raise ValueError("Original exclusion reasons differ from frozen metadata")
            trajectory_path = _text(group.attrs["source_trajectory"])
            trajectory_hash = _text(group.attrs["source_trajectory_sha256"])
            trajectory_file = _safe_path(source_root, trajectory_path)
            if file_sha256(trajectory_file) != trajectory_hash:
                raise ValueError("Original trajectory hash mismatch")
            reference = np.load(trajectory_file, allow_pickle=False)
            if reference.shape != (2, n) or not np.isfinite(reference).all():
                raise ValueError("Original trajectory must be finite 2 by N")
            if not np.array_equal(reference.T, group["reference_trajectory_xy"][:]):
                raise ValueError("Original JPG/coordinate association differs from frozen metadata")
            volume_path = _text(group.attrs["source_volume"])
            volume_hash = _text(group.attrs["source_volume_sha256"])
            if file_sha256(_safe_path(source_root, volume_path)) != volume_hash:
                raise ValueError("Original volume dependency hash mismatch")
            log_path = _text(group.attrs["source_log"])
            log_hash = _text(group.attrs["source_log_sha256"])
            log_file = _safe_path(source_root, log_path)
            if file_sha256(log_file) != log_hash:
                raise ValueError("Original acquisition log hash mismatch")
            log = np.loadtxt(log_file, delimiter=",", ndmin=2)
            if log.shape != (n, 3) or not np.isfinite(log).all() or not np.array_equal(log[:, 0], np.arange(1, n + 1)):
                raise ValueError("Original acquisition log frame rows do not match JPG numbering")
            positions = np.stack([-reference[1], reference[0]], axis=1)
            manifest["scans"][scan_id] = {
                "scan_id": scan_id, "source_scan": source_scan,
                "split": _text(group.attrs["split"]), "split_group_id": _text(group.attrs["split_group_id"]),
                "object": _text(group.attrs["object"]), "method": _text(group.attrs["method"]),
                "acquisition_date": _text(group.attrs["acquisition_date"]),
                "eligible": eligible.tolist(), "eligibility_reasons": reasons,
                "source_indices": indices.tolist(), "positions_yz": positions.tolist(),
                "jpg_sha256": jpg_hashes, "decoded_sha256": decoded_hashes,
                "image_paths": paths, "image_size_xy": size_xy,
                "source_trajectory": trajectory_path, "trajectory_sha256": trajectory_hash,
                "source_volume": volume_path, "source_volume_sha256": volume_hash,
                "source_log": log_path, "source_log_sha256": log_hash,
                "nominal_geometry": _geometry(spec, size_xy, float(config["width_mm"])),
            }
            if progress:
                progress(scan_id, source_scan, n)
        for split in SPLITS:
            group = ref["samples"][split]
            manifest["windows"][split] = [
                {"scan_id": _text(scan_id), "frame_indices": rows.tolist()}
                for scan_id, rows in zip(group["scan_ids"][:], group["frame_indices"][:])]
    validate_manifest(manifest)
    summary = {split: {
        "scans": sum(row["split"] == split for row in manifest["scans"].values()),
        "windows": len(manifest["windows"][split]),
        "frames": sum(len(row["eligible"]) for row in manifest["scans"].values() if row["split"] == split),
        "eligible_frames": sum(sum(row["eligible"]) for row in manifest["scans"].values() if row["split"] == split),
        "strata": dict(Counter(row["object"] + "/" + row["method"] for row in manifest["scans"].values() if row["split"] == split)),
    } for split in SPLITS}
    report = {"schema": SCHEMA, "source_validation": "all original JPEG bytes/decoded pixels, NPY labels, logs and volume dependency hashes match frozen reference",
              "reference_images_read": False, "source_images_resized_or_copied": False,
              "split_crossings": 0, "summary": summary, "preprocessing": spec,
              "builder_sha256": file_sha256(__file__)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report["manifest_sha256"] = file_sha256(manifest_path)
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest_path, report


class MDPIOriginalDataset(Dataset):
    """Lazy ten-frame original-JPEG windows with immutable preprocessing."""

    def __init__(self, manifest_path, split="train", allow_test=False, verify_sources=True):
        if split not in SPLITS or (split == "test" and not allow_test):
            raise ValueError("Test input requires explicit allow_test=True")
        self.manifest_path = Path(manifest_path).resolve()
        if self.manifest_path.is_dir():
            self.manifest_path = self.manifest_path / "manifest.json"
        self.manifest_sha256 = file_sha256(self.manifest_path)
        self.manifest = validate_manifest(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        self.split = split
        self.source_root = Path(self.manifest["source_root"]).resolve()
        self.preprocessing = self.manifest["preprocessing"]
        self.scans = {scan_id: row for scan_id, row in self.manifest["scans"].items() if row["split"] == split}
        self.windows = self.manifest["windows"][split]
        if verify_sources:
            self.verify_sources()

    def verify_sources(self):
        """Check the selected split only; model training never opens test JPEGs."""
        for scan in self.scans.values():
            path = _safe_path(self.source_root, scan["source_trajectory"])
            if file_sha256(path) != scan["trajectory_sha256"]:
                raise ValueError("Original trajectory changed since dataset preparation")
            reference = np.load(path, allow_pickle=False)
            expected = np.stack([-reference[1], reference[0]], axis=1)
            if not np.array_equal(expected, np.asarray(scan["positions_yz"])):
                raise ValueError("Prepared label rows differ from original NPY")
            for relative, digest in zip(scan["image_paths"], scan["jpg_sha256"]):
                if file_sha256(_safe_path(self.source_root, relative)) != digest:
                    raise ValueError("Original JPEG changed since dataset preparation: " + relative)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        return self.load_window(window["scan_id"], window["frame_indices"], window_id=index)

    def load_window(self, scan_id, frame_indices, window_id=-1):
        if scan_id not in self.scans:
            raise ValueError("Cannot load a scan outside the selected split")
        scan = self.scans[scan_id]
        indices = np.asarray(frame_indices)
        if (indices.shape != (10,) or indices.dtype.kind not in "iu" or indices.min() < 0
                or indices.max() >= len(scan["eligible"]) or np.any(np.diff(indices) <= 0)
                or np.any(np.diff(np.asarray(scan["source_indices"])[indices]) > 3)
                or not np.asarray(scan["eligible"])[indices].all()):
            raise ValueError("Expected ten increasing eligible source rows with maximum gap three")
        local, global_ = [], []
        lx, ly, lw, lh = self.preprocessing["local_crop_xywh"]
        gx, gy, gw, gh = self.preprocessing["global_roi_xywh"]
        for index in indices:
            image = _image(_safe_path(self.source_root, scan["image_paths"][int(index)]))
            if list(image.shape[::-1]) != scan["image_size_xy"]:
                raise ValueError("Original JPEG dimensions changed")
            local.append(image[ly:ly + lh, lx:lx + lw])
            global_.append(cv2.resize(image[gy:gy + gh, gx:gx + gw], (224, 224), interpolation=cv2.INTER_AREA))
        def tensor(images):
            return torch.from_numpy(np.stack(images).copy()).unsqueeze(1).float().div_(255)
        return {"local_encoder_images": tensor(local), "global_encoder_images": tensor(global_),
                "positions_yz": torch.tensor(np.asarray(scan["positions_yz"])[indices], dtype=torch.float32),
                "source_indices": torch.tensor(np.asarray(scan["source_indices"])[indices], dtype=torch.long),
                "scan_id": scan_id, "window_id": int(window_id)}
