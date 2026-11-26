"""Small source fixtures verify indexing, frozen splits and original pixels."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import torch

from src.datasets.mdpi_original import (
    DEFAULT_PREPROCESSING, MDPIOriginalDataset, file_sha256, prepare_dataset,
    validate_manifest,
)


def source_fixture(root):
    source = root / "original"
    reference = root / "reference.h5"
    strings = h5py.string_dtype("utf-8")
    spec = copy.deepcopy(DEFAULT_PREPROCESSING)
    spec.update(local_crop_xywh=[22, 22, 256, 256], global_roi_xywh=[0, 0, 300, 300])
    with h5py.File(reference, "x") as handle:
        handle.attrs["schema_version"] = "mdpi_interpolated_v4"
        handle.attrs["config_json"] = json.dumps({"source_size_xy": [300, 300], "width_mm": 38.5})
        for ordinal, split in enumerate(("train", "val", "test")):
            scan_id = f"{ordinal:04d}"
            relative = f"5mm/V1/scan_{split}"
            folder = source / relative
            (folder / "Images").mkdir(parents=True)
            random = np.random.default_rng(ordinal)
            for i in range(11):
                image = random.integers(0, 256, size=(300, 300), dtype=np.uint8)
                cv2.imwrite(str(folder / "Images" / f"{i}.jpg"), image)
            (folder / "Images/11.jpg").write_bytes((folder / "Images/10.jpg").read_bytes())
            trajectory = np.stack([np.arange(12) * 0.7, np.arange(12) * -0.2])
            np.save(folder / "coords_interpolated.npy", trajectory)
            log = np.stack([np.arange(1, 13), np.arange(12) * 0.7, np.zeros(12)], axis=1)
            np.savetxt(folder / "data_log.csv", log, delimiter=",")
            (folder / "source.nrrd").write_bytes(f"unique source volume {ordinal}".encode())
            jpg = [folder / "Images" / f"{i}.jpg" for i in range(12)]
            hashes = [file_sha256(path) for path in jpg]
            decoded = [hashlib.sha256(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).tobytes()).hexdigest() for path in jpg]
            group = handle.create_group("scans/" + scan_id)
            for key, value in {"original_source_scan": relative, "split": split,
                               "split_group_id": "dependency_" + scan_id, "object": "5mm", "method": "V1",
                               "acquisition_date": "shared_date", "source_trajectory": relative + "/coords_interpolated.npy",
                               "source_trajectory_sha256": file_sha256(folder / "coords_interpolated.npy"),
                               "source_volume": relative + "/source.nrrd", "source_volume_sha256": file_sha256(folder / "source.nrrd"),
                               "source_log": relative + "/data_log.csv", "source_log_sha256": file_sha256(folder / "data_log.csv")}.items():
                group.attrs[key] = value
            group.create_dataset("source_frame_indices", data=np.arange(12))
            group.create_dataset("original_jpg_sha256", data=np.array(hashes, dtype=strings))
            group.create_dataset("decoded_jpg_sha256", data=np.array(decoded, dtype=strings))
            group.create_dataset("geometry_valid", data=[True] * 10 + [False] * 2)
            group.create_dataset("geometry_reason", data=np.array([""] * 10 + ["duplicate_decoded_image"] * 2, dtype=strings))
            group.create_dataset("reference_trajectory_xy", data=trajectory.T)
            # A broken link proves preparation never dereferences resized frames.
            group["frames"] = h5py.ExternalLink("does_not_exist.h5", "/frames")
            samples = handle.create_group("samples/" + split)
            samples.create_dataset("scan_ids", data=np.array([scan_id], dtype=strings))
            samples.create_dataset("frame_indices", data=np.arange(10)[None])
    return source, reference, spec


class OriginalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.reference, self.spec = source_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return prepare_dataset(self.source, self.reference, self.root / "prepared", self.spec)

    def test_original_pixels_labels_and_frozen_windows(self):
        path, report = self.prepare()
        self.assertFalse(report["reference_images_read"])
        self.assertEqual(report["summary"]["train"]["windows"], 1)
        self.assertEqual(report["summary"]["train"]["eligible_frames"], 10)
        dataset = MDPIOriginalDataset(path)
        sample = dataset[0]
        self.assertEqual(sample["local_encoder_images"].shape, (10, 1, 256, 256))
        self.assertEqual(sample["global_encoder_images"].shape, (10, 1, 224, 224))
        self.assertEqual(sample["source_indices"].tolist(), list(range(10)))
        torch.testing.assert_close(sample["positions_yz"][:, 0], torch.arange(10) * 0.2)
        torch.testing.assert_close(sample["positions_yz"][:, 1], torch.arange(10) * 0.7)
        original = cv2.imread(str(self.source / "5mm/V1/scan_train/Images/0.jpg"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(sample["local_encoder_images"][0, 0].numpy(), original[22:278, 22:278].astype(np.float32) / 255)
        self.assertEqual(dataset.manifest_sha256, report["manifest_sha256"])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_wrong_source_hash_is_rejected(self):
        path, _ = self.prepare()
        jpg = self.source / "5mm/V1/scan_train/Images/0.jpg"
        jpg.write_bytes(jpg.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "JPEG changed"):
            MDPIOriginalDataset(path)

    def test_test_opt_in_precedes_file_read_and_train_verification_is_scoped(self):
        with self.assertRaisesRegex(ValueError, "explicit allow_test"):
            MDPIOriginalDataset(self.root / "missing.json", split="test")
        path, _ = self.prepare()
        (self.source / "5mm/V1/scan_test/Images/0.jpg").unlink()
        self.assertEqual(len(MDPIOriginalDataset(path, split="train")), 1)
        with self.assertRaises(FileNotFoundError):
            MDPIOriginalDataset(path, split="test", allow_test=True)

    def test_dependencies_and_frozen_window_policy_are_enforced(self):
        path, _ = self.prepare()
        manifest = json.loads(path.read_text())
        changed = copy.deepcopy(manifest)
        changed["scans"]["0001"]["split_group_id"] = changed["scans"]["0000"]["split_group_id"]
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["scans"]["0001"]["decoded_sha256"][0] = changed["scans"]["0000"]["decoded_sha256"][0]
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["windows"]["train"].append(changed["windows"]["train"][0])
        with self.assertRaisesRegex(ValueError, "frozen nonoverlap"):
            validate_manifest(changed)

    def test_reference_hash_and_crop_errors_fail_before_output_publish(self):
        with h5py.File(self.reference, "r+") as handle:
            handle["scans/0000"].attrs["source_trajectory_sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "trajectory hash"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_load_window_rejects_invalid_or_cross_split_rows(self):
        path, _ = self.prepare()
        dataset = MDPIOriginalDataset(path)
        with self.assertRaisesRegex(ValueError, "outside the selected split"):
            dataset.load_window("0001", list(range(10)))
        with self.assertRaisesRegex(ValueError, "eligible"):
            dataset.load_window("0000", list(range(2, 12)))
        with self.assertRaisesRegex(ValueError, "eligible"):
            dataset.load_window("0000", [0] * 10)


if __name__ == "__main__":
    unittest.main()
