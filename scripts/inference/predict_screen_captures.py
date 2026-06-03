from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from src.submission.predictor import get_predictor


def main() -> None:
    parser = ArgumentParser(description="Run DualTrack inference on one converted H5 sweep")
    parser.add_argument("--config", required=True, help="Predictor config path")
    parser.add_argument("--input_h5", required=True, help="Input sweep .h5 with key 'images'")
    parser.add_argument("--output_h5", required=True, help="Output .h5 to save predictions")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    predictor = get_predictor(cfg)

    input_h5 = Path(args.input_h5)
    output_h5 = Path(args.output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as f:
        images = f["images"][:]
        spacing = f["spacing"][:] if "spacing" in f else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        dimensions = f["dimensions"][:] if "dimensions" in f else np.array([images.shape[2], images.shape[1], 1], dtype=np.float32)
        pixel_to_image = f["pixel_to_image"][:] if "pixel_to_image" in f else np.eye(4, dtype=np.float32)

    predictor.run_inference(images)

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("spacing", data=spacing)
        f.create_dataset("dimensions", data=dimensions)
        f.create_dataset("pixel_to_image", data=pixel_to_image)
        f.create_dataset("pred_tracking_glob", data=predictor.pred_tracking_matrices_glob, compression="gzip")
        f.create_dataset("pred_tracking_loc", data=predictor.pred_tracking_matrices_loc, compression="gzip")

    print(f"Saved prediction file: {output_h5}")
    print(f"pred_tracking_glob shape: {predictor.pred_tracking_matrices_glob.shape}")
    print(f"pred_tracking_loc shape: {predictor.pred_tracking_matrices_loc.shape}")


if __name__ == "__main__":
    main()
