from argparse import ArgumentParser
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd


def build_pixel_to_image(width: int, height: int, spacing_x: float, spacing_y: float) -> np.ndarray:
    """Build pixel->image transform with image origin at center in mm."""
    tx = (-width / 2 - 0.5) * spacing_x
    ty = (-height / 2 - 0.5) * spacing_y
    return np.array(
        [
            [spacing_x, 0.0, 0.0, tx],
            [0.0, spacing_y, 0.0, ty],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        out = img
    elif img.ndim == 3:
        # Simple luminance conversion for RGB or RGBA-like arrays.
        out = img[..., :3].mean(axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def main() -> None:
    parser = ArgumentParser(description="Convert a folder of screenshot frames to DualTrack .h5 format")
    parser.add_argument("--input_dir", required=True, help="Directory containing ordered frame images")
    parser.add_argument("--output_dir", required=True, help="Where to write output .h5 and metadata.csv")
    parser.add_argument("--sweep_id", default="screen-captures", help="Sweep ID name")
    parser.add_argument("--split", default="val", help="Dataset split label")
    parser.add_argument("--spacing_x", type=float, default=0.037, help="Pixel spacing in mm (x)")
    parser.add_argument("--spacing_y", type=float, default=0.037, help="Pixel spacing in mm (y)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )
    if not frame_paths:
        raise ValueError(f"No image frames found in: {input_dir}")

    frames = [to_grayscale_uint8(iio.imread(p)) for p in frame_paths]
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"All frames must have the same shape, got: {sorted(shapes)}")

    images = np.stack(frames, axis=0)
    n_frames, height, width = images.shape

    tracking = np.repeat(np.eye(4, dtype=np.float32)[None, ...], n_frames, axis=0)
    spacing = np.array([args.spacing_x, args.spacing_y, 1.0], dtype=np.float32)
    dimensions = np.array([width, height, 1], dtype=np.float32)
    pixel_to_image = build_pixel_to_image(width, height, args.spacing_x, args.spacing_y)

    sweep_path = output_dir / f"{args.sweep_id}.h5"
    with h5py.File(sweep_path, "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("tracking", data=tracking, compression="gzip")
        f.create_dataset("spacing", data=spacing)
        f.create_dataset("dimensions", data=dimensions)
        f.create_dataset("pixel_to_image", data=pixel_to_image)

    metadata = pd.DataFrame(
        [
            {
                "sweep_id": args.sweep_id,
                "processed_sweep_path": str(sweep_path.resolve()),
                "split": args.split,
            }
        ]
    )
    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path)

    print(f"Wrote {sweep_path}")
    print(f"Wrote {metadata_path}")
    print(f"Frames: {n_frames}, shape: {height}x{width}")


if __name__ == "__main__":
    main()
